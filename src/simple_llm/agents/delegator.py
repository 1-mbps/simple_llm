from __future__ import annotations # Allows usage of Qdrant and ChromaDB classes in type hints, even if they're not installed
from .universal import UniversalAgent
from ..agent import Agent
# from ..embeddings.openai import openai_embedding

import math
from typing import Type, TypeVar, Callable
from uuid import uuid4

errors = 0

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
except Exception:
    errors += 1

try:
    from chromadb import Collection
except Exception:
    errors += 1

if errors == 2:
    raise ImportError("Both qdrant_client and chromadb are not installed. Please install at least one to use the Delegator agent.")

T = TypeVar('T', bound=Agent)

class DelegatorClient:
    def __init__(
        self,
        vector_client: Collection | QdrantClient,
        collection_name: str,
        embed_func: Callable[[str, str], list[float]] | Callable,
        embed_model: str = "text-embedding-3-small"
    ):
        self.vector_client = vector_client
        self.collection_name = collection_name
        self.embed_func = embed_func
        self.embed_model = embed_model

    def insert_vectors(self, vectors: list[list[float]], categories: list[str]) -> None:
        if isinstance(self.vector_client, QdrantClient):
            self.vector_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid4()), vector = vectors[i], payload = {"category": categories[i]}
                    ) for i in range(len(vectors))
                ]
            )
        elif isinstance(self.vector_client, Collection):
            self.vector_client.upsert(
                ids = [str(uuid4()) for _ in range(len(vectors))],
                embeddings = vectors,
                metadatas = [{"category": categories[i]} for i in range(len(categories))]
            )

    def build_collection(self, prompts: list[str], categories: list[str]) -> None:
        if len(prompts) != len(categories):
            raise ValueError("Length of prompts and categories must be equal.")
        vectors = [self.embed_func(self.embed_model, prompt) for prompt in prompts]
        self.insert_vectors(vectors, categories)

    def scale_cos_sim(self, cosine: float) -> float:
        """
        Converts cosine similarities to a scale where one item with high cosine similarity
        dominates over several items with lower cosine similarities.

        Graph: https://www.desmos.com/calculator/soopm7c3es
        Any item with a cosine similarity greater than roughly 0.75 will have a scaled value
        greater than the sum of the scaled values of 4 items whose original cosine similarities
        are half as high.

        Notice how on the graph, g(x) is below the x-axis for x < 0.75.
        """
        c = 1.5
        a = 2.5
        b = 0.124
        return ((cosine*c)**a) + b

    def query(self, prompt: str, k: int = 5) -> dict[str, float]:
        query_vector = self.embed_func(self.embed_model, prompt)
        if isinstance(self.vector_client, QdrantClient):
            hits = self.vector_client.search(collection_name = self.collection_name, query_vector=query_vector, limit=k)
            similarities = [hit.score for hit in hits]
            categories = [hit.payload["category"] for hit in hits]
        elif isinstance(self.vector_client, Collection):
            hits = self.vector_client.query(query_embeddings=query_vector, n_results=k)

            # Keep hits whose cosine similarity >= 0
            # Chroma uses distance = 1 - cosine similarity, so the actual cossim is 1-dist
            similarities = [1-dist for dist in hits["distances"][0] if dist <= 1]
            
            # Extract category of each hit
            categories = [meta["category"] for meta in hits["metadatas"][0]]

        category_scores = {}

        for i in range(len(categories)):
            # scale the cosine similarities to make sure 1 really good result is (roughly)
            # better than 4 results that are each half as good.
            sim = self.scale_cos_sim(similarities[i])
            if categories[i] not in category_scores:
                category_scores[categories[i]] = sim
            else:
                category_scores[categories[i]] += sim

        return category_scores
        
    def normalize_cosine(self, cosine: float) -> float:
        """
        Convert cosine similarity to a scale between 0 and 1.
        """
        return math.acos(-cosine) / math.pi

class Delegator(UniversalAgent):
    def __init__(
        self,
        agent_cls: Type[T],
        model: str,
        system_message: str,
        delegator_client: DelegatorClient,
    ) -> T:
        super().__init__(agent_cls, name="assistant", model=model, system_message=system_message)
        self.delegator_client = delegator_client

    def linearize_logprob(self, raw_logprob: float, round_to: int = 8) -> float:
        """
        Convert OpenAI's log probability to a linear scale between 0 and 1.
        """
        return round(math.exp(raw_logprob), round_to)
    
    def delegate(self, query: str, number_bias: dict[str, float] = {}, print_scores: bool = False) -> str | int:
        self.add_user_message(query)

        llm_response = self.completion(self._messages, False, logprobs=True, top_logprobs=5)

        # this adds the LLM response to the message list
        self.process_completion(llm_response)

        logprobs = self.get_logprobs(llm_response)

        knn_scores = self.k_nearest_prompts(query)
        max_score = 0
        delegated_category = None

        for logtoken in logprobs:
            category = logtoken.token

            prob = self.linearize_logprob(logtoken.logprob)

            if category in number_bias and prob >= number_bias[category]:
                return category

            knn_score = knn_scores.get(category, 0)

            weighted_score = self.score(prob, knn_score)
            
            if print_scores:
                print(f"Category: {category}, Weighted Score: {weighted_score}      Logprob: {prob}      kNN Score: {knn_score}")
            
            if weighted_score > max_score:
                max_score = weighted_score
                delegated_category = category

        return delegated_category

    def score(self, prob: float, knn_score: float, logprob_weight: float = 2.2, knn_weight: float = 1) -> float:
        """
        Compute the final score for a category based on the LLM's log probability
        and the kNN / vector search score.
        """
        return (logprob_weight*prob + knn_weight*knn_score) / (logprob_weight + knn_weight)
    
    def k_nearest_prompts(self, prompt: str, k: int = 5):
        return self.delegator_client.query(prompt, k)





