from openai import OpenAI

def openai_embedding(model: str | list[str], prompt: str, dimensions: int = 768, api_key: str = None) -> list[float]:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=prompt, dimensions=dimensions)
    return response.data[0].embedding