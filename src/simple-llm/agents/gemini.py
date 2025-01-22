from agent import Agent, AsyncAgent

from abc import ABC
from google.generativeai import GenerativeModel, GenerationConfig, configure
from google.generativeai.types.generation_types import GenerateContentResponse

class GeminiAgent(Agent, ABC):
    def __init__(
        self,
        name: str,
        model: str,
        system_message: str,
        stream: bool = False,
        default_params: dict = {},
        api_key: str = None
    ):
        
        # Set default parameters for Gemini model
        self.llm_config = GenerationConfig(**default_params)

        # Initialize model
        self.gemini_model = GenerativeModel(
            model_name=model, system_instruction=system_message,
            generation_config=self.llm_config
        )

        # Initialize chat object. This saves and responds to messages
        client = self.gemini_model.start_chat(history=[])

        # Initialize base agent object
        super().__init__("gemini", name, model, client, system_message, stream, api_key, default_params, track_msgs=False)

        # Make self._messages point to client's message list
        self.set_messages_pointer(client.history)

        self.default_params = default_params

        # Configure Gemini API key
        configure(api_key=self.api_key)
    
    def add_user_message(self, query):
        self._messages.append({"role": "user", "parts": query})

    def add_agent_message(self, response: str) -> None:
        self._messages.append({"role": "model", "parts": response})

    def get_config(self, new_params: dict) -> GenerationConfig:
        return GenerationConfig(**new_params) if new_params else self.llm_config
    
    def get_completion_function(self):
        return getattr(self.client, "send_message")
    
    def get_completion_args(self, stream: bool, query: str, messages: list[dict] = None, **kwargs) -> dict:
        kwargs["content"] = query
        kwargs["stream"] = stream
        return kwargs
    
    def process_completion(self, completion: GenerateContentResponse):
        return completion.text
    
    def process_chunk(self, chunk: GenerateContentResponse):
        return chunk.text

    # def nostream_reply(self, query, **kwargs):
    #     response = self.client.send_message(query, generation_config=self.get_config(kwargs))
    #     return response.text
    
    # def stream_reply(self, query, **kwargs):
    #     stream = self.client.send_message(query, generation_config=self.get_config(kwargs))
    #     response = ""
    #     for chunk in stream:
    #         delta = chunk.text
    #         if delta:
    #             yield delta
    #             response += delta
    #             continue