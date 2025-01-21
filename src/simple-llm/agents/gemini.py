from agent import Agent, AsyncAgent

from abc import ABC
from google.generativeai import GenerativeModel, GenerationConfig, configure

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
        super().__init__("gemini", name, model, system_message, stream, api_key, default_params)
        self.default_params = default_params
        configure(api_key=self.api_key)
        self.llm_config = GenerationConfig(**default_params)
        self.model = GenerativeModel(
            model_name=model, system_instruction=system_message,
            generation_config=self.llm_config
        )
        self.chat_client = self.model.start_chat(history=self.messages)
    
    def add_user_message(self, query):
        self._messages.append({"role": "user", "parts": query})

    def add_agent_message(self, response: str) -> None:
        self._messages.append({"role": "model", "parts": response})

    def get_config(self, new_params: dict) -> GenerationConfig:
        return GenerationConfig(**new_params) if new_params else self.llm_config

    def nostream_reply(self, query, **kwargs):
        response = self.chat_client.send_message(query, generation_config=self.get_config(kwargs))
        return response.text
    
    def stream_reply(self, query, **kwargs):
        stream = self.chat_client.send_message(query, generation_config=self.get_config(kwargs))
        response = ""
        for chunk in stream:
            delta = chunk.text
            if delta:
                yield delta
                response += delta
                continue