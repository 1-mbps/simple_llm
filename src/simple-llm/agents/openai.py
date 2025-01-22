from agent import Agent, AsyncAgent

from abc import ABC
from typing import Optional, AsyncGenerator, Generator, Any
from openai import OpenAI, AsyncOpenAI

class BaseOpenAIAgent(Agent, ABC):
    def __init__(
            self,
            name: str,
            model: str,
            client: OpenAI | AsyncOpenAI,
            system_message: str,
            stream: bool,
            track_usage: bool,
            default_params: dict,
            api_key: Optional[str] = None,
        ):
        super().__init__("openai", name, model, client, system_message, stream, api_key, default_params)
        self._messages.append({"role": "system", "content": self.system_message})
        self.track_usage = track_usage

    def add_user_message(self, query: str) -> None:
        self._messages.append({"role": "user", "content": query})

    def add_agent_message(self, response: str) -> None:
        self._messages.append({"role": "user", "content": response})

    def get_completion_function(self):
        return getattr(self.client.chat.completions, "create")
    
    def get_completion_args(self, stream: bool, messages: list[dict], query: str = None, **kwargs) -> dict:
        kwargs["model"] = self.model
        kwargs["messages"] = messages
        kwargs["stream"] = stream
        if stream:
            kwargs["stream_options"] = {"include_usage": self.track_usage}
        return kwargs

class OpenAIAgent(BaseOpenAIAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, default_params: dict = {}, api_key = None):
        client = OpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_usage, default_params, api_key)

    def process_completion(self, completion):
        return completion.choices[0].message.content
    
    def process_chunk(self, chunk):
        if chunk.choices and chunk.choices[0].delta.content is not None:
            return chunk.choices[0].delta.content

class AsyncOpenAIAgent(BaseOpenAIAgent, AsyncAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, default_params: dict = {}, api_key = None):
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_usage, default_params, api_key)



    
