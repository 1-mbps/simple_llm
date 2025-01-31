from ..agent import Agent, AsyncAgent

from abc import ABC
from typing import Optional

try:
    from openai import OpenAI, AsyncOpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install simple-llm[openai] to use the OpenAI agents.")

class BaseOpenAIAgent(Agent, ABC):
    def __init__(
        self,
        name: str,
        model: str,
        client: OpenAI | AsyncOpenAI,
        system_message: str,
        stream: bool,
        track_msgs: bool,
        track_usage: bool,
        tools: list,
        default_params: dict,
        api_key: Optional[str] = None,
        api_type: str = "openai"
    ):
        super().__init__(api_type, name, model, client, system_message, stream, api_key, tools, default_params, track_msgs)
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
    
    def process_completion(self, completion):
        return completion.choices[0].message.content
    
    def process_chunk(self, chunk):
        if chunk.choices and chunk.choices[0].delta.content is not None:
            return chunk.choices[0].delta.content
    
    def add_tool(self, tool):
        pass
        # Rita TODO

class OpenAIAgent(BaseOpenAIAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_msgs: bool = True, track_usage: bool = True, tools: list = [], default_params: dict = {}, api_key = None):
        client = OpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_msgs, track_usage, tools, default_params, api_key)

class AsyncOpenAIAgent(BaseOpenAIAgent, AsyncAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_msgs: bool = True, track_usage: bool = True, tools: list = [], default_params: dict = {}, api_key = None):
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_msgs, track_usage, tools, default_params, api_key)