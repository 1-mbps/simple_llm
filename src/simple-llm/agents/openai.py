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
        super().__init__("openai", name, model, system_message, stream, api_key, default_params)
        self._messages.append({"role": "system", "content": self.system_message})
        self.client = client
        self.track_usage = track_usage

    def add_user_message(self, query: str) -> None:
        self._messages.append({"role": "user", "content": query})

    def add_agent_message(self, response: str) -> None:
        self._messages.append({"role": "user", "content": response})

class OpenAIAgent(BaseOpenAIAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, default_params: dict = {}, api_key = None):
        client = OpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_usage, default_params, api_key)

    def nostream_reply(self, query: str, **kwargs) -> str:
        self.add_user_message(query)
        params = self.merge_params(kwargs)
        completion = self.client.chat.completions.create( # type: ignore
            messages=self._messages, # type: ignore
            model=self.model,
            **params
        )
        response = completion.choices[0].message.content
        self.add_agent_message(response)
        return response
    
    def stream_reply(self, query: str, **kwargs) -> Generator[str, Any, Any]:
        self.add_user_message(query)
        params = self.merge_params(kwargs)
        stream = self.client.chat.completions.create( # type: ignore
            messages=self._messages, # type: ignore
            model=self.model,
            stream=True,
            stream_options={"include_usage": self.track_usage},
            **params
        )
        response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                yield delta
                response += delta
                continue
        self.add_agent_message(response)

class AsyncOpenAIAgent(BaseOpenAIAgent, AsyncAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, default_params: dict = {}, api_key = None):
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_usage, default_params, api_key)

    async def stream_reply(self, query: str, **kwargs) -> AsyncGenerator[str]:
        self.add_user_message(query)
        params = self.merge_params(kwargs)
        stream = await self.client.chat.completions.create( # type: ignore
            messages=self._messages, # type: ignore
            model=self.model,
            stream=True,
            stream_options={"include_usage": self.track_usage},
            **params
        )
        response = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                yield delta
                response += delta
                continue
            if self.track_usage and chunk.usage:
                print(chunk)
        self.add_agent_message(response)
        

    async def nostream_reply(self, query: str, **kwargs):
        self.add_user_message(query)
        params = self.merge_params(kwargs)
        completion = await self.client.chat.completions.create( # type: ignore
            messages=self._messages, # type: ignore
            model=self.model,
            **params
        )
        response = completion.choices[0].message.content
        self.add_agent_message(response)
        return response



    
