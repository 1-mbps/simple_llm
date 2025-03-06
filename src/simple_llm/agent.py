from __future__ import annotations
from .printing import print_role, print_dashed_line, print_message

from abc import ABC, abstractmethod
import os
from typing import Any, List, Dict, AsyncGenerator, Generator, Callable, TYPE_CHECKING
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    try:
        from openai import OpenAI, AsyncOpenAI
    except Exception:
        raise ImportError("Run `pip install simple_llm[openai]` to use OpenAI agents.")
    try:
        from google.genai import Client
    except Exception:
        raise ImportError("Run `pip install simple_llm[gemini]` to use Gemini agents.")

class Agent(ABC):
    def __init__(
        self,
        api_type: str,
        name: str,
        model: str,
        client: OpenAI | AsyncOpenAI | Client,
        system_message: str,
        stream: bool,
        api_key: str | None,
        tools: list = [],
        default_params: dict = {},
        track_msgs: bool = True
    ):
        key = api_key or os.getenv(f"{api_type.upper()}_API_KEY")
        if not key:
            raise KeyError(f"{api_type} API key not found. Either pass `api_key` as an argument to this class or set {api_type.upper()}_API_KEY as an environment variable.")
        self.api_key = key
        self.name = name
        self.model = model
        self.client = client
        self.api_type = api_type
        self.system_message = system_message
        self.stream = stream
        self.default_params = default_params
        self.track_msgs = track_msgs

        self._messages: List[Dict[str, str]] = []
        self.completion_fn = self.get_completion_function()

        for tool in tools:
            self.add_tool(tool)

    @abstractmethod
    def add_user_message(self, query: str) -> None:
        pass

    @abstractmethod
    def add_agent_message(self, response: str) -> None:
        pass

    def reply(self, query: str, **kwargs) -> str | Generator:
        return self.stream_reply(query, **kwargs) if self.stream else self.nostream_reply(query, **kwargs)
    
    def nostream_reply(self, query: str, **kwargs) -> str:
        self.add_user_message(query)

        # Unpack completion arguments and feed them into completion function
        completion = self.completion(self._messages, False, **kwargs)
        response = self.process_completion(completion)
        
        if self.track_msgs:
            self.add_agent_message(response)

        return response

    def stream_reply(self, query: str, **kwargs) -> Generator[str]:
        self.add_user_message(query)

        # Unpack completion arguments and feed them into completion function
        stream = self.completion(self._messages, True, **kwargs)
        
        return self.process_stream(stream)
    
    def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs) -> Any:
        return self.completion_fn(**self.get_completion_args(stream, messages=messages, **kwargs))

    @abstractmethod
    def get_completion_function(self) -> Callable:
        """
        Override this to return the client's completion method, like OpenAI().chat.completions.create()
        """
        pass

    @abstractmethod
    def get_completion_args(self, stream: bool, messages: list[dict] = None, query: str = None, **kwargs) -> dict:
        """
        Return a dictionary containing arguments passed to the completion function
        """
        pass

    def process_stream(self, stream: AsyncGenerator):
        response = ""
        for chunk in stream:
            delta = self.process_chunk(chunk)
            if delta:
                yield delta
                response += delta
        if self.track_msgs:
            self.add_agent_message(response)

    @abstractmethod
    def process_completion(self, completion) -> str:
        """
        Extract and return string content from completion and do any additional logic, if required
        """
        pass

    @abstractmethod
    def process_chunk(self, chunk) -> str:
        """
        Extract and return string content from chunk and do any additional logic, if required
        """
        pass

    @abstractmethod
    def add_tool(self, tool):
        pass
    
    @abstractmethod
    def get_logprobs(self, completion) -> list:
        pass
    
    def start_chat(self, init_message: str = None, stream: bool = False, **kwargs) -> None:
        """
        For user-agent conversations that get printed to the console.
        """
        if not init_message:
            msg = input("Enter initial message for agent. Type 'exit' to end the conversation: ")
        else:
            msg = init_message

        while msg != "exit":
            
            # Print initial user message
            print_message("user", msg)

            if stream:
                # Print name of agent right before agent message
                print_role(self.name)

                # Initiate streaming of tokens
                stream = self.stream_reply(msg, **kwargs)
                
                # Print each token
                for chunk in stream:
                    print(chunk, end="")

                # Print dashed line at the end of the message
                print("\n")
                print_dashed_line()

            else:
                response = self.nostream_reply(msg, **kwargs)
                print_message(self.name, response)

            msg = input("Enter message for agent. Type 'exit' to end the conversation: ")
            
    def update_model(self, new_model: str) -> None:
        self.model = new_model

    def update_system_message(self, new_system_message: str) -> None:
        self.system_message = new_system_message

    @property
    def messages(self):
        return self._messages
    
    def set_messages_pointer(self, msg_list: List) -> None:
        self._messages = msg_list

    async def sse_stream(self, query: str, chunk_event_name: str = "delta", **kwargs) -> AsyncGenerator[str]:
        """
        Stream output as server-sent events
        """
        yield "event: start\ndata: \n\n"
        for chunk in self.stream_reply(query, **kwargs):
            yield f"event: {chunk_event_name}\ndata: {chunk}\n\n"
        yield "event: done\ndata: \n\n"

    def fastapi_stream(self, query: str, sse: bool = True, chunk_event_name: str = "delta", **kwargs) -> StreamingResponse:
        """
        Streams agent responses as a FastAPI StreamingResponse type.

        Args:
            query (str): user's query, to be sent to the agent
            sse (bool): whether to stream as server-sent events
            chunk_event_name (str): the event name of each streamed chunk, if server-sent events are used
        """
        if sse:
            return StreamingResponse(content=self.sse_stream(query, chunk_event_name, **kwargs), media_type="text/event-stream")
        else:
            return StreamingResponse(content=self.stream_reply(query, **kwargs), media_type="text/event-stream")

class AsyncAgent(Agent):
    def __init__(
        self,
        api_type: str,
        name: str,
        model: str,
        client: OpenAI | AsyncOpenAI | Client,
        system_message: str,
        stream: bool,
        api_key: str | None,
        tools: list = [],
        default_params: dict = {},
        track_msgs: bool = True
    ):
        super().__init__(api_type, name, model, client, system_message, stream, api_key, tools, default_params, track_msgs)

    async def reply(self, query: str, **kwargs) -> str | AsyncGenerator:
        return await self.stream_reply(query, **kwargs) if self.stream else await self.nostream_reply(query, **kwargs)

    async def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs) -> Any:
        return await self.completion_fn(**self.get_completion_args(stream, messages=messages, **kwargs))

    async def nostream_reply(self, query: str, **kwargs):
        self.add_user_message(query)

        # Unpack completion arguments and feed them into completion function
        completion = await self.completion(self._messages, False, **kwargs)
        response = self.process_completion(completion)

        if self.track_msgs:
            self.add_agent_message(response)

        return response

    async def stream_reply(self, query: str, **kwargs) -> Generator[str]:
        self.add_user_message(query)

        # Unpack completion arguments and feed them into completion function
        stream = await self.completion(self._messages, True, **kwargs)
        
        return self.process_stream(stream)

    async def process_stream(self, stream: AsyncGenerator) -> AsyncGenerator[str]:
        response = ""
        async for chunk in stream:
            delta = self.process_chunk(chunk)
            if delta:
                yield delta
                response += delta
        if self.track_msgs:
            self.add_agent_message(response)
    
    async def start_chat(self, init_message: str = None, stream: bool = False, **kwargs) -> None:
        """
        For user-agent conversations that get printed to the console.
        """
        if not init_message:
            msg = input("Enter initial message for agent. Type 'exit' to end the conversation: ")
        else:
            msg = init_message

        while msg != "exit":
            
            # Print initial user message
            print_message("user", msg)

            if stream:
                # Print name of agent right before agent message
                print_role(self.name)

                # Initiate streaming of tokens
                stream = await self.stream_reply(msg, **kwargs)
                
                # Print each token
                async for chunk in stream:
                    print(chunk, end="")

                # Print dashed line at the end of the message
                print("\n")
                print_dashed_line()

            else:
                response = await self.nostream_reply(msg, **kwargs)
                print_message(self.name, response)

            msg = input("Enter message for agent. Type 'exit' to end the conversation: ")

    async def sse_stream(self, query: str, chunk_event_name: str = "delta", **kwargs) -> AsyncGenerator[str]:
        """
        Stream output as server-sent events
        """
        yield "event: start\ndata: \n\n"
        async for chunk in self.stream_reply(query, **kwargs):
            yield f"event: {chunk_event_name}\ndata: {chunk}\n\n"
        yield "event: done\ndata: \n\n"