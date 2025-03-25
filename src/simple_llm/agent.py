from __future__ import annotations
from .printing import print_role, print_dashed_line, print_message
from .tools.models import Tool
from .tools.toolcalls import get_function_schema

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
        tools: list[Tool] = [],
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
        self.tools = []
        self.tool_matrix: Dict[str, Callable] = {}

        for tool in tools:
            self.add_tool(tool.function, tool.name, tool.description, **tool.settings)

    @abstractmethod
    def add_user_message(self, query: str) -> None:
        pass

    @abstractmethod
    def add_agent_message(self, response: str) -> None:
        pass

    def reply(self, query: str, **kwargs) -> str | Generator:
        return self.stream_reply(query, **kwargs) if self.stream else self.nostream_reply(query, **kwargs)
    
    def nostream_reply(self, query: str, **kwargs) -> str:
        self.add_user_message(query=query)

        # Unpack completion arguments and feed them into completion function
        completion = self.completion(self._messages, False, **kwargs)
        return self.process_completion(completion)

    def stream_reply(self, query: str, **kwargs) -> Generator[str]:
        self.add_user_message(query=query)

        # Unpack completion arguments and feed them into completion function
        stream = self.completion(self._messages, True, **kwargs)
        
        return self.process_stream(stream)
    
    @abstractmethod
    def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs) -> Any:
        pass

    @abstractmethod
    def process_stream(self, stream: AsyncGenerator):
        pass

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

    def add_tool(self, tool: Callable, name: str, description: str, **kwargs):
        schema = get_function_schema(tool, name=name, description=description)
        self.tools.append(schema)
        self.update_tool_matrix(tool, name)

    def add_tool_dict(self, tool: Callable, name: str, tool_dict: dict) -> None:
        self.tools.append(tool_dict)
        self.update_tool_matrix(tool, name)

    def update_tool_matrix(self, tool: Callable, name: str) -> None:
        self.tool_matrix[name] = tool
    
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
    
    @property
    @abstractmethod
    def last_response(self) -> str:
        pass

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

    @abstractmethod
    async def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs) -> Any:
        pass

    async def nostream_reply(self, query: str, **kwargs) -> str:
        self.add_user_message(query=query)

        # Unpack completion arguments and feed them into completion function
        completion = await self.completion(self._messages, False, **kwargs)

        return self.process_completion(completion)

    async def stream_reply(self, query: str, **kwargs) -> Generator[str]:
        self.add_user_message(query=query)

        # Unpack completion arguments and feed them into completion function
        stream = await self.completion(self._messages, True, **kwargs)
        
        return self.process_stream(stream)

    @abstractmethod
    async def process_stream(self, stream: AsyncGenerator) -> AsyncGenerator[str]:
        pass
    
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