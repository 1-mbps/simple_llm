from ..agent import Agent, AsyncAgent
from ..tools.models import Tool

from abc import ABC
from typing import Optional, AsyncGenerator
import json

try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat.chat_completion import ChatCompletion
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
        tools: list[Tool],
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
        self._messages.append({"role": "assistant", "content": response})

    @property
    def last_response(self) -> str:
        if not self._messages:
            raise ValueError("No messages have been sent yet.")
        last_msg = self.messages[-1]
        if last_msg["role"] != "assistant":
            raise ValueError("The last message was not from the assistant.")
        return self._messages[-1]["content"]

    def completion(self, messages: list[dict], stream: bool = False, **kwargs):
        kwargs["model"] = kwargs.get("model", self.model)
        kwargs["messages"] = messages
        kwargs["stream"] = stream
        kwargs["tools"] = self.tools
        if stream:
            kwargs["stream_options"] = {"include_usage": self.track_usage}
        return self.client.chat.completions.create(**kwargs)
    
    def process_completion(self, completion: ChatCompletion):
        text = completion.choices[0].message.content
        tool_calls = completion.choices[0].message.tool_calls
        output_calls = []
        tool_outputs = []
        completion = {"role": "assistant", "content": text}

        # clean up tool calls to match OpenAI API format,
        # and call tools at the same time
        if tool_calls:
            for tool_call in tool_calls:
                output_calls.append(tool_call)
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
                fxn = self.tool_matrix[name]
                output = fxn(**args)
                tool_output = {"role": "tool", "tool_call_id": tool_id, "content": output}
                tool_outputs.append(tool_output)

            completion["tool_calls"] = output_calls

        self._messages.append(completion)
        self._messages += tool_outputs

        if tool_outputs:
            response = self.completion(self._messages, False)
            return self.process_completion(response)
        else:
            return text
    
    def process_chunk(self, chunk):
        if chunk.choices and chunk.choices[0].delta.content is not None:
            return chunk.choices[0].delta.content
        
    def get_logprobs(self, completion):
        return completion.choices[0].logprobs.content[0].top_logprobs
    
    def process_stream(self, stream: AsyncGenerator):
        tool_calls = {}
        output_calls = []
        tool_outputs = []
        chunk_acc = {"role": "assistant"}
        text = ""

        for chunk in stream:
            if chunk.choices:
                self.accumulate_chunks(chunk, chunk_acc, tool_calls)
            delta = self.process_chunk(chunk)
            if delta:
                text += delta
                yield delta

        if text:
            chunk_acc["content"] = text

        # clean up tool calls to match OpenAI API format,
        # and call tools at the same time
        if tool_calls:
            for _, tool_call in tool_calls.items():
                output_calls.append(tool_call)
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
                fxn = self.tool_matrix[name]
                output = fxn(**args)
                tool_output = {"role": "tool", "tool_call_id": tool_id, "content": output}
                tool_outputs.append(tool_output)

            chunk_acc["tool_calls"] = output_calls

        self._messages.append(chunk_acc)
        self._messages += tool_outputs

        if tool_outputs:
            stream = self.completion(self._messages, True)
            yield from self.process_stream(stream)
    
    def accumulate_chunks(self, chunk, chunk_acc: dict, tool_call_dict: dict) -> None:
        delta = self.process_chunk(chunk)

        if delta:
            if chunk_acc.get("content"):
                chunk_acc["content"] += delta
            else:
                chunk_acc["content"] = delta

        for tool_call in chunk.choices[0].delta.tool_calls or []:
            index = tool_call.index

            if index not in tool_call_dict:
                tool_call_dict[index] = tool_call

            tool_call_dict[index].function.arguments += tool_call.function.arguments

class OpenAIAgent(BaseOpenAIAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_msgs: bool = True, track_usage: bool = True, tools: list[Tool] = [], default_params: dict = {}, api_key = None):
        client = OpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_msgs, track_usage, tools, default_params, api_key)

class AsyncOpenAIAgent(BaseOpenAIAgent, AsyncAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_msgs: bool = True, track_usage: bool = True, tools: list[Tool] = [], default_params: dict = {}, api_key = None):
        client = AsyncOpenAI(api_key=api_key)
        super().__init__(name, model, client, system_message, stream, track_msgs, track_usage, tools, default_params, api_key)


def create_openai_like_agent(class_name: str, base_url: str):
    """
    Dynamically creates a class that inherits from BaseOpenAIAgent.
    This allows for the creation of agents that are compatible with the OpenAI standard.
    """

    def __init__(
        self,
        name: str,
        model: str,
        system_message: str,
        stream: bool = False,
        track_msgs: bool = True,
        track_usage: bool = True,
        tools: list[Tool] = [],
        default_params: dict = {},
        api_key: Optional[str] = None,
    ):
        client = OpenAI(base_url=base_url, api_key=api_key)
        BaseOpenAIAgent.__init__(self, name, model, client, system_message, stream, track_msgs, track_usage, tools, default_params)
        
    return type(class_name, (BaseOpenAIAgent,), {"__init__": __init__})