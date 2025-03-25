from ..agent import Agent
from ..tools.models import Tool, ToolCall
from ..tools.toolcalls import get_function_schema, capitalize_type_values

import os
from typing import Callable, Iterator
try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install simple-llm[gemini] to use the Gemini agents.")

class GeminiAgent(Agent):
    def __init__(
        self,
        name: str,
        model: str,
        system_message: str,
        stream: bool = False,
        tools: list[Tool] = [],
        default_params: dict = {},
        api_key: str = None,
        client_args: dict = {}
    ):

        # Initialize client object
        try:
            client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"), **client_args)
        except Exception as e:
            raise Exception(f"Error initializing Gemini client: {e}")

        # Initialize base agent object
        super().__init__("gemini", name, model, client, system_message, stream, api_key, tools, default_params, track_msgs=False)

        # Set default parameters for Gemini model
        self.llm_config = types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=system_message)], 
            tools=[types.Tool(function_declarations=self.tools)],
            **default_params
        )

    def add_user_message(self, query = None, tool_call: ToolCall = None):
        parts = []
        if query:
            parts.append(types.Part.from_text(text=query))
        if tool_call:
            parts.append(types.Part.from_function_call(name=tool_call.name, args=tool_call.args))

        self._messages.append(types.Content(role="user", parts=parts))

    def add_agent_message(self, response: str) -> None:
        self._messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response)]))

    @property
    def last_response(self) -> str:
        if not self._messages:
            raise ValueError("No messages have been sent yet.")
        last_msg = self.messages[-1]
        if last_msg.role != "model":
            raise ValueError("The last message was not from the model.")
        return last_msg.text

    def _add_agent_content(self, completion: types.GenerateContentResponse):
        self._messages.append(completion.candidates[0].content)

    def get_config(self, new_params: dict) -> types.GenerationConfig:
        return types.GenerateContentConfig(**new_params) if new_params else self.llm_config
        
    def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs):
        kwargs["model"] = kwargs.get("model", self.model)
        kwargs["contents"] = messages
        kwargs["config"] = kwargs.get("config", self.llm_config)
        if stream:
            return self.client.models.generate_content_stream(**kwargs)
        else:
            return self.client.models.generate_content(**kwargs)
        
    def process_function_calls(self, fcalls: list[types.FunctionCall]) -> None:
        """
        Produces a list of tool outputs as Part objects, and adds these to the messages list
        """
        tool_outputs = []
        for fcall in fcalls:
            # Get function name and arguments
            name = fcall.name
            args = fcall.args

            # Retrieve function from tool matrix
            fxn = self.tool_matrix[name]

            # Initialize output dictionary - Gemini API has fields for
            # "output" and "error". See FunctionResponse definition in google.genai.types
            output_dict = {}

            try:
                output = fxn(**args)
                output_dict["output"] = output
            except Exception as e:
                output_dict["error"] = str(e)

            # add each tool output as a Part object
            tool_outputs.append(types.Part.from_function_response(name=name, response=output_dict))

        # add tool outputs to the messages list
        self._messages.append(types.Content(role="user", parts=tool_outputs))

        return tool_outputs
    
    def process_completion(self, completion: types.GenerateContentResponse):
        self._add_agent_content(completion)
        fcalls = completion.function_calls
        
        if fcalls:
            # Process function calls and add tool outputs to messages list
            self.process_function_calls(fcalls)
            
            # Call completion function again if there are tool outputs
            # So the model can summarize the tool outputs and return a proper response to the user
            completion = self.completion(self._messages, False)
            return self.process_completion(completion)
        else:
            # base recursive case - if no tool outputs, return completion text
            return completion.text
    
    def process_chunk(self, chunk: types.GenerateContentResponse):
        return chunk.text
    
    def process_stream(self, stream: Iterator[types.GenerateContentResponse]):
        response = ""
        last_chunk = None
        parts = []
        fcalls = []

        for chunk in stream:
            if chunk and chunk.candidates:
                last_chunk = chunk
                text = chunk.text
                if text:
                    # accumulate text - all text chunks will be concatenated and saved to a single Part object
                    response += text
                if chunk.function_calls:
                    # keep track of all function calls
                    fcalls += chunk.function_calls
                for part in chunk.candidates[0].content.parts:
                    # all non-text content gets its own Part object
                    if not part.text:
                        parts.append(part)
                delta = self.process_chunk(chunk)
                if delta:
                    yield delta

        # save all text to a single Part object
        if response:
            parts.insert(0, types.Part.from_text(text=response))

        self.process_last_chunk(last_chunk)
        
        # add all parts to the messages list
        self._messages.append(types.Content(role="model", parts=parts))

        # if there are function calls, call the functions, get their outputs, and call
        # the LLM again to summarize the function outputs
        if fcalls:
            self.process_function_calls(fcalls)
            stream = self.completion(self._messages, True)
            yield from self.process_stream(stream)

    def process_last_chunk(self, chunk: types.GenerateContentResponse):
        pass
    
    def get_logprobs(self, completion: types.GenerateContentResponse):
        pass

    def add_tool(self, tool: Callable, name: str, description: str) -> None:
        func_schema = types.FunctionDeclaration.from_callable(client=self.client, callable=tool)
        func_schema.name = name
        func_schema.description = description
        self.tools.append(func_schema)
        self.update_tool_matrix(tool, name)

    def add_tool_dict(self, tool: Callable, name: str, tool_dict: dict) -> None:
        self.tools.append(types.FunctionDeclaration(**tool_dict))
        self.update_tool_matrix(tool, name)
