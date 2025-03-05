from ..agent import Agent

import os
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
        tools: list = [],
        default_params: dict = {},
        api_key: str = None,
        client_args: dict = {}
    ):
        
        # Set default parameters for Gemini model
        self.llm_config = types.GenerateContentConfig(
            system_instruction=[types.Part.from_text(text=system_message)], 
            **default_params
        )

        # Initialize client object
        try:
            client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"), **client_args)
        except Exception as e:
            raise Exception(f"Error initializing Gemini client: {e}")

        # Initialize base agent object
        super().__init__("gemini", name, model, client, system_message, stream, api_key, tools, default_params, track_msgs=False)

    def add_user_message(self, query):
        self._messages.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

    def add_agent_message(self, response: str) -> None:
        self._messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response)]))

    def get_config(self, new_params: dict) -> types.GenerationConfig:
        return types.GenerateContentConfig(**new_params) if new_params else self.llm_config
    
    def get_completion_function(self):
        if self.stream:
            return getattr(self.client.models, "generate_content_stream")
        else:
            return getattr(self.client.models, "generate_content")
        
    def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs):
        completion_args = self.get_completion_args(stream, messages, **kwargs)
        if stream:
            return self.client.models.generate_content_stream(**completion_args)
        else:
            return self.client.models.generate_content(**completion_args)
    
    def get_completion_args(self, stream: bool, messages: list[dict] = None, **kwargs) -> dict:
        kwargs["model"] = kwargs.get("model", self.model)
        kwargs["contents"] = messages
        kwargs["config"] = kwargs.get("config", self.llm_config)
        return kwargs
    
    def process_completion(self, completion: types.GenerateContentResponse):
        # print(f"COMPLETION: {completion}")
        return completion.text
    
    def process_chunk(self, chunk: types.GenerateContentResponse):
        return chunk.text
    
    def get_logprobs(self, completion: types.GenerateContentResponse):
        pass
    
    def add_tool(self, tool):
        pass
        # Rita TODO