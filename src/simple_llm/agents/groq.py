from ..agent import AsyncAgent
from .openai import BaseOpenAIAgent
from openai import OpenAI, AsyncOpenAI

import os

class GroqAgent(BaseOpenAIAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, tools: list = [], default_params: dict = {}, api_key = None):
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key or os.getenv("GROQ_API_KEY"))
        super().__init__(name, model, client, system_message, stream, track_usage, tools, default_params, api_key, api_type="groq")

class AsyncGroqAgent(BaseOpenAIAgent, AsyncAgent):
    def __init__(self, name: str, model: str, system_message: str, stream: bool = False, track_usage: bool = True, tools: list = [], default_params: dict = {}, api_key = None):
        client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key or os.getenv("GROQ_API_KEY"))
        super().__init__(name, model, client, system_message, stream, track_usage, tools, default_params, api_key, api_type="groq")