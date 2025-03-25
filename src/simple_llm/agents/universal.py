from ..agent import Agent

from typing import Type, TypeVar

T = TypeVar('T', bound=Agent)

class UniversalAgent(Agent):
    def __init__(
        self,
        agent_cls: Type[T],
        name: str,
        model: str,
        system_message: str,
        stream: bool = False,
        api_key: str = None,
        tools: list = [],
        default_params: dict = {},
        track_msgs: bool = True
    ):
        self.agent = agent_cls(name=name, model=model, system_message=system_message, stream=stream, api_key=api_key, tools=tools, default_params=default_params, track_msgs=track_msgs)

    def add_tool(self, tool):
        return self.agent.add_tool(tool)
    
    def add_user_message(self, query):
        return self.agent.add_user_message(query)
    
    def add_agent_message(self, response):
        return self.agent.add_agent_message(response)
    
    def process_chunk(self, chunk):
        return self.agent.process_chunk(chunk)
    
    def process_stream(self, stream):
        return self.agent.process_stream(stream)
    
    def process_completion(self, completion):
        return self.agent.process_completion(completion)
    
    def get_logprobs(self, completion):
        return self.agent.get_logprobs(completion)
    
    def completion(self, messages: list[dict[str, str]], stream: bool, **kwargs):
        return self.agent.completion(messages, stream, **kwargs)
    
    def __getattr__(self, name):
        """
        Redirects all other method calls to the agent.
        """
        return getattr(self.agent, name)