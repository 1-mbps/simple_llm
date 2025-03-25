from abc import ABC
from typing import Generator

from ..agent import Agent
from ..agents.delegator import Delegator

class GroupChat(ABC):
    def __init__(self, agents: list[Agent]):
        self.agents = agents

class DelegatorGroupChat(GroupChat):
    def __init__(
        self,
        agents: list[Agent],
        delegator: Delegator,
        delegator_fallback: int = 9,
        share_chat_history: bool = True,
        exclude_from_chat_history: list[int] = []
    ):
        self.delegator = delegator
        self.delegator_fallback = delegator_fallback
        self.share_chat_history = share_chat_history
        self.exclude_from_chat_history = exclude_from_chat_history
        self.last_agent = 0
        super().__init__(agents)

    def delegate_to_agent(self, query: str) -> int:
        index = int(self.delegator.delegate(query))

        if index != self.delegator_fallback:
            self.last_agent = index

        return index
    
    def update_shared_chat_history(self, query: str, response: str) -> None:
        for i, agent in enumerate(self.agents): 
            if i != self.last_agent and i not in self.exclude_from_chat_history:
                agent.add_user_message(query)
                agent.add_agent_message(response)

    def reply(self, query: str, **kwargs) -> str | Generator[str]:
        index = self.delegate_to_agent(query)
        delegated_agent = self.agents[index]

        if delegated_agent.stream:
            yield from delegated_agent.stream_reply(query, **kwargs)
            response = delegated_agent.most_recent_response()
            if self.share_chat_history:
                self.update_shared_chat_history(query, response)
        else:
            response = delegated_agent.nostream_reply(query, **kwargs)
            if self.share_chat_history:
                self.update_shared_chat_history(query, response)
            return response
    
    def start_chat(self, query: str, **kwargs) -> None:
        pass
        # recreate CLI chats
        # delegated_agent = self.delegate_to_agent(query)
        # delegated_agent.start_chat(query, **kwargs)