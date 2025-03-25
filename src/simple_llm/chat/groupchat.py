from abc import ABC
from typing import Generator

from ..printing import print_dashed_line, print_message, print_role
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
        # index = 0

        if index != self.delegator_fallback:
            self.last_agent = index

        return index
    
    def update_shared_chat_history(self, query: str, response: str) -> None:
        for i, agent in enumerate(self.agents): 
            if i != self.last_agent and i not in self.exclude_from_chat_history:
                agent.add_user_message(query)
                agent.add_agent_message(response)

    def reply(self, query: str, stream_override: bool = None, **kwargs) -> str | Generator[str]:
        """
        Delegate to an agent and get a reply.

        Args:
            query (str): The user query.
            stream_override (bool): Whether to force streaming, regardless of the agent's stream attribute.
            **kwargs: Additional arguments to pass to the agent's reply method.
        """

        index = self.delegate_to_agent(query)
        delegated_agent = self.agents[index]

        if (delegated_agent.stream or stream_override) and stream_override is not False:
            return self._stream_reply(query, delegated_agent, **kwargs)
        else:
            response = delegated_agent.nostream_reply(query, **kwargs)
            if self.share_chat_history:
                self.update_shared_chat_history(query, response)
            return response
        
    def _stream_reply(self, query: str, delegated_agent: Agent, **kwargs) -> Generator[str]:
        yield from delegated_agent.stream_reply(query, **kwargs)
        response = delegated_agent.last_response
        if self.share_chat_history:
            self.update_shared_chat_history(query, response)
    
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

            completion = self.reply(msg, stream_override=stream, **kwargs)
            agent_name = self.agents[self.last_agent].name

            if stream:
                # Print name of agent right before agent message
                print_role(agent_name)
                
                # Print each token
                for chunk in completion:
                    print(chunk, end="")

                # Print dashed line at the end of the message
                print("\n")
                print_dashed_line()

            else:
                print_message(agent_name, completion)

            msg = input("Enter message for agent. Type 'exit' to end the conversation: ")