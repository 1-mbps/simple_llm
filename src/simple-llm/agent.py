from abc import ABC, abstractmethod
import os
from typing import List, Dict
from termcolor import colored

class Agent(ABC):
    def __init__(
        self,
        api_type: str,
        name: str,
        model: str,
        system_message: str,
        stream: bool,
        api_key: str | None,
        default_params: dict = {}
    ):
        key = api_key or os.getenv(f"{api_type.upper()}_API_KEY")
        if not key:
            raise KeyError(f"{api_type} API key not found. Either pass `api_key` as an argument to this class or set {api_type.upper()}_API_KEY as an environment variable.")
        self.api_key = key
        self.name = name
        self.model = model
        self.api_type = api_type
        self.system_message = system_message
        self.stream = stream
        self.default_params = default_params

        self._messages: List[Dict[str, str]] = []

    @abstractmethod
    def add_user_message(self, query: str) -> None:
        pass

    @abstractmethod
    def add_agent_message(self, response: str) -> None:
        pass
    
    @abstractmethod
    def nostream_reply(self, query: str, **kwargs):
        pass

    @abstractmethod
    def stream_reply(self, query: str, **kwargs) -> str:
        pass

    def merge_params(self, new_params: dict) -> dict:
        return {**self.default_params, **(new_params or {})}

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
            self.print_message("user", msg)

            if stream:
                # Print name of agent right before agent message
                print(colored("\n"+self.name+":", "yellow"), end="\n\n")

                # Initiate streaming of tokens
                stream = self.stream_reply(msg, **kwargs)
                
                # Print each token
                for chunk in stream:
                    print(chunk, end="")

                # Print dashed line at the end of the message
                print("\n")
                print("-"*os.get_terminal_size().columns)

            else:
                response = self.nostream_reply(msg, **kwargs)
                self.print_message(self.name, response)

            msg = input("Enter message for agent. Type 'exit' to end the conversation: ")
            

    def update_model(self, new_model: str) -> None:
        self.model = new_model

    def update_system_message(self, new_system_message: str) -> None:
        self.system_message = new_system_message

    def print_message(self, role: str, content: str) -> None:
        """
        Print messages for chat display
        """
        
        # Print role in yellow
        print(colored("\n"+role+":", "yellow"), end="\n\n")

        # Print main message content
        print(content, end="\n\n")

        # Print dashes covering the width of the terminal
        print("-"*os.get_terminal_size().columns)

    @property
    def messages(self):
        return self._messages

class AsyncAgent(Agent):
    def __init__(self, api_type, name, model, system_message, stream, api_key):
        super().__init__(api_type, name, model, system_message, stream, api_key)

    @abstractmethod
    async def nostream_reply(self, query: str, **kwargs):
        pass

    @abstractmethod
    async def stream_reply(self, query: str, **kwargs) -> str:
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
            self.print_message("user", msg)

            if stream:
                # Print name of agent right before agent message
                print(colored("\n"+self.name+":", "yellow"), end="\n\n")

                # Initiate streaming of tokens
                stream = await self.stream_reply(msg, **kwargs)
                
                # Print each token
                for chunk in stream:
                    print(chunk, end="")

                # Print dashed line at the end of the message
                print("\n")
                print("-"*os.get_terminal_size().columns)

            else:
                response = await self.nostream_reply(msg, **kwargs)
                self.print_message(self.name, response)

            msg = input("Enter message for agent. Type 'exit' to end the conversation: ")

    
