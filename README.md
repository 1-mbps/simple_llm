# simple-llm
A lightweight Python package providing a bare-bones implementation of LLM agents. Provides a unified, highly customizable interface for various LLM providers with minimal abstraction.

We've purposely only implemented the basic stuff - streaming, conversation memory, function calling, and API calls – so that you can easily customize and modify the agents for your own use case.

## Quickstart
The following code allows you to have a conversation with an agent on the terminal:
```python
from simple_llm.agents.openai import OpenAIAgent
import dotenv

dotenv.load_dotenv()

agent = OpenAIAgent(
    name="assistant",
    model="gpt-4o-mini",
    system_message="You are a helpful assistant.",
    default_params={"temperature": 1}
)

QUERY = "Do universals exist as real and distinct entities, or only as mental constructs?"

agent.start_chat(stream=True, init_message=QUERY)
```
All agents check the environment variables for API keys. You can also specify an `api_key` parameter on each agent and set the key there.

## Design
Other LLM frameworks add a tremendous amount of abstraction and overhead. This is great for novice developers, but for more complex LLM applications, it can get in your way.

This package is completely different. It's extremely lightweight, highly customizable, and designed for you to override class methods and get *really* close to the original LLM clients.

### Example
```python
class Agent:
    def stream_reply(self, query: str, **kwargs) -> Generator[str]:
        if self.track_msgs:
            self.add_user_message(query)

        # Unpack completion arguments and feed them into completion function
        stream = self.completion_fn(**self.get_completion_args(stream=True, messages=self._messages, query=query, **kwargs))
        
        return self.process_stream(stream)

    def process_stream(self, stream: AsyncGenerator):
        response = ""
        for chunk in stream:
            delta = self.process_chunk(chunk)
            if delta:
                yield delta
                response += delta
        if self.track_msgs:
            self.add_agent_message(response)
```
Customization examples:
- You can override the `process_stream` method to do additional computations while streaming.
- You can override the `process_chunk` method to modify the operations done on each chunk.
- You can override `add_user_message` and `add_agent_message` to modify how the agent records conversation history.