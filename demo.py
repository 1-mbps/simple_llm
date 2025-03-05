import asyncio
from src.simple_llm.clients.openai import OpenAIAgent, AsyncOpenAIAgent
from src.simple_llm.clients.gemini import GeminiAgent
import dotenv

dotenv.load_dotenv()

# agent = AsyncOpenAIAgent("assistant", "gpt-4o-mini", system_message="You are a helpful assistant.", default_params={"temperature": 1})
agent = GeminiAgent("assistant", "gemini-2.0-flash", system_message="Respond in a condescending way.", default_params={})

QUERY = "Do universals exist as real and distinct entities, or only as mental constructs?"

agent.start_chat(stream=True, init_message=QUERY)
# asyncio.run(agent.start_chat(stream=True, init_message=QUERY))