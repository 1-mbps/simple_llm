from src.simple_llm.clients.openai import OpenAIAgent, AsyncOpenAIAgent
from src.simple_llm.clients.gemini import GeminiAgent
import dotenv

dotenv.load_dotenv()

openai_agent = OpenAIAgent("assistant", "gpt-4o-mini", system_message="You are a helpful assistant.", default_params={"temperature": 1})
gemini_agent = GeminiAgent("assistant", "gemini-1.5-flash-002", system_message="Respond in a condescending way.")

QUERY = "Do universals exist as real and distinct entities, or only as mental constructs?"

openai_agent.start_chat(stream=False, init_message=QUERY)