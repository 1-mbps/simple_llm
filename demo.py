from src.simple_llm.agents.openai import OpenAIAgent
import dotenv

dotenv.load_dotenv()

agent = OpenAIAgent("assistant", "gpt-4o-mini", system_message="You are a helpful assistant.", default_params={"temperature": 1})

QUERY = "Do universals exist as real and distinct entities, or only as mental constructs?"

agent.start_chat(stream=True, init_message=QUERY)