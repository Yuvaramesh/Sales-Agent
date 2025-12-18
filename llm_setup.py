from typing import List, Dict
from config import LLM_MODEL_NAME, OPENAI_API_KEY

# Try to import LangChain/OpenAI
llm = None
LC_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain.tools import tool

    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
    LC_AVAILABLE = True
except Exception:
    import openai

    openai.api_key = OPENAI_API_KEY

    class SimpleOpenAIWrapper:
        def __init__(self, model=LLM_MODEL_NAME, temperature=0):
            self.model = model
            self.temperature = temperature

        def __call__(self, messages: List[Dict[str, str]]):
            return openai.ChatCompletion.create(
                model=self.model, messages=messages, temperature=self.temperature
            )

    llm = SimpleOpenAIWrapper()
    LC_AVAILABLE = False
