import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

if not MONGO_URI:
    raise RuntimeError("MONGODB_CONNECTION_STRING not set in .env")

# Conversation limits
QUESTION_LIMIT = 6

# JSON markers
CAR_JSON_MARKER = "===CAR_JSON==="
WEB_JSON_MARKER = "===WEB_JSON==="
