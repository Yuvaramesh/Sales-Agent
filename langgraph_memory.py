# LangGraph Memory setup
LANGGRAPH_AVAILABLE = False
InMemorySaver = None
InMemoryStore = None
HumanMessage = None
AIMessage = None

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langchain_core.messages import HumanMessage, AIMessage

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    InMemorySaver = InMemoryStore = None

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class AIMessage:
        def __init__(self, content: str):
            self.content = content
