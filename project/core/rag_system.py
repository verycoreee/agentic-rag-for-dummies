import uuid
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChunker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability

def _build_llm():
    """按 config.LLM_PROVIDER 构造对话模型。

    - ``openai``：任何兼容 OpenAI Chat Completions 的端点（本地网关、DeepSeek、Qwen、vLLM 等）
    - ``ollama``：本地 Ollama（默认，与原实现一致）
    """
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.LLM_MAX_TOKENS:
            kwargs["max_tokens"] = config.LLM_MAX_TOKENS
        return ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        seed=config.LLM_SEED,
    )


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChunker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = _build_llm()
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
