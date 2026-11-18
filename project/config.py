import os

# --- 目录配置 ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant 配置 ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- 模型配置 ---
# 支持填 HF 仓库名或本地模型目录（本地目录可完全离线，Windows 上快很多）
DENSE_MODEL = os.environ.get("DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")

# LLM 提供方：ollama（本地，默认）| openai（任何 OpenAI 兼容端点）
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "granite4.1:8b")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
# 大模型输出上限，留空则用提供方默认值
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "0")) or None

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "ministral-3:3b-instruct-2512-q8_0")
LLM_TEMPERATURE = 0
# 结构化输出方式：留空用 LangChain 默认（json_schema）；
# 部分 OpenAI 兼容端点只支持工具调用，这时填 function_calling
STRUCTURED_OUTPUT_METHOD = os.environ.get("STRUCTURED_OUTPUT_METHOD", "").strip() or None
LLM_SEED = 42

# --- 检索配置 ---
RETRIEVAL_SCORE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 7
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"

# --- Agent 配置 ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- 终端执行日志 ---
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- 文本切分器配置 ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse 可观测性 ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
