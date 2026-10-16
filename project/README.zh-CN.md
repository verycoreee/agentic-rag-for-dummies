# Agentic RAG 系统文档

> 本文是 `README.md` 的中文翻译副本，供学习参考。原文以英文版为准。

一个用 **LangGraph** 构建的 **Agentic 检索增强生成（RAG）** 系统，具备 **父子分块（parent–child chunking）**、**稠密 + 稀疏混合检索**，默认以本地 Ollama 优先，也可适配其他 LLM 提供方。

## 目录

[快速开始](#快速开始) | [架构概览](#架构概览) | [项目结构](#项目结构) | [配置指南](#配置指南) | [常见改造](#常见改造) | [可观测性](#可观测性) | [进阶主题](#进阶主题) | [问题排查](#问题排查)

---

## 快速开始

### 安装

用 `pip` 安装全部依赖：

```bash
pip install -r requirements.txt
```

或者用 `uv`：

```bash
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 运行应用

在本地启动 Gradio 界面：

```bash
python project/app.py
```

应用会运行在 `http://localhost:7860`（Gradio 默认端口）。

> 这个内存版 demo 面向单个本地用户。若要部署给多用户使用，应为每个会话分配独立的 LangGraph thread ID。

### 前置条件

- Python 3.11+
- 本地 Ollama，或 OpenAI / Anthropic / Google 的 API Key

---

## 架构概览

本系统实现了一条进阶 RAG 流水线，核心特性如下：

- **父子分块**：文档被切成小的子块（用于精确检索），并关联到更大的父块（用于提供丰富上下文）
- **混合检索**：结合稠密向量与稀疏（BM25）检索以获得更优结果
- **LangGraph Agent**：编排查询重写、检索与回答生成
- **紧凑对话记忆**：用滚动摘要 + 有界的近期对话历史来重写追问
- **澄清记忆**：在人在回路（human-in-the-loop）暂停期间保留未解决的查询，汇总其澄清内容，并在给出回答后丢弃这些临时回复
- **提供方可替换路径**：可运行的 app 默认使用 Ollama，并附有将其适配到 OpenAI、Google Gemini 或 Anthropic Claude 的示例
- **向量存储**：使用 Qdrant 做高效相似度检索

### 数据流

```
PDF → Markdown 转换 → 父/子分块 → 向量索引 → Agent 检索 → LLM 回答
```

---

## 项目结构

### 入口与配置

| 文件 | 作用 |
|------|---------|
| `project/app.py` | 应用入口，启动 Gradio UI |
| `project/config.py` | **配置中枢** —— 要改提供方/模型/分块参数就动这里 |
| `project/utils.py` | PDF 转换，以及带离线兜底方案的上下文字数缓存估算 |
| `project/document_chunker.py` | 父/子分块逻辑，含清洗与合并规则 |
| `project/Dockerfile` | 含 Ollama 的本地部署 Dockerfile |

### 核心系统

| 文件 | 作用 |
|------|---------|
| `project/core/rag_system.py` | 系统启动引导 —— 创建各管理器并编译 LangGraph agent |
| `project/core/document_manager.py` | 文档入库流水线（转换、分块、索引） |
| `project/core/chat_interface.py` | 流式输出聚合后的回答，同时把查询分析与工具活动同内部节点输出分离 |
| `project/core/observability.py` | 可选的 Langfuse 追踪 —— 回调处理器生命周期 |

### 数据层

| 文件 | 作用 |
|------|---------|
| `project/db/vector_db_manager.py` | Qdrant 客户端封装，含 embedding 初始化 |
| `project/db/parent_store_manager.py` | 父块的本地文件存储 |

### RAG Agent（LangGraph）

| 文件 | 作用 |
|------|---------|
| `project/rag_agent/graph.py` | 图的构建与编译逻辑 |
| `project/rag_agent/graph_state.py` | 共享状态与各 agent 的状态，含滚动记忆、待澄清项、回答的累积与重置逻辑 |
| `project/rag_agent/nodes.py` | 各节点实现（摘要、重写、agent 执行、聚合） |
| `project/rag_agent/edges.py` | 条件边路由逻辑（例如根据查询是否清晰来路由） |
| `project/rag_agent/tools.py` | 检索工具（`search_child_chunks`、`retrieve_parent_chunks`） |
| `project/rag_agent/prompts.py` | agent 行为的系统提示词 |
| `project/rag_agent/schemas.py` | 结构化输出 schema（Pydantic 模型） |

### 用户界面

| 文件 | 作用 |
|------|---------|
| `project/ui/css.py` | Gradio 界面的自定义 CSS 样式 |
| `project/ui/gradio_app.py` | Gradio UI 实现，含文档上传与对话 |

---

## 配置指南

所有主要设置都在 `project/config.py` 里。关键参数：

### 目录配置

```python
MARKDOWN_DIR = "markdown_docs"        # 存放 PDF → Markdown 转换后的文件
PARENT_STORE_PATH = "parent_store"    # 父块的本地文件存储
QDRANT_DB_PATH = "qdrant_db"          # 本地 Qdrant 向量数据库路径
```

### Qdrant 配置

```python
CHILD_COLLECTION = "document_child_chunks"  # 子块所属的 collection 名
SPARSE_VECTOR_NAME = "sparse"               # 命名的稀疏向量字段（BM25）
```

### 模型配置

```python
# 默认：单模型配置
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "granite4.1:8b"
JUDGE_MODEL = "ministral-3:3b-instruct-2512-q8_0"
LLM_TEMPERATURE = 0  # 0 = 确定性输出，1 = 有创造性
LLM_SEED = 42
```

### 检索配置

```python
RETRIEVAL_SCORE_THRESHOLD = 0.4  # 越低召回越多，越高越精确
DEFAULT_RETRIEVAL_K = 7          # 检索与评估默认使用的子块数量
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"  # 让排序后的子块结果保持可分割，便于评估
```

### Agent 配置

```python
# 防止无限循环的硬性上限
MAX_TOOL_CALLS = 8       # 单次 agent 运行最多调用工具次数
MAX_ITERATIONS = 10      # agent 循环最大迭代次数
GRAPH_RECURSION_LIMIT = 50 # 触发停止条件前的最大步数
MAIN_HISTORY_MESSAGES_TO_KEEP = 4  # 每次回答后保留的原始消息数；最小为 2

# 上下文压缩阈值
BASE_TOKEN_THRESHOLD = 2000     # 触发压缩的初始 token 阈值
TOKEN_GROWTH_FACTOR = 0.9       # 每次压缩后应用的倍率
```

### 终端执行日志

```python
EXECUTION_LOGGING_ENABLED = False  # 打印图的步骤、状态预览、工具调用与输出
EXECUTION_LOG_MAX_CHARS = 1200     # 长值的最大显示字符数
EXECUTION_LOG_USE_COLOR = True     # 终端中使用 ANSI 颜色
```

### 文本切分器配置

```python
CHILD_CHUNK_SIZE = 500              # 用于检索的块大小
CHILD_CHUNK_OVERLAP = 100           # 块之间的重叠（防止上下文丢失）
MIN_PARENT_SIZE = 2000              # 目标最小值；极短的文档可能更小
MAX_PARENT_SIZE = 4000             # 父块最大尺寸

# Markdown 标题切分策略
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
```

### Langfuse 可观测性（可选）

```python
import os

LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = ""               # 来自你的 Langfuse 项目设置
LANGFUSE_SECRET_KEY = ""               # 来自你的 Langfuse 项目设置
LANGFUSE_BASE_URL = "http://localhost:3000"  # Langfuse Cloud 或自托管地址
```

---

## 常见改造

### 1. 切换 LLM 提供方（单一提供方）

> **性能提示：** 参数量 8B 以上的 LLM，通常在推理能力、上下文理解与回答质量上都优于更小的模型。这一点对闭源和开源模型同样成立——前提是它们**原生支持工具/函数调用**，而这是 agentic RAG 流程所必需的。

如果你想从某一提供方永久切换到另一个（例如 Ollama → Google Gemini），按以下步骤操作：

**第 1 步：** 安装该提供方的 SDK

```bash
pip install langchain-google-genai
```

**第 2 步：** 设置环境变量

```bash
export GOOGLE_API_KEY="your-google-key"
```

**第 3 步：** 修改 `project/config.py`

```python
LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0
LLM_SEED = 42
```

**第 4 步：** 修改 `project/core/rag_system.py`

把：

```python
llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE, seed=config.LLM_SEED)
```

替换为：

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
```

### 2. 可选的多提供方配置

本仓库提供的是 Ollama 优先的实现。如果你想把手动切换改造为"提供方工厂"，请参考本节。

这种方式可以让你维护多套提供方配置，并轻松切换。

**第 1 步：** 安装所需的 SDK

```bash
pip install langchain-openai langchain-anthropic langchain-google-genai
```

**第 2 步：** 设置环境变量

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

**第 3 步：** 在 `project/config.py` 中加入多提供方配置

```python
# --- 多提供方 LLM 配置 ---
LLM_CONFIGS = {
    "ollama": {
        "model": "granite4.1:8b",
        "url":"http://localhost:11434",
        "temperature": 0
    },
    "openai": {
        "model": "gpt-5.2",
        "temperature": 0
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    }
}

# 改这一行即可切换提供方
ACTIVE_LLM_CONFIG = "ollama"
```

**第 4 步：** 修改 `project/core/rag_system.py` 的 `initialize()` 方法

把原有的 LLM 初始化替换为：

```python
def initialize(self):
    self.vector_db.create_collection(self.collection_name)
    collection = self.vector_db.get_collection(self.collection_name)

    # 加载当前生效的配置
    active_config = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
    model = active_config["model"]
    temperature = active_config["temperature"]

    if config.ACTIVE_LLM_CONFIG == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=temperature, base_url=active_config["url"])

    elif config.ACTIVE_LLM_CONFIG == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)

    elif config.ACTIVE_LLM_CONFIG == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=temperature)

    elif config.ACTIVE_LLM_CONFIG == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    else:
        raise ValueError(f"Unsupported LLM provider: {config.ACTIVE_LLM_CONFIG}")

    # 继续初始化工具与图
    tools = ToolFactory(collection).create_tools()
    self.agent_graph = create_agent_graph(llm, tools)
```

**切换提供方：** 只需修改 `config.py` 里的 `ACTIVE_LLM_CONFIG`：

```python
ACTIVE_LLM_CONFIG = "google"  # 切换到 Gemini Pro
# ACTIVE_LLM_CONFIG = "anthropic"  # 或 Claude Sonnet
# ACTIVE_LLM_CONFIG = "openai"  # 或 GPT-4o
```

---

**提供方对照表：**

| 提供方 | 环境变量 | 导入语句 | 模型示例 |
|----------|---------------------|------------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `from langchain_openai import ChatOpenAI` | `gpt-5.2`, `gpt-5-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `from langchain_anthropic import ChatAnthropic` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| Google | `GOOGLE_API_KEY` | `from langchain_google_genai import ChatGoogleGenerativeAI` | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Ollama | 无（本地） | `from langchain_ollama import ChatOllama` | `granite4.1:8b`, `llama3.1:8b-instruct-q6_K` |

---

### 3. 更换 Embedding 模型

**为什么要换？** 在速度、成本与质量之间做权衡。

**第 1 步：** 修改 `project/config.py`

```python
# 示例：更快、更小的模型
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 示例：另一款 Sentence Transformers 模型
# DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# 示例：Gemma embedding（Google 的开源模型）
# DENSE_MODEL = "google/embeddinggemma-300m"

# 默认：Qwen embedding（阿里的多语言模型）
# DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

SPARSE_MODEL = "Qdrant/bm25"  # 通常无需修改
```

**第 2 步：** 重新为文档建索引

⚠️ **重要：** 更换 embedding 必须重新建索引。在教学版 UI 里请点 **Clear All**，如果旧 collection 导致启动失败就重启，然后重新上传文档。

**实现细节**（位于 `project/db/vector_db_manager.py`）：

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
import config

self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)
```

**常用 Embedding 模型：**

| 模型 | 上下文长度 | 向量维度 | 速度 | 质量 | 适用场景 |
|-------|--------------|------------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 256 tokens | 384 | 快 | 良好 | 通用，快速语义相似度 |
| all-mpnet-base-v2 | 512 tokens | 768 | 中 | 优秀 | 高精度语义检索 |
| bge-large-en-v1.5 | 512 tokens | 1024 | 慢 | 最佳 | GPU 上的生产级检索 |
| google/embeddinggemma-300m | 2048 tokens | 768 | 快 | 很好 | 轻量、高效的多语言检索 |
| Qwen/Qwen3-Embedding-0.6B | 32768 tokens | 1024 | 中 | 优秀 | 默认的多语言/代码检索模型 |
| Qwen/Qwen3-Embedding-8B | 32768 tokens | 4096 | 很慢 | 优秀 / SOTA | 面向 GPU 的重型检索实验 |

---

### 4. 调整分块策略

**为什么要调？** 在检索精度与上下文丰富度之间取平衡。

> **验证工具：** 可用 🐿️ [**Chunky**](https://github.com/GiovanniPasq/chunky) 清洗 Markdown、检查分块效果、对比不同分块策略，并在重建索引前丰富元数据。

**第 1 步：** 在 `project/config.py` 中修改块大小

```python
# 适合简短的事实型查询（例如技术文档）
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60
MIN_PARENT_SIZE = 1500
MAX_PARENT_SIZE = 8000

# 适合叙述性或强上下文的查询（例如法律文档）
# CHILD_CHUNK_SIZE = 800
# CHILD_CHUNK_OVERLAP = 150
# MIN_PARENT_SIZE = 3000
# MAX_PARENT_SIZE = 15000
```

**第 2 步（可选）：** 替换 `project/document_chunker.py` 中的切分器

**默认（按字符切分）：**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

self.__child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**替代方案（按句子切分）：**
```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

self.__child_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**第 3 步：** 重新执行入库

在 Gradio 界面里点 **Clear All**，然后重新上传文档。已存在的文档会被有意跳过，避免重复。

**分块准则：**

> ⚠️ **免责声明：** 以下为经验性准则。最优值取决于：
> - **子块** → embedding 模型的上下文窗口（例如 all-MiniLM-L6-v2 为 256 tokens，bge-large-en-v1.5 为 512）：子块尺寸不应超过它
> - **父块** → 生成式模型的上下文窗口（例如 8K、32K、128K tokens）：父块连同查询一起发给 LLM 时必须放得下
>
> 请务必在自己的语料上实测验证。

分块器会强制 `MAX_PARENT_SIZE`、对合并的标题元数据去重，并在可能时重新平衡相邻的小块。短于 `MIN_PARENT_SIZE` 的文档会保持为单个较小的块。

| 文档类型 | 子块尺寸 | 父块尺寸 | 理由 |
|---------------|-----------|-------------|-----------|
| 技术文档 | 300-500 | 2000-4000 | 精确查找、代码片段 |
| 法律合同 | 600-1000 | 5000-15000 | 强上下文、定义条款 |
| 研究论文 | 400-600 | 3000-8000 | 兼顾精度与上下文 |
| FAQ / 知识库 | 200-400 | 1500-4000 | 简短、聚焦的答案 |

---

### 5. Agent 配置

在 `project/config.py` 中调整 agent 行为：

```python
# 防止无限循环的硬性上限
MAX_TOOL_CALLS = 8       # 单次 agent 运行最多调用工具次数
MAX_ITERATIONS = 10      # agent 循环最大迭代次数
GRAPH_RECURSION_LIMIT = 50 # 触发停止条件前的最大步数
MAIN_HISTORY_MESSAGES_TO_KEEP = 4  # 每次回答后保留的原始消息数；最小为 2

# 上下文压缩阈值
BASE_TOKEN_THRESHOLD = 2000     # 触发压缩的初始 token 阈值
TOKEN_GROWTH_FACTOR = 0.9       # 每次压缩后应用的倍率
```

| 参数 | 作用 |
|-----------|--------|
| `MAX_TOOL_CALLS` | 累计请求的工具调用上限；会超出该值的请求不会被执行 |
| `MAX_ITERATIONS` | LLM 推理迭代的最大次数；在边界处产生的最终回答仍会被接受 |
| `GRAPH_RECURSION_LIMIT` | 复杂图可调高（见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)） |
| `BASE_TOKEN_THRESHOLD` | 调高该值可推迟压缩 |
| `TOKEN_GROWTH_FACTOR` | 值越低，压缩越激进 |

---

## 可观测性

可选的 [Langfuse](https://langfuse.com) 追踪会记录每一次 LLM 调用、工具调用与图的流转。它有助于调试 agent 行为、跟踪成本以及评估检索质量。

### 启用 Langfuse

1. 在 [Langfuse Cloud](https://cloud.langfuse.com/) 注册，创建组织，然后创建项目并从项目设置里生成 API Key。
2. 设置环境变量（或把 `.env.example` 复制为 `.env`）：

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或你的自托管地址
```

3. 正常启动应用。追踪数据会出现在你的 [Langfuse 面板](https://cloud.langfuse.com/) 中。

关闭追踪可设 `LANGFUSE_ENABLED=false`，或干脆不设这些变量。两种情况应用运行完全一致。

关于 Langfuse 与 LangChain / LangGraph 集成的更多细节，见[官方文档](https://docs.langchain.com/oss/python/integrations/providers/langfuse)。

### 会被追踪的内容

| 组件 | 被追踪的操作 |
|-----------|-------------------|
| 图节点 | `summarize_history`、`rewrite_query`、`orchestrator`、`compress_context`、`fallback_response`、`aggregate_answers` |
| 工具 | `search_child_chunks`、`retrieve_parent_chunks`（参数 + 结果） |
| 结构化输出 | 重写步骤中的 `QueryAnalysis` 解析 |
| 子图扇出 | 通过 `Send()` 触发的并行 agent 调用 |

### 托管方案

- **Langfuse Cloud** —— 在 [cloud.langfuse.com](https://cloud.langfuse.com) 注册，并在那里查看当前的套餐限额。
- **自托管** —— MIT 许可，用 Docker Compose 部署。见[官方自托管文档](https://langfuse.com/self-hosting)。

关于概念性的追踪指南与平台背景，见 [`notebooks/observability.ipynb`](../notebooks/observability.ipynb)。

---

## 进阶主题

### 定制 RAG Agent

**位置：** `project/rag_agent/`

**增删节点：** 编辑 `graph.py` 和 `nodes.py`

示例：增加一个事实核查节点
```python
# 在 nodes.py 中
def fact_check_node(state):
    # 你的事实核查逻辑
    return {"fact_checked": True}

# 在 graph.py 中
builder.add_node("fact_check", fact_check_node)
builder.add_edge("retrieve", "fact_check")
```

**修改条件路由：** 编辑 `edges.py` 来改变图的流转逻辑

系统中的示例 —— 根据查询是否清晰来路由：
```python
def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """问题不清晰则转人工输入，否则处理所有重写后的查询"""
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        # 扇出：把每个重写后的问题送去并行处理
        return [
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
```

这个模式让 agent 要么向用户请求澄清，要么把多个查询变体扇出去并行检索。

**修改提示词：** 编辑 `prompts.py` 来改变 agent 行为与回答风格

**增加自定义工具：** 在 `tools.py` 中扩展新的检索策略或外部集成

### 替换存储后端

**向量数据库：**
- 默认：本地 Qdrant
- 可选：远程 Qdrant Cloud、Pinecone、Weaviate
- 编辑：`project/db/vector_db_manager.py`

**父块存储：**
- 默认：JSON 文件
- 可选：PostgreSQL、MongoDB、S3
- 编辑：`project/db/parent_store_manager.py`

### 扩展 UI

**位置：** `project/ui/gradio_app.py`

可添加运行时设置、管理面板或数据分析：
```python
with gr.Accordion("Advanced Settings", open=False):
    provider_dropdown = gr.Dropdown(
        choices=["openai", "anthropic", "google", "ollama"],
        label="LLM Provider"
    )
```

### Docker 部署

> ⚠️ **系统要求**：至少给 Docker 分配 8GB 内存；若用 Qwen embedding 建索引并在本地运行默认 Ollama 模型，建议 12GB。实际内存占用取决于 Ollama 模型版本与量化方式。

#### 构建与运行
```bash
# 构建镜像
docker build -t agentic-rag -f project/Dockerfile .

# 运行容器
docker run --name rag-assistant -p 7860:7860 agentic-rag
```

**可选：GPU 加速**（仅 NVIDIA）：
```bash
docker run --gpus all --name rag-assistant -p 7860:7860 agentic-rag
```

**常用命令：**
```bash
docker stop rag-assistant      # 停止
docker start rag-assistant     # 重启
docker logs -f rag-assistant   # 查看日志
docker rm -f rag-assistant     # 删除
```

> ⚠️ **性能提示**：在 Windows/Mac 上，Docker 通过 Linux 虚拟机运行，可能拖慢文档索引这类 I/O 操作。LLM 推理速度基本不受影响。在 Linux 上，性能与本地运行相当。

启动后打开 `http://localhost:7860`。

---

## 问题排查

| 问题 | 原因 | 解决方案 |
|-------|-------|----------|
| 报错 "Model not found" | 提供方的模型名写错 | 确认 `LLM_MODEL` 与该提供方 API 一致（例如是 `gpt-4o-mini` 而非 `gpt4-mini`） |
| 检索结果质量差 | embedding 模型或分块配置不佳 | 换更好的 embedding 重建索引，或调整块尺寸 |
| 向量维度不匹配 | 已有的 Qdrant collection 是用另一个 embedding 模型建的 | 清空 collection，修改 `DENSE_MODEL` 后重建索引 |
| 响应慢 | embedding 模型过大或 `top_k` 过高 | 换更小的 embedding 模型（如 all-MiniLM-L6-v2），或降低检索工具里的 `top_k` |
| 超出 API 限流 | 对外部提供方请求过多 | 加重试逻辑并做指数退避，或改用本地 Ollama 模型 |
| 内存溢出 | 文档量过大或 embedding 模型过大 | 用更小的 embedding、降低批大小，或启用 GPU 加速 |
| 检索结果为空 | collection 未建索引，或 collection 名不对 | 确认文档已上传，且配置里的 `CHILD_COLLECTION` 名称一致 |
| 切换提供方后导入报错 | 缺少 SDK | 安装对应包：`pip install langchain-{provider}` |
| 多次运行回答不一致 | temperature 设置过高 | 配置中设 `LLM_TEMPERATURE = 0` 以获得确定性输出 |
