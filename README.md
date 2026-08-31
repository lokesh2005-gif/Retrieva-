# Retrieva | AI-Powered PDF Q&A Assistant

Retrieva is a high-performance Retrieval-Augmented Generation (RAG) web application tailored for querying and chatting with PDF documents. It parses documents, creates chunked vector embeddings locally using HuggingFace models, stores them in ChromaDB, and uses Groq's low-latency LLM inference to provide accurate answers with exact page citations.

---

## Key Features

- **Multi-Document & Single-Document Filtering**: Chat across all uploaded PDFs or select specific files to isolate search contexts.
- **Precise Citations**: Responses indicate exact source documents and referenced page numbers.
- **Expandable Passage Verification**: Inspect the exact chunk passages retrieved by the vector similarity search.
- **In-App PDF Upload**: Upload new PDF files directly from the Streamlit UI and index them on the fly.
- **Conversation State & Memory**: Managed seamlessly using LangGraph message workflows and `MemorySaver`.
- **Ultra-Fast LLM Inference**: Powered by the Groq API (`groq/compound-mini`).

---

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive chat interface and file management |
| **Orchestration & State** | [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph) | RAG pipeline, state graph flow, and memory management |
| **LLM Provider** | [Groq](https://groq.com/) | Ultra-low latency model inference (`groq/compound-mini`) |
| **Embeddings** | [HuggingFace](https://huggingface.co/) | `intfloat/multilingual-e5-large` sentence transformer |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Local embedded vector database |
| **Document Processing** | `pypdf` | Document loader and recursive character text splitter |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) | Fast Python package management and virtual environments |

---

## Project Structure

```text
RAG/
├── .streamlit/
│   └── config.toml         # Streamlit server configurations
├── data/                   # Directory where source PDF documents are stored
├── docs/
│   └── chroma_db/          # Persistent local Chroma vector database
├── .env                    # API keys and environment variables (excluded from git)
├── .env.example            # Example template for environment configuration
├── app.py                  # Main Streamlit web application & LangGraph workflow
├── config.py               # Configuration for chunking, paths, and embedding models
├── ingest.py               # Document loading, chunking, and embedding generation script
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Project documentation
```

---

## Prerequisites

Ensure you have **Python 3.11+** installed and [`uv`](https://github.com/astral-sh/uv) installed.

### Install `uv`

- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows (PowerShell):**
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

## Setup & Installation

### 1. Clone & Sync Dependencies

```bash
git clone <repository-url>
cd RAG
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

### 1. Ingest Documents (CLI)

Place your PDF documents inside the `data/` directory and run:

```bash
uv run python ingest.py
```

This will:
1. Load all PDFs from `data/`.
2. Split content using `RecursiveCharacterTextSplitter` (chunk size: `2028`, chunk overlap: `250`).
3. Compute embeddings using `intfloat/multilingual-e5-large`.
4. Persist vectors into `docs/chroma_db`.

### 2. Launch the Web Interface

Start the Streamlit application:

```bash
uv run streamlit run app.py
```

Open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## Performance & Optimization Notes

- **Model Resource Caching**: Embeddings and chat models use `@st.cache_resource` to ensure models are initialized only once across Streamlit reruns, preventing PyTorch meta tensor conflicts.
- **Local Persistence**: Vector embeddings are saved directly in `docs/chroma_db/`, avoiding external vector cloud dependencies.
