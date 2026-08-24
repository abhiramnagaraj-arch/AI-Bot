# 🚀 AI-Bot: Modular RAG Pipeline with Guardrails

AI-Bot is a robust, production-ready Retrieval-Augmented Generation (RAG) system designed for intelligent document querying. It features a modular architecture, sophisticated safety guardrails, and a clean web interface.

## ✨ Key Features

- **Modular RAG Pipeline:** Easily swap or upgrade components for chunking, embedding, and retrieval.
- **Advanced Guardrails:** Built-in protection against PII leakage, prompt injection, and hallucination (grounding validation).
- **Context-Aware Memory:** Maintains conversation flow with sophisticated follow-up detection and query expansion.
- **Vector Search:** High-performance similarity search powered by **ChromaDB**.
- **Web & CLI Interfaces:** Interact with the bot via a modern web UI or a comprehensive command-line menu.
- **Dockerized Deployment:** Simple setup using Docker and Docker Compose.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **LLM & Embeddings:** Sentence Transformers, HuggingFace, Ollama
- **Vector Database:** ChromaDB
- **Data Processing:** NLTK, PyPDF
- **Frontend:** HTML5, CSS3, JavaScript
- **DevOps:** Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM execution)
- Docker & Docker Compose (optional)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/AI-Bot.git
   cd AI-Bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**
   Update `app/config.py` with your settings if necessary.

5. **Run the CLI application:**
   ```bash
   python -m app.main
   ```

6. **Run the Web Server:**
   ```bash
   python -m app.server
   ```
   The web UI will be available at `http://localhost:8000`.

### Docker Setup

```bash
docker compose up --build
```

## 📂 Project Structure

```text
AI-Bot/
├── app/                # Python application package
│   ├── main.py         # Pipeline orchestration and CLI
│   ├── server.py       # FastAPI server and API endpoints
│   ├── pdf_loader.py   # PDF extraction and cleanup
│   ├── chunker.py      # Document splitting
│   ├── embedder.py     # Embeddings and intent classification
│   ├── retriever.py    # Context retrieval
│   ├── vector_store.py # ChromaDB access
│   ├── llm.py          # Ollama interface
│   ├── guardrails.py   # Input/output safety checks
│   ├── memory.py       # Conversation history
│   ├── logger.py       # JSONL event logging
│   └── config.py       # Paths and model settings
├── documents/          # Local source PDFs
├── frontend/           # Static web UI
├── docs/               # Project documentation
├── runtime/            # Generated database and logs
├── deleted/            # Recoverable discarded artifacts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🛡️ Guardrails & Safety

The system implements multiple layers of security:
- **PII Redaction:** Automatically detects and masks sensitive information in inputs and outputs.
- **Injection Detection:** Identifies and blocks malicious prompt injection attempts.
- **Grounding Validation:** Ensures the LLM's response is strictly based on the retrieved context to prevent hallucinations.
- **Context Relevance:** Validates that retrieved chunks are actually relevant to the user's query.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
