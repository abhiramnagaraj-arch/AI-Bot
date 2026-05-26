# 🏗️ AI-Bot Architecture

This document provides a detailed overview of the internal architecture of the AI-Bot RAG system.

## High-Level Overview

AI-Bot follows a standard Retrieval-Augmented Generation (RAG) pattern, enhanced with modularity and safety layers. The system is designed to process user queries, retrieve relevant information from a knowledge base (PDFs), and generate grounded responses using a Large Language Model (LLM).

## System Components

### 1. Data Ingestion & Preprocessing
- **Source:** PDF documents.
- **Loader (`data.py`):** Uses `pypdf` to extract raw text from documents.
- **Chunker (`chunker.py`):** Segments text into manageable pieces using semantic or character-based strategies (supported by `nltk`).

### 2. Knowledge Storage (Vector DB)
- **Embedder (`embedder.py`):** Converts text chunks into high-dimensional vectors using `sentence-transformers`.
- **Vector Store (`vector_store.py`):** Manages **ChromaDB** for persistent storage and fast similarity searching of embeddings.

### 3. Request Orchestration (`main.py`)
The pipeline orchestrator manages the flow of data through several stages:
1. **Input Guardrails:** Initial safety checks on the user's query.
2. **Intent Classification:** Determines if the query is a greeting, a meta-question, or a knowledge request.
3. **Query Expansion:** Handles context-dependent queries (follow-ups) by combining them with previous conversation history.
4. **Retrieval:** Fetches the most relevant chunks from ChromaDB based on embedding similarity.
5. **Context Filtering:** Filters out chunks that fall below a similarity threshold or are deemed irrelevant.

### 4. Safety & Guardrails (`guardrails.py`)
- **PII Redaction:** Uses regex and pattern matching to mask sensitive data.
- **Injection Shield:** Detects common prompt injection patterns.
- **Relevance Check:** Heuristic-based validation to ensure retrieved context matches the query intent.
- **Grounding Validation:** (Post-processing) Checks if the generated answer is supported by the context.

### 5. LLM Interface (`llm.py`)
- Connects to LLM providers (defaulting to **Ollama** for local execution).
- Handles prompt construction, injecting retrieved context into the system prompt for grounded generation.

### 6. Memory Management (`memory.py`)
- Maintains a sliding window of the last `N` messages to provide conversational continuity.

### 7. API & Frontend
- **FastAPI (`server.py`):** Exposes a RESTful `/chat` endpoint and serves the static frontend.
- **Frontend (`frontend/`):** A lightweight SPA (Single Page Application) that communicates with the backend via JSON.

## Data Flow (Query to Answer)

1. **User Input:** User sends a query via Web UI or CLI.
2. **Guardrails (In):** Query is checked for injections and PII.
3. **Intent Check:** System decides how to handle the query (e.g., greet vs. search).
4. **Search Expansion:** If it's a follow-up, the query is enriched with history.
5. **Vector Search:** The query is embedded and matched against ChromaDB.
6. **Context Selection:** Top chunks are selected and validated for relevance.
7. **LLM Generation:** LLM receives the prompt: `Context + Query -> Answer`.
8. **Guardrails (Out):** Answer is checked for grounding and PII.
9. **Delivery:** Final answer is returned to the user and stored in memory.

## Tech Stack Decisions

- **ChromaDB:** Chosen for its simplicity, "batteries-included" nature, and excellent Python integration.
- **FastAPI:** Used for its high performance, ease of use with Pydantic, and automatic OpenAPI documentation.
- **Sentence Transformers:** Provides high-quality embeddings suitable for similarity search.
- **Ollama:** Enables running powerful LLMs locally, ensuring privacy and reducing latency/costs.
