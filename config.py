import os

DB_PATH = "./chroma_db"

CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

MAX_TOKENS = 400

TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Support Docker service name fallback
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = "qwen:1.8b"