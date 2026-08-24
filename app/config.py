import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(RUNTIME_DIR / "chroma_db")
LOG_FILE = RUNTIME_DIR / "pipeline_logs.jsonl"

CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

MAX_TOKENS = 400

TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Support Docker service name fallback
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = "qwen:1.8b"
