"""Explicit one-time model provisioning. Runtime does not download model artifacts."""

from ollama import Client

from app.config import Settings
from app.documents import create_embeddings


def main():
    settings = Settings()
    settings.allow_model_downloads = True
    print("Provisioning the pinned embedding model...")
    embeddings = create_embeddings(settings)
    print(f"Embedding dimensions: {len(embeddings.embed_query('health check'))}")
    print(f"Provisioning Ollama model {settings.model}...")
    Client(host=settings.ollama_base_url, timeout=1800).pull(settings.model)
    print("Model artifacts are ready. Start the application and check /health/ready.")


if __name__ == "__main__":
    main()
