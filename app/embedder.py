from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(settings):
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.embedding_device,
            "revision": settings.embedding_revision,
            "local_files_only": not settings.allow_model_downloads,
        },
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_encode_kwargs={"normalize_embeddings": True},
    )
