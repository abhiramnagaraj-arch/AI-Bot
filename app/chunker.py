from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_splitter(settings, embeddings):
    # SentenceTransformer's effective max_seq_length, not just tokenizer.model_max_length.
    model = embeddings._client
    tokenizer = model.tokenizer
    limit = min(settings.chunk_tokens, model.max_seq_length - tokenizer.num_special_tokens_to_add())
    if limit <= settings.chunk_overlap:
        raise ValueError("Embedding token limit is smaller than configured overlap")
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=limit,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )
