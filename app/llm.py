from langchain_ollama import ChatOllama


def create_model(settings):
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=settings.output_tokens,
        num_ctx=settings.context_tokens,
        client_kwargs={"timeout": settings.model_timeout},
    )
