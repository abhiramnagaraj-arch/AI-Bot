from vector_store import get_collection
from config import TOP_K

def retrieve_context(query_embedding):
    """
    Retrieves the top-k most relevant chunks from the cosine similarity collection.
    """
    col = get_collection()
    res = col.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "distances"]
    )

    docs = res["documents"][0]
    distances = res["distances"][0]

    # Convert distances to scores (Cosine distance to similarity)
    # Cosine distance in Chroma is 1 - Cosine Similarity
    results = []
    for doc, dist in zip(docs, distances):
        score = 1 - dist
        results.append({"doc": doc, "score": score})

    return results