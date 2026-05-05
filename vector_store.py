import chromadb
import hashlib
from config import DB_PATH

client = chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    return client.get_or_create_collection(
        name="cosine_docs",
        metadata={"hnsw:space": "cosine"}
    )

def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def hash_id(text):
    return hashlib.sha256(text.encode()).hexdigest()

def clear_collections():
    try:
        client.delete_collection("cosine_docs")
        print("Deleted collection: cosine_docs")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    
    # Re-initialize
    get_collection()
    print("Collection reset.")

def store_embeddings(chunks, embeddings):
    col = get_collection()
    ids = [hash_id(c) for c in chunks]

    existing = col.get(ids=ids)
    existing_ids = set(existing["ids"] or [])

    new_idx = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

    if new_idx:
        col.add(
            documents=[chunks[i] for i in new_idx],
            embeddings=[to_list(embeddings[i]) for i in new_idx],
            ids=[ids[i] for i in new_idx],
        )
        print(f"Added {len(new_idx)} new chunks to cosine collection.")
    else:
        print("No new chunks to add.")