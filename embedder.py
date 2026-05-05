from sentence_transformers import SentenceTransformer, util
from config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)

INTENTS = {
    "greeting": [
        "hello", "hi", "hey", "yo", "what's up",
        "good morning", "good evening"
    ],
    "meta": [
        "what can you do",
        "how can you help",
        "what are your abilities",
        "what can i ask",
        "who are you",
        "about you",
        "what is this document",
        "what should i ask",
        "what is the uploaded document",
        "tell me what u can assist me with",
        "what is the context about",
        "what are the specifications of the model?",
        "what questions can you answer",
        "what do you know",
        "tell me about yourself",
        "what is this system"
    ]
}

INTENT_EMBEDDINGS = {k: model.encode(v) for k, v in INTENTS.items()}

def embed_texts(texts):
    return model.encode(texts)

def embed_query(query):
    return model.encode([query]).tolist()

def get_intent(query):
    query_emb = model.encode(query)
    best_intent = "uncertain"
    best_score = 0.0
    
    for intent, embs in INTENT_EMBEDDINGS.items():
        sims = util.cos_sim(query_emb, embs)
        score = sims.max().item()
        if score > best_score:
            best_score = score
            best_intent = intent
            
    if best_score >= 0.6:
        return best_intent
    return "knowledge"