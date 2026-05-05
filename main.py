import sys
from data import load_pdf
from chunker import chunk_text
from embedder import embed_texts, embed_query, get_intent
from vector_store import store_embeddings
from retriever import retrieve_context
from llm import generate_answer, check_connection, generate_llm_direct
from memory import ChatMemory
from guardrails import redact_pii, detect_injection, validate_grounding, detect_leakage
from logger import log_event, get_logs
from config import SIMILARITY_THRESHOLD
import time

# Initialize Chat Memory
memory = ChatMemory(max_history=5)
last_user_query = None
root_query = None

DOCUMENT_SUMMARY = "This document is about astronomy, cosmological models, and celestial objects like comets and the Sun."

def normalize_text(text):
    return text.strip().lower()

def is_context_dependent(query):
    """Detect if a query is a follow-up or contains vague references."""
    vague_terms = ["this", "that", "it", "they", "he", "she", "these", "those", "them", "its"]
    follow_up_starts = [
        "explain", "more", "detail", "why", "how", "tell", "what about", 
        "tell me more", "can", "could", "would", "any more", "further"
    ]
    
    q = normalize_text(query)
    words = q.split()
    
    # 1. Contains pronouns
    if any(word in words for word in vague_terms):
        return True
        
    # 2. Starts with follow-up indicators
    if any(q.startswith(ind) for ind in follow_up_starts):
        return True
        
    # 3. Contains "the [something]" which often refers to a previously mentioned entity
    if "the " in q and len(words) < 8:
        return True
        
    # 4. Very short queries
    if len(words) < 5:
        return True
        
    return False

def setup(filepath):
    text = load_pdf(filepath)
    chunks = chunk_text(text)
    print(f"Chunks: {len(chunks)}")
    embeddings = embed_texts(chunks)
    store_embeddings(chunks, embeddings)
    print("Setup complete\n")

def classify_query(query: str) -> str:
    q = query.lower().strip()
    
    # 1. Use Embedding-based model logic
    intent = get_intent(query)
    
    # 2. Heuristic check: simple
    simple_patterns = ["what is", "define", "tell me about", "explain"]
    if intent == "knowledge" and any(q.startswith(p) for p in simple_patterns) and len(q.split()) < 6:
        return "simple"
        
    return intent

def is_context_relevant_detailed(context, query):
    """
    Detailed relevance check with diagnostic feedback.
    """
    import re
    CORE_KEYWORDS = ["astronomy", "sun", "comet", "star", "planet", "galaxy", "orbit", "celestial", "space", "cosmic", "asteroid", "meteor", "universe"]
    
    context_lower = context.lower()
    clean_q = re.sub(r'[^\w\s]', '', query.lower())
    query_terms = [t for t in clean_q.split() if len(t) > 3]
    
    has_core_keyword = any(kw in context_lower for kw in CORE_KEYWORDS)
    
    if not query_terms:
        if has_core_keyword:
            return True, "Core keyword found in short query context"
        return False, "Short query and no core keywords in context"

    matching_terms = [term for term in query_terms if term in context_lower]
    match_count = len(matching_terms)
    
    if match_count >= 1 and has_core_keyword:
        return True, f"Matched {matching_terms} + core keyword"
    if match_count >= 2:
        return True, f"Matched {matching_terms} (multi-match)"
        
    return False, f"Insufficient matches (matches: {matching_terms}, core_kw: {has_core_keyword})"

def is_context_relevant(context, query):
    relevant, _ = is_context_relevant_detailed(context, query)
    return relevant

def run_pipeline(query):
    global last_user_query, root_query
    start_time = time.time()
    print(f"User: {query}")
    
    # 1. Input Guardrails
    if detect_injection(query):
        return "System Warning: Potential prompt injection detected. Request blocked."
    
    clean_query = redact_pii(query)
    history_str = memory.get_formatted_history()
    
    query_type = classify_query(clean_query)
    print("Query Type:", query_type)
    
    if query_type == "greeting":
        final_answer = "Hi! How can I assist you today?"
        memory.add_message("user", clean_query)
        memory.add_message("assistant", final_answer)
        print(f"⏱️ Terminal logs - Response time: {time.time() - start_time:.1f}s")
        return final_answer
        
    if query_type == "meta":
        final_answer = DOCUMENT_SUMMARY
        memory.add_message("user", clean_query)
        memory.add_message("assistant", final_answer)
        print(f"⏱️ Terminal logs - Response time: {time.time() - start_time:.1f}s")
        return final_answer
        
    if query_type == "uncertain":
        pass

    if query_type == "simple":
        print("Simple query detected, proceeding to RAG for grounding...")
        pass 
        
    # 4. Retrieval (Knowledge/Complex Query)
    print("Retrieving context...")
    search_query = clean_query
    
    is_dependent = is_context_dependent(search_query)
    
    if last_user_query and is_dependent:
        # Use root_query + last_query + current_query for maximum context
        effective_root = root_query if root_query else ""
        search_query = f"{effective_root} {last_user_query} {clean_query}".strip()
        print(f"Expanded Query: {search_query}")
    else:
        # This is a new primary query, update root context
        root_query = clean_query
        print(f"New Root Subject: {root_query}")
        
    q_emb = embed_query(search_query)
    retrieved_results = retrieve_context(q_emb)
    
    # 1. Apply Similarity Threshold Filter (from config)
    filtered_results = [item for item in retrieved_results if item['score'] >= SIMILARITY_THRESHOLD]
    
    if not filtered_results:
        print(f"Confidence below threshold ({SIMILARITY_THRESHOLD}), refusing (Strict RAG)...")
        final_answer = "I don't know based on available data."
        context_count = 0
    else:
        # 2. Limit to max 2 chunks for latency
        context_chunks = [item['doc'] for item in filtered_results][:2]
        context_str = "\n".join(context_chunks)
        context_count = len(context_chunks)
        
        # Use expanded query for relevance check if available
        relevance_query = search_query if 'search_query' in locals() else clean_query
        
        if not is_context_relevant(context_str, relevance_query):
            print("Context not relevant, refusing (Strict RAG)...")
            final_answer = "I don't know based on available data."
        else:
            # 5. Prompt Construction & LLM Call
            print(f"Generating answer from RAG (using {context_count} chunks)...")
            answer = generate_answer(context_chunks, relevance_query)
            final_answer = redact_pii(answer)
    
    # 6. Logging
    log_event({
        "query": query,
        "clean_query": clean_query,
        "standalone_query": clean_query,
        "context_count": context_count,
        "answer": final_answer,
        "history_used": history_str
    })
    
    # 7. Update Memory
    memory.add_message("user", clean_query)
    memory.add_message("assistant", final_answer)
    
    if final_answer != "I don't know based on available data." and not final_answer.startswith("System Warning"):
        last_user_query = clean_query
    
    print(f"⏱️ Terminal logs - Response time: {time.time() - start_time:.1f}s")
    
    return final_answer

def run():
    print("\n" + "="*50)
    print("🚀 MODULAR RAG PIPELINE WITH GUARDRAILS")
    print("="*50)
    
    while True:
        print("\n--- Menu ---")
        print("1. Run Setup (PDF Path)")
        print("2. Reset Database")
        print("3. Chat / Query")
        print("4. View Logs")
        print("5. Clear Chat History")
        print("6. Check LLM Status")
        print("7. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == "1":
            path = input("PDF path: ")
            setup(path)
        elif choice == "2":
            from vector_store import clear_collections
            confirm = input("Are you sure? This will delete all stored chunks. (y/n): ")
            if confirm == "y":
                clear_collections()
        elif choice == "3":
            print("\nEntering Chat Mode. Type 'back' to return to menu.")
            while True:
                query = input("\nYou: ")
                if query.lower() == "back":
                    break
                
                answer = run_pipeline(query)
                print(f"\nAI: {answer}")
                print("-" * 20)
        elif choice == "4":
            logs = get_logs()
            print(f"\nLast 5 logs:")
            for log in logs[-5:]:
                print(f"[{log['timestamp']}] Q: {log['query'][:50]}... | A: {log['answer'][:50]}...")
        elif choice == "5":
            memory.clear()
            print("Chat history cleared.")
        elif choice == "6":
            print("Checking LLM connection...")
            ok, msg = check_connection()
            print(f"[{'PASSED' if ok else 'FAILED'}] {msg}")
        elif choice == "7":
            sys.exit()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    run()