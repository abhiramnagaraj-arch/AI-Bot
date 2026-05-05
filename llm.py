import requests
import json
from config import OLLAMA_URL, LLM_MODEL, MAX_TOKENS


def check_connection():
    try:
        res = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            if LLM_MODEL in models or f"{LLM_MODEL}:latest" in models:
                return True, f"Connected! Model '{LLM_MODEL}' is ready."
            return False, f"Model '{LLM_MODEL}' not found. Available: {models}"
        return False, f"Ollama returned status {res.status_code}"
    except Exception as e:
        return False, f"Connection error: {e}"


def generate_answer(context_chunks, query):
    """
    Optimized RAG response generation with balanced prompt
    """

    context = "\n".join(context_chunks) if context_chunks else "No context available."

    prompt = f"""
You are a specialized astronomy assistant.

STRICT RULES for your response:
1. Answer ONLY using the provided context. 
2. If the context does not contain the answer, or is about an unrelated topic (like textbooks, authors, or general software), say exactly: "I don't know based on available data."
3. Do NOT use ANY outside knowledge.
4. IMPORTANT: You are an ASTRONOMY assistant. If the context contains Astrology (pseudoscience like zodiacs/horoscopes), ignore it and do NOT use it to answer unless specifically asked about the difference between the two. Say you don't know if the context is only about astrology.
5. Do NOT mention authors, publishers, or the text source unless specifically asked about them.
6. If the context is irrelevant to the question, do NOT attempt to answer.
7. If the question is vague or ambiguous, ask for clarification.
8. If the question is a follow-up question, use the previous conversation history to answer the question.
9. If the question is a greeting, respond politely and courteously.
10. If the question is a meta question, respond politely and courteously.
11. Answer the question fully within 400 tokens (approx 600 words).

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": MAX_TOKENS,
                    "top_k": 40,
                    "top_p": 0.9
                }
            },
            timeout=180
        )
        return _parse_response(res)

    except Exception as e:
        return f"LLM Error: {e}"


def _parse_response(res):
    """Robustly parse Ollama response, handling both single JSON and newline-delimited JSON."""
    if res.status_code != 200:
        return f"LLM Error: {res.status_code} - {res.text}"

    try:
        # Standard case: single JSON object
        return res.json()["response"].strip()
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: concatenated JSON objects (common in some streaming edge cases)
        try:
            full_response = ""
            lines = res.text.strip().split('\n')
            for line in lines:
                if line.strip():
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
            return full_response.strip()
        except Exception as e:
            return f"LLM Error: Failed to parse response: {e}"

def generate_llm_direct(query):
    prompt = f"Answer clearly and concisely:\\n\\n{query}"
    
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": MAX_TOKENS,
                    "top_k": 40,
                    "top_p": 0.9
                }
            },
            timeout=180
        )
        return _parse_response(res)
    except Exception as e:
        return f"LLM Error: {e}"
