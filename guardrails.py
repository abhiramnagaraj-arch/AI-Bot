import re

def redact_pii(text):
    # Redact email
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_REDACTED]', text)
    # Redact phone (basic)
    text = re.sub(r'\b\d{10}\b|\b\d{3}-\d{3}-\d{4}\b', '[PHONE_REDACTED]', text)
    return text

def detect_injection(text):
    # Basic injection patterns
    patterns = [
        r'ignore previous instructions',
        r'disregard all rules',
        r'system prompt',
        r'you are now a',
        r'acting as'
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def validate_grounding(answer, context_chunks):
    # Simple check: if context is empty and answer isn't "I don't know" style, it's a hallucination
    # In a real system, we'd use another LLM or NLI model here.
    if not context_chunks:
        if "don't know" in answer.lower() or "not present" in answer.lower():
            return True
        return False
    return True # Assume grounded for now if context exists, LLM rules should handle it

def detect_leakage(text):
    # Phrases from the system prompt that should not be in the output
    forbidden_phrases = [
        "Answer ONLY from the provided context",
        "If the answer is not present in the context",
        "I don't know based on available data",
        "Stay in character as a support bot",
        "INTERNAL DATA PROTECTION",
        "Your goal is to provide helpful, accurate answers based ONLY on the provided context"
    ]
    
    for phrase in forbidden_phrases:
        if phrase.lower() in text.lower():
            return True
    return False
