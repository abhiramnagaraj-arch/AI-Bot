import nltk
from transformers import AutoTokenizer
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# Ensure nltk data is downloaded (silent check)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def split_sentences(text):
    return nltk.tokenize.sent_tokenize(text)

def chunk_text(text):
    sentences = split_sentences(text)
    
    # Cache token lengths to avoid repeated tokenization
    sentence_tokens = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in sentences
    ]
    sentence_token_lengths = [len(t) for t in sentence_tokens]

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        tokens = sentence_tokens[i]
        length = sentence_token_lengths[i]

        # If a single sentence is too long → split it with hard cap
        if length > CHUNK_SIZE_TOKENS:
            # Save pending chunk
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0

            # Split overflowing sentence
            for j in range(0, length, CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS):
                sub_tokens = tokens[j : j + CHUNK_SIZE_TOKENS]
                # HARD CAP check
                if len(sub_tokens) > CHUNK_SIZE_TOKENS:
                    sub_tokens = sub_tokens[:CHUNK_SIZE_TOKENS]
                
                chunks.append(tokenizer.decode(sub_tokens))
                if j + CHUNK_SIZE_TOKENS >= length:
                    break
            continue

        # Check if adding this sentence exceeds chunk size
        # Use a more accurate token count by joining and re-encoding
        test_chunk = " ".join(current_chunk_sentences + [sentence])
        test_tokens = tokenizer.encode(test_chunk, add_special_tokens=False)
        
        if len(test_tokens) > CHUNK_SIZE_TOKENS:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

            # Overlap logic: keep sentences until overlap limit
            overlap_sentences = []
            overlap_tokens = 0
            for s_idx in range(len(current_chunk_sentences) - 1, -1, -1):
                s = current_chunk_sentences[s_idx]
                s_len = sentence_token_lengths[sentences.index(s)] # This index lookup is slow but sentences are few
                if overlap_tokens + s_len > CHUNK_OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_len
            
            current_chunk_sentences = overlap_sentences
            # Recalculate accurate token count for overlap
            current_tokens = len(tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)) if current_chunk_sentences else 0

        current_chunk_sentences.append(sentence)
        current_tokens = len(tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False))

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # FINAL HARD CAP ENFORCEMENT
    final_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        if len(tokens) > CHUNK_SIZE_TOKENS:
            final_chunks.append(tokenizer.decode(tokens[:CHUNK_SIZE_TOKENS]))
        else:
            final_chunks.append(chunk)

    # DEBUG
    if final_chunks:
        lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in final_chunks]
        print(f"Max tokens per chunk: {max(lengths)}. Total chunks: {len(final_chunks)}")

    return final_chunks