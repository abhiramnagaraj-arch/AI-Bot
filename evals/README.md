# Synthetic RAG evaluation set

`cases.json` contains 50 questions: 45 answerable paraphrases across fictional leave, travel and security policies, plus 5 questions whose answers are absent. `documents/` contains only synthetic fixtures, not company policy or user uploads.

Run:

```bash
python -m scripts.evaluate --mode retrieval --output runtime/retrieval-results.json
python -m scripts.evaluate --mode generation --output runtime/generation-results.json
```

Retrieval evaluates whether the expected source appears within the configured top-k results after the score cutoff. This is a small source-level smoke test; because the corpus has few documents, a high recall score is easy to achieve and does not establish passage precision or production accuracy.

Generation records status agreement, reference-term matches, outputs and citations. Term matching is only a diagnostic. Manually assess correctness, complete conditions/exceptions, claim support, citation alignment, refusal/clarification and response usefulness. Add representative private documents locally and keep held-out release questions separate from tuning data before using this as a release gate.

`python -m scripts.smoke_live` separately checks actual ingestion and a multi-turn conversation against Ollama using an isolated temporary synthetic policy. `python -m scripts.browser_smoke` tests the UI/API/parser using a deterministic model, not model quality.
