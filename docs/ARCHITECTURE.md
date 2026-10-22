# Architecture

`models.schemas` carries validated articles, critique decisions, evidence, and responses. `retrieval` handles PubMed syntax, E-utilities and XML records. `taxonomy` downloads/parses NLM annual files and performs JSONL lookup. `llm` contains the sole live integration, the official Groq SDK, plus a clearly labeled mock. `pipeline.orchestrator.Reasoner` coordinates query planning/refinement, ranked batched retrieval, evidence extraction/sufficiency and cited response generation. `app.py` is the interactive Streamlit adapter; the CLI supports taxonomy and retrieval.

Each run writes an atomic JSON trace to `OUTPUT_DIR` with a run ID, time, query history, returned records, model mode/usage and stop reason. No secrets are persisted. Current gaps: query/article caching is in-memory only; there is no claim-level entailment evaluator or automatic Groq pricing calculator.

