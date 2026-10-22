# Architecture

`models.schemas` carries validated articles, critique decisions, evidence, and responses. `retrieval` handles PubMed syntax, E-utilities and XML records. `taxonomy` downloads/parses NLM annual files and performs JSONL lookup. `llm` contains the official Groq and Google Gen AI SDK integrations plus a clearly labeled mock; both live clients implement the same provider interface. `pipeline.orchestrator.Reasoner` coordinates query planning/refinement, ranked batched retrieval, evidence extraction/sufficiency and cited response generation. `app.py` is the interactive Streamlit adapter; the CLI supports taxonomy and retrieval.

Each run writes an atomic JSON trace to `OUTPUT_DIR` with a run ID, time, query history, returned records, model mode/usage and stop reason. No secrets are persisted. PubMedQA benchmark evaluation currently uses Groq only. Current gaps: query/article caching is in-memory only; there is no claim-level entailment evaluator or automatic provider pricing calculator.

