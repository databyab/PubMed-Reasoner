# Experiments

The supported initial dataset interface reads official PubMedQA JSON and preserves the source path, labels where present, class distribution, and duplicate-question count. Start with 50 fixed examples after downloading and recording the official split and a deterministic sample. Check for train/test leakage and never infer labels for unlabeled files.

Run actual labeled evaluation with a configured active Groq model:

```bash
LLM_MODE=groq GROQ_MODEL='<active-model-id>' python scripts/run_evaluation.py data/evaluation/<official-labeled-split>.json --limit 50 --seed 13 --methods full,standard_rag,direct_llm
python scripts/run_ablations.py data/evaluation/<official-labeled-split>.json --limit 50 --seed 13
```

Each method uses the same deterministic sample and model, and results include accuracy, macro F1, per-class metrics, citation validity by retrieved PMID, evidence-grounded rate, article count, calls, tokens, PubMed requests, latency, errors, and a secret-free manifest. Model calls are real, incur provider quotas/costs, and should be started only after checking the sample and budgets. Citation membership is not entailment. Entailment and retrieval precision/recall remain unavailable absent annotations or an explicitly documented human-review protocol. The published paper's scores are never copied into measured files.

To generate a Markdown comparison from the latest metric summary, run `python scripts/generate_report.py data/results/evaluation/metrics.json`.

The Streamlit demo can execute full, no-self-critic, no-early-stop, single-query RAG, direct model, and post-hoc self-reflection on a question with either supported provider. The self-reflection variant is an adaptation: it disables early stopping then performs one post-hoc evidence review. Abstract-only is the retrieval scope of the current implementation; a standalone taxonomy-disabled toggle is provided. Baseline comparability requires the same model, question, article/call budgets, output constraints and decoding settings. Never interpret LLM-as-judge output as objective truth.

The latest user-observed live comparison reported Gemini at 15.487 seconds, 10 LLM calls and 3 accepted excerpts, and an earlier Groq run at 150.21 seconds, 23 calls and 0 accepted excerpts. Treat these as observations from individual runs only: they are not a controlled benchmark or evidence of general model superiority. Existing saved local traces include different questions and outcomes, so they do not reproduce that comparison. Do not combine runs into a provider ranking without a controlled, repeated evaluation.

