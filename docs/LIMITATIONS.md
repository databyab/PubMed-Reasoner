# Limitations and safety

- Retrieval bias: PubMed ranking and query vocabulary can omit relevant lower-ranked or newly indexed studies.
- Abstract-only evidence omits methods, tables, corrections and full-text nuance. Evidence extraction can misread abstracts.
- LLMs can produce invalid or unsupported output; citation membership proves only that a PMID was retrieved, not entailment. Human review is needed for biomedical use.
- Multiple articles may report the same cohort; this implementation does not deduplicate underlying studies.
- LLM-as-judge metrics are model-, prompt- and sampling-dependent and are not objective truth.
- API responses, indexed records, model behavior and pricing vary over time; manifests should retain exact settings.
- Token usage is logged, but cost remains unavailable unless a current per-model price schedule is added; the project does not hardcode transient pricing.
- Benchmark leakage and PubMedQA label imbalance can inflate apparent performance. The loader does not by itself prevent leakage.
- The tool is not validated for clinical decisions and must not provide patient-specific diagnosis or treatment instructions.
- This is an educational reproduction attempt, not a verified reproduction of the paper's empirical results.

