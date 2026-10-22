# Methodology mapping and reproduction status

The cited paper describes (A) self-critic query refinement against partial PubMed metadata using coverage, alignment and redundancy; (B) coarse filtering and evidence extraction in batches, with sufficiency checks and early stopping; and (C) grouped evidence summaries and cited answer generation. The inspected official code is a compact Python implementation using LangChain OpenAI, `PubMedClient`, prompt modules, `max_self_critic_iterations=3`, `max_articles=20`, and `batch_size=5`. It uses direct JSON parsing, ESearch/EFetch title/abstract retrieval, and includes baseline/evaluation folders. It does not ship a MeSH release workflow or the complete reproducibility framework requested here. This adaptation uses Groq only, so provider/model results are not numerically comparable to the GPT-4o experiments.

## Faithful methodological components

- Three stage flow, metadata feedback dimensions, evidence batch size default of five, article budget, early-stop concept, and source-linked PubMed citations follow the paper and official implementation's central structure.
- The implementation uses E-utilities and article abstracts, matching the source implementation's retrieval scope.

## Approximations

- This prototype's default limits are conservative and configurable, but its LLM prompts/contracts are new and not the original prompt text. Query planning currently generates a structured query; the optional downloaded taxonomy is searchable but not yet incorporated to automatically canonicalize every model term.
- Evidence filtering is performed per record by provider call, rather than the original code's batched coarse-filter prompt followed by per-article extraction.
- The short official implementation itself has incomplete validation, retry, cache, provenance and error handling. This project adds these engineering layers, so it is not line-for-line reproduction.
- Baselines are implemented adaptations and are not verified to match the paper's complete protocols.

## Not reproduced / not measured

No published accuracy or judge result is copied into our result artifacts. We have not run a real benchmark. The paper reports GPT-4o and comparisons (including PubMedQA); claims in the paper are not results from this project. Exact prompts, experimental splits, decoding parameters, evaluation scripts and all appendix configurations must be independently verified before calling this an exact replication.

