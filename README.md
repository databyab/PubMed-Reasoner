# PubMed Reasoner

A Streamlit research prototype for answering biomedical questions with PubMed abstracts. It extends a basic one-query RAG workflow with iterative query critique, per-article relevance assessment, and an evidence sufficiency check that can stop retrieval early.

This is a methodology prototype, not a clinical tool or an exact reproduction of the paper. Its judgments are model-generated heuristics and must be checked against the source articles.

## Why add reasoning around RAG?

A basic RAG workflow usually searches once, retrieves a fixed set of records, and asks a model to answer from them. PubMed Reasoner adds feedback around that workflow:

| Basic single-query RAG | PubMed Reasoner |
| --- | --- |
| Uses one initial search query | Creates a query plan and can critique/refine the query using preliminary PubMed results |
| Passes retrieved records to the answer step | Assesses abstracts individually and keeps relevant, verbatim evidence passages |
| Typically uses a fixed retrieval set | Can assess evidence sufficiency after batches and stop early when its checks pass |
| Often hides why a result was retrieved | Shows query history, records, excerpts, assessments, citations, budgets, and call counts |

These features are intended to make retrieval more adaptive and the process easier to inspect. They do **not** establish that it is more accurate than RAG; that requires evaluation on labeled data. The app includes a Standard RAG baseline for same-question comparisons.

## Data and deployment size

**No article dataset or local database is required to deploy the demo.** PubMed search and record fetching happen live through NCBI E-utilities. The GitHub repository can contain the source code and dependency files; result JSON files are generated at runtime and are not needed in the repository.

The local MeSH taxonomy is optional. Without it, PubMed search still works, but local descriptor validation and MeSH query expansion are unavailable. The runtime descriptor index is about **20 MB**. The full downloaded taxonomy directory in this project was about **319 MB**, mostly source XML; that XML is not needed by the app after the descriptor index is built. The repository ignores `data/mesh/`, so do not commit the full taxonomy release just to deploy.

PubMedQA or another labeled dataset is only needed if you want to run benchmark evaluations. It is not needed for the interactive demo.

## Run locally

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add the key for the provider you want to use to `.env`. Gemini is the recommended and preselected live provider; provider selection stays configurable:

```dotenv
GROQ_API_KEY=your-key
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
DEFAULT_LLM_PROVIDER=gemini
LLM_MODE=mock
MIN_USABLE_EVIDENCE=3
MIN_UNIQUE_EVIDENCE_PMIDS=2
NCBI_EMAIL=you@example.com
NCBI_API_KEY=
```

The sidebar lets you select Groq or Gemini and loads models available to the configured API key. Set the matching provider key. `LLM_MODE=mock` is the demo-safe default; switch off **Mock/demo mode** to run live calls, with Gemini selected by default. For CLI use, set `LLM_MODE=gemini` with `GEMINI_MODEL`, or `LLM_MODE=groq` with `GROQ_MODEL`. The deterministic early-stop gate requires the configured minimum usable excerpts and distinct PMIDs before asking the model whether evidence is sufficient. Mock mode retrieves records but makes no real relevance or sufficiency judgments and produces no biomedical conclusion. NCBI credentials are optional; `NCBI_EMAIL` and `NCBI_API_KEY` are recommended for responsible API use and higher request limits.

Start the app:

```bash
streamlit run app.py
```

Optional: build the local MeSH index once if you want descriptor validation and expansion:

```bash
python scripts/download_mesh.py
```

## Deploy from GitHub with Streamlit Community Cloud

1. Push the project to a GitHub repository. Keep `.env` and credentials out of GitHub; `.env` is ignored by Git.
2. In Streamlit Community Cloud, create an app from the repository and set the entrypoint to `app.py`. The `requirements.txt` installs the project package and its dependencies.
3. In the app's **Advanced settings → Secrets**, add your credentials as TOML, for example:

   ```toml
   GROQ_API_KEY = "your-key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   GEMINI_API_KEY = "your-gemini-key"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   DEFAULT_LLM_PROVIDER = "gemini"
   LLM_MODE = "mock"
   MIN_USABLE_EVIDENCE = 3
   MIN_UNIQUE_EVIDENCE_PMIDS = 2
   NCBI_EMAIL = "you@example.com"
   ```

   The app reads these values from Community Cloud secrets. Add either or both provider keys. `DEFAULT_LLM_PROVIDER` sets the recommended live selection; `GEMINI_MODEL` and `GROQ_MODEL` can be changed in the sidebar. Keep `LLM_MODE="mock"` if you want the demo to start in mock mode; turn off the mock toggle to use the selected provider. Leave `NCBI_API_KEY` out if you do not have one.

4. Deploy. No PubMed corpus or MeSH download is required. The app calls PubMed and, when live mode is selected, the configured Groq or Gemini provider at runtime. Without the selected provider's API key, use mock/demo mode.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) for current steps.

## How a run works

1. **Plan and refine:** the LLM creates a PubMed query plan. In each enabled refinement round, the app retrieves preliminary records and asks the LLM to critique query coverage, alignment, and redundancy. Optional local MeSH data can validate or expand terms.
2. **Retrieve and assess:** PubMed returns article IDs and abstracts. The LLM assesses each abstract individually and extracts a short passage. The app accepts an excerpt only when it matches the abstract belonging to that retrieved PMID. Before the model is asked to judge sufficiency, deterministic checks require the configured minimum excerpt count and unique source PMID count.
3. **Answer and inspect:** the LLM drafts a cited answer. Code checks citation identifiers against accepted evidence and retrieved records. Citations are source-linked; claim-level entailment is not verified. The UI exposes the records, excerpts, query history, termination reason, provider/model, and usage details.

There is no vector database or embedding-based nearest-neighbor search. PubMed performs the initial matching against its own index. The optional MeSH JSONL file is a terminology index, not a vector store.

A run's model calls vary with refinement rounds, abstracts assessed, sufficiency checks, and answer generation. **Maximum model calls** caps the run. Run details report provider, model, app-initiated generation calls, available token counts, latency, PubMed searches/requests, and cache hits. Provider SDK retries may create additional HTTP requests beyond the app-initiated call count. Output-token totals use the provider-reported definition; reasoning tokens may be included in that total. PubMed requests are separate from model calls. Provider-specific cost is reported as unavailable unless pricing is explicitly configured. Cached searches are keyed by the exact query and date filters; a larger result set can serve a smaller request for that same query, but not a different query.

## Provider selection and observed runs

Gemini is the demo's recommended default based on the project's latest user-observed run: 15.487 seconds, 10 LLM calls, and 3 accepted evidence excerpts, compared with an earlier Groq observation of 150.21 seconds, 23 calls, and zero accepted excerpts. These are individual observations under potentially different run conditions, not a controlled benchmark and not evidence of general model superiority. Earlier saved local traces also show variable outcomes across questions and runs. See `docs/EXPERIMENTS.md` for limitations. Groq remains supported; citation diagnostics now record inline/schema PMID mismatches, and a citation failure no longer erases accepted excerpts.

## Modes and comparisons

The sidebar lets you select the Full Reasoner, Standard RAG (single query), no self-critic refinement, no early stopping, post-hoc self-reflection, or Direct LLM. Run methods on the same question and provider/model to compare their outputs and resource use. Accuracy and macro F1 are shown only when supported by a labeled evaluation; the interactive comparison does not estimate them. The included PubMedQA evaluation scripts currently support Groq only.

## Benchmark evaluation

PubMedQA is not bundled. Obtain the dataset from the [official PubMedQA repository](https://github.com/pubmedqa/pubmedqa). The included evaluation scripts currently support Groq predictions; mock mode is refused. The examples below show the expected workflow after placing an authorized labeled split under `data/evaluation/`:

```bash
python scripts/run_evaluation.py data/evaluation/<labeled-split>.json --limit 50 --seed 13 --methods full,standard_rag,direct_llm
python scripts/run_ablations.py data/evaluation/<labeled-split>.json --limit 50 --seed 13
python scripts/generate_report.py data/results/evaluation/metrics.json
```

Read `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`, and `docs/EXPERIMENTS.md` for implementation details and evaluation caveats. Never use this prototype for patient care.
