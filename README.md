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

Add your Groq key to `.env` to use live model reasoning:

```dotenv
GROQ_API_KEY=your-key
GROQ_MODEL=openai/gpt-oss-120b
LLM_MODE=mock
NCBI_EMAIL=you@example.com
NCBI_API_KEY=
```

`GROQ_MODEL` is optional in the UI: with a key present, the app fetches the active model list and lets you choose a model. `LLM_MODE=mock` is the safe default; switch off **Mock/demo mode** in the sidebar to run live Groq calls. Mock mode can demonstrate the interface and retrieve PubMed records, but it does not make real relevance or sufficiency judgments or generate a biomedical conclusion. NCBI credentials are optional; `NCBI_EMAIL` and `NCBI_API_KEY` are recommended for responsible API use and higher request limits.

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
   LLM_MODE = "mock"
   NCBI_EMAIL = "you@example.com"
   ```

   The app reads these values from Community Cloud secrets. Keep `LLM_MODE="mock"` if you want the demo to start in mock mode; turn off the mock toggle in the app to use Groq. Leave `NCBI_API_KEY` out if you do not have one.

4. Deploy. No PubMed corpus or MeSH download is required. The app calls PubMed and, when live mode is selected, Groq at runtime. Without `GROQ_API_KEY`, use mock/demo mode.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) for current steps.

## How a run works

1. **Plan and refine:** the LLM creates a PubMed query plan. In each enabled refinement round, the app retrieves preliminary records and asks the LLM to critique query coverage, alignment, and redundancy. Optional local MeSH data can validate or expand terms.
2. **Retrieve and assess:** PubMed returns article IDs and abstracts. The LLM assesses each abstract individually and extracts a short passage. The app only accepts an extracted passage if it appears verbatim in that abstract. After a batch, the LLM can assess evidence sufficiency; early stopping also requires at least two distinct PMIDs.
3. **Answer and inspect:** if accepted evidence exists, the LLM drafts a cited answer. Code checks that cited PMIDs belong to the retrieved evidence set. The UI exposes the records, excerpts, query history, termination reason, and usage details.

There is no vector database or embedding-based nearest-neighbor search. PubMed performs the initial matching against its own index. The optional MeSH JSONL file is a terminology index, not a vector store.

A run's Groq calls vary with the number of refinement rounds, abstracts assessed, sufficiency checks, and whether an answer is generated. **Maximum Groq calls** caps the run. The Run details view reports actual calls and token counts. PubMed requests are separate from LLM calls.

## Modes and comparisons

The sidebar lets you select the Full Reasoner, Standard RAG (single query), no self-critic refinement, no early stopping, post-hoc self-reflection, or Direct Groq LLM. Run methods on the same question and model to compare their outputs and resource use. Accuracy and macro F1 are shown only when supported by a labeled evaluation; the interactive comparison does not estimate them.

## Benchmark evaluation

PubMedQA is not bundled. Obtain the dataset from the [official PubMedQA repository](https://github.com/pubmedqa/pubmedqa). Evaluation requires real Groq predictions; mock mode is refused. The examples below show the expected workflow after placing an authorized labeled split under `data/evaluation/`:

```bash
python scripts/run_evaluation.py data/evaluation/<labeled-split>.json --limit 50 --seed 13 --methods full,standard_rag,direct_llm
python scripts/run_ablations.py data/evaluation/<labeled-split>.json --limit 50 --seed 13
python scripts/generate_report.py data/results/evaluation/metrics.json
```

Read `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`, and `docs/EXPERIMENTS.md` for implementation details and evaluation caveats. Never use this prototype for patient care.
