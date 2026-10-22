# Dataset acquisition

PubMedQA has expert-labeled, unlabeled and artificially generated examples. The official repository is the source of truth for split names and format; verify files before use. Download manually or through official instructions and store under `data/evaluation/` (ignored by Git). `load_pubmedqa` handles top-level mappings or arrays and optional `final_decision`/`LABEL` fields. It reports labeled count, label distribution and duplicate question count. Do not train on the official test split or use unlabeled examples as labels.

