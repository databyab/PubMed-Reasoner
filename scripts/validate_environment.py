#!/usr/bin/env python3
import sys
print(f"Python {sys.version}")
try:
 import pubmed_reasoner
 from pubmed_reasoner.config import settings
 print(f"pubmed_reasoner {pubmed_reasoner.__version__}; LLM mode={settings.llm_mode}; MeSH={'available' if __import__('pathlib').Path('data/mesh/descriptors.jsonl').exists() else 'not downloaded'}")
except Exception as exc: raise SystemExit(f"Environment invalid: {exc}")

