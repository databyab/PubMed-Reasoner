#!/usr/bin/env python3
"""Run a reproducible labeled PubMedQA pilot with configured Groq credentials."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pubmed_reasoner.evaluation.runner import run_pubmedqa

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--limit",type=int,default=50); p.add_argument("--seed",type=int,default=13); p.add_argument("--output",default="data/results/evaluation"); p.add_argument("--model"); p.add_argument("--methods",default="full",help="Comma-separated: full,standard_rag,self_reflection,direct_llm,no_refinement,no_mesh,no_early_stop"); a=p.parse_args()
    result=run_pubmedqa(a.dataset,a.output,methods=tuple(x.strip() for x in a.methods.split(",") if x.strip()),limit=a.limit,seed=a.seed,model=a.model)
    print(json.dumps({"run_id":result["run_id"],"metrics":result["metrics"],"manifest":result["manifest"]},indent=2))
if __name__=="__main__": main()
