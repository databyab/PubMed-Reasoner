#!/usr/bin/env python3
"""Run the implemented method/ablation set on the same labeled sample."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pubmed_reasoner.evaluation.runner import run_pubmedqa
from pubmed_reasoner.evaluation.ablations import ABLATION_METHODS

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--limit",type=int,default=50); p.add_argument("--seed",type=int,default=13); p.add_argument("--output",default="data/results/ablations"); p.add_argument("--model"); a=p.parse_args()
    result=run_pubmedqa(a.dataset,a.output,methods=ABLATION_METHODS,limit=a.limit,seed=a.seed,model=a.model)
    print(json.dumps({"run_id":result["run_id"],"metrics":result["metrics"]},indent=2))
if __name__=="__main__": main()
