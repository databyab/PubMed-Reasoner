"""Command line for taxonomy, search, and reasoning."""
import argparse, json
from pathlib import Path
from .pipeline.orchestrator import Reasoner
from .retrieval.pubmed_client import PubMedClient
from .taxonomy.search import MeshIndex
from .logging_config import configure_logging

def main():
    configure_logging()
    parser=argparse.ArgumentParser(prog="pubmed-reasoner"); sub=parser.add_subparsers(dest="cmd",required=True)
    m=sub.add_parser("mesh-search"); m.add_argument("term"); m.add_argument("--limit",type=int,default=10)
    s=sub.add_parser("search"); s.add_argument("query"); s.add_argument("--limit",type=int,default=10)
    r=sub.add_parser("reason"); r.add_argument("question"); r.add_argument("--context",default="")
    args=parser.parse_args()
    if args.cmd=="mesh-search":
        index=MeshIndex.from_jsonl(Path("data/mesh/descriptors.jsonl")); print(json.dumps([x.model_dump() for x in index.search(args.term,args.limit)],indent=2))
    elif args.cmd=="search": print(json.dumps([x.model_dump(mode="json") for x in PubMedClient().search_and_fetch(args.query,args.limit)],indent=2))
    else: print(Reasoner().run(args.question,args.context).model_dump_json(indent=2))

if __name__=="__main__": main()

