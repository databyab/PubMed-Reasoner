#!/usr/bin/env python3
"""Download official annual MeSH XML, parse, validate and write JSONL."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pubmed_reasoner.taxonomy.downloader import download_mesh
from pubmed_reasoner.taxonomy.parser import parse_mesh_xml
from pubmed_reasoner.taxonomy.search import MeshIndex
from pubmed_reasoner.taxonomy.validator import validate_descriptors

def main():
    p=argparse.ArgumentParser(); p.add_argument("--year",type=int,default=2026); p.add_argument("--output",default="data/mesh"); a=p.parse_args()
    out=Path(a.output); manifest=download_mesh(a.year,out)
    desc=parse_mesh_xml(out/f"desc{a.year}.xml"); quals=parse_mesh_xml(out/f"qual{a.year}.xml","qualifier")
    report=validate_descriptors(desc)
    if not report["valid"]: raise SystemExit(f"MeSH validation failed: {report}")
    MeshIndex(desc).save_jsonl(out/"descriptors.jsonl"); MeshIndex(quals).save_jsonl(out/"qualifiers.jsonl")
    print(json.dumps({**report,"qualifiers":len(quals),"manifest":manifest},indent=2))
if __name__=="__main__": main()

