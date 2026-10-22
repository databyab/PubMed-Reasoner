"""Read official PubMedQA JSON layouts without pretending unlabeled subsets are labeled."""
import json
from pathlib import Path

def load_pubmedqa(path: str|Path):
    p=Path(path); raw=json.loads(p.read_text(encoding="utf8")); rows=[]
    if isinstance(raw,dict):
        for key,value in raw.items():
            if isinstance(value,dict): rows.append({"id":key,**value})
    elif isinstance(raw,list): rows=raw
    validated=[]
    for row in rows:
        question=row.get("question")
        if question is None and isinstance(row.get("QUESTION"),str): question=row["QUESTION"]
        if not question: continue
        label=row.get("final_decision",row.get("LABEL"))
        validated.append({"id":str(row.get("id",row.get("PMID",""))),"question":question,"label":label,"context":row.get("context",row.get("CONTEXT")),"raw":row})
    return {"examples":validated,"provenance":str(p),"n":len(validated),"labeled":sum(x["label"] is not None for x in validated),"label_distribution":dict(__import__("collections").Counter(x["label"] for x in validated if x["label"] is not None)),"duplicates":len(validated)-len(set(x["question"] for x in validated))}

