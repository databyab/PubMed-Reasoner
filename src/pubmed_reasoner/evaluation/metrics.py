"""Transparent classification metrics without silently manufacturing labels."""
from collections import Counter

def classification_metrics(gold: list[str], pred: list[str]) -> dict:
    if len(gold)!=len(pred): raise ValueError("gold and prediction lengths differ")
    if not gold: return {"accuracy":None,"macro_f1":None,"n":0,"per_class":{}}
    labels=sorted(set(gold)|set(pred)); per={}; f1s=[]
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(gold,pred)); fp=sum(g!=label and p==label for g,p in zip(gold,pred)); fn=sum(g==label and p!=label for g,p in zip(gold,pred))
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f=2*prec*rec/(prec+rec) if prec+rec else 0
        per[label]={"precision":prec,"recall":rec,"f1":f,"support":sum(g==label for g in gold)}; f1s.append(f)
    return {"accuracy":sum(g==p for g,p in zip(gold,pred))/len(gold),"macro_f1":sum(f1s)/len(f1s),"n":len(gold),"per_class":per}

