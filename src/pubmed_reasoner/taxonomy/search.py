"""Normalized lookup over a JSONL MeSH dump."""
import json, re
from pathlib import Path
from .models import MeshDescriptor

def normalize(s: str) -> str: return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()

class MeshIndex:
    def __init__(self, descriptors: list[MeshDescriptor]):
        self.items = descriptors
        self.by_id = {x.descriptor_id: x for x in descriptors}
    @classmethod
    def from_jsonl(cls, path: str | Path):
        with open(path, encoding="utf8") as f: return cls([MeshDescriptor.model_validate_json(line) for line in f if line.strip()])
    def search(self, query: str, limit: int = 20):
        q = normalize(query); hits = []
        for d in self.items:
            name = normalize(d.name); terms = [normalize(t) for t in d.entry_terms]
            score = 3 if name == q else 2 if q and (q in name or name in q) else 0
            if q in terms: score = max(score, 2)
            elif any(q and q in term for term in terms): score = max(score, 1)
            if score: hits.append((score, d))
        return [d for _,d in sorted(hits, key=lambda p: (-p[0], p[1].name))[:limit]]
    def save_jsonl(self, path: str | Path):
        target=Path(path); target.parent.mkdir(parents=True, exist_ok=True); tmp=target.with_suffix(target.suffix+".tmp")
        with tmp.open("w", encoding="utf8") as f:
            for x in self.items: f.write(x.model_dump_json()+"\n")
        tmp.replace(target)

