"""Conservative normalization for candidate PubMed boolean queries."""
import re

def normalize_query(query: str) -> str:
    q=" ".join(query.strip().split())
    q=re.sub(r"\b(and|or|not)\b", lambda m:m.group(1).upper(), q, flags=re.I)
    q=re.sub(r"\s+(AND|OR|NOT)\s+", r" \1 ", q)
    if not q: raise ValueError("PubMed query cannot be empty")
    if q.count("(") != q.count(")"): raise ValueError("Unbalanced parentheses in PubMed query")
    if re.search(r"\b(AND|OR|NOT)\s*$|^\s*(AND|OR|NOT)\b",q): raise ValueError("Boolean operator has no operand")
    return q

