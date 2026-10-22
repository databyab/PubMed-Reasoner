from .query_normalizer import normalize_query

def build_query(concepts: list[str], mesh_terms: list[str] | None = None) -> str:
    groups=[]
    for term in mesh_terms or []: groups.append(f'"{term.replace(chr(34), "")}"[MeSH Terms]')
    for concept in concepts:
        if concept.strip(): groups.append(f'"{concept.strip().replace(chr(34), "")}"[Title/Abstract]')
    if not groups: raise ValueError("At least one search concept is required")
    # Concept clauses are ANDed; MeSH evidence is treated as another candidate concept in this MVP.
    return normalize_query(" AND ".join(groups))

