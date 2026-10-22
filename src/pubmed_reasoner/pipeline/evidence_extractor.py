"""Deterministic checks for excerpt-to-record source linking."""
import re
import unicodedata
from ..models.schemas import Evidence

def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()

def validate_evidence(evidence: Evidence, sources) -> bool:
    """Require an accepted nonempty passage tied to a retrieved source record.

    A PMID set remains supported for compatibility, but a source mapping also
    verifies that the passage occurs in that PMID's actual abstract.
    """
    if not evidence.aligned or not evidence.passage.strip():
        return False
    if isinstance(sources, (set, frozenset)):
        return evidence.pmid in sources
    source=sources.get(evidence.pmid) if sources else None
    if source is None:
        return False
    abstract=getattr(source,"abstract","") or ""
    passage=_normalized(evidence.passage)
    return bool(passage and passage in _normalized(abstract))

def usable_evidence(evidence_items, sources):
    """Return only excerpts linked to retrieved records and their abstracts."""
    return [item for item in evidence_items if validate_evidence(item,sources)]

def evidence_sufficiency_gate(evidence_items, sources, min_excerpts: int, min_unique_pmids: int):
    usable=usable_evidence(evidence_items,sources)
    unique_pmids={item.pmid for item in usable}
    return {
        "usable_excerpts":len(usable),
        "unique_retrieved_pmids":len(unique_pmids),
        "minimum_usable_excerpts":min_excerpts,
        "minimum_unique_pmids":min_unique_pmids,
        "passed":len(usable)>=min_excerpts and len(unique_pmids)>=min_unique_pmids,
    }

