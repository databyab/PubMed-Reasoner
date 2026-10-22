"""Evidence model and strict citation membership validation."""
from ..models.schemas import Evidence

def validate_evidence(evidence: Evidence, known_pmids: set[str]) -> bool:
    return evidence.aligned and bool(evidence.passage.strip()) and evidence.pmid in known_pmids

