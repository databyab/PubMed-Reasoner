"""Citation validation for generated answers."""
import re
def cited_pmids(answer: str) -> set[str]: return set(re.findall(r"PMID\s*:\s*(\d+)",answer))
def citations_supported(answer: str, evidence_pmids: set[str]) -> bool:
    cited=cited_pmids(answer)
    return bool(cited) and cited.issubset(evidence_pmids)

