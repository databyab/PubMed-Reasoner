"""Citation validation for generated answers."""
import re
def cited_pmids(answer: str) -> set[str]: return set(re.findall(r"PMID\s*:\s*(\d+)",answer,re.I))
def citations_supported(answer: str, evidence_pmids: set[str]) -> bool:
    cited=cited_pmids(answer)
    return bool(cited) and cited.issubset(evidence_pmids)

def valid_citation_mapping(answer: str, declared_pmids, evidence_pmids, retrieved_pmids) -> bool:
    """Check citation strings, structured IDs, validated evidence, and sources agree."""
    return not citation_mapping_diagnostics(answer,declared_pmids,evidence_pmids,retrieved_pmids)["reasons"]

def citation_mapping_diagnostics(answer: str, declared_pmids, evidence_pmids, retrieved_pmids) -> dict:
    """Return non-content diagnostics for a citation mapping decision."""
    inline=cited_pmids(answer)
    declared={str(pmid).strip() for pmid in declared_pmids if str(pmid).strip()}
    evidence_ids=set(evidence_pmids)
    retrieved_ids=set(retrieved_pmids)
    reasons=[]
    if not inline: reasons.append("no_inline_pmid_citations")
    if inline!=declared: reasons.append("inline_and_structured_pmids_differ")
    if not inline.issubset(evidence_ids): reasons.append("citation_not_in_usable_evidence")
    if not inline.issubset(retrieved_ids): reasons.append("citation_not_in_retrieved_records")
    return {"valid":not reasons,"inline_pmids":sorted(inline),"declared_pmids":sorted(declared),"reasons":reasons}

