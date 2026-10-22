def citation_validity(cited_pmids, retrieved_pmids):
    cited=list(cited_pmids)
    return None if not cited else sum(p in set(retrieved_pmids) for p in cited)/len(cited)

