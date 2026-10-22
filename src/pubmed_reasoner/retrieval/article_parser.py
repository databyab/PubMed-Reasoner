"""Parse PubMed EFetch XML without losing mixed-content titles or abstracts."""
import xml.etree.ElementTree as ET
from ..models.schemas import Article

def _alltext(node): return "".join(node.itertext()).strip() if node is not None else ""

def parse_pubmed_xml(xml: str | bytes, query: str = "") -> list[Article]:
    root=ET.fromstring(xml); out=[]
    for item in root.findall(".//PubmedArticle"):
        med=item.find("MedlineCitation"); art=med.find("Article") if med is not None else None
        if med is None or art is None: continue
        pmid=_alltext(med.find("PMID")); title=_alltext(art.find("ArticleTitle"))
        abstracts=[_alltext(n) for n in art.findall(".//Abstract/AbstractText")]
        journal=_alltext(art.find("Journal/Title")) or None
        authors=[]
        for au in art.findall("AuthorList/Author"):
            name=" ".join(filter(None,[_alltext(au.find("ForeName")),_alltext(au.find("LastName"))]))
            if name: authors.append(name)
        doi=None
        for aid in item.findall(".//ArticleId"):
            if aid.get("IdType")=="doi": doi=_alltext(aid); break
        date_node=art.find("Journal/JournalIssue/PubDate")
        date=" ".join(_alltext(x) for x in list(date_node or [])) or None
        headings=[_alltext(x.find("DescriptorName")) for x in med.findall(".//MeshHeading") if x.find("DescriptorName") is not None]
        out.append(Article(pmid=pmid,title=title,abstract=" ".join(filter(None,abstracts)),publication_date=date,authors=authors,journal=journal,doi=doi,mesh_headings=headings,source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",original_query=query))
    return out

