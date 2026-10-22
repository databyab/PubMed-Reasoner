"""Simple baseline runners sharing the selected Groq SDK client and budgets."""
from ..pipeline.orchestrator import Reasoner
from ..models.schemas import AnswerDraft,Citation
from ..llm.base import LLMError
import re
import time

def run_baseline(method,question,*,llm,pubmed=None,context="",**options):
    if method=="direct_llm":
        started=time.monotonic()
        text=llm.generate_text("Answer this biomedical question directly. Do not invent citations. State uncertainty.",f"QUESTION: {question}\nCONTEXT: {context or 'None'}")
        return {"method":method,"answer":text,"usage":llm.get_usage(),"citations":[],"latency_seconds":round(time.monotonic()-started,3)}
    if method=="standard_rag":
        config={**options,"refinement_rounds":0,"reflective_filtering":False,"early_stopping":False}
        result=Reasoner(llm=llm,pubmed=pubmed,**config).run(question,context)
        result.trace["method"]="standard_rag_single_query"
        return result
    if method=="self_reflection":
        config={**options,"early_stopping":False}
        result=Reasoner(llm=llm,pubmed=pubmed,**config).run(question,context)
        if result.evidence and llm.calls < options.get("max_llm_calls",30):
            review=llm.generate_structured_output(AnswerDraft,"Review the draft answer for unsupported claims or missing citations. Rewrite only from these evidence excerpts. Cite only supplied PMIDs.",f"QUESTION: {question}\nDRAFT: {result.answer}\nEVIDENCE: {[e.model_dump() for e in result.evidence]}")
            known={e.pmid for e in result.evidence}
            inline=set(re.findall(r"PMID\s*:\s*(\d+)",review.answer))
            if review.cited_pmids and set(review.cited_pmids).issubset(known) and inline==set(review.cited_pmids):
                result.answer=review.answer
                source={a.pmid:a for a in result.articles}
                result.citations=[Citation(pmid=p,title=source[p].title,url=source[p].source_url) for p in dict.fromkeys(inline) if p in source]
            result.trace["posthoc_reflection"]={"limitations":review.limitations,"cited_pmids":review.cited_pmids,"accepted":bool(review.cited_pmids and set(review.cited_pmids).issubset(known) and inline==set(review.cited_pmids))}
        result.trace["method"]="self_reflection_posthoc_approximation"
        return result
    raise ValueError(f"Unknown baseline: {method}")

