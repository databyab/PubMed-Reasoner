"""The three research stages with explicit limits and source-grounded evidence."""
import json, logging, re, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from ..config import settings
from ..llm.factory import create_llm
from ..llm.base import LLMError
from ..models.schemas import QueryPlan,CriticFeedback,SufficiencyDecision,Evidence,EvidenceExtraction,AnswerDraft,ReasonResponse,Citation
from .evidence_extractor import validate_evidence,usable_evidence,evidence_sufficiency_gate
from .answer_generator import cited_pmids,citation_mapping_diagnostics
from ..retrieval.pubmed_client import PubMedClient
from ..retrieval.query_normalizer import normalize_query
from ..retrieval.query_builder import build_query

LOG=logging.getLogger(__name__)

class Reasoner:
    def __init__(self,pubmed=None,llm=None,max_articles=None,batch_size=None,refinement_rounds=None,max_llm_calls=None,min_usable_evidence=None,min_unique_evidence_pmids=None,mesh_expansion=True,reflective_filtering=True,early_stopping=True,progress_callback=None):
        self.pubmed=pubmed or PubMedClient(); self.llm=llm or create_llm()
        self.max_articles=max_articles if max_articles is not None else settings.max_articles
        self.batch_size=batch_size if batch_size is not None else settings.batch_size
        self.refinement_rounds=settings.max_refinement_rounds if refinement_rounds is None else refinement_rounds
        self.max_llm_calls=max_llm_calls if max_llm_calls is not None else settings.max_llm_calls
        self.min_usable_evidence=min_usable_evidence if min_usable_evidence is not None else settings.min_usable_evidence
        self.min_unique_evidence_pmids=min_unique_evidence_pmids if min_unique_evidence_pmids is not None else settings.min_unique_evidence_pmids
        self.mesh_expansion=mesh_expansion; self.reflective_filtering=reflective_filtering; self.early_stopping=early_stopping
        self.progress_callback=progress_callback or (lambda *_:None)
        self.mesh=None
        p=Path("data/mesh/descriptors.jsonl")
        if p.exists():
            from ..taxonomy.search import MeshIndex
            self.mesh=MeshIndex.from_jsonl(p)
    def _budget(self):
        if self.llm.calls>=self.max_llm_calls: raise LLMError("Configured LLM-call budget exhausted")
    def _structured(self,schema,system,prompt): self._budget(); return self.llm.generate_structured_output(schema,system,prompt)
    def _text(self,system,prompt): self._budget(); return self.llm.generate_text(system,prompt)
    def _plan_query(self,plan):
        validation=[]; confirmed=[]
        for term in plan.mesh_terms:
            matches=self.mesh.search(term,limit=3) if self.mesh and self.mesh_expansion else []
            exact=next((m for m in matches if m.name.casefold()==term.casefold()),None)
            selected=exact or (matches[0] if matches and term.casefold() in {s.casefold() for s in matches[0].entry_terms} else None)
            validation.append({"candidate":term,"validated":bool(selected),"descriptor_id":selected.descriptor_id if selected else None,"preferred_term":selected.name if selected else None,"status":"validated_exact_or_synonym" if selected else ("not_found" if self.mesh and self.mesh_expansion else "taxonomy_unavailable_or_disabled")})
            if selected: confirmed.append(selected.name)
        # Candidate terms are always searched as text unless matched to an official heading.
        if plan.concepts:
            query=build_query(plan.concepts[:4],confirmed)
        else: query=normalize_query(plan.query)
        query,_=self._sanitize_mesh_tags(query)
        return query,validation
    def _sanitize_mesh_tags(self,query):
        """Keep a MeSH field tag only when a local official descriptor confirms it."""
        unverified=[]
        pattern=re.compile(r'(["\']?)([^"\'()]+?)\1\s*\[MeSH(?: Terms)?\]',re.I)
        def replace(match):
            term=match.group(2).strip(); matches=self.mesh.search(term,limit=5) if self.mesh else []
            descriptor=next((m for m in matches if m.name.casefold()==term.casefold()),None)
            descriptor=descriptor or next((m for m in matches if term.casefold() in {s.casefold() for s in m.entry_terms}),None)
            if descriptor: return f'"{descriptor.name}"[MeSH Terms]'
            unverified.append(term); return f'"{term}"[Title/Abstract]'
        return normalize_query(pattern.sub(replace,query)),unverified
    def run(self,question:str,context:str="") -> ReasonResponse:
        if not question.strip(): raise ValueError("Question cannot be empty")
        started=time.monotonic(); request_start=getattr(self.pubmed,"requests",None); request_log_start=len(getattr(self.pubmed,"request_log",[])); search_log_start=len(getattr(self.pubmed,"search_log",[])); cache_start=getattr(self.pubmed,"cache_hits",None)
        trace={"run_id":str(uuid.uuid4()),"started_at":datetime.now(timezone.utc).isoformat(),"provider":self.llm.mode,"llm_mode":self.llm.mode,"model":getattr(self.llm,"model",None),"article_limit":self.max_articles,"batch_size":self.batch_size,"max_refinement_rounds":self.refinement_rounds,"max_llm_calls":self.max_llm_calls,"min_usable_evidence":self.min_usable_evidence,"min_unique_evidence_pmids":self.min_unique_evidence_pmids,"mesh_expansion":self.mesh_expansion,"reflective_filtering":self.reflective_filtering,"early_stopping":self.early_stopping}
        history=[]; articles=[]; evidence=[]; assessments=[]; validation=[]; batches=0; termination=""
        plan=self._structured(QueryPlan,"Identify biomedical concepts and candidate MeSH descriptor names. Candidate phrases are unverified.",f"QUESTION: {question}\nCONTEXT: {context or 'None'}")
        self.progress_callback("query_plan",0.18,"Query plan generated")
        query,validation=self._plan_query(plan); history.append({"query":query,"stage":"initial","rationale":plan.rationale,"candidate_mesh_terms":plan.mesh_terms,"term_rationales":plan.term_rationales})
        # Stage 1: partial PubMed metadata informs self-critic refinement.
        for round_no in range(self.refinement_rounds):
            self.progress_callback("query_refinement",0.2+0.1*round_no,f"Refinement round {round_no+1}")
            try: preliminary=self.pubmed.search_and_fetch(query,min(10,self.max_articles))
            except Exception as exc:
                LOG.exception("Preliminary PubMed search failed"); termination=f"pubmed_error_during_refinement:{type(exc).__name__}"; break
            metadata="\n".join(f"PMID {a.pmid} | {a.title} | {a.abstract[:240]}" for a in preliminary[:5]) or "No preliminary records."
            try:
                critique=self._structured(CriticFeedback,"Assess query coverage, alignment, and redundancy. Return proposed query and concrete rationale.",f"QUESTION: {question}\nCURRENT_QUERY: {query}\nRESULTS:\n{metadata}\nHISTORY: {json.dumps(history)}")
            except LLMError as exc: termination=f"llm_budget_or_error:{exc}"; break
            history.append({"round":round_no+1,"query":critique.query,"stage":"self_critic","coverage":critique.coverage,"alignment":critique.alignment,"redundancy":critique.redundancy,"rationale":critique.rationale,"terms_to_add":critique.terms_to_add,"terms_to_remove":critique.terms_to_remove})
            try: proposed,unverified_mesh=self._sanitize_mesh_tags(normalize_query(critique.query))
            except ValueError as exc: history[-1]["validation_error"]=str(exc); break
            if unverified_mesh: history[-1]["unverified_mesh_terms_demoted"]=unverified_mesh
            if (critique.coverage,critique.alignment,critique.redundancy)==(1,1,1) or proposed==query: break
            query=proposed
        history.append({"query":query,"stage":"final"})
        # Stage 2: process the relevance-ranked records in bounded batches.
        try: articles=self.pubmed.search_and_fetch(query,self.max_articles)
        except Exception as exc: termination=termination or f"pubmed_error:{type(exc).__name__}"; articles=[]
        source_by_pmid={a.pmid:a for a in articles}
        if not articles: termination=termination or "no_pubmed_results"
        self.progress_callback("article_retrieval",0.42,f"Retrieved {len(articles)} article records")
        for start in range(0,len(articles),self.batch_size):
            batch=articles[start:start+self.batch_size]; batches+=1
            self.progress_callback("evidence_review",min(0.84,0.44+0.4*batches/max(1,(self.max_articles+self.batch_size-1)//self.batch_size)),f"Assessing batch {batches}")
            for article in batch:
                if not article.abstract.strip(): continue
                if self.reflective_filtering:
                    try:
                        extracted=self._structured(EvidenceExtraction,"Assess relevance and extract a short verbatim passage from the abstract only. If it does not directly address the question, set aligned=false. Never invent text.",f"QUESTION: {question}\nPMID: {article.pmid}\nTITLE: {article.title}\nABSTRACT: {article.abstract}")
                    except LLMError as exc: termination=f"llm_budget_or_error:{exc}"; break
                    passage=extracted.passage.strip()
                    rationale=extracted.rationale
                    item=Evidence(pmid=article.pmid,title=article.title,passage=passage,aligned=extracted.aligned,rationale=rationale)
                    if item.aligned and not validate_evidence(item,source_by_pmid):
                        item.aligned=False; item.rationale=(rationale+"; passage did not match the retrieved PMID's abstract").strip("; ")
                    assessments.append(item)
                    if item.aligned: evidence.append(item)
                else:
                    # RAG ablation carries source abstracts without pretending there was a relevance judgment.
                    item=Evidence(pmid=article.pmid,title=article.title,passage=article.abstract,aligned=False,rationale="Reflective assessment disabled; raw abstract context")
                    evidence.append(item); assessments.append(item)
            if termination.startswith("llm_budget_or_error"): break
            if self.reflective_filtering and evidence and self.early_stopping and self.llm.mode!="mock":
                gate=evidence_sufficiency_gate(evidence,source_by_pmid,self.min_usable_evidence,self.min_unique_evidence_pmids)
                if not gate["passed"]:
                    history.append({"stage":"sufficiency_gate","batch":batches,**gate,"accepted_for_early_stop":False,"reason":"deterministic_minimum_not_met"})
                    continue
                try:
                    decision=self._structured(SufficiencyDecision,"Assess whether current evidence covers the major aspects. Consider independent articles and conflicting results. This is a model assessment, not proof.",f"QUESTION: {question}\nEVIDENCE: {json.dumps([e.model_dump() for e in evidence])}")
                    accepted=bool(gate["passed"] and decision.is_sufficient)
                    history.append({"stage":"sufficiency","batch":batches,"model_says_sufficient":decision.is_sufficient,**gate,"accepted_for_early_stop":accepted,"covered_aspects":decision.covered_aspects,"missing_aspects":decision.missing_aspects,"rationale":decision.rationale})
                    if accepted: termination="model_assessed_sufficient_across_multiple_pmids"; break
                except LLMError as exc: termination=f"llm_budget_or_error:{exc}"; break
        if not termination: termination="article_budget_exhausted" if len(articles)>=self.max_articles else "results_exhausted"
        # Stage 3: compose a cited response only from accepted extracted passages.
        self.progress_callback("answer_generation",0.88,"Preparing evidence-grounded response")
        answer="Insufficient evidence: no verified, relevant abstract passage supported a response."
        status="insufficient_evidence"
        citation_draft_diagnostics=None
        if self.llm.mode=="mock":
            status="mock_demo"
            answer="Mock/demo run: retrieved records and query planning are shown below, but this mode does not make relevance or sufficiency judgments and does not produce a biomedical conclusion."
            evidence=[]
        elif evidence and self.reflective_filtering:
            try:
                evidence=[item for item in evidence if validate_evidence(item,source_by_pmid)]
                draft=self._structured(AnswerDraft,"Answer using only supplied evidence. Attach an inline citation in exactly [PMID:123] format to each factual biomedical claim. The cited_pmids array must contain exactly the unique PMIDs used inline. Preserve uncertainty/conflicts and explain limitations. Cite only supplied PMIDs.",f"QUESTION: {question}\nEVIDENCE: {json.dumps([e.model_dump() for e in evidence])}")
                allowed={e.pmid for e in evidence}; retrieved=set(source_by_pmid)
                citation_draft_diagnostics=citation_mapping_diagnostics(draft.answer,draft.cited_pmids,allowed,retrieved)
                if citation_draft_diagnostics["valid"]:
                    answer=draft.answer+"\n\nLimitations: "+("; ".join(draft.limitations) if draft.limitations else "Evidence is limited to retrieved abstracts and model-assessed excerpts.")
                    status="answered"
                else:
                    answer="Insufficient evidence: the generated answer citations did not map to validated evidence from retrieved records. Accepted excerpts are retained for inspection."
            except LLMError as exc: termination=f"answer_generation_error:{exc}"
        elif evidence and not self.reflective_filtering:
            try:
                draft=self._structured(AnswerDraft,"Answer using only these retrieved PubMed abstracts. Attach an inline citation in exactly [PMID:123] format to each factual biomedical claim. The cited_pmids array must contain exactly the unique PMIDs used inline. Preserve uncertainty and never cite outside the supplied article set.",f"QUESTION: {question}\nABSTRACT CONTEXT: {json.dumps([{'pmid':e.pmid,'title':e.title,'abstract':e.passage} for e in evidence])}")
                allowed={e.pmid for e in evidence if e.pmid in source_by_pmid}
                citation_draft_diagnostics=citation_mapping_diagnostics(draft.answer,draft.cited_pmids,allowed,set(source_by_pmid))
                if citation_draft_diagnostics["valid"]:
                    answer=draft.answer+"\n\nThis single-query RAG baseline did not apply reflective relevance filtering or evidence sufficiency checks."
                    status="raw_context_only"
                else: answer="Insufficient evidence: the RAG draft citations did not map to supplied retrieved abstracts."
            except LLMError as exc: termination=f"answer_generation_error:{exc}"
        cited_for_answer=cited_pmids(answer) if status in {"answered","raw_context_only"} else set()
        allowed_for_answer={e.pmid for e in evidence if e.pmid in source_by_pmid}
        cited_for_answer &= allowed_for_answer
        citations=[Citation(pmid=p,title=source_by_pmid[p].title,url=f"https://pubmed.ncbi.nlm.nih.gov/{p}/") for p in dict.fromkeys(cited_for_answer) if p in source_by_pmid]
        request_end=getattr(self.pubmed,"requests",None)
        request_delta=request_end-request_start if isinstance(request_end,int) and isinstance(request_start,int) else request_end
        search_log=getattr(self.pubmed,"search_log",[])
        request_log=getattr(self.pubmed,"request_log",[])
        cache_end=getattr(self.pubmed,"cache_hits",None)
        usable=usable_evidence(evidence,source_by_pmid)
        trace.update({"llm_usage":self.llm.get_usage(),"pubmed_requests":request_delta,"pubmed_searches":list(search_log[search_log_start:]),"pubmed_request_log":list(request_log[request_log_start:]),"pubmed_cache_hits":cache_end-cache_start if isinstance(cache_end,int) and isinstance(cache_start,int) else None,"usable_evidence_excerpts":len(usable),"unique_evidence_pmids":len({e.pmid for e in usable}),"citation_validation":{"mode":"source_linked_membership_only","cited_pmids_in_retrieved_records":bool(citations) and all(c.pmid in source_by_pmid for c in citations),"claim_entailment_verified":False,"draft_diagnostics":citation_draft_diagnostics},"estimated_cost":None,"cost_status":"unavailable_no_provider_pricing_configured","duration_seconds":round(time.monotonic()-started,3),"termination_reason":termination,"meSH_taxonomy_loaded":self.mesh is not None})
        result=ReasonResponse(question=question,answer=answer,status=status,query=query,query_history=history,mesh_validation=validation,articles=articles,evidence=evidence,relevance_assessments=assessments,citations=citations,termination_reason=termination,batches_processed=batches,trace=trace)
        LOG.info("run=%s status=%s articles=%s accepted_evidence=%s batches=%s stop=%s",trace["run_id"],status,len(articles),len(evidence),batches,termination)
        self._save(result); return result
    def _save(self,result):
        out=settings.output_dir; out.mkdir(parents=True,exist_ok=True)
        path=out/f"{result.trace['run_id']}.json"; tmp=path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=2),encoding="utf8"); tmp.replace(path)
