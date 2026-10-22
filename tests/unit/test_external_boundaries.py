import pytest
from pubmed_reasoner.llm.base import GroqLLM,LLMError,MockLLM,LLMProvider
from pubmed_reasoner.models.schemas import QueryPlan,EvidenceExtraction,Article,CriticFeedback,SufficiencyDecision,AnswerDraft
from pubmed_reasoner.retrieval.pubmed_client import PubMedClient
from pubmed_reasoner.pipeline.orchestrator import Reasoner
from pubmed_reasoner.evaluation.runner import run_pubmedqa

def test_mock_structured_validation_and_extractive_demo():
    llm=MockLLM(); plan=llm.generate_structured_output(QueryPlan,"","QUESTION: Does treatment improve sleep?")
    assert plan.query and plan.concepts and llm.get_usage()["mode"]=="mock"
    extraction=llm.generate_structured_output(EvidenceExtraction,"","ABSTRACT: Treatment improved sleep.")
    assert extraction.aligned is False

def test_groq_requires_key_and_model_without_network():
    with pytest.raises(LLMError,match="GROQ_API_KEY"):
        GroqLLM("","model")
    with pytest.raises(LLMError,match="Select an active"):
        GroqLLM("nonempty"," ")

def test_malformed_structured_response_is_not_silently_repaired():
    class Bad(LLMProvider):
        def generate_text(self,*args): return "not json"
    with pytest.raises(LLMError,match="Invalid QueryPlan"):
        Bad().generate_structured_output(QueryPlan,"","question")

def test_groq_provider_error_is_sanitized():
    class Failure(Exception): status_code=429
    class Completions:
        def create(self,**kwargs): raise Failure("secret provider detail")
    class Chat: completions=Completions()
    class Client: chat=Chat()
    llm=GroqLLM.__new__(GroqLLM); LLMProvider.__init__(llm); llm.model="dynamic"; llm.client=Client()
    with pytest.raises(LLMError,match="rate limited") as err: llm.generate_text("system","prompt")
    assert "secret provider detail" not in str(err.value)

class Response:
    status_code=200
    text='{"esearchresult":{"idlist":[]}}'
    def raise_for_status(self): pass
    def json(self): return {"esearchresult":{"idlist":[]}}
class Session:
    def get(self,*args,**kwargs): return Response()

def test_pubmed_empty_result_and_id_cache():
    client=PubMedClient(session=Session()); assert client.search("test",2)==[]; first=client.requests
    assert client.search("test",2)==[] and client.requests==first

def test_pubmed_http_error_does_not_expose_request_url_or_key(monkeypatch):
    import requests
    from pubmed_reasoner.retrieval import pubmed_client
    class BadResponse:
        status_code=400
        def raise_for_status(self):
            raise requests.HTTPError("secret=ncbi-key",response=self)
    class BadSession:
        def get(self,*args,**kwargs): return BadResponse()
    monkeypatch.setattr(pubmed_client.time,"sleep",lambda *_:None)
    client=PubMedClient(api_key="private",session=BadSession())
    with pytest.raises(RuntimeError) as error: client._get("esearch.fcgi",{})
    assert "private" not in str(error.value) and "secret" not in str(error.value)

class EmptyPubMed:
    requests=0
    def search_and_fetch(self,query,max_results): return []

def test_empty_retrieval_returns_mock_demo_without_citations(tmp_path,monkeypatch):
    from pubmed_reasoner import config
    monkeypatch.setattr(config.settings,"output_dir",tmp_path)
    result=Reasoner(pubmed=EmptyPubMed(),llm=MockLLM(),refinement_rounds=0).run("Does this work?")
    assert result.termination_reason=="no_pubmed_results"
    assert result.articles==[] and result.citations==[] and result.status=="mock_demo"

def test_unvalidated_mesh_query_is_demoted_to_text():
    reasoner=Reasoner(pubmed=EmptyPubMed(),llm=MockLLM(),refinement_rounds=0)
    query,unverified=reasoner._sanitize_mesh_tags('"Unverified Heading"[MeSH Terms] AND asthma')
    assert '[Title/Abstract]' in query and '[MeSH Terms]' not in query
    assert unverified==["Unverified Heading"]

def test_benchmark_runner_refuses_mock_mode_before_accessing_dataset():
    with pytest.raises(RuntimeError,match="requires LLM_MODE=groq"):
        run_pubmedqa("unavailable.json","unused-output")

def test_sufficiency_early_stop_after_two_distinct_sources(tmp_path,monkeypatch):
    from pubmed_reasoner import config
    monkeypatch.setattr(config.settings,"output_dir",tmp_path)
    class PubMed:
        requests=2
        def search_and_fetch(self,query,max_results):
            return [Article(pmid=str(i),title=f"Study {i}",abstract=f"Evidence from study {i}.") for i in range(1,6)]
    class LLM(LLMProvider):
        mode="fake-test"
        def generate_structured_output(self,schema,system,prompt):
            self.calls+=1
            if schema is QueryPlan: return QueryPlan(query="study",concepts=[])
            if schema is CriticFeedback: return CriticFeedback(coverage=1,alignment=1,redundancy=1,query="study")
            if schema is EvidenceExtraction:
                import re
                pmid=re.search(r"PMID: (\d+)",prompt).group(1)
                return EvidenceExtraction(aligned=True,passage=f"Evidence from study {pmid}.")
            if schema is SufficiencyDecision: return SufficiencyDecision(is_sufficient=True,covered_aspects=["finding"])
            if schema is AnswerDraft: return AnswerDraft(answer="Evidence supports the finding [PMID: 1] [PMID: 2].",cited_pmids=["1","2"])
            raise AssertionError(schema)
    result=Reasoner(pubmed=PubMed(),llm=LLM(),refinement_rounds=0,batch_size=2,max_articles=5).run("What do the studies report?")
    assert result.status=="answered" and result.batches_processed==1
    assert {c.pmid for c in result.citations}=={"1","2"}
    assert result.termination_reason=="model_assessed_sufficient_across_multiple_pmids"
