from pubmed_reasoner.pipeline.orchestrator import Reasoner
from pubmed_reasoner.llm.base import MockLLM

class FakePubMed:
    requests=2
    def search_and_fetch(self,query,max_results): return []

def test_mock_pipeline_stops_without_fabricating_answer(tmp_path,monkeypatch):
    from pubmed_reasoner import config
    monkeypatch.setattr(config.settings,"output_dir",tmp_path)
    r=Reasoner(pubmed=FakePubMed(),llm=MockLLM(),refinement_rounds=1).run("Does a test work?")
    assert r.status=="mock_demo" and not r.evidence and not r.citations
    assert "Mock/demo" in r.answer and r.trace["llm_mode"]=="mock"
    assert list(tmp_path.glob("*.json"))

