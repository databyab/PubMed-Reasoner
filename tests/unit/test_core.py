import json
from pathlib import Path
import pytest
from pubmed_reasoner.taxonomy.parser import parse_mesh_xml
from pubmed_reasoner.taxonomy.search import MeshIndex
from pubmed_reasoner.taxonomy.validator import validate_descriptors
from pubmed_reasoner.retrieval.query_builder import build_query
from pubmed_reasoner.retrieval.query_normalizer import normalize_query
from pubmed_reasoner.retrieval.article_parser import parse_pubmed_xml
from pubmed_reasoner.models.schemas import CriticFeedback, Evidence
from pubmed_reasoner.pipeline.evidence_extractor import validate_evidence
from pubmed_reasoner.pipeline.answer_generator import citations_supported
from pubmed_reasoner.evaluation.metrics import classification_metrics
from pubmed_reasoner.evaluation.dataset_loader import load_pubmedqa
from pubmed_reasoner.evaluation.cost_tracker import CostTracker

FIX=Path(__file__).parents[1]/"fixtures"
def test_mesh_parse_lookup_and_validation():
    items=parse_mesh_xml(FIX/"mesh.xml"); assert items[0].descriptor_id=="D000001"; assert "Example Synonym" in items[0].entry_terms; assert items[0].tree_numbers==["A01"]
    idx=MeshIndex(items); assert idx.search("example synonym")[0].name=="Test Term"; assert idx.by_id["D000001"].qualifiers==["therapy"]; assert validate_descriptors(items)["valid"]
def test_query_construction_and_invalid_parenthesis():
    q=build_query(["sleep deprivation","obesity"]); assert " AND " in q and "[Title/Abstract]" in q
    with pytest.raises(ValueError): normalize_query("(asthma AND obesity")
def test_article_parser_mixed_content():
    a=parse_pubmed_xml((FIX/"pubmed.xml").read_bytes(),"test")[0]; assert a.pmid=="123456"; assert a.title=="Example mixed title"; assert a.abstract=="Abstract evidence."; assert a.doi=="10.1/example"; assert a.authors==["Ada Lovelace"]
def test_structured_output_and_evidence_citations():
    c=CriticFeedback.model_validate({"coverage":1,"alignment":1,"redundancy":0,"query":"test"}); assert c.redundancy==0
    e=Evidence(pmid="12",title="t",passage="p"); assert validate_evidence(e,{"12"}); assert not validate_evidence(e,{"13"}); assert citations_supported("[PMID:12]",{"12"}); assert not citations_supported("[PMID:99]",{"12"})
def test_metrics_cost_and_dataset_loader(tmp_path):
    m=classification_metrics(["yes","no"],["yes","yes"]); assert m["accuracy"]==.5 and m["macro_f1"]>0
    tracker=CostTracker(1_000_000,500_000,.1,.2); assert tracker.estimated_cost()==pytest.approx(.2)
    p=tmp_path/"data.json"; p.write_text(json.dumps({"1":{"question":"Does it work?","final_decision":"yes","context":{"contexts":[]}}}))
    loaded=load_pubmedqa(p); assert loaded["n"]==1 and loaded["labeled"]==1

