"""Shared validated data contracts used across retrieval and reasoning."""
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Literal

def now_utc() -> datetime: return datetime.now(timezone.utc)

class Article(BaseModel):
    pmid: str
    title: str = ""
    abstract: str = ""
    publication_date: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    doi: str | None = None
    mesh_headings: list[str] = Field(default_factory=list)
    source_url: str = ""
    retrieval_timestamp: datetime = Field(default_factory=now_utc)
    original_query: str = ""

class QueryPlan(BaseModel):
    query: str
    rationale: str = ""
    concepts: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    term_rationales: dict[str, str] = Field(default_factory=dict)

class CriticFeedback(BaseModel):
    coverage: int = Field(ge=-1, le=1)
    alignment: int = Field(ge=-1, le=1)
    redundancy: int = Field(ge=-1, le=1)
    rationale: str = ""
    query: str
    terms_to_add: list[str] = Field(default_factory=list)
    terms_to_remove: list[str] = Field(default_factory=list)

class Evidence(BaseModel):
    pmid: str
    title: str
    passage: str
    aligned: bool = True
    rationale: str = ""

class EvidenceExtraction(BaseModel):
    aligned: bool
    passage: str = ""
    rationale: str = ""

class AnswerDraft(BaseModel):
    answer: str
    cited_pmids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

class PubMedQAClassification(BaseModel):
    label: Literal["yes", "no", "maybe"]
    rationale: str = ""

class SufficiencyDecision(BaseModel):
    is_sufficient: bool
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    rationale: str = ""

class Citation(BaseModel):
    pmid: str
    title: str
    url: str

class ReasonResponse(BaseModel):
    question: str
    answer: str
    status: str
    query: str
    query_history: list[dict] = Field(default_factory=list)
    mesh_validation: list[dict] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    relevance_assessments: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    termination_reason: str
    batches_processed: int = 0
    trace: dict = Field(default_factory=dict)

