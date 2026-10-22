"""Interactive Streamlit demonstration for the PubMed Reasoner methodology."""
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

# Community Cloud stores credentials in st.secrets. Copy supported values into
# the environment before importing the shared settings used by the CLI/app.
try:
    for _key in ("GROQ_API_KEY", "GROQ_MODEL", "LLM_MODE", "NCBI_EMAIL", "NCBI_API_KEY"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except (FileNotFoundError, ImportError):
    pass

from pubmed_reasoner.config import settings
from pubmed_reasoner.llm.base import GroqLLM, MockLLM, LLMError
from pubmed_reasoner.pipeline.orchestrator import Reasoner
from pubmed_reasoner.retrieval.pubmed_client import PubMedClient
from pubmed_reasoner.baselines import run_baseline
from pubmed_reasoner.logging_config import configure_logging

configure_logging()

st.set_page_config(page_title="PubMed Reasoner",page_icon="🧬",layout="wide")
st.title("🧬 PubMed Reasoner")
st.caption("Evidence-grounded biomedical literature QA · undergraduate research prototype")
st.warning("Research demonstration only. Not for diagnosis, treatment, or patient-specific decisions.")

@st.cache_data(ttl=300,show_spinner=False)
def load_active_models(api_key):
    return GroqLLM.active_models(api_key)

@st.cache_resource
def shared_pubmed_client():
    return PubMedClient()

with st.sidebar:
    st.header("Run configuration")
    mock=st.toggle("Mock/demo mode",value=settings.llm_mode=="mock",help="Uses deterministic synthetic control logic and never reports live model output.")
    active_models=[]
    if not mock:
        if not settings.groq_api_key:
            st.error("GROQ_API_KEY is missing. Add it to .env; the key is never displayed.")
        else:
            try: active_models=load_active_models(settings.groq_api_key)
            except LLMError as exc: st.error(str(exc))
    model=""
    if not mock and active_models:
        default_idx=active_models.index(settings.groq_model) if settings.groq_model in active_models else 0
        model=st.selectbox("Active Groq model",active_models,index=default_idx,help="Loaded from your authenticated Groq models endpoint at runtime.")
    elif mock: st.info("MOCK MODEL MODE · no model conclusions; PubMed retrieval is still live")
    max_rounds=st.slider("Refinement rounds",0,5,min(settings.max_refinement_rounds,5))
    batch_size=st.slider("Articles per batch",1,10,min(settings.batch_size,10))
    max_articles=st.slider("Maximum articles",1,50,min(settings.max_articles,50))
    llm_budget=st.slider("Maximum Groq calls",1,60,min(settings.max_llm_calls,60))
    mesh_enabled=st.checkbox("Validate and expand with local MeSH",True)

st.subheader("Biomedical question answering")
examples=["Are children with sleep deprivation at higher risk for obesity?","Do statins reduce atrial fibrillation after coronary artery bypass grafting?","Does vitamin D supplementation prevent respiratory infections?"]
if "research_question" not in st.session_state: st.session_state["research_question"]=""
def set_example():
    example=st.session_state.get("example_question")
    if example and example!="Choose an example…": st.session_state["research_question"]=example
st.selectbox("Example questions",["Choose an example…"]+examples,key="example_question",on_change=set_example)
question=st.text_area("Research question",height=95,placeholder="Enter a biomedical research question",key="research_question")
context=st.text_input("Optional context or task requirement",placeholder="e.g., answer yes/no and summarize evidence")
method=st.selectbox("Pipeline / baseline",["Full PubMed Reasoner","Standard RAG (single query)","No self-critic refinement","No early stopping","Self-reflection (post-hoc review)","Direct Groq LLM"])
run=st.button("Run research pipeline",type="primary",disabled=not question.strip() or (not mock and (not active_models or not model)))

if run:
    progress=st.progress(0,text="Preparing configuration…")
    status=st.empty()
    try:
        llm=MockLLM() if mock else GroqLLM(settings.groq_api_key or "",model,settings.request_timeout,available_models=active_models)
        if method=="Direct Groq LLM":
            if mock: raise LLMError("Direct LLM baseline requires live Groq; switch off mock mode.")
            status.info("Running direct-answer baseline…"); progress.progress(.3,text="Groq direct answer")
            data=run_baseline("direct_llm",question,llm=llm,context=context)
            runs=st.session_state.setdefault("comparison_runs",[])
            runs.append({"question":question,"method":method,"model":model,"mode":"groq","status":"direct_answer","answer":data["answer"],"articles":0,"evidence":0,"llm_calls":data["usage"]["calls"],"input_tokens":data["usage"]["input_tokens"],"output_tokens":data["usage"]["output_tokens"],"latency_seconds":data["latency_seconds"],"budgets":{"max_articles":max_articles,"batch_size":batch_size,"refinement_rounds":max_rounds,"max_llm_calls":llm_budget}})
            del runs[:-100]
            progress.progress(1.0,text="Complete")
            st.subheader("Direct model answer"); st.write(data["answer"])
            st.caption("No PubMed retrieval or citation validation was used for this baseline.")
            st.json(data["usage"])
            st.download_button("Download run JSON",json.dumps({"question":question,"method":method,"model":model,"answer":data["answer"],"usage":data["usage"],"latency_seconds":data["latency_seconds"]},indent=2),file_name="direct-llm-run.json",mime="application/json")
            current=[r for r in runs if r["question"]==question and r["model"]==model and r["mode"]=="groq"]
            if len(current)>1:
                st.subheader("Same-question comparison")
                st.dataframe(pd.DataFrame([{**{k:r[k] for k in ("method","status","articles","evidence","llm_calls","latency_seconds")},"tokens":r["input_tokens"]+r["output_tokens"]} for r in current]),width="stretch",hide_index=True)
        else:
            options={"max_articles":max_articles,"batch_size":batch_size,"refinement_rounds":max_rounds,"max_llm_calls":llm_budget,"mesh_expansion":mesh_enabled,"reflective_filtering":method!="Standard RAG (single query)","early_stopping":method!="No early stopping"}
            if method=="No self-critic refinement": options["refinement_rounds"]=0
            status.info("Stage 1/3 · self-critic query planning and PubMed search…"); progress.progress(.12,text="Searching PubMed")
            callback=lambda stage,value,text:(progress.progress(value,text=text),status.info(f"{stage.replace('_',' ').title()} · {text}"))
            if method=="Standard RAG (single query)":
                result=run_baseline("standard_rag",question,llm=llm,pubmed=shared_pubmed_client(),context=context,progress_callback=callback,**options)
            elif method=="Self-reflection (post-hoc review)":
                result=run_baseline("self_reflection",question,llm=llm,pubmed=shared_pubmed_client(),context=context,progress_callback=callback,**options)
            else:
                result=Reasoner(pubmed=shared_pubmed_client(),llm=llm,progress_callback=callback,**options).run(question,context)
            runs=st.session_state.setdefault("comparison_runs",[])
            runs.append({"question":question,"method":method,"model":model if not mock else "mock","mode":"mock" if mock else "groq","status":result.status,"answer":result.answer,"articles":len(result.articles),"evidence":len(result.evidence),"llm_calls":result.trace["llm_usage"]["calls"],"input_tokens":result.trace["llm_usage"]["input_tokens"],"output_tokens":result.trace["llm_usage"]["output_tokens"],"latency_seconds":result.trace["duration_seconds"],"budgets":{"max_articles":max_articles,"batch_size":batch_size,"refinement_rounds":max_rounds,"max_llm_calls":llm_budget}})
            del runs[:-100]
            progress.progress(1.0,text="Run complete")
            tabs=st.tabs(["Final answer","Query refinement","Retrieved evidence","Retrieval process","Evaluation & comparison","Run details"])
            with tabs[0]:
                st.subheader("Answer")
                if result.status=="mock_demo": st.info("MOCK/DEMO RESULT — this is not a live Groq experiment and includes no scientific conclusion.")
                elif result.status!="answered": st.warning(result.status.replace("_"," ").title())
                st.write(result.answer)
                if result.citations:
                    st.markdown("**PubMed references**")
                    for c in result.citations: st.markdown(f"[{c.pmid} · {c.title}]({c.url})")
                if result.status=="answered": st.caption("These are model-extracted abstract passages; citation membership is not proof of entailment.")
            with tabs[1]:
                st.markdown("**Final query**"); st.code(result.query)
                st.markdown("**Query history**"); st.dataframe(pd.DataFrame(result.query_history),width="stretch",hide_index=True)
                st.markdown("**Candidate MeSH validation**")
                if result.mesh_validation: st.dataframe(pd.DataFrame(result.mesh_validation),width="stretch",hide_index=True)
                else: st.info("No candidate MeSH terms were supplied, or the local 2026 taxonomy is unavailable.")
            with tabs[2]:
                st.markdown("Retrieved source records are shown separately from generated interpretations.")
                if result.articles:
                    for a in result.articles:
                        with st.expander(f"{a.title or 'Untitled'} · PMID {a.pmid}"):
                            st.markdown(f"[Open PubMed record]({a.source_url})")
                            st.caption(f"{a.journal or 'Journal unavailable'} · {a.publication_date or 'Date unavailable'}")
                            st.write(a.abstract or "No abstract indexed.")
                            matches=[e for e in result.relevance_assessments if e.pmid==a.pmid]
                            for e in matches:
                                label="Accepted relevant excerpt" if e.aligned else "Rejected or unassessed context"
                                st.info(f"{label}: {e.passage}\n\nAssessment: {e.rationale}")
                else: st.info("No PubMed records returned.")
            with tabs[3]:
                st.metric("Articles retrieved",len(result.articles)); st.metric("Accepted evidence excerpts",sum(e.aligned for e in result.evidence)); st.metric("Batches assessed",result.batches_processed)
                st.write("**Termination reason:**",result.termination_reason)
                st.write("**PubMed requests:**",result.trace.get("pubmed_requests","unavailable"))
                st.write("**LLM calls:**",result.trace["llm_usage"]["calls"])
                st.write("**Input / output tokens:**",f"{result.trace['llm_usage']['input_tokens']} / {result.trace['llm_usage']['output_tokens']}")
                st.write("**Estimated cost:** unavailable (no Groq price schedule configured)")
            with tabs[4]:
                st.write("Selected method:",method)
                current=[r for r in runs if r["question"]==question and r["mode"]==("mock" if mock else "groq") and r["model"]==(model if not mock else "mock")]
                st.info("Run additional methods on this same question and model to build a side-by-side comparison. Accuracy and macro F1 remain unavailable without labeled benchmark evaluation.")
                st.dataframe(pd.DataFrame([{**{k:r[k] for k in ("method","status","articles","evidence","llm_calls","latency_seconds")},"tokens":r["input_tokens"]+r["output_tokens"],"accuracy":"unavailable","macro F1":"unavailable","budgets":json.dumps(r["budgets"])} for r in current]),width="stretch",hide_index=True)
                if len(current)>1:
                    for start in range(0,len(current),2):
                        cols=st.columns(min(2,len(current)-start))
                        for col,item in zip(cols,current[start:start+2]):
                            with col:
                                st.markdown(f"**{item['method']}**"); st.caption(f"{item['status']} · {item['llm_calls']} calls · {item['latency_seconds']} sec")
                                st.write(item["answer"])
            with tabs[5]:
                st.json(result.trace)
                st.download_button("Download run JSON",result.model_dump_json(indent=2),file_name=f"{result.trace['run_id']}.json",mime="application/json")
        status.success("Run finished")
    except Exception as exc:
        status.error(f"Run failed: {exc}")
        st.exception(exc)
    finally: progress.empty()

st.divider()
if not Path("data/mesh/descriptors.jsonl").exists():
    st.info("Local MeSH taxonomy not initialized. Run `python scripts/download_mesh.py` once to enable descriptor validation and MeSH query expansion.")
