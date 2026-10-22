"""Reproducible PubMedQA driver and atomic CSV/JSON artifact writer."""
import json, random, subprocess, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .metrics import classification_metrics
from .citation_metrics import citation_validity
from .dataset_loader import load_pubmedqa
from ..config import settings
from ..llm.base import GroqLLM
from ..models.schemas import PubMedQAClassification
from ..pipeline.orchestrator import Reasoner
from ..retrieval.pubmed_client import PubMedClient
from ..baselines import run_baseline

def save_evaluation(rows, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    _atomic_text(out/"evaluation.json",json.dumps(rows,indent=2))
    _atomic_csv(pd.DataFrame(rows),out/"evaluation.csv")
    labeled=[r for r in rows if r.get("gold") is not None and r.get("prediction") is not None]
    metrics=classification_metrics([r["gold"] for r in labeled],[r["prediction"] for r in labeled])
    _atomic_text(out/"metrics.json",json.dumps(metrics,indent=2))
    return metrics

def _mean(values):
    items=[float(v) for v in values if v is not None]
    return sum(items)/len(items) if items else None

def _atomic_text(path,text):
    path=Path(path); temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(text,encoding="utf8"); temp.replace(path)

def _atomic_csv(frame,path):
    path=Path(path); temp=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temp,index=False); temp.replace(path)

def run_pubmedqa(dataset_path,output_dir,*,methods=("full",),limit=50,seed=13,model=None):
    """Run labeled examples through Groq; refuses mock-mode evaluation."""
    if settings.llm_mode!="groq" or not settings.groq_api_key:
        raise RuntimeError("Evaluation requires LLM_MODE=groq and GROQ_API_KEY; mock output is not a model experiment.")
    selected_model=model or settings.groq_model
    if not selected_model: raise RuntimeError("Set GROQ_MODEL to an active model ID")
    active=GroqLLM.active_models(settings.groq_api_key)
    if selected_model not in active: raise RuntimeError("Selected model is not currently active on Groq")
    data=load_pubmedqa(dataset_path)
    labeled=[e for e in data["examples"] if e["label"] in {"yes","no","maybe"}]
    rng=random.Random(seed); rng.shuffle(labeled); sample=labeled[:limit]
    pubmed=PubMedClient(); rows=[]
    for index,example in enumerate(sample,1):
        for method in methods:
            started=time.monotonic(); llm=GroqLLM(settings.groq_api_key,selected_model,settings.request_timeout,available_models=active)
            requests_before=getattr(pubmed,"requests",0)
            options={"pubmed":pubmed,"llm":llm,"max_articles":settings.max_articles,"batch_size":settings.batch_size,"max_llm_calls":settings.max_llm_calls,"mesh_expansion":True,"reflective_filtering":True,"early_stopping":True,"refinement_rounds":settings.max_refinement_rounds}
            try:
                context=json.dumps(example.get("context") or {},ensure_ascii=False)
                if method=="direct_llm":
                    result=run_baseline("direct_llm",example["question"],llm=llm,context=context)
                    answer=result["answer"]; article_count=0; evidence=[]; citations=[]; retrieved=set()
                else:
                    if method=="standard_rag": result=run_baseline("standard_rag",example["question"],context=context,**options)
                    elif method=="self_reflection": result=run_baseline("self_reflection",example["question"],context=context,**options)
                    elif method in {"full","no_refinement","no_mesh","no_early_stop"}:
                        if method=="no_refinement": options["refinement_rounds"]=0
                        if method=="no_mesh": options["mesh_expansion"]=False
                        if method=="no_early_stop": options["early_stopping"]=False
                        result=Reasoner(**options).run(example["question"],context)
                    else: raise ValueError(f"Unknown method {method}")
                    answer=result.answer; article_count=len(result.articles); evidence=result.evidence; citations=result.citations
                    retrieved={a.pmid for a in result.articles}
                if llm.calls < settings.max_llm_calls:
                    classified=llm.generate_structured_output(PubMedQAClassification,"Classify the answer to the biomedical question as exactly yes, no, or maybe based on the question and answer. Do not add facts.",f"QUESTION: {example['question']}\nANSWER: {answer}")
                    prediction=classified.label
                else: prediction=None
                cited=[c.pmid for c in citations]
                accepted_count=sum(bool(getattr(e,"aligned",False)) for e in evidence)
                grounded=bool(cited) and getattr(result,"status",None) in {"answered","raw_context_only"}
                row={"id":example["id"],"question":example["question"],"gold":example["label"],"prediction":prediction,"method":method,"status":getattr(result,"status","direct_answer"),"termination_reason":getattr(result,"termination_reason","direct_answer"),"article_count":article_count,"accepted_evidence_count":accepted_count,"citation_validity":citation_validity(cited,retrieved) if cited else None,"evidence_grounded":grounded,"llm_calls":llm.calls,"input_tokens":llm.input_tokens,"output_tokens":llm.output_tokens,"pubmed_requests":getattr(pubmed,"requests",0)-requests_before,"latency_seconds":round(time.monotonic()-started,3),"error":None}
            except Exception as exc:
                row={"id":example["id"],"question":example["question"],"gold":example["label"],"prediction":None,"method":method,"status":"error","termination_reason":"error","article_count":0,"accepted_evidence_count":0,"citation_validity":None,"evidence_grounded":False,"llm_calls":llm.calls,"input_tokens":llm.input_tokens,"output_tokens":llm.output_tokens,"pubmed_requests":getattr(pubmed,"requests",0)-requests_before,"latency_seconds":round(time.monotonic()-started,3),"error":f"{type(exc).__name__}: {exc}"}
            rows.append(row)
        if index%10==0: print(f"Processed {index}/{len(sample)} labeled questions")
    per_method={}
    for method in methods:
        subset=[r for r in rows if r["method"]==method and r["prediction"]]
        per_method[method]={**classification_metrics([r["gold"] for r in subset],[r["prediction"] for r in subset]),"citation_validity":_mean(r["citation_validity"] for r in subset),"evidence_grounded_rate":_mean(float(r["evidence_grounded"]) for r in subset),"avg_retrieved_articles":_mean(r["article_count"] for r in subset),"avg_accepted_evidence":_mean(r["accepted_evidence_count"] for r in subset),"avg_pubmed_requests":_mean(r["pubmed_requests"] for r in subset),"avg_llm_calls":_mean(r["llm_calls"] for r in subset),"avg_tokens":_mean(r["input_tokens"]+r["output_tokens"] for r in subset),"avg_latency_seconds":_mean(r["latency_seconds"] for r in subset),"estimated_cost_usd":None,"n_evaluated":len(subset)}
    run_id=str(uuid.uuid4()); created=datetime.now(timezone.utc).isoformat()
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: commit=None
    manifest={"run_id":run_id,"timestamp":created,"git_commit":commit,"software":"pubmed-reasoner-reproduction 0.1.0","provider":"Groq","model":selected_model,"dataset":data["provenance"],"label_distribution":data["label_distribution"],"sample_size":len(sample),"labeled_available":len(labeled),"seed":seed,"methods":list(methods),"budgets":{"max_articles":settings.max_articles,"batch_size":settings.batch_size,"max_refinement_rounds":settings.max_refinement_rounds,"max_llm_calls_per_question":settings.max_llm_calls}}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    _atomic_text(out/f"{run_id}.json",json.dumps({"manifest":manifest,"metrics":per_method,"predictions":rows},indent=2))
    _atomic_csv(pd.DataFrame(rows),out/f"{run_id}.csv")
    _atomic_text(out/f"{run_id}_manifest.json",json.dumps(manifest,indent=2))
    _atomic_text(out/"metrics.json",json.dumps(per_method,indent=2))
    return {"run_id":run_id,"manifest":manifest,"metrics":per_method,"predictions":rows}
