"""Groq and Gemini model integrations, plus a synthetic test/offline provider."""
from typing import Type, TypeVar
import re
import json
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMError(RuntimeError):
    """Provider, budget, or response-validation error."""

class LLMProvider:
    mode = "abstract"
    def __init__(self): self.calls=0; self.input_tokens=0; self.output_tokens=0; self.reasoning_tokens=0
    def generate_text(self, system: str, prompt: str) -> str: raise NotImplementedError
    def generate_structured_output(self, schema: Type[T], system: str, prompt: str) -> T:
        raw=self.generate_text(system + " Return one JSON object only. Required schema: " + schema.model_json_schema().__str__(),prompt)
        try:
            match=re.search(r"\{.*\}",raw,re.S)
            return schema.model_validate_json(match.group(0) if match else raw)
        except Exception as exc: raise LLMError(f"Invalid {schema.__name__} response: {exc}") from exc
    def get_usage(self):
        return {"calls":self.calls,"request_count":self.calls,"request_count_scope":"provider generation calls initiated by the app; provider SDK retries may issue additional HTTP requests","input_tokens":self.input_tokens,"output_tokens":self.output_tokens,"output_tokens_scope":"provider-reported generated tokens; reasoning tokens may be included","reasoning_tokens":self.reasoning_tokens,"mode":self.mode}

class GroqLLM(LLMProvider):
    mode="groq"
    def __init__(self,api_key: str,model: str,timeout: float=60,available_models=None):
        super().__init__()
        if not api_key: raise LLMError("GROQ_API_KEY is not configured")
        model=model.strip()
        if not model: raise LLMError("Select an active Groq model in the sidebar or set GROQ_MODEL")
        try:
            from groq import Groq
            self.client=Groq(api_key=api_key,timeout=timeout,max_retries=2)
        except Exception as exc: raise LLMError(f"Could not initialize Groq SDK: {exc}") from exc
        active=available_models if available_models is not None else self.active_models(api_key)
        if model not in active: raise LLMError("Configured model is not in the currently active Groq model list")
        self.model=model
    @classmethod
    def active_models(cls,api_key: str):
        """Query the authenticated Groq models endpoint; no model IDs are assumed."""
        try:
            from groq import Groq
            result=Groq(api_key=api_key,timeout=20,max_retries=1).models.list()
            # The Groq model list also contains audio/small moderation models; the
            # reasoning workflow needs a text-generation context of practical size.
            return sorted(m.id for m in result.data if getattr(m,"active",True) and getattr(m,"context_window",0)>=8192)
        except Exception as exc: raise LLMError(f"Could not load active Groq models ({type(exc).__name__})") from None
    def generate_text(self,system: str,prompt: str) -> str:
        self.calls+=1
        try:
            result=self.client.chat.completions.create(model=self.model,temperature=0,messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
            usage=result.usage
            if usage:
                self.input_tokens+=getattr(usage,"prompt_tokens",0) or 0
                self.output_tokens+=getattr(usage,"completion_tokens",0) or 0
                details=getattr(usage,"completion_tokens_details",None)
                self.reasoning_tokens+=getattr(details,"reasoning_tokens",0) or 0
            return result.choices[0].message.content or ""
        except Exception as exc:
            status=getattr(exc,"status_code",None)
            detail="rate limited by Groq" if status==429 else f"Groq request failed ({status or type(exc).__name__})"
            raise LLMError(detail) from exc
    def generate_structured_output(self,schema:Type[T],system:str,prompt:str)->T:
        self.calls+=1
        full_system=system+" Return exactly one valid JSON object matching this JSON schema: "+str(schema.model_json_schema())
        try:
            result=self.client.chat.completions.create(model=self.model,temperature=0,response_format={"type":"json_object"},messages=[{"role":"system","content":full_system},{"role":"user","content":prompt}])
            usage=result.usage
            if usage:
                self.input_tokens+=getattr(usage,"prompt_tokens",0) or 0; self.output_tokens+=getattr(usage,"completion_tokens",0) or 0
                details=getattr(usage,"completion_tokens_details",None)
                self.reasoning_tokens+=getattr(details,"reasoning_tokens",0) or 0
            return schema.model_validate_json(result.choices[0].message.content or "")
        except Exception as exc:
            raise LLMError(f"Groq structured {schema.__name__} request/validation failed: {type(exc).__name__}") from exc

class GeminiLLM(LLMProvider):
    """Google Gemini API adapter implementing the shared LLMProvider interface."""
    mode="gemini"
    def __init__(self,api_key: str,model: str,timeout: float=60,available_models=None):
        super().__init__()
        if not api_key: raise LLMError("GEMINI_API_KEY is not configured")
        model=model.strip()
        if not model: raise LLMError("Select an active Gemini model in the sidebar or set GEMINI_MODEL")
        try:
            from google import genai
            self.client=genai.Client(api_key=api_key)
        except Exception as exc: raise LLMError(f"Could not initialize Gemini SDK: {type(exc).__name__}") from exc
        active=available_models if available_models is not None else self.active_models(api_key)
        if model not in active: raise LLMError("Configured model is not in the currently available Gemini model list")
        self.model=model
    @classmethod
    def active_models(cls,api_key: str):
        """List models available to this key that support text generation."""
        try:
            from google import genai
            client=genai.Client(api_key=api_key)
            models=[]
            for item in client.models.list():
                actions=getattr(item,"supported_actions",None) or []
                name=getattr(item,"name","")
                model_id=name.removeprefix("models/")
                if "generateContent" in actions and model_id:
                    models.append(model_id)
            return sorted(set(models))
        except Exception as exc: raise LLMError(f"Could not load available Gemini models ({type(exc).__name__})") from None
    @staticmethod
    def _response_schema(schema:Type[T]):
        """Build Gemini-compatible JSON schema for this app's Pydantic outputs."""
        result=schema.model_json_schema()
        # Gemini structured output rejects free-form maps (additionalProperties).
        # QueryPlan.term_rationales is optional presentation metadata, so omit it;
        # Pydantic supplies its empty-dict default when validating the response.
        properties=result.get("properties",{})
        if schema.__name__=="QueryPlan":
            properties.pop("term_rationales",None)
            result["required"]=[name for name in result.get("required",[]) if name!="term_rationales"]
        def clean(value):
            if isinstance(value,dict):
                value.pop("additionalProperties",None)
                for child in value.values(): clean(child)
            elif isinstance(value,list):
                for child in value: clean(child)
        clean(result)
        return result
    def _generate(self,system,prompt,schema=None):
        from google.genai import types
        config={"system_instruction":system,"temperature":0}
        if schema is not None:
            config.update(response_mime_type="application/json",response_schema=self._response_schema(schema))
        self.calls+=1
        try:
            result=self.client.models.generate_content(model=self.model,contents=prompt,config=types.GenerateContentConfig(**config))
            usage=getattr(result,"usage_metadata",None)
            if usage:
                self.input_tokens+=getattr(usage,"prompt_token_count",0) or 0
                candidate_tokens=getattr(usage,"candidates_token_count",0) or 0
                thought_tokens=getattr(usage,"thoughts_token_count",0) or 0
                self.output_tokens+=candidate_tokens+thought_tokens
                self.reasoning_tokens+=thought_tokens
            return result.text or ""
        except Exception as exc:
            raise LLMError(f"Gemini request failed ({type(exc).__name__}): {exc}") from exc
    def generate_text(self,system: str,prompt: str)->str:
        return self._generate(system,prompt)
    def generate_structured_output(self,schema:Type[T],system:str,prompt:str)->T:
        raw=self._generate(system,prompt,schema)
        try: return schema.model_validate_json(raw)
        except Exception as exc: raise LLMError(f"Gemini structured {schema.__name__} response validation failed") from exc

class MockLLM(LLMProvider):
    """Synthetic provider for tests/demo. Never represents a model experiment."""
    mode="mock"
    def generate_text(self,system,prompt):
        self.calls+=1
        # Echo supplied abstract excerpts only; no medical conclusion or synthetic citation.
        if "ABSTRACT:" in prompt:
            abstract=prompt.split("ABSTRACT:",1)[1].strip()
            sentence=re.split(r"(?<=[.!?])\s+",abstract)[0] if abstract else ""
            return json.dumps({"aligned":bool(sentence),"passage":sentence,"rationale":"MOCK extractive demonstration; relevance is not assessed"})
        if "EVIDENCE:" in prompt: return "Mock mode does not synthesize a biomedical answer from evidence."
        raise LLMError("Mock mode only supports clearly labeled extractive demonstration")
    def generate_structured_output(self,schema:Type[T],system:str,prompt:str)->T:
        self.calls+=1
        if schema.__name__=="QueryPlan":
            question=prompt.split("QUESTION:",1)[-1].splitlines()[0]
            words=list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z-]{3,}",question)))[:4]
            value={"query":" AND ".join(f'"{w}"[Title/Abstract]' for w in words) or 'biomedical[Title/Abstract]',"rationale":"MOCK lexical plan; no clinical interpretation","concepts":words,"mesh_terms":[],"term_rationales":{}}
        elif schema.__name__=="CriticFeedback":
            query=prompt.split("CURRENT_QUERY:",1)[-1].splitlines()[0].strip()
            value={"coverage":0,"alignment":0,"redundancy":0,"rationale":"MOCK: query dimensions are not assessed","query":query,"terms_to_add":[],"terms_to_remove":[]}
        elif schema.__name__=="SufficiencyDecision":
            value={"is_sufficient":False,"covered_aspects":[],"missing_aspects":["independent relevance and coverage review"],"rationale":"MOCK does not decide scientific sufficiency"}
        elif schema.__name__=="EvidenceExtraction":
            abstract=prompt.split("ABSTRACT:",1)[-1].strip()
            passage=re.split(r"(?<=[.!?])\s+",abstract)[0] if abstract else ""
            value={"aligned":False,"passage":passage,"rationale":"MOCK mode displays an extractive candidate but does not assess relevance"}
        else: raise LLMError(f"No mock structured behavior for {schema.__name__}")
        return schema.model_validate(value)
