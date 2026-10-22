"""Rate-limited NCBI E-utilities client with modest diskless query cache."""
import time
import threading
import requests
from .article_parser import parse_pubmed_xml
from ..config import settings

BASE="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

class PubMedClient:
    def __init__(self, api_key: str | None = None, email: str | None = None, timeout: float | None = None, session=None):
        self.api_key=api_key if api_key is not None else settings.ncbi_api_key; self.email=email or settings.ncbi_email
        self.timeout=timeout or settings.request_timeout; self.session=session or requests.Session(); self._last=0.0; self.requests=0; self._cache={}; self._article_cache={}; self._request_lock=threading.Lock()
    def _get(self, endpoint, params):
        # Shared Streamlit clients keep all sessions under NCBI's per-client request pace.
        with self._request_lock:
            return self._get_locked(endpoint,params)
    def _get_locked(self, endpoint, params):
        interval=0.11 if self.api_key else 0.34
        delay=interval-(time.monotonic()-self._last)
        if delay>0: time.sleep(delay)
        params={**params,"tool":"pubmed-reasoner-research"}
        if self.email: params["email"]=self.email
        if self.api_key: params["api_key"]=self.api_key
        for attempt in range(4):
            self._last=time.monotonic(); self.requests+=1
            try:
                resp=self.session.get(BASE+endpoint,params=params,timeout=self.timeout)
                if resp.status_code==429 or resp.status_code>=500:
                    if attempt<3: time.sleep(0.5*(2**attempt)); continue
                resp.raise_for_status(); return resp.text
            except requests.RequestException as exc:
                status=getattr(getattr(exc,"response",None),"status_code",None)
                if status and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"NCBI rejected the request (HTTP {status})") from None
                if attempt==3:
                    raise RuntimeError(f"NCBI request failed after retries ({type(exc).__name__}, HTTP {status or 'unavailable'})") from None
                time.sleep(0.5*(2**attempt))
        raise RuntimeError("NCBI request retries exhausted")
    def search(self, query: str, max_results: int=20, date_from: str|None=None, date_to: str|None=None) -> list[str]:
        key=(query,max_results,date_from,date_to)
        if key in self._cache: return list(self._cache[key])
        params={"db":"pubmed","term":query,"retmax":max_results,"retmode":"json","sort":"relevance"}
        if date_from: params.update(datetype="pdat",mindate=date_from,maxdate=date_to or date_from)
        import json
        payload=json.loads(self._get("esearch.fcgi",params)); ids=payload.get("esearchresult",{}).get("idlist",[])
        self._cache[key]=ids; return ids
    def fetch_articles(self, pmids: list[str], query: str=""):
        if not pmids: return []
        key=tuple(pmids)
        if key in self._article_cache:
            return [a.model_copy(update={"original_query":query}) for a in self._article_cache[key]]
        xml=self._get("efetch.fcgi",{"db":"pubmed","id":",".join(pmids),"retmode":"xml","rettype":"abstract"})
        articles=parse_pubmed_xml(xml,query); self._article_cache[key]=articles
        return articles
    def search_and_fetch(self, query: str, max_results: int=20, **date_filters):
        ids=self.search(query,max_results,**date_filters)
        return self.fetch_articles(ids,query)

