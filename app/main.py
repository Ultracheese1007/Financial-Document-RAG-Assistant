import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import httpx
from .core import DEFAULT_CORPUS, Retriever, answer, load_chunks

class Question(BaseModel):
    question: str = Field(min_length=3,max_length=1000)
    company: str|None = None
    fiscal_year: int|None = None
    k: int = Field(default=3,ge=1,le=10)
    mode: Literal['extractive','llm'] = 'extractive'

def create_app(corpus: Path|None=None):
    path=corpus or Path(os.environ.get('CORPUS_PATH',str(DEFAULT_CORPUS)))
    index=Retriever(load_chunks(path))
    api=FastAPI(title='Financial Document RAG Assistant',version='0.1.0')
    @api.get('/health')
    def health(): return {'status':'ok','chunks':len(index.chunks),'corpus':path.name,'llm_configured':bool(os.environ.get('LLM_MODEL'))}
    @api.post('/ask')
    def ask(q:Question):
        if not q.question.strip(): raise HTTPException(422,'Question must not be blank')
        if q.mode=='llm' and not os.environ.get('LLM_MODEL'): raise HTTPException(503,'LLM not configured; use extractive mode')
        try: return answer(index,q.question,q.k,q.company,q.fiscal_year,q.mode)
        except (ValueError,KeyError,TypeError,IndexError,httpx.HTTPError):
            raise HTTPException(502,'Generation failed or citation validation rejected the response')
    @api.get('/',response_class=HTMLResponse)
    def home(): return (Path(__file__).parent/'index.html').read_text(encoding='utf-8')
    return api

app=create_app()
