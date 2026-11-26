import json
import httpx
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from app.core import DEFAULT_CORPUS, Retriever, load_chunks, ingest_pdf, merge_chunks, save_chunks, answer, llm_claims
from app.main import create_app

@pytest.fixture
def index(): return Retriever(load_chunks(DEFAULT_CORPUS))

def test_revenue(index):
    out=answer(index,'What was total revenue in 2025?',k=1)
    assert '281.7' in out['answer'] and out['sources'][0]['id']=='msft25-revenue'

def test_profit_synonym(index):
    assert answer(index,'What was operating profit?',k=1)['sources'][0]['id']=='msft25-operating'

def test_risk(index):
    assert 'operating margins' in answer(index,'financial risk from AI infrastructure',k=1)['answer']

def test_abstain(index):
    assert answer(index,'Microsoft 2025 favorite pizza topping')['status']=='insufficient_evidence'

def test_filters(index):
    assert not index.search('revenue',company='Not Microsoft')
    assert not index.search('revenue',fiscal_year=2024)

def test_pdf_ingestion(tmp_path):
    # Synthetic test document, not a public-company report.
    path=tmp_path/'synthetic.pdf'; c=canvas.Canvas(str(path)); c.drawString(40,700,'ExampleCo revenue USD 10 million.'); c.showPage(); c.drawString(40,700,'ExampleCo operating profit USD 2 million.'); c.save()
    chunks=ingest_pdf(path,'ExampleCo',2025,'https://example.com/report.pdf',size=5,overlap=2)
    assert {x.page for x in chunks}=={1,2}
    assert len(merge_chunks(chunks,chunks))==len(chunks)
    target=tmp_path/'index.json'; save_chunks(chunks,target); assert load_chunks(target)==chunks
    assert answer(Retriever(chunks),'revenue',k=1)['sources'][0]['page']==1

def test_invalid_window(tmp_path):
    with pytest.raises(ValueError): ingest_pdf(tmp_path/'no.pdf','Example',2025,size=10,overlap=10)

def test_api():
    c=TestClient(create_app(DEFAULT_CORPUS)); assert c.get('/health').json()['chunks']==6
    assert c.get('/').status_code==200
    assert c.post('/ask',json={'question':'revenue','k':1}).json()['status']=='evidence_found'
    assert c.post('/ask',json={'question':'revenue','k':100}).status_code==422

def test_llm_requires_config(monkeypatch):
    monkeypatch.delenv('LLM_MODEL',raising=False)
    assert TestClient(create_app(DEFAULT_CORPUS)).post('/ask',json={'question':'revenue','mode':'llm'}).status_code==503

@pytest.mark.parametrize('claim,accepted',[
    ({'text':'Revenue was USD 281.7 billion.','source_ids':['msft25-revenue']},True),
    ({'text':'Revenue was USD 999 billion.','source_ids':['msft25-revenue']},False),
    ({'text':'Revenue was USD 281.7 billion.','source_ids':['fake-id']},False),
])
def test_llm_validation(monkeypatch,index,claim,accepted):
    monkeypatch.setenv('LLM_MODEL','mock-model')
    def fake(*args,**kwargs):
        assert 'evidence' in kwargs['json']['messages'][1]['content']
        return httpx.Response(200,request=httpx.Request('POST','http://localhost'),json={'choices':[{'message':{'content':json.dumps({'claims':[claim]})}}]})
    monkeypatch.setattr('app.core.httpx.post',fake)
    if accepted: assert llm_claims('revenue',index.search('revenue',k=1))==[claim]
    else:
        with pytest.raises(ValueError): llm_claims('revenue',index.search('revenue',k=1))
