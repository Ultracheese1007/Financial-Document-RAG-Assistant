import json
from .core import ROOT, DEFAULT_CORPUS, Retriever, load_chunks, answer

def run():
    cases=json.loads((ROOT/'data/evaluation.json').read_text())
    index=Retriever(load_chunks(DEFAULT_CORPUS)); results=[]
    for c in cases:
        out=answer(index,c['question'],k=3)
        ids=out['retrieved_ids']; expected=c['expected']; rank=ids.index(expected)+1 if expected in ids else None
        passed=(out['status']=='insufficient_evidence') if expected is None else bool(rank and c['expected_text'].lower() in out['answer'].lower())
        results.append({'question':c['question'],'expected':expected,'rank':rank,'passed':passed,'latency_ms':out['latency_ms']})
    positives=[r for r in results if r['expected'] is not None]
    metrics={'cases':len(results),'hit_at_1':sum(r['rank']==1 for r in positives)/len(positives),'hit_at_3':sum(r['rank'] is not None for r in positives)/len(positives),'checks_passed':sum(r['passed'] for r in results),'mean_latency_ms':round(sum(r['latency_ms'] for r in results)/len(results),2),'results':results}
    print(json.dumps(metrics,indent=2)); return metrics

if __name__=='__main__':
    m=run()
    if m['checks_passed']!=m['cases']: raise SystemExit(1)
