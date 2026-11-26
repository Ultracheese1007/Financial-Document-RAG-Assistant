import argparse, json, os
from pathlib import Path
from .core import DEFAULT_CORPUS, Retriever, answer, ingest_pdf, load_chunks, merge_chunks, save_chunks


def main():
    p=argparse.ArgumentParser(description='Financial Document RAG Assistant')
    sub=p.add_subparsers(dest='command',required=True)
    ingest=sub.add_parser('ingest'); ingest.add_argument('pdf',type=Path); ingest.add_argument('--company',required=True); ingest.add_argument('--year',required=True,type=int); ingest.add_argument('--url',default=''); ingest.add_argument('--out',type=Path,default=Path('data/corpus.json'))
    ask=sub.add_parser('ask'); ask.add_argument('question'); ask.add_argument('--corpus',type=Path,default=Path(os.environ.get('CORPUS_PATH',str(DEFAULT_CORPUS)))); ask.add_argument('--company'); ask.add_argument('--year',type=int); ask.add_argument('--mode',choices=['extractive','llm'],default='extractive'); ask.add_argument('--k',type=int,default=3)
    args=p.parse_args()
    if args.command=='ingest':
        chunks=ingest_pdf(args.pdf,args.company,args.year,args.url)
        previous=load_chunks(args.out) if args.out.exists() else []
        merged=merge_chunks(previous,chunks); save_chunks(merged,args.out)
        print(json.dumps({'extracted_chunks':len(chunks),'corpus_chunks':len(merged),'output':str(args.out)}))
    else:
        if not 1<=args.k<=10: p.error('--k must be 1–10')
        print(json.dumps(answer(Retriever(load_chunks(args.corpus)),args.question,args.k,args.company,args.year,args.mode),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
