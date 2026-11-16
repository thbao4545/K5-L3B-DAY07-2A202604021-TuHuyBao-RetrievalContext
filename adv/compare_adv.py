"""Advanced challenge: compare dense / BM25 / hybrid retrieval on one split of the Gemini 1.5 benchmark.

  <day6>/starter/.venv/Scripts/python.exe -I adv/compare_adv.py --split dev --output adv/results/dev-compare

Reports Hit@1, Hit@3 (all gold entries covered), and context evidence at several budgets, overall and per category.
Unanswerable questions are not in the retrieval metrics. The test split needs --i-locked-config.
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advlib  # noqa: E402,F401  (sets sys.path)
from advlib import ROOT, mark, question_hit  # noqa: E402
from baseline import pack_context  # noqa: E402
from compare import BM25, rrf  # noqa: E402
from dense import DenseRetriever  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

STRATEGY = 'ADV_gemini15'
BUDGETS = (3000, 6000)
CAND = 20
SOURCE = 'Gemini 1.5 technical report (arXiv 2403.05530v5)'


def load_chunks():
    chunks = json.loads((ROOT / 'adv/out/index_chunks.json').read_text(encoding='utf-8'))
    for c in chunks:
        c['source_id'], c['source'] = 'gemini15', SOURCE
    return chunks


def make_search(method, dense, bm25):
    def search(query, top_k=3, filters=None):
        if method == 'dense':
            return dense.search(query, filters=filters, top_k=top_k)
        if method == 'bm25':
            return bm25.search(query, filters=filters, top_k=top_k)
        return rrf([dense.search(query, filters=filters, top_k=CAND), bm25.search(query, filters=filters, top_k=CAND)], top_k)
    return search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--i-locked-config', action='store_true')
    args = ap.parse_args()
    if args.split == 'test' and not args.i_locked_config:
        sys.exit('Refusing to run test before the configuration is locked (--i-locked-config).')
    qs = [q for q in json.loads((ROOT / 'adv/benchmark.json').read_text(encoding='utf-8'))
          if q['split'] == args.split and q['answerable']]
    chunks = load_chunks()
    env = dotenv_values(ROOT / 'starter/.env')
    dense = DenseRetriever(env['DATABASE_URL'], env.get('EMBEDDING_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2', STRATEGY)
    bm25 = BM25(chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ('dense', 'bm25', 'hybrid'):
        search = make_search(method, dense, bm25)
        traces = []
        for q in qs:
            t0 = time.perf_counter()
            res = search(q['question'], top_k=3)
            ms = (time.perf_counter() - t0) * 1000
            marked = mark(res, q)
            ctx = {}
            for b in BUDGETS:
                text, ids = pack_context(res, b)
                keep = [c for c in res if c['id'] in ids]
                ctx[str(b)] = {'ids': ids, 'hit': question_hit(keep, q, strict_text=True)}
            traces.append({'query_id': q['id'], 'cat': q['cat'], 'question': q['question'], 'method': method,
                           'results': [{k: v for k, v in c.items() if k != 'text'} | {'text': c['text'][:400]} for c in marked],
                           'hit1': question_hit(res[:1], q), 'hit3': question_hit(res[:3], q), 'context': ctx, 'ms': ms})
        (args.output / f'traces_{method}.json').write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding='utf-8')
        by = defaultdict(list)
        for t in traces:
            by[t['cat']].append(t)
        row = {'method': method, 'n': len(traces), 'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context': {b: sum(t['context'][b]['hit'] for t in traces) for b in map(str, BUDGETS)},
               'median_ms': round(statistics.median(t['ms'] for t in traces), 1),
               'by_cat': {c: {'n': len(v), 'hit1': sum(t['hit1'] for t in v), 'hit3': sum(t['hit3'] for t in v),
                              'ctx3000': sum(t['context']['3000']['hit'] for t in v)} for c, v in sorted(by.items())},
               'misses_hit3': [t['query_id'] for t in traces if not t['hit3']]}
        rows.append(row)
    (args.output / 'summary.json').write_text(json.dumps({'split': args.split, 'rows': rows}, indent=2), encoding='utf-8')
    for r in rows:
        print(f"{r['method']:7} n={r['n']} H@1={r['hit1']} H@3={r['hit3']} ctx3000={r['context']['3000']} ctx6000={r['context']['6000']} "
              f"{r['median_ms']}ms misses={r['misses_hit3']}")
        print('        ', {c: f"{v['hit3']}/{v['n']}" for c, v in r['by_cat'].items()})


if __name__ == '__main__':
    main()
