"""Compare retrieval methods on one split: dense, BM25 (lexical), hybrid (RRF).

Run with the Day 6 venv from the Day 7 root:
  <day6>/starter/.venv/Scripts/python.exe -I starter/compare.py --split dev --output starter/results/dev-compare

Gold/evidence is used only to score; filters come from question['filter'] (user-supplied).
No API is called (local embeddings + local BM25).
"""
import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baseline import hit, pack_context, tokenize, validate  # noqa: E402
from dense import load_retriever  # noqa: E402

BUDGETS = (800, 1500, 3000)
CAND = 20          # candidates taken from each retriever before fusion
RRF_K = 60         # standard RRF constant (not tuned)


class BM25:
    """Okapi BM25 over the exported B_paragraph chunks (k1=1.5, b=0.75, defaults, not tuned)."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c['text'])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(chunks)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, *, filters=None, top_k=3):
        q = [t for t in tokenize(query) if t in self.idf]
        scored = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if filters and any(c['metadata'].get(k) != v for k, v in filters.items()):
                continue
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append({**c, 'score': s, 'strategy': 'bm25'})
        return sorted(scored, key=lambda c: (-c['score'], c['id']))[:top_k]


def rrf(lists, top_k, k=RRF_K):
    score, keep = Counter(), {}
    for lst in lists:
        for rank, c in enumerate(lst, 1):
            score[c['id']] += 1 / (k + rank)
            keep.setdefault(c['id'], c)
    ranked = sorted(score, key=lambda i: (-score[i], i))[:top_k]
    return [{**keep[i], 'score': score[i], 'strategy': 'hybrid_rrf'} for i in ranked]


def run(method, q, dense, bm25, use_filter):
    flt = q.get('filter') if use_filter else None
    if method == 'dense':
        return dense.search(q['question'], filters=flt, top_k=3)
    if method == 'bm25':
        return bm25.search(q['question'], filters=flt, top_k=3)
    return rrf([dense.search(q['question'], filters=flt, top_k=CAND),
                bm25.search(q['question'], filters=flt, top_k=CAND)], 3)


def evaluate(method, use_filter, questions, dense, bm25):
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        res = run(method, q, dense, bm25, use_filter)
        ms = (time.perf_counter() - t0) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in res]
        ranks = [i for i, c in enumerate(marked, 1) if c['hit']]
        flt = q.get('filter') if use_filter else None
        ctx = {}
        for b in BUDGETS:
            text, ids = pack_context(res, b)
            ctx[b] = {'chars': len(text), 'ids': ids,
                      'hit': any(c['hit'] and c['id'] in ids for c in marked)}
        traces.append({'query_id': q['id'], 'question': q['question'], 'filter': flt, 'results': marked,
                       'hit1': bool(marked and marked[0]['hit']), 'hit3': bool(ranks),
                       'rr': 1 / ranks[0] if ranks else 0.0, 'elapsed_ms': ms,
                       'context': {str(b): v for b, v in ctx.items()},
                       'filter_respected': None if not flt else all(
                           all(c['metadata'].get(k) == v for k, v in flt.items()) for c in res)})
    return traces


def summarize(name, traces):
    n = len(traces)
    f = [t for t in traces if t['filter']]
    return {'config': name, 'n': n, 'hit1': sum(t['hit1'] for t in traces),
            'hit3': sum(t['hit3'] for t in traces), 'mrr@3': round(sum(t['rr'] for t in traces) / n, 3),
            'context_hit': {b: sum(t['context'][b]['hit'] for t in traces) for b in map(str, BUDGETS)},
            'filter_questions': len(f), 'filter_respected': sum(bool(t['filter_respected']) for t in f) if f else None,
            'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 1),
            'misses': [t['query_id'] for t in traces if not t['hit3']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--chunks', type=Path, default=HERE / 'data/day6/chunks.json')
    ap.add_argument('--questions', type=Path, default=HERE / 'data/benchmark.json')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    qs = [q for q in json.loads(args.questions.read_text(encoding='utf-8'))
          if q['split'] == args.split and q.get('answerable')]
    validate(chunks, qs)
    dense, bm25 = load_retriever(), BM25(chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ('dense', 'bm25', 'hybrid'):
        for use_filter in (False, True):
            name = f"{method}{'+filter' if use_filter else ''}"
            tr = evaluate(method, use_filter, qs, dense, bm25)
            (args.output / f'traces_{name}.json').write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding='utf-8')
            rows.append(summarize(name, tr))
    (args.output / 'summary.json').write_text(json.dumps(
        {'split': args.split, 'n_questions': len(qs), 'cand': CAND, 'rrf_k': RRF_K, 'rows': rows},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{'config':16} {'H@1':>5} {'H@3':>5} {'MRR':>6}  ctx@800/1500/3000  filt  ms  misses")
    for r in rows:
        c = r['context_hit']
        print(f"{r['config']:16} {r['hit1']:>2}/{r['n']} {r['hit3']:>2}/{r['n']} {r['mrr@3']:>6}  "
              f"{c['800']:>2}/{c['1500']:>2}/{c['3000']:>2}          "
              f"{r['filter_respected'] if r['filter_respected'] is not None else '-'}/{r['filter_questions']}  {r['median_ms']}  {r['misses']}")


if __name__ == '__main__':
    main()
