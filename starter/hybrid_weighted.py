"""Dev experiment: weighted RRF (dense weight 1, BM25 weight w). Run with the Day 6 venv."""
import json, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare
from compare import BM25, evaluate, summarize, hit
from dense import load_retriever

def wrrf(dense_l, bm_l, w, top_k=3, k=60):
    score, keep = Counter(), {}
    for lst, wt in ((dense_l, 1.0), (bm_l, w)):
        for r, c in enumerate(lst, 1):
            score[c['id']] += wt / (k + r); keep.setdefault(c['id'], c)
    return [{**keep[i], 'score': score[i], 'strategy': 'hybrid_wrrf'} for i in sorted(score, key=lambda i: (-score[i], i))[:top_k]]

chunks = json.loads((HERE / 'data/day6/chunks.json').read_text(encoding='utf-8'))
qs = [q for q in json.loads((HERE / 'data/benchmark.json').read_text(encoding='utf-8')) if q['split'] == 'dev' and q['answerable']]
dense, bm = load_retriever(), BM25(chunks)
out = []
for w in (1, 2, 3):
    def run(method, q, dense, bm25, use_filter, w=w):
        f = q.get('filter') if use_filter else None
        return wrrf(dense.search(q['question'], filters=f, top_k=20), bm25.search(q['question'], filters=f, top_k=20), w)
    compare.run = run
    s = summarize(f'hybrid_w{w}', evaluate('hybrid', True, qs, dense, bm)); out.append(s)
    print(s['config'], f"H@1 {s['hit1']}/20 H@3 {s['hit3']}/20 MRR {s['mrr@3']} ctx {s['context_hit']} misses {s['misses']}")
(HERE / 'results/dev-compare/weighted_hybrid.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
