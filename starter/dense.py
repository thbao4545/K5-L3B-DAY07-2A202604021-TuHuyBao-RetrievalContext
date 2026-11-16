"""Dense retrieval over the Day 6 pgvector index (B_paragraph by default).

Reuses Day 6's real embedder (MyEmbedder) and PgVectorStore, so query vectors are
compatible with the stored document vectors. No mock/TF-IDF fallback: any model or
database error propagates. Run with the Day 6 venv (psycopg, sentence-transformers,
python-dotenv):

  <day6>/starter/.venv/Scripts/python.exe -I starter/dense.py --split dev --output starter/results/dev-dense
  ... --use-filter --output starter/results/dev-dense-filter     # apply question['filter']

The filter comes from the question's `filter` field (user-supplied context); gold and
evidence are only used afterwards to score hits.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
DAY6 = (HERE.parents[1] / 'K5-L3-Day06-DataIngestionEngineering' / 'starter')
sys.path.insert(0, str(HERE))   # baseline (hit, validate, pack_context)
sys.path.insert(0, str(DAY6))   # src.embeddings, src.store

from baseline import hit, pack_context, validate  # noqa: E402
from src.embeddings import MyEmbedder  # noqa: E402
from src.store import PgVectorStore  # noqa: E402


class DenseRetriever:
    """Retriever.search(query, filters=, top_k=) as expected by agentic.run_agent."""

    def __init__(self, database_url, model_name, strategy='B_paragraph'):
        self.embedder = MyEmbedder(model_name)
        self.store = PgVectorStore(database_url, self.embedder.dimensions)
        self.strategy = strategy
        self.model_name = model_name

    def search(self, query, *, filters=None, top_k=3):
        vector = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vector, top_k=top_k, filters=filters)
        return [{'id': r['chunk_id'], 'source_id': r['document_id'], 'source': r['source'],
                 'page': r['page'], 'text': r['text'], 'metadata': r['metadata'],
                 'score': r['score'], 'strategy': f'dense:{self.strategy}'} for r in rows]


def load_retriever(strategy='B_paragraph'):
    env = dotenv_values(HERE / '.env')
    if not env.get('DATABASE_URL'):
        raise SystemExit('DATABASE_URL missing in starter/.env')
    model = env.get('EMBEDDING_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
    return DenseRetriever(env['DATABASE_URL'], model, strategy)


def evaluate(retriever, questions, *, use_filter, top_k, budget, name):
    traces = []
    for q in questions:
        filters = q.get('filter') if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        top3 = marked[:3]
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'strategy': name, 'top_k': top_k, 'budget_chars': budget, 'filter': filters,
                       'gold': q['gold'], 'results': marked, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'hit1': bool(marked and marked[0]['hit']),
                       'hit3': any(c['hit'] for c in top3),
                       'context_hit': any(c['hit'] and c['id'] in ids for c in marked),
                       'filter_respected': (None if not filters else all(
                           all(c['metadata'].get(k) == v for k, v in filters.items()) for c in results))})
    return traces


def main():
    ap = argparse.ArgumentParser(description='Dense retrieval baseline on one explicit split')
    ap.add_argument('--chunks', type=Path, default=HERE / 'data/day6/chunks.json',
                    help='exported chunks, used only to validate gold evidence')
    ap.add_argument('--questions', type=Path, default=HERE / 'data/benchmark.json')
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--strategy', default='B_paragraph')
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--budget', type=int, default=800)
    ap.add_argument('--use-filter', action='store_true', help="apply each question's user-supplied `filter`")
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split and q.get('answerable')]
    validate(chunks, selected)

    t0 = time.perf_counter()
    retriever = load_retriever(args.strategy)
    load_seconds = time.perf_counter() - t0
    name = f"dense{'+filter' if args.use_filter else ''}:{args.strategy}"
    traces = evaluate(retriever, selected, use_filter=args.use_filter, top_k=args.top_k,
                      budget=args.budget, name=name)
    filtered = [t for t in traces if t['filter']]
    summary = {'strategy': name, 'split': args.split, 'n': len(traces),
               'embedding_model': retriever.model_name, 'dimensions': retriever.embedder.dimensions,
               'top_k': args.top_k, 'budget_chars': args.budget,
               'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context_hit': sum(t['context_hit'] for t in traces),
               'n_filter_questions': len(filtered),
               'filter_applied': args.use_filter,
               'filter_respected': (sum(bool(t['filter_respected']) for t in filtered)
                                    if args.use_filter else None),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'model_load_seconds': load_seconds, 'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for fname, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / fname).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    misses = [t['query_id'] for t in traces if not t['hit3']]
    print('Hit@3 misses:', misses or 'none')


if __name__ == '__main__':
    main()
