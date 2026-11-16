"""Build the advanced-challenge index: text + table + chart/figure chunks -> pgvector (strategy ADV_gemini15).

Run with the Day 6 venv from the Day 7 root (needs psycopg + sentence-transformers):
  <day6>/starter/.venv/Scripts/python.exe -I adv/index_adv.py

Chunk kinds
  text   : paragraph chunks of the PDF text layer (<= 1000 chars), page + section + bbox
  table  : markdown table from the vision pass, split by rows (header repeated) into <= 1500 chars; label + caption in every part
  chart / figure : vision description (chart/diagram/image) split into <= 1500 chars; label + caption in every part
Every chunk keeps: page, kind, label ("Table 12"/"Figure 3"/""), page_image (rendered original page) so that evidence can be
re-checked against the original. Re-running upserts the same IDs (no duplicates) and deletes stale chunks of the strategy.
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parent / 'K5-L3-Day06-DataIngestionEngineering' / 'starter'
sys.path.insert(0, str(DAY6))
sys.path.insert(0, str(ROOT / 'starter'))

from dotenv import dotenv_values  # noqa: E402
from src.embeddings import MyEmbedder  # noqa: E402
from src.models import Chunk  # noqa: E402
from src.store import PgVectorStore  # noqa: E402

STRATEGY = 'ADV_gemini15'
DOC = 'gemini15'
SOURCE = 'Gemini 1.5 technical report (arXiv 2403.05530v5)'
MAXV = 1500
LABEL = re.compile(r'(Figure|Table)\s*(\d+)', re.I)


def norm_label(label, caption=''):
    m = LABEL.search(label or '') or LABEL.match(caption or '')
    return f'{m.group(1).capitalize()} {m.group(2)}' if m else ''


def split_table(md, header_prefix, limit=MAXV):
    lines = [l for l in md.strip().split('\n') if l.strip()]
    if len(lines) < 3 or not lines[1].lstrip().startswith('|'):
        return [header_prefix + md]
    head, rows = lines[:2], lines[2:]
    parts, cur = [], []
    for r in rows:
        if cur and len(header_prefix) + sum(len(x) + 1 for x in head + cur) + len(r) > limit:
            parts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        parts.append(cur)
    return [header_prefix + '\n'.join(head + p) for p in parts]


def split_text(txt, prefix, limit=MAXV):
    sents, parts, cur = re.split(r'(?<=[.!?])\s+', txt.strip()), [], ''
    for s in sents:
        if cur and len(prefix) + len(cur) + len(s) + 1 > limit:
            parts.append(prefix + cur)
            cur = ''
        cur = (cur + ' ' + s).strip()
    if cur:
        parts.append(prefix + cur)
    return parts


def build_chunks():
    out = []
    for c in json.loads((ROOT / 'adv/out/text_chunks.json').read_text(encoding='utf-8')):
        out.append({'id': c['id'], 'page': c['page'], 'kind': 'text', 'label': '', 'text': c['text'],
                    'metadata': {'kind': 'text', 'section': c['section'], 'bbox': c['bbox'], 'label': '',
                                 'page_image': f"adv/out/pages/p{c['page']:03d}.png"}})
    vis = json.loads((ROOT / 'adv/out/visual_items.json').read_text(encoding='utf-8'))
    for p, v in sorted(vis.items(), key=lambda kv: int(kv[0])):
        for i, it in enumerate(v.get('items', [])):
            kind = 'table' if it.get('kind') == 'table' else 'chart' if it.get('kind') == 'chart' else 'figure'
            label = norm_label(it.get('label', ''), it.get('caption', ''))
            cap = (it.get('caption') or '').strip()
            prefix = f"[{label or kind} | page {p}] {cap}\n"
            body = (it.get('content') or '').strip()
            if not body:
                continue
            parts = split_table(body, prefix) if kind == 'table' else split_text(body, prefix)
            for j, t in enumerate(parts):
                out.append({'id': f'{DOC}:p{p}:{kind}:{i}:{j}', 'page': int(p), 'kind': kind, 'label': label, 'text': t,
                            'metadata': {'kind': kind, 'label': label, 'caption': cap[:200], 'part': j, 'parts': len(parts),
                                         'section': '', 'page_image': f'adv/out/pages/p{int(p):03d}.png'}})
    return out


def main():
    chunks = build_chunks()
    ids = [c['id'] for c in chunks]
    assert len(ids) == len(set(ids)), 'duplicate chunk ids'
    (ROOT / 'adv/out/index_chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding='utf-8')
    env = dotenv_values(ROOT / 'starter/.env')
    model = env.get('EMBEDDING_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
    t0 = time.perf_counter()
    emb = MyEmbedder(model)
    vectors = emb.embed_documents([c['text'] for c in chunks])
    t_embed = time.perf_counter() - t0
    store = PgVectorStore(env['DATABASE_URL'], emb.dimensions)
    store.create_schema()
    objs = [Chunk(chunk_id=c['id'], document_id=DOC, source=SOURCE, page=c['page'], text=c['text'], metadata=c['metadata'])
            for c in chunks]
    t1 = time.perf_counter()
    store.upsert(STRATEGY, objs, vectors)
    t_store = time.perf_counter() - t1
    import psycopg
    with psycopg.connect(env['DATABASE_URL']) as conn:
        n, pages = conn.execute('SELECT count(*), count(DISTINCT page) FROM chunks WHERE strategy=%s', (STRATEGY,)).fetchone()
        dup = conn.execute('SELECT count(*) - count(DISTINCT chunk_id) FROM chunks WHERE strategy=%s', (STRATEGY,)).fetchone()[0]
    import collections
    kinds = collections.Counter(c['kind'] for c in chunks)
    lens = [len(c['text']) for c in chunks]
    stats = {'strategy': STRATEGY, 'chunks': len(chunks), 'by_kind': dict(kinds), 'in_db': n, 'pages_in_db': pages,
             'duplicates_in_db': dup, 'chars_total': sum(lens), 'chars_median': sorted(lens)[len(lens) // 2], 'chars_max': max(lens),
             'embedding_model': model, 'dimensions': emb.dimensions, 'embed_seconds': round(t_embed, 1), 'store_seconds': round(t_store, 1),
             'service_cost_usd': 0}
    (ROOT / 'adv/out/index_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
