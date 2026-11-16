"""Export the Day 6 pgvector index (chunks) to the Day 7 JSON contract and verify
that query embeddings are compatible with the stored vectors.

Read-only against the database. Run with the Day 6 venv (needs psycopg,
python-dotenv, sentence-transformers):

  <day6>/starter/.venv/Scripts/python.exe starter/export_day6.py --day6 <day6>/starter --output starter/data/day6
"""
import argparse
import json
import sys
from pathlib import Path

import psycopg
from dotenv import dotenv_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--day6', required=True, type=Path, help='Day 6 starter/ directory')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--strategy', default='B_paragraph', help='Day 6 chunk strategy to export as main index')
    args = ap.parse_args()

    env = dotenv_values(args.day6 / '.env')
    url, model = env['DATABASE_URL'], env.get('EMBEDDING_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
    args.output.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(url, connect_timeout=5) as conn:
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        pg = conn.execute("SHOW server_version").fetchone()[0]
        stats = conn.execute("SELECT strategy, count(*), count(DISTINCT page), min(vector_dims(embedding)), "
                             "max(vector_dims(embedding)) FROM chunks GROUP BY 1 ORDER BY 1").fetchall()
        for strategy, *_ in stats:
            rows = conn.execute("SELECT chunk_id, document_id, source, page, text, metadata FROM chunks "
                                "WHERE strategy=%s ORDER BY page, chunk_id", (strategy,)).fetchall()
            chunks = [{'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'text': r[4],
                       'metadata': r[5]} for r in rows]
            (args.output / f'chunks_{strategy}.json').write_text(
                json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
            if strategy == args.strategy:
                (args.output / 'chunks.json').write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

        # Compatibility check: embed a stored chunk with the same model; its own vector must be ~1.0 cosine.
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        dims = int(m.get_sentence_embedding_dimension())
        cid, text = conn.execute("SELECT chunk_id, text FROM chunks WHERE strategy=%s LIMIT 1",
                                 (args.strategy,)).fetchone()
        vec = '[' + ','.join(repr(float(x)) for x in m.encode([text], normalize_embeddings=True)[0]) + ']'
        sim = conn.execute("SELECT 1 - (embedding <=> %s::vector) FROM chunks WHERE strategy=%s AND chunk_id=%s",
                           (vec, args.strategy, cid)).fetchone()[0]

    info = {'embedding_model': model, 'dimensions': dims, 'distance': 'cosine (<=>), normalized vectors',
            'vector_store': f'PostgreSQL {pg} + pgvector {ext}', 'table': 'chunks(strategy, chunk_id, ...)',
            'connection': 'DATABASE_URL in Day 6 starter/.env (docker compose -p day6_student_starter, 127.0.0.1:55433)',
            'strategies': [{'strategy': s, 'chunks': n, 'pages': p, 'dims_min': a, 'dims_max': b}
                           for s, n, p, a, b in stats],
            'main_index': args.strategy,
            'self_similarity_check': {'chunk_id': cid, 'cosine': round(float(sim), 6)}}
    (args.output / 'index_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if dims != stats[0][3] or sim < 0.999:
        sys.exit('Query embedding NOT compatible with stored vectors')


if __name__ == '__main__':
    main()
