"""Shared helpers for the advanced challenge: retrievers over the ADV index and the evidence-hit definition.

Hit definition (decided before any run; gold is used ONLY here, after retrieval):
  A gold entry g of question q is satisfied by a chunk c when c.page == g.page and
    * g has text_evidence strings -> c is a text chunk that contains at least one of them (normalised), or
    * g.kind == table             -> c is a table chunk whose label equals g.label and that contains the entry's last cell
                                     (the target value), or
    * g.kind in (chart, figure)   -> c is a chart/figure chunk with the same label (Figure N), or with no label when
                                     the page has a single figure item.
  Question hit@k: EVERY gold entry is satisfied by some chunk in the top-k (single-evidence questions = usual hit;
  multi-hop = all required evidence). Context evidence = the same test on the final context chunks, where text entries
  need ALL their strings present in the context.
Whether a vision description contains the exact answer is NOT assumed; the final answer is graded separately.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parent / 'K5-L3-Day06-DataIngestionEngineering' / 'starter'
for p in (str(ROOT / 'starter'), str(DAY6), str(ROOT / 'adv')):
    if p not in sys.path:
        sys.path.insert(0, p)

VISUAL = {'chart', 'figure'}


def norm(s):
    return re.sub(r'\s+', ' ', s.replace('−', '-').replace('–', '-')).strip().lower()


def kind_of(c):
    return (c.get('metadata') or {}).get('kind', 'text')


def label_of(c):
    return (c.get('metadata') or {}).get('label', '')


def entry_satisfied(c, g):
    if c['page'] != g['page']:
        return False
    k, text = kind_of(c), norm(c['text'])
    ev = g.get('text_evidence')
    if ev and k == 'text' and any(norm(s) in text for s in ev):
        return True
    if g['kind'] == 'table':
        if k != 'table' or label_of(c) != g.get('label'):
            return False
        last = norm(g['cells'][-1]) if g.get('cells') else None
        return last is None or last in text
    if g['kind'] in ('chart', 'figure'):
        return k in VISUAL and label_of(c) in (g.get('label'), '')
    return False


def question_hit(chunks, q, strict_text=False):
    """All gold entries satisfied by the given chunks (union)."""
    for g in q['gold']:
        if strict_text and g.get('text_evidence') and g['kind'] == 'text':
            blob = ' '.join(norm(c['text']) for c in chunks if c['page'] == g['page'] and kind_of(c) == 'text')
            if not all(norm(s) in blob for s in g['text_evidence']):
                return False
        elif not any(entry_satisfied(c, g) for c in chunks):
            return False
    return True


def any_entry_hit(c, q):
    return any(entry_satisfied(c, g) for g in q['gold'])


def mark(results, q):
    return [{**c, 'hit': any_entry_hit(c, q)} for c in results]
