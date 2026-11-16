"""Build the Day 7 benchmark: Day 6 answerable questions (converted) + unanswerable questions.

Gold/evidence lives only in this benchmark file; it must never be passed to the agent or retrievers.
"""
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
CHECK = ('Full-text scan of all 47 pages (252 fixed-size + 199 paragraph chunks, case-insensitive) found no '
         'occurrence of: {terms}.')

UNANSWERABLE = [
    {'id': 'u01', 'split': 'dev',
     'question': 'What execution accuracy did GPT-4o reach on the KaggleDBQA benchmark?',
     'absence_reason': 'The survey never mentions KaggleDBQA or GPT-4o. ' + CHECK.format(terms="'kaggledbqa', 'kaggle', 'gpt-4o', 'gpt4o'"),
     'terms': ['kaggledbqa', 'kaggle', 'gpt-4o', 'gpt4o']},
    {'id': 'u02', 'split': 'dev',
     'question': 'How many annotator hours were spent building the Spider 2.0 benchmark?',
     'absence_reason': 'Spider 2.0 is described (632 workflows) but no annotation effort or hours are reported. ' + CHECK.format(terms="'hours', 'person-hour'"),
     'terms': ['hours', 'person-hour']},
    {'id': 'u03', 'split': 'test',
     'question': 'How many questions does the Dr.Spider perturbation benchmark contain for each perturbation type?',
     'absence_reason': 'Dr.Spider is not covered by the survey. ' + CHECK.format(terms="'dr.spider', 'dr. spider', 'perturb'"),
     'terms': ['dr.spider', 'dr. spider', 'perturb']},
    {'id': 'u04', 'split': 'test',
     'question': 'What accuracy does the survey report for systems evaluated on the EHRSQL electronic health record benchmark?',
     'absence_reason': 'Healthcare is mentioned only as an application domain; EHRSQL is not covered. ' + CHECK.format(terms="'ehrsql', 'electronic health'"),
     'terms': ['ehrsql', 'electronic health']},
]

SRC = '2026Reimagining-Text-to-SQL'
# Fresh Day 7 test questions (written from pages 17-25, none used for Day 6 reporting/grading).
# The Day 6 test questions (q16-q20) were already used to report and hand-grade Day 6, so they are
# demoted to dev (extra dev data) and the held-out test is these new questions. Decided before any Day 7 run.
DEMOTED_TO_DEV = {'q16', 'q17', 'q18', 'q19', 'q20'}
NEW_TEST = [
    {'id': 't01', 'split': 'test',
     'question': 'In Arctic-Text2SQL-R1, what reward does a query get if it is syntactically valid and executes but returns a different result from the gold result?',
     'gold': [{'source_id': SRC, 'page': 17, 'evidence': '0.1 if the query is syntactically valid and executes but returns a different result'}],
     'expected_answer': '0.1 (1 for a matching result, 0 otherwise)'},
    {'id': 't02', 'split': 'test',
     'question': 'What execution accuracy did Databricks RLVR-32B reach on the BIRD test set, both single-generation and with self-consistency?',
     'gold': [{'source_id': SRC, 'page': 17, 'evidence': '73.56% execution accuracy on the BIRD test set in the singlemodel, single-generation setting (surpassing the previous best single-model result of 71.83%) and 75.68% with self-consistency'}],
     'expected_answer': '73.56% single-generation and 75.68% with self-consistency'},
    {'id': 't03', 'split': 'test',
     'question': 'In MAC-SQL, what does the Selector agent do?',
     'gold': [{'source_id': SRC, 'page': 24, 'evidence': 'The Selector filters the schema to retain only relevant tables and columns, reducing token overhead'}],
     'expected_answer': 'It filters the schema to keep only relevant tables and columns, reducing token overhead'},
    {'id': 't04', 'split': 'test',
     'question': 'What execution accuracy does SQL-of-Thought report on Spider-Syn?',
     'gold': [{'source_id': SRC, 'page': 25, 'evidence': 'strong robustness on Spider-Realistic (90.2%) and Spider-Syn (82.0%)'}],
     'expected_answer': '82.0%'},
    {'id': 't05', 'split': 'test',
     'question': 'What are the four phases of the interaction protocol that TRUST-SQL imposes?',
     'gold': [{'source_id': SRC, 'page': 20, 'evidence': 'Explore(schema-discovery tool calls such as table listing, column inspection, and row sampling),Propose(explicit commitment to a verified subset of tables, columns, and join paths),Generate(SQL conditioned on the proposed schema), andConfirm(execution and validation of the result)'}],
     'expected_answer': 'Explore, Propose, Generate and Confirm'},
    {'id': 't06', 'split': 'test',
     'question': 'In the section on agentic methods, what format is the semantic model at the core of Snowflake Cortex Analyst encoded in?',
     'gold': [{'source_id': SRC, 'page': 23, 'evidence': 'a semantic model encoded in YAML that captures domain-specific table and column metadata, relationships, and synonyms'}],
     'expected_answer': 'YAML',
     'filter': {'section': '5.4 Agentic Methods'}},
]


def verify_new(questions, chunks_b):
    """Each evidence must sit verbatim in exactly one B_paragraph chunk of the stated page; filters must match it."""
    for q in questions:
        for g in q['gold']:
            hits = [c for c in chunks_b if c['page'] == g['page'] and g['evidence'] in c['text']]
            if len(hits) != 1:
                sys.exit(f"{q['id']}: evidence found in {len(hits)} chunks on page {g['page']} (need exactly 1)")
            for k, v in q.get('filter', {}).items():
                if hits[0]['metadata'].get(k) != v:
                    sys.exit(f"{q['id']}: chunk {hits[0]['id']} has {k}={hits[0]['metadata'].get(k)!r}, not {v!r}")


def main() -> None:
    chunks = json.loads((DATA / 'day6/chunks_A_fixed.json').read_text(encoding='utf-8')) + \
        json.loads((DATA / 'day6/chunks_B_paragraph.json').read_text(encoding='utf-8'))
    corpus = ' '.join(re.sub(r'\s+', ' ', c['text']).lower() for c in chunks)
    out = []
    chunks_b = json.loads((DATA / 'day6/chunks_B_paragraph.json').read_text(encoding='utf-8'))
    for q in json.loads((DATA / 'converted/questions.json').read_text(encoding='utf-8')):
        q['answerable'] = True
        if q['id'] in DEMOTED_TO_DEV:
            q['day6_split'] = q['split']
            q['split'] = 'dev'
        out.append(q)
    verify_new(NEW_TEST, chunks_b)
    for q in NEW_TEST:
        out.append({**q, 'answerable': True})
    for u in UNANSWERABLE:
        hits = [t for t in u.pop('terms') if t in corpus]
        if hits:
            sys.exit(f"{u['id']} is not unanswerable: corpus contains {hits}")
        u.update(answerable=False, expected_behavior='search_and_abstain')
        out.append(u)
    ids = [q['id'] for q in out]
    assert len(ids) == len(set(ids))
    (DATA / 'benchmark.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split in ('dev', 'test'):
        a = sum(q['split'] == split and q['answerable'] for q in out)
        n = sum(q['split'] == split and not q['answerable'] for q in out)
        print(f'{split}: {a} answerable, {n} unanswerable')


if __name__ == '__main__':
    main()
