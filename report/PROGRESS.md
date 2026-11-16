# Day 7 – Progress report (retrieval comparison on dev)

Date: 2026-11-16. Scope so far: GUIDE.md §1 (prepare) and §2 (retrieval methods). The agent (§3) is not started. **The test split has not been run or inspected.**

## 1. What was done

| Item | File | Status |
|---|---|---|
| `.env` for Day 7 (DB URL, embedding model, LLM keys) | `starter/.env` (git-ignored), `starter/.env.example` | done; LLM key filled by user |
| New held-out test set t01–t06 (+u03, u04); Day 6 test q16–q20 moved to dev | `starter/build_benchmark.py` → `starter/data/benchmark.json` | done, evidence verified |
| Dense retrieval on the Day 6 pgvector index | `starter/dense.py` | done |
| BM25 + hybrid (RRF) + comparison on dev | `starter/compare.py` | done |
| Weighted-RRF experiment | `starter/hybrid_weighted.py` | done (no gain) |
| Unit tests (15) | `tests/` | pass (`PYTHONUTF8=1`, Day 6 venv, `unittest`) |

Raw results: `starter/results/dev-dense*/`, `starter/results/dev-compare/` (summary.json + one traces file per configuration). These folders are git-ignored; regenerate with the commands in §5.

## 2. Setup used (kept identical across methods)

- Corpus: 1 PDF, 47 pages; index `B_paragraph`, 199 chunks (section metadata, 28 values).
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`, local, 384-d, cosine, cost 0 USD. Postgres 16.16 + pgvector 0.8.7 in Docker.
- Dev = 20 answerable questions (q01–q20). 5 of them carry a user-supplied `section` filter. Top-3 for Hit@1/Hit@3; `hit` = gold source+page and the full evidence string inside the chunk text.
- BM25: own implementation (Okapi, k1=1.5, b=0.75, plain `\w+` tokenizer, no stemming). Parameters are defaults, not tuned.
- Hybrid: RRF (k=60) over the top-20 of dense and of BM25. Parameters are defaults.
- No API is called by any method here.

## 3. Results on dev (n = 20; one question = 5 points)

| Method | Hit@1 | Hit@3 | MRR@3 | Context hit @800 / 1500 / 3000 chars | Misses (Hit@3) |
|---|---:|---:|---:|---|---|
| dense | 11/20 | 16/20 | 0.658 | 3 / 11 / 16 | q06 q09 q13 q16 |
| dense + filter | 11/20 | 16/20 | 0.658 | 3 / 11 / 16 | q06 q09 q13 q16 |
| **bm25** | **16/20** | **19/20** | **0.867** | 4 / 16 / 18 | q05 |
| bm25 + filter | 16/20 | 19/20 | 0.867 | 4 / 16 / 18 | q05 |
| hybrid (RRF) | 14/20 | 16/20 | 0.750 | 3 / 14 / 16 | q05 q06 q09 q16 |
| hybrid + filter | 14/20 | 16/20 | 0.750 | 3 / 14 / 16 | q05 q06 q09 q16 |

Filter compliance: 5/5 filtered questions had all top-3 chunks inside the requested section, for all three methods. Latency (median): dense ≈ 32 ms, BM25 ≈ 1 ms, hybrid ≈ 35 ms. Model load ≈ 14–18 s once.

Weighted RRF (BM25 weight 1, 2, 3 against dense weight 1): identical to plain hybrid (14/20, 16/20) – no improvement (`results/dev-compare/weighted_hybrid.json`).

## 4. Findings

1. **BM25 beat dense and hybrid on this corpus.** Questions often reuse the document's rare terms (BEAVER, "name … actor.name", "closed API"), which BM25 matches exactly. MiniLM is a small English model and returned related-topic chunks instead. Caution: n = 20, a single English document, defaults not tuned; this does not generalise beyond this corpus.
2. **Hybrid was worse than BM25.** In q06, q09, q16 BM25 ranked the gold chunk first, but chunks that appeared in both lists (at lower ranks) outscored it under RRF, because RRF rewards agreement. Raising the BM25 weight to 3 did not change the outcome (for q06 a chunk present in both lists still wins). Only q13 was fixed by hybrid (dense missed it, BM25 and hybrid found it).
3. **Context budget is the main loss for context, not retrieval.** `pack_context` drops any chunk that does not fit the budget; paragraph chunks are 892 chars (median, max 1000) before the header line, so with 800 chars the top-1 chunk was too big in 13/20 questions and 9 questions got an empty context. Context hit rises from 3/20 to 11–16/20 at 1500 chars and to 16–18/20 at 3000. Evidence Hit@3 is unchanged by the budget.
4. **Remaining failures to analyse (candidates for the required two dev error analyses):**
   - q05 (BM25/hybrid miss): "How is a predicted SQL query judged correct under execution accuracy?" – generic words; dense found it (p10) but BM25 ranked p13/p9/p12 above.
   - q06 (dense/hybrid miss): paraphrase ("overly lenient" vs "overestimate/coincidence"); BM25 found it.
   - q16 (dense/hybrid miss): rare entity name BEAVER; BM25 found it.
5. **The metadata filter had no effect on rankings** on dev (same Hit@1/Hit@3 with and without) but it guaranteed the section constraint (5/5). Limitation: `section` is the nearest preceding heading, so a chunk at a section boundary can carry the previous section's label (seen in page 16, whose first chunks are labelled 5.2 while the paragraph belongs to 5.1 text).

## 5. Reproduce

From the Day 7 root, with the Day 6 venv and the Docker Postgres up (`docker compose -p day6_student_starter up -d --wait` in the Day 6 `starter/`):

```bash
PY=../K5-L3-Day06-DataIngestionEngineering/starter/.venv/Scripts/python.exe
export PYTHONUTF8=1
python -I starter/build_benchmark.py
$PY -I starter/dense.py   --split dev --output starter/results/dev-dense
$PY -I starter/compare.py --split dev --output starter/results/dev-compare
$PY -I starter/hybrid_weighted.py
```

## 6. Limits and risks to state in the final report

- Single author wrote the test questions after reading the corpus and chunks; test has only 6 answerable + 2 unanswerable questions.
- Dev now contains q16–q20 (used for Day 6 grading); this is documented in `PREPARE.md`.
- All numbers are single runs on a small dev set; differences of 1–2 questions are within noise. Hyperparameters (BM25, RRF) were not tuned except the weight sweep above, which was done on dev.
- English questions and an English embedding model: effect of Vietnamese tokenisation is not tested.
- Evidence matching is exact-string; manual grading of context/grounding is still required.

## 7. Agent on dev (GUIDE §3–4) – added after the first report

**Agent.** `starter/run_agent.py` + `starter/llm.py` + the `starter/agentic.py` loop. Controller = real LLM `gpt-6-luna` (OpenAI chat-completions, called over plain HTTP) with two tools: `search(query)` and `answer(answer, citations)`; abstention = `answer` with empty citations. `reasoning_effort="none"` is required by this model for function tools on that endpoint (so no reasoning tokens were used). Up to 3 searches, top-k 3 per search, the loop checks that every cited ID is in the final context. Filters are the user-supplied `filter` of a question; gold is never given to the model or retriever. Provider/retriever errors are stored as `status: "error"` (0 errors in all runs). Documents are labelled as untrusted data in the system prompt. Every trace stores queries, tool calls, retrieved chunks, final context, answer, citations, token usage and seconds (`starter/results/agent-dev-<config>/traces.json`).

**Configurations (dev: 20 answerable + u01, u02):**

| Config | Retriever | Budget | Context policy |
|---|---|---|---|
| A (baseline) | dense | 800 chars | keep earlier chunks first (starter default) |
| B | dense | 3000 | keep first |
| C | BM25 | 3000 | keep first |
| D | BM25 | 3000 | **newest results first** |

**Automatic proxies (gold used after the run; these are NOT the manual grades the guide requires):**

| | A | B | C | D |
|---|---:|---:|---:|---:|
| Answered / 20 answerable | 4 | 17 | 19 | **20** |
| Final context contains gold evidence | 4/20 | 16/20 | 17/20 | **18/20** |
| A cited chunk contains gold evidence | 4/20 | 16/20 | 17/20 | **18/20** |
| Abstained on unanswerable (u01, u02) | 2/2 | 2/2 | 2/2 | 2/2 |
| False abstentions (answerable but refused) | 16/20 | 3/20 | 1/20 | **0/20** |
| Searches per question (0/1/2/3 calls) | 0/2/0/20 | 0/14/1/5 | 0/15/2/5 | 0/16/3/3 |
| Total tokens (22 questions) | 39,060 | 44,156 | 42,176 | 38,786 |
| Median seconds / question | 6.4 | 3.9 | 3.7 | 3.3 |

Cost in USD is not computed because the price of `gpt-6-luna` is not known to me; token counts are recorded.

**Improvement experiments (dev, with hypotheses):**
1. *Raise budget 800 → 3000 (A → B)*. Hypothesis: the agent fails because of context packing, not retrieval. Result: answered 4 → 17, false abstentions 16 → 3. Supported. With 800 chars the agent searched 3 times on 20 of 22 questions and then gave up.
2. *Dense → BM25 (B → C)*. Hypothesis: BM25 has better Hit@3 (section 3). Result: context evidence 16 → 17, answered 17 → 19. Small gain, within noise for n = 20.
3. *Newest-first context packing (C → D)*. Hypothesis (from the q02 error below): the right chunk is retrieved on a later search but does not fit because earlier chunks fill the budget. Result: q02 fixed, false abstentions 1 → 0, context evidence 17 → 18, tokens slightly lower. Not a controlled proof: one question changed.

**Dev error analysis (config C, read from traces):**
- **q02 (false abstention).** Gold chunk `p6:paragraph:1` was retrieved on searches 2 and 3 but never entered the context: the three chunks from search 1 (p7, p9, p7) already used the budget, and `pack_context` drops chunks that do not fit. The model then correctly said "not enough evidence" for the context it had. Cause: starter `keep_first` policy, not the retriever or the model. Fix = D; after D q02 is answered with the gold chunk in context.
- **q05 (answered, wrong page).** The answer ("results match") is semantically right but cites `p13:paragraph:2`; gold chunk `p10:paragraph:0` was retrieved on searches 2–3 and again did not fit. Still unresolved in D (context hit False). Needs manual judgement: the answer may be acceptable if p13 states the same rule, which the guide lets you accept as an alternative valid source only if you verify it.
- **q09 (answered, gold not in context).** Long, keyword-stuffed rewrite queries kept returning p11 chunks; gold `p12:paragraph:1` never appeared in results. The answer ("embedding and schema proxies are cheaper fallbacks") matches the expected answer but is cited to p11. Needs manual check of whether p11 supports it. Not fixed.
- Observation: for unanswerable questions the agent used all 3 searches before abstaining (u01, u02), so abstention is correct but costs the full search budget.

## 8. Decision rule and next step (needs your approval before test)

Criteria I suggest locking **before reading test results**: (1) Evidence Hit@3 / final-context evidence rate on dev, then (2) answered correctly without false abstention, then (3) tokens/time. On that rule **D (BM25, 3000 chars, newest-first, max 3 searches, top-k 3)** is the best dev configuration and **A (dense, 800, keep-first)** is the baseline. Caveats: 20 dev questions, 1 to 2 question differences, config chosen on the same dev set it was tuned on.

If you agree:
1. Run the agent once on test for the baseline and the chosen config (the runner refuses test unless `--i-locked-config` is passed):
   `$PY -I starter/run_agent.py --split test --config A_dense_b800 --i-locked-config`
   `$PY -I starter/run_agent.py --split test --config D_bm25_b3000_latest --i-locked-config`
2. Hand-grade (you, as author; note the bias risk): ≥ 5 test answers (answer 0–2, each citation supported yes/no, context 0–2) and the 2 unanswerable (did it really refuse, without added claims). Fill `reviews_template.json` and run `starter/agent_metrics.py`.
3. Write the final report from `REPORT_TEMPLATE.md` (use sections 3–7 here as raw material, include one answered example, one abstention and one error such as q02).

## 9. Test results (run once, after locking A = baseline and D = chosen)

Run on 2026-11-16 with `--i-locked-config`; configuration, criteria and code were not changed afterwards. Test = 6 answerable (t01–t06) + 2 unanswerable (u03, u04). 0 provider errors. Output: `starter/results/agent-test-<config>/` (`traces.json`, `summary_auto.json`, `agent_metrics.json`, `reviews_ai_prelim.json`) and `starter/results/test-compare/` (retrieval only).

**Retrieval only (test, top-3, n = 6):**

| Method | Hit@1 | Hit@3 | Context hit @800 / 1500 / 3000 |
|---|---:|---:|---|
| dense | 5/6 | 6/6 | 0 / 5 / 6 |
| BM25 | 6/6 | 6/6 | 0 / 6 / 6 |
| hybrid (RRF) | 6/6 | 6/6 | 0 / 6 / 6 |

Filter (t06): 1/1 respected for all methods. All methods already reach Hit@3 = 6/6, so **test cannot separate the retrieval methods** (ceiling effect; the test questions name their targets explicitly, e.g. "MAC-SQL", "TRUST-SQL"). Differences between methods seen on dev (BM25 > dense, hybrid < BM25) are therefore neither confirmed nor refuted by test.

**Agent (test):**

| | A baseline (dense, 800 chars, keep-first) | D chosen (BM25, 3000, newest-first) |
|---|---:|---:|
| Tool use (called search) | 8/8 | 8/8 |
| Searches per question (0/1/2/3) | 0/0/0/8 | 0/6/0/2 |
| Answered, answerable | 0/6 | 6/6 |
| Grounded answer grade 0 / 1 / 2 (mean /2) | 6 / 0 / 0 (0.00) | 0 / 0 / 6 (2.00) |
| Citation valid (supported/total); answers with citation | 0/0 (n/a); 0/0 | 6/6; 6/6 |
| Abstention on unanswerable | 2/2 | 2/2 |
| False abstention (answerable refused) | 6/6 | 0/6 |
| Context grade 0 / 1 / 2 | 6 / 0 / 0 | 0 / 0 / 6 |
| Total tokens (8 questions) | 14,290 | 14,968 |
| Median seconds / question | 7.0 | 3.4 |

Cost in USD: not computed (price of `gpt-6-luna` unknown); token counts above. Embedding and BM25 cost 0.

**Why A fails.** Retrieval is fine (dense Hit@3 = 6/6) but at an 800-character budget `pack_context` drops every chunk that does not fit, so the final context was empty for 5 of the 8 questions (t01, t02, t03, t05, t06); t04 and the two unanswerable questions got a single chunk, and for t04 it did not contain the gold evidence. The model then, correctly for the context it saw, said there was not enough evidence. This repeats the dev result (A: 4/20 answered).

**Honest limits of these numbers.**
- n = 6 answerable, all easy and explicit; D = 6/6 is not evidence that D is generally better than other retrieval methods, only that a 3000-char budget fixes the 800-char packing loss. The A-vs-D gap is dominated by the budget, not by dense vs BM25.
- The labels above are **AI-assisted preliminary labels** (Claude read each answer against the cited chunk; method: compare answer with `expected_answer`, check the cited chunk contains the gold evidence, check abstentions contain no extra claim). The guide asks the author to grade; please review `reviews_ai_prelim.json`, change any label you disagree with, and re-run `starter/agent_metrics.py` before submitting. Bias risk: the same assistant wrote the test questions, the pipeline and these labels.
- The raw abstention text in the table of traces is the fixed string from `agentic.py`; the model's own words ("not enough evidence in the retrieved context …") are in `llm_calls` of each trace.
- Test questions were written by the same author who knew the corpus; test set is small.

## 10. Still to do before submission
1. Review/adjust the preliminary labels; keep `reviews.json` of your own (copy `reviews_ai_prelim.json`) and re-run `agent_metrics.py`.
2. Fill `REPORT_TEMPLATE.md` → `report/individual/<MSSV>_<HoTen>.md` using sections 3–9; include one answered example (t05), one abstention (u03) and one error (dev q02, or A's refusals on test).
3. Follow `SUBMISSION.md` (README run commands, results; `starter/data` and `starter/results` are git-ignored, add with `git add -f` only what you may share).
