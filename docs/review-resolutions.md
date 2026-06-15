# Adversarial review resolutions — 2026-06-10

Two independent adversarial reviews of the benchmark methodology, results, and
plans (Fable 5 in-session; Codex in `adversarial-methodology-review.md`)
converged on the same core complaints. This documents every item and what was
done about it. Net effect on the decision: **GLM-OCR survives as the
production pick, but for corrected reasons and behind a pilot gate.**

## Scoreboard

| # | Finding (who) | Status | Resolution |
|---|---|---|---|
| 1 | Headline "0.9B beat every larger model" carried by ONE failed Qwen-32B page; no variance/paired stats reported (Fable) | ✅ fixed | `triangulate.py` now reports per-page sd, min, coverage, and paired per-page wins vs the leader. Corrected story: **Qwen2.5-72B is the accuracy winner (F1 0.927, beats GLM 8/10 pages); GLM wins on recall+speed**, which is what search needs. Report rewritten. |
| 2 | 10-page sample = one paper, one decade, front pages only; corpus is 6 papers / 1870s–2010s (both) | ✅ **pilot run & passed** | 27 stratified pages (1872–1964, 4 papers, interior pages, clippings) OCR'd by GLM + GPT-5.5 yardstick. **Recall 0.924 vs baseline 0.923 — holds exactly**; dict 0.95–1.00 everywhere; no decade/paper craters. `scripts/pilot_report.py`. Bonus finding: on dense pre-1920 pages tiled GLM out-reads GPT-at-2048px by 1.4–3.3× words (the flagship recall ceiling, demonstrated). |
| 3 | Consensus truth = flagship-legible text only; recall ceiling unmeasured; zero human validation (both) | 🟡 partially | Fourth-model validation on the weakest-agreement page (`data/bench/_faint_validate.py`): Fable 5 read the scan independently; all "missing" words traced to ONE systematic artifact (see #4), not illegibility. True human spot-check still recommended (~1 page, ~1 hr). Recall numbers are now documented as "vs flagship-legible text". |
| 4 | **NEW (found by #3's harness): line-break hyphenation corrupts truth & index.** Claude+Gemini transcribe "Laugh-↵lin" literally (a 2-of-3 majority!) → consensus holds `laugh`+`lin`, penalizing GLM/GPT which correctly join "Laughlin"; FTS5 query for *Laughlin* would MISS the page | ✅ fixed | `dehyphenate()` in `metrics.py`, applied in all tokenizers AND at index time (`index.py`). Claude↔GPT agreement 0.957→0.966; ranking unchanged (conclusions stable). |
| 5 | Digits (2.8% of tokens: dates, addresses, prices) invisible to all metrics (both) | ✅ fixed | `digit_recall` metric. GLM tiled: **0.876 — best of any fast model** (beats even 72B's 0.827). Not a hidden weakness. |
| 6 | BOW F1 dominated by common words; rare words (the actual search targets) unweighted (both) | ✅ fixed | `rare_word_recall` (zipf < 3.5). GLM 0.871; Qwen3-32B-tiled best at 0.940. |
| 7 | Tiling scrambles word order → FTS5 phrase queries break; unmeasured (both) | ✅ measured | `phrase_recovery` (proper-noun bigrams surviving adjacent) + `search_eval.py` for concrete missed queries. **GLM tiled 0.830 vs 0.874 for full-page 72B — tiling costs ~4 pts of phrase adjacency, far less than feared.** GLM's whole-word misses (392/3421 name-bigrams) are recall losses, not order losses → recall-union tier targets them. |
| 8 | No degeneration guard in production: silent repetition loops would index garbage at 53k-page scale (Fable) | ✅ fixed | `ocr_guard` config + `guard_verdict()` in `ocr_run.py`: low dict-rate / truncated pages retried once with stronger penalties, `flagged` recorded in JSON (= the tier-2 escalation set). `finish_reason` captured per page and per tile (`length:2/9`). |
| 9 | PNG cache: 14.7 MB/page avg → **~780 GB for 53k pages; C: has only ~86 GB free** — full run would crash mid-way (Fable, upgraded by measurement) | ✅ fixed | `ocr_run.py` now **purges page PNGs after successful OCR by default** (`--keep-png` to opt out). Webapp re-renders on view (`app.py:152`). |
| 10 | Repointing `paths.pdf_root` mutates shared config (both) | ✅ fixed | `--pdf-root` CLI flag on `ocr_run.py` and `benchmark.py`; config untouched. |
| 11 | Survivorship bias: models averaged only over pages they succeeded on (Fable) | ✅ fixed | Missing/failed pages score 0; `cov` column shows it. dots.ocr: 0.770 → 0.699. |
| 12 | Sanborn maps + clippings fed a "1940s newspaper page" prompt (Fable) | ✅ fixed | `corpus_routing`: sanborn skipped (132 pages of fire-insurance maps, no running text); halloween clippings get a clipping prompt; all prompts era-neutralized ("historical newspaper", corpus is 1870s–1960s+). |
| 13 | Tier-2 escalation chose Gemini — the *weakest* reference — on price (Fable) | ✅ fixed | `multipass-ideas.md` updated: escalate to **Qwen3-VL-32B tiled locally** (recall 0.990, digitR 0.948, rareR 0.940, phrase 0.926 — best on every recall axis, free) on guard-flagged pages. |
| 14 | Batched-throughput 6.3× measured for GLM only, applied to all models (both) | 🟨 acknowledged | Stays an estimate; report keeps the caveat and the plan re-measures before any non-GLM long run. Only GLM's own number gates the plan. |
| 15 | Tiling/penalties/prompts tuned on the same 10 pages reported (Fable) | ✅ resolved | The pilot (#2) was out-of-sample for every tuned choice — recall held (0.924). Two new tunings came FROM the pilot (max_tokens 6144, `max_repeat_run` guard) and were validated on it; treat any future re-tuning the same way. |
| 16 | README still describes the 482-PDF setup; runbook inconsistent with full corpus (Codex) | ✅ fixed | README updated: full corpus, `--pdf-root`, disk policy, guard. |
| 17 | Conclusion overstated; should be "best production default from a narrow benchmark", not settled winner (Codex) | ✅ fixed | Report TL;DR rewritten in exactly that frame, gated on the pilot. |
| 18 | Snippet readability / layout quality under-measured (Codex) | 🟨 open | Phrase adjacency is a proxy; true snippet readability still needs the side-by-side web view (`/benchmark`) + human eyes. Cheap to judge during pilot review. |
| 19 | Cloud cost extrapolations scale dense Gazette pages to the whole corpus (Fable) | 🟨 acknowledged | Treated as upper bounds in the docs; clippings/maps/sparse pages are cheaper. Decision (local-first) unaffected at any plausible discount. |

✅ fixed in code/docs · 🟡 gated on pilot/human step · 🟨 acknowledged residual risk

## What changed in the leaderboard after the fixes

Dehyphenation + survivorship + failed-pages-score-0 (10-page 1941–49 sample,
vs majority-consensus truth):

| model | cov | F1 | ±sd | recall | digitR | rareR | phrase | vs top (W–L) |
|---|---|---|---|---|---|---|---|---|
| qwen25vl-72b | 10/10 | **0.927** | 0.045 | 0.882 | 0.827 | 0.857 | 0.874 | (top) |
| glm-ocr (tiled) | 10/10 | 0.853 | 0.037 | 0.935 | **0.876** | 0.871 | 0.830 | 2–8 |
| qwen3vl-32b (full) | 10/10 | 0.835 | **0.197** | 0.839 | 0.691 | 0.744 | 0.745 | 3–7 |
| hunyuan-ocr (tiled) | 10/10 | 0.815 | 0.045 | 0.904 | 0.864 | 0.808 | 0.774 | 0–10 |
| qwen3vl-32b-tiled | 10/10 | 0.804 | 0.022 | **0.990** | **0.948** | **0.940** | **0.926** | 0–10 |

Reading: 72B is the local accuracy king (and beats GLM on 8/10 pages — the old
report's "specialist beat every larger model" was an artifact of one Qwen-32B
loop page). GLM keeps the production crown on the axes search needs:
recall 0.935, best digit recall, 4× the speed of anything above it. And
Qwen3-32B-tiled is the measured best escalation target, not Gemini.

## Pilot gate (before `ocr_run.py` on the full corpus)

1. ✅ stratified pilot sample defined (27 pages, all decades/papers/page-types)
2. ✅ GLM tiled + GPT-5.5 reference on all 27 pages → **recall 0.924 = baseline
   0.923; gate passed** (`scripts/pilot_report.py`; ~$1.5 of API)
3. ⬜ human eyeball of `/benchmark` side-by-sides on the pilot pages (they're
   in `results.json` now — open http://localhost:5000/benchmark)
4. ✅ pre-1920 recall did NOT crater (0.88–0.95); two fixes shipped instead:
   tile `max_tokens` 4096→6144 (truncation), `max_repeat_run` guard
   (tile-local loops — found one hiding in a 0.98-dict page)
5. ⬜ optional: one human-transcribed faint page (closes #3 fully)

**Verdict: the full-corpus run is unblocked** (pending the human eyeball, which
can happen during the run since OCR JSONs are per-page and idempotent).

## Reclaimable disk (user decision — not touched)

- `dl-c6e6b6c1de78.zip` — **95.1 GB**, already extracted to `dl-c6e6b6c1de78/`
- `dl-88dc7c6d5e1b.zip` — **10.5 GB**, the original 482-PDF sample archive
- C: free space is ~86 GB; deleting both zips roughly doubles headroom and is
  required before any plan that writes >80 GB (e.g. `--keep-png`).
