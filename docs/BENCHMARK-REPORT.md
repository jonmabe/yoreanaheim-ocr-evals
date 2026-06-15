# YoreAnaheim OCR benchmark — final report (rev. 2, post-adversarial-review)

_Goal: choose the OCR engine to make the full scanned-newspaper corpus
full-text searchable. Metric that matters: **word recall** (find all the text) at
viable **throughput**, locally on 2× RTX 5090._

> **Rev. 2 (2026-06-10):** two adversarial reviews (Fable 5, Codex) audited the
> methodology; every finding and its resolution is in
> `docs/review-resolutions.md`. Headline corrections: the leaderboard now uses
> dehyphenated tokens, failed pages score 0, per-page dispersion + paired wins
> are reported, and digit/rare-word/phrase metrics were added. The conclusion
> is **narrower**: GLM-OCR is the best *production default from this
> (still-narrow) benchmark*, gated on an out-of-distribution pilot — not a
> settled corpus-wide winner.

---

## TL;DR / recommendation

**Use GLM-OCR (`zai-org/GLM-OCR`, 0.9B) with 3×3 tiling** for the full-corpus
pass. **The stratified pilot has now been run (27 pages, 1872–1964, all four
papers, interior pages, clippings) and GLM passed**: out-of-distribution
recall 0.924 vs the 1941–49 baseline's 0.923, dict rate 0.95–1.00 on every
page, no decade or paper craters. Details in "Pilot results" below.

Why GLM, stated precisely:

- **It is NOT the accuracy winner.** Qwen2.5-VL-72B is (F1 0.927, beats GLM on
  8/10 pages) — and full-page Qwen3-VL-32B beats GLM on 7/10 pages *when it
  doesn't loop* (it catastrophically looped on 1 of 10 pages, sd 0.197). Big
  models give the cleanest text but are 3–4× slower and have a fat failure
  tail that needs guarding either way.
- **It IS the search winner.** Best recall-per-second by a wide margin:
  recall 0.935 vs consensus truth, **best digit recall of any model tested
  (0.876** — dates/addresses/prices), rare-word recall 0.871, and 9.9
  pages/min/GPU batched. Search wants recall and coverage, not prose.
- Tiling's cost is measured and acceptable: proper-noun **phrase adjacency
  0.830 vs 0.874** for the best full-page model (~4 pts) — phrase queries
  mostly survive. GLM's name misses are *recall* losses (word absent), which
  the tier-2 escalation pass targets.

---

## The corpus (cost driver)

| Publication | PDFs | Pages | Size | pg/pdf |
|---|---:|---:|---:|---:|
| anaheim-gazette | 5,428 | 38,771 | 74.3 GB | 7.1 |
| oc-plain-dealer | 1,382 | 10,488 | 16.2 GB | 7.6 |
| anaheim-bulletin | 163 | 2,657 | 3.3 GB | 16.3 |
| anaheim-daily-herald | 101 | 788 | 1.3 GB | 7.8 |
| halloween (clippings) | 184 | 193 | 0.05 GB | 1.0 |
| sanborn (maps) | 132 | 132 | 0.02 GB | 1.0 |
| **Total** | **7,390** | **53,029** | **95.1 GB** | — |

Spans the **1870s–2010s** (heaviest in the 1920s: 15.3k pages, and 1950s: 12.1k).
Image scans with **no text layer** → OCR is mandatory. Scan:
`scripts/scan_corpus.py`; data in `data/bench/corpus_scan.json`.
Sanborn maps are **skipped** (`corpus_routing` — no running text); clippings
get their own prompt.

---

## Ground truth and its limits

Three flagships transcribed the 10-page 1941–49 sample; models are scored
against a **majority-consensus** word multiset (≥2 of 3 agree), which filters
single-model hallucinations (`scripts/triangulate.py`).

- Claude Opus 4.8 ↔ GPT-5.5: **F1 0.966** (after dehyphenation) — tight core.
- Claude ↔ Gemini-3.1-Pro 0.860 · GPT ↔ Gemini 0.854 — Gemini is the noisy
  third vote. **Never escalate hard pages to Gemini** (earlier plan did, on
  price); the measured best escalation target is Qwen3-VL-32B tiled.

Two caveats the numbers inherit:

1. **The consensus contains only flagship-legible text.** Words all (or 2 of 3)
   flagships missed aren't in the denominator, so "recall 0.93" = recall of
   what flagships could read; true page recall is strictly lower. A
   fourth-model spot check on the weakest-agreement page
   (`data/bench/_faint_validate.py`) found no flagship blind spots beyond the
   hyphenation artifact below — but a human-transcribed faint page is still
   the missing calibration (~1 hour, recommended during the pilot).
2. **Line-break hyphenation was silently corrupting the truth AND the index**
   (found by that spot check). Claude+Gemini transcribe "Laugh-↵lin" literally
   — a 2-of-3 majority — so the truth held `laugh`+`lin` and *penalized*
   GLM/GPT for correctly writing "Laughlin"; FTS5 would likewise miss a
   "Laughlin" query. Fixed: `dehyphenate()` is applied in every metric
   tokenizer and at index time. Ranking unchanged; agreement improved.

---

## Results (10-page 1941–49 sample, vs consensus truth)

Failed/missing pages score 0 (`cov` shows coverage). `digitR` = recall of
digit tokens (dates, addresses, prices — 2.8% of tokens, previously unmeasured);
`rareR` = recall of rare words (zipf<3.5 — the proper nouns people search for);
`phrase` = proper-noun bigrams surviving **adjacent** (FTS5 phrase queries);
`vs top` = per-page wins–losses against the leader (on n=10, this beats means).

| Model | Params | cov | F1 | ±sd | recall | digitR | rareR | phrase | s/pg | vs top |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-72B (full, TP=2) | 72B | 10/10 | **0.927** | 0.045 | 0.882 | 0.827 | 0.857 | 0.874 | 165 | (top) |
| **GLM-OCR (tiled)** | 0.9B | 10/10 | 0.853 | 0.037 | 0.935 | **0.876** | 0.871 | 0.830 | **38** | 2–8 |
| Qwen3-VL-32B (full) | 32B | 10/10 | 0.835 | 0.197⚠ | 0.839 | 0.691 | 0.744 | 0.745 | 111 | 3–7 |
| HunyuanOCR (tiled) | 1B | 10/10 | 0.815 | 0.045 | 0.904 | 0.864 | 0.808 | 0.774 | 39 | 0–10 |
| Qwen3-VL-32B (tiled) | 32B | 10/10 | 0.804 | 0.022 | **0.990** | **0.948** | **0.940** | **0.926** | 166 | 0–10 |
| PaddleOCR-VL base (tiled) | 0.9B | 10/10 | 0.777 | 0.098 | 0.825 | 0.731 | 0.715 | 0.692 | 37 | 0–10 |
| olmOCR-2 (full) | 7B | 10/10 | 0.723 | 0.242⚠ | 0.702 | 0.616 | 0.562 | 0.598 | 61 | 0–10 |
| dots.ocr | 3B | 9/10 | 0.699 | 0.298⚠ | 0.624 | 0.478 | 0.555 | 0.513 | 35 | 2–8 |
| PaddleOCR-VL-1.5 / 1.6 | 0.9B | 10/10 | 0.61 | — | 0.56/0.77 | — | — | — | 33/43 | — |
| GLM-4.6V-Flash | 10B | 10/10 | 0.416 | 0.025 | 0.982 | 0.969 | 0.934 | 0.883 | 352 | 0–10 |

⚠ sd ≥ 0.19 = fat failure tail (full-page repetition loops on ≥1 page).

### What this corrects from rev. 1

Rev. 1's #2 finding ("specialized 0.9B beat every larger general VLM") was an
**artifact of one Qwen-32B loop page** dragging its mean below GLM's. Page-by-
page, the big full-page models beat GLM on most pages; their means lose to
their own tail risk. The durable findings are:

1. **Dense broadsheets break single-pass OCR for small models** (recall
   0.07–0.19 untiled) — 3×3 tiling fixes it (GLM 0.07→0.93). Unchanged.
2. **Full-page decoding of dense pages carries loop risk at every size**
   (32B: 1/10 pages; olmOCR: high variance). Any production run needs the
   degeneration guard, regardless of model.
3. **Small specialists win recall-per-second, not accuracy.** GLM's edge is
   the speed–recall product plus the lowest variance of any fast model
   (sd 0.037, min 0.783 — no catastrophic pages on this sample).
4. **Concrete search impact** (`scripts/search_eval.py`): of 3,421 realistic
   name-phrase queries from the reference, GLM misses 392 outright (word never
   transcribed) and breaks adjacency on 180 (phrase query fails, AND-query
   still hits). The misses cluster in dense rosters/society columns — the
   recall-union/escalation tier exists for exactly these.

---

## Pilot results (2026-06-10) — GATE PASSED

27 stratified pages OCR'd by GLM tiled and by GPT-5.5 as the yardstick
(Claude↔GPT agree 0.966, so one flagship suffices; ~$1.5 of API). Scored with
`scripts/pilot_report.py`; the 1941–49 sample re-scored identically as baseline:

| | F1 | recall | digitR | rareR | phrase |
|---|---:|---:|---:|---:|---:|
| **Pilot (1872–1964, 4 papers, interior, clippings)** | 0.803 | **0.924** | 0.870 | 0.783 | 0.789 |
| Baseline (1941–49 sample, same scoring) | 0.844 | 0.923 | 0.860 | 0.805 | 0.787 |

- **Recall holds exactly** out-of-distribution — the gate's core question.
  Worst page recall 0.871; 1872 typography reads clean (dict 0.98).
  Clippings: recall 0.92–0.98. Interior Bulletin pages: 0.94–0.97.
- **The F1 dip on 1880s–1890s pages is a REFERENCE artifact, not a GLM
  failure**: dense pre-1920 broadsheets run 6–10k words, and GPT at
  `max_side 2048` (≈85 dpi effective) transcribes only 45–70% as many words
  as tiled GLM (1898 p1: GPT 3,006 vs GLM 10,045 words, with 11k of GPT's 16k
  token budget unused). GLM's surplus was verified mostly real text
  (dict 0.899, genuine 1890s names: Randsburg, Whittaker, Oleson…). This
  empirically confirms the "flagship recall ceiling" caveat — on the densest
  pages, **tiled GLM out-reads the flagship reference**; "precision vs GPT"
  is not meaningful there.
- **Two production fixes came out of the pilot:**
  1. Dense tiles overflowed `max_tokens 4096` (3 pages at `length:1/9`) →
     raised to 6144, re-ran, +200–600 words/page recovered.
  2. One **tile-local repetition loop** found hiding inside a clean page
     (45× "shouldn't"; page dict rate still 0.98 — the dict-rate guard alone
     misses it). Added `max_repeat_run` to metrics + guard: the looped page
     scores 45, all 36 clean pages ≤3, threshold 10. Observed incidence
     ~1/37 pages (~3%) → expect roughly ~1.5k flagged pages corpus-wide;
     guard auto-retries, survivors form the tier-2 escalation queue.
- Pre-1920 pages are denser → 45–68 s/pg vs the 38 s average; expect the
  full-corpus wall-clock estimate to run ~10–20% over.

Remaining (optional) gate items: human eyeball of `/benchmark` side-by-sides
on the pilot pages; one human-transcribed faint page.

## Execution plan for the full corpus (53,029 pages)

| Model | sec/pg | Single-stream | Batched, 1×5090 | Batched, 2×5090 |
|---|---:|---:|---:|---:|
| **GLM-OCR (tiled)** | 38 | ~24 days | **~3.7 days** | **~1.9 days** |
| HunyuanOCR (tiled) | 39 | ~24 days | ~3.8 days | ~1.9 days |
| Qwen3-VL-32B (tiled) | 166 | ~102 days | ~16 days | ~8 days |
| Qwen2.5-VL-72B | 165 | ~101 days | ~16 days | ~8 days |

> GLM's batched figure (9.9 pages/min/GPU, 6.3× single-stream) is **measured**;
> other rows reuse GLM's 6.3× as an estimate — re-measure before any non-GLM
> long run. The 2-GPU column requires pausing the production LLM on GPU 0 for
> ~2 days; the 1-GPU plan (~3.7 days) avoids that. Decide explicitly.

Operational safeguards now built into `ocr_run.py` (rev. 2):

- **`--pdf-root dl-c6e6b6c1de78`** — no config mutation.
- **PNG purge by default**: 300-dpi pages average **14.7 MB → ~780 GB for the
  corpus, vs ~86 GB free on C:**. Without purging, the run dies at ~6k pages.
  (`--keep-png` to opt out. The webapp re-renders pages on view.)
- **Degeneration guard** (`ocr_guard`): pages with dict-rate < 0.5 or truncated
  tiles (`finish_reason=length`) retry once with stronger penalties and carry a
  `flagged` reason in their JSON — that set is the tier-2 escalation queue.
- **Routing** (`corpus_routing`): sanborn skipped; clippings prompt; prompts
  era-neutralized (corpus is 1870s–1960s+, not "1940s").
- Reclaimable: the two `dl-*.zip` archives (~105 GB) duplicate extracted data.

### How to run (after the pilot passes)

```bash
# WSL: serve GLM-OCR on GPU 1
cd ~/vllm && MODEL=zai-org/GLM-OCR PORT=4003 GPU=1 GPUUTIL=0.5 PRESET=auto bash scripts/serve-any.sh
```
```powershell
python scripts/ocr_run.py --model glm-ocr --pdf-root dl-c6e6b6c1de78 --workers 8
python scripts/index.py  --model glm-ocr --rebuild
python webapp/app.py       # search at http://localhost:5000
```

---

## Appendix A: cost reference

Cloud SOTA full-corpus (measured tokens × list price, **upper bounds** — dense
Gazette pages extrapolated): Gemini ~$3.9k / GPT ~$9.6k / Claude ~$13k.
Per-page measured: Claude 4920/8865 tok @119s · GPT 2830/5568 @183s · Gemini
1168/5930 @135s. Local GLM ≈ $0 + electricity. Tiered escalation design (with
Qwen3-32B-tiled, not Gemini, as tier 2): `docs/multipass-ideas.md`.

## Appendix B: what couldn't run

- **Qwen3-Omni-30B / Nemotron-Omni-30B (FP8 MoE):** flashinfer can't JIT the
  FP8 group-GEMM for Blackwell `sm_120` on vLLM 0.22.1. Revisit on newer stack.
- **Surya 2:** now a vLLM/Docker-backed VLM; needs Docker WSL integration.
- **Kraken:** weak base models on this typography; the real play is
  fine-tuning on the flagship transcriptions (`docs/multipass-ideas.md` #7).
- **MinerU2.5:** toolchain not yet wired (`mineru` entry disabled).
