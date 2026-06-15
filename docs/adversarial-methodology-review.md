# Adversarial Methodology Review

## Summary Take

The current benchmark supports GLM-OCR as the best practical production candidate under the tested conditions, but it does not yet support treating GLM-OCR as the settled winner for the whole corpus.

The main issue is not that the results are useless. The issue is that the benchmark is narrow: ten Anaheim Gazette pages from 1941-1949, scored mostly by alphabetic bag-of-words overlap against VLM-generated reference transcripts. That is enough to identify promising candidates and reject obvious poor fits. It is not enough to lock the final leaderboard across a 53,029-page corpus spanning multiple publications, decades, page layouts, scan qualities, clippings, and maps.

## Leaderboard Risk

Methodology changes could affect the overall leaderboard.

The most likely shifts are in the production/search ranking, especially among:

- GLM-OCR
- HunyuanOCR
- Qwen3-VL-32B full-page
- Qwen3-VL-32B tiled
- PaddleOCR-VL

Qwen2.5-VL-72B appears to be the strongest accuracy reference in the cached results, but its speed makes it a poor full-corpus default. GLM-OCR is still a strong practical candidate because it combines high recall with viable throughput. The distinction matters: GLM is the apparent production winner, not the measured absolute-accuracy winner.

## Main Methodology Weaknesses

1. The sample is too narrow.

   The benchmark uses ten pages from five Gazette issues, taking only the first two pages from each issue. That leaves major corpus categories untested: other publications, other decades, interior pages, classifieds, ads, clippings, Sanborn maps, older typography, and degraded scans.

2. The reference is not true ground truth.

   The Claude/GPT/Gemini consensus is useful, but it is still model-generated. Agreement between strong VLMs reduces random hallucination risk, but it does not eliminate shared blind spots or systematic omissions.

3. Bag-of-words F1 matches the search goal only partially.

   Alphabetic word overlap is reasonable for broad full-text search, but it underweights dates, addresses, prices, numbers, initials, abbreviations, punctuation, and exact phrase recoverability. Those are important in historical newspaper search.

4. Layout and reading order are under-measured.

   For pure hit discovery, bag-of-words recall may be enough. For page display, snippets, article reading, and user trust, layout quality matters. Tiled methods can score well while producing duplicated, fragmented, or scrambled text.

5. Runtime projections are strongest only for GLM.

   GLM batching was measured directly. Applying that same batching speedup to other models is a planning estimate, not a result. Other models may saturate differently.

6. The runbook and corpus target are inconsistent.

   The README still describes the smaller `data/pdfs` setup, while the report discusses the larger 7,390-PDF corpus. The full-corpus run depends on changing `pdf_root`, which should be explicit and hard to miss.

## Practical Recommendation

Keep GLM-OCR as the default next production candidate, but describe the conclusion more narrowly:

> GLM-OCR is the best production default from the current narrow benchmark because it has strong recall and viable throughput. The final corpus-wide leaderboard is not settled until the benchmark is expanded.

Before committing to a full run, run a stratified validation set across:

- Publication
- Decade
- Page number or page type
- Scan quality
- Text density
- Numeric-heavy pages
- Maps and clippings

Add search-oriented checks in addition to bag-of-words F1:

- Known-name recall
- Date/address/price recall
- Exact phrase recovery
- Query hit rate on realistic searches
- Duplicate/noise rate from tiling
- Snippet readability

If GLM holds up on that expanded sample, the production plan becomes much stronger. If it slips on older pages, maps, or numeric-heavy material, the likely adjustment is not necessarily replacing GLM everywhere. A tiered plan may be better: GLM for all pages, then escalation or alternate OCR only for page classes where GLM underperforms.
