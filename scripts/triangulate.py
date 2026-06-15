"""Triangulated benchmark: score every local model against THREE SOTA references
(Claude / GPT / Gemini) plus a majority-consensus ground truth.

The consensus = words that >=2 of the 3 flagships agree on, on the same page.
This filters single-model hallucinations -> a sturdier truth than any one model.
Also reports pairwise reference agreement (validates the truth itself).

Caveats the numbers inherit (see docs/review-resolutions.md):
  * The consensus contains only flagship-legible text. Words ALL flagships
    missed are absent from the denominator -> recall is vs what flagships could
    read, not vs the page. True page recall is strictly lower.
  * Pages a model FAILED on score 0 and stay in its mean (no survivorship);
    the `cov` column shows pages-scored/pages-total.

Beyond mean F1 this reports per-page sd / min, paired per-page wins vs the
leader (10 pages -> means are noisy; the paired column is the honest view),
and search-oriented metrics: digit recall, rare-word recall, and proper-noun
phrase recovery (FTS5 phrase queries need ADJACENT tokens; bag-of-words
scoring is blind to tiling scrambling word order).

  python scripts/triangulate.py
"""
from __future__ import annotations
import json
import statistics as st
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, write_json
from metrics import (words, digit_tokens, proper_phrases, dehyphenate,
                     TOKEN_RE, zipf_frequency)

REFS = ["reference-claude", "reference-gpt", "reference-gemini"]
RARE_ZIPF = 3.5


def lwords(t):
    return [w.lower() for w in words(t)]


def prf(hyp_counter, ref_counter):
    if not ref_counter:
        return (0.0, 0.0, 0.0)
    ov = sum((hyp_counter & ref_counter).values())
    p = ov / max(1, sum(hyp_counter.values()))
    r = ov / sum(ref_counter.values())
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return (f, r, p)


def load_model(raw, model):
    out = {}
    d = raw / model
    if not d.exists():
        return out
    for jf in d.rglob("p*.json"):
        r = json.loads(jf.read_text(encoding="utf-8"))
        if r.get("ok"):
            out[(r["issue_id"], str(r["page"]))] = r["text"]
    return out


def majority_multiset(counters: list[Counter]) -> Counter:
    """Words/tokens >=2 of the refs contain, at the 2nd-highest count
    (majority floor)."""
    merged = Counter()
    for w in set().union(*[set(c) for c in counters]):
        present = [c[w] for c in counters if c[w] > 0]
        if len(present) >= 2:
            merged[w] = sorted(present)[len(present) - 2]
    return merged


def main():
    cfg = load_config()
    raw = Path(cfg["paths"]["bench"]) / "raw"
    texts = {m.name: load_model(raw, m.name) for m in sorted(raw.glob("*"))}
    refs = {r: texts.get(r, {}) for r in REFS if texts.get(r)}
    if len(refs) < 2:
        print("Need >=2 reference models with results. Have:", list(refs))
        return
    pages = sorted(set.intersection(*[set(t) for t in refs.values()]))
    rl = list(refs)
    print(f"References present: {rl} | common pages: {len(pages)}\n")

    # --- pairwise reference agreement (validates the ground truth) ---
    print("=== Reference agreement (F1) — how much the flagships concur ===")
    for i in range(len(rl)):
        for j in range(i + 1, len(rl)):
            fs = [prf(Counter(lwords(refs[rl[i]][pg])),
                      Counter(lwords(refs[rl[j]][pg])))[0] for pg in pages]
            print(f"  {rl[i].split('-')[1]:<7} vs {rl[j].split('-')[1]:<7} "
                  f"F1={st.mean(fs):.3f}")

    # --- consensus multisets per page: words, digit tokens, rare words ---
    consensus, cons_digits, cons_rare = {}, {}, {}
    for pg in pages:
        wcnts = [Counter(lwords(refs[r][pg])) for r in rl]
        consensus[pg] = majority_multiset(wcnts)
        dcnts = [Counter(digit_tokens(refs[r][pg])) for r in rl]
        cons_digits[pg] = majority_multiset(dcnts)
        cons_rare[pg] = Counter({w: c for w, c in consensus[pg].items()
                                 if zipf_frequency(w, "en") < RARE_ZIPF})
    # phrase truth: proper-noun bigrams from Claude (consensus has no order)
    phrase_truth = {pg: proper_phrases(refs["reference-claude"][pg])
                    for pg in pages}

    # --- score every non-reference model; missing/failed pages count as 0 ---
    models = [m for m in texts if m not in REFS and texts[m]]
    rows = []
    for mdl in models:
        per_page_f1, per_ref = {}, {}
        rec_l, digit_l, rare_l, phr_hit, phr_tot = [], [], [], 0, 0
        for r in rl:
            per_ref[r] = st.mean(
                [prf(Counter(lwords(texts[mdl].get(pg, ""))),
                     Counter(lwords(refs[r][pg])))[0] for pg in pages])
        for pg in pages:
            txt = texts[mdl].get(pg, "")          # absent -> "" -> scores 0
            f, rcl, _ = prf(Counter(lwords(txt)), consensus[pg])
            per_page_f1[pg] = f
            rec_l.append(rcl)
            if cons_digits[pg]:
                _, dr, _ = prf(Counter(digit_tokens(txt)), cons_digits[pg])
                digit_l.append(dr)
            if cons_rare[pg]:
                _, rr, _ = prf(Counter(lwords(txt)), cons_rare[pg])
                rare_l.append(rr)
            if phrase_truth[pg]:
                toks = [t.lower() for t in TOKEN_RE.findall(dehyphenate(txt))]
                bigrams = set(zip(toks, toks[1:]))
                phr_hit += sum(1 for p in phrase_truth[pg] if p in bigrams)
                phr_tot += len(phrase_truth[pg])
        fs = list(per_page_f1.values())
        rows.append({
            "model": mdl,
            "coverage": f"{len(texts[mdl].keys() & set(pages))}/{len(pages)}",
            "f1_cons": st.mean(fs),
            "f1_sd": st.stdev(fs) if len(fs) > 1 else 0.0,
            "f1_min": min(fs),
            "recall_cons": st.mean(rec_l),
            "digit_recall": st.mean(digit_l) if digit_l else None,
            "rare_recall": st.mean(rare_l) if rare_l else None,
            "phrase_recovery": phr_hit / phr_tot if phr_tot else None,
            "per_ref": per_ref,
            "per_page_f1": {f"{k[0]}|{k[1]}": v for k, v in per_page_f1.items()},
        })
    rows.sort(key=lambda r: r["f1_cons"], reverse=True)

    # paired per-page wins vs the leader — the honest small-sample comparison
    if rows:
        lead = rows[0]
        for r in rows:
            d = [lead["per_page_f1"][pg] - r["per_page_f1"][pg]
                 for pg in lead["per_page_f1"]]
            r["leader_wins"] = sum(x > 0 for x in d)
            r["leader_mean_diff"] = st.mean(d)

    print("\n=== Models vs majority-consensus truth (failed pages score 0) ===")
    hdr = (f"{'model':<20}{'cov':>6}{'F1':>7}{'±sd':>6}{'min':>6}{'recall':>7}"
           f"{'digitR':>7}{'rareR':>7}{'phrase':>7}{'vs-top':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        fmt = lambda v: f"{v:7.3f}" if v is not None else "      —"
        vs = ("  (top)" if r is rows[0]
              else f" {len(rows[0]['per_page_f1']) - r['leader_wins']}–{r['leader_wins']}")
        print(f"{r['model']:<20}{r['coverage']:>6}{r['f1_cons']:>7.3f}"
              f"{r['f1_sd']:>6.3f}{r['f1_min']:>6.3f}{r['recall_cons']:>7.3f}"
              f"{fmt(r['digit_recall'])}{fmt(r['rare_recall'])}"
              f"{fmt(r['phrase_recovery'])}{vs:>8}")
    print("\ncov=pages scored/total (missing pages count as F1 0)  "
          f"rareR=recall on words w/ zipf<{RARE_ZIPF} (search targets)\n"
          "phrase=proper-noun bigrams from the Claude ref surviving ADJACENT "
          "(FTS5 phrase queries)\nvs-top=per-page wins–losses against the "
          "leader — on 10 pages this matters more than the mean")

    out = Path(cfg["paths"]["bench"]) / "triangulate.json"
    write_json(out, {"pages": [f"{p[0]}|{p[1]}" for p in pages], "models": rows})
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
