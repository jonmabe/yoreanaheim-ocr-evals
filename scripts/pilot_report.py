"""Score the stratified PILOT (out-of-distribution pages) and compare against
the original 1941-49 benchmark sample scored the same way.

Single yardstick: reference-gpt (Claude<->GPT agree 0.966 on the bench sample,
so GPT alone is a solid ruler; running all three flagships on the pilot would
triple cost for little extra signal). Every metric is GLM (or --model) vs GPT.

The GATE: does recall hold up per decade/paper vs the 1941-49 baseline?

  python scripts/pilot_report.py [--model glm-ocr]
"""
from __future__ import annotations
import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from metrics import (words, digit_tokens, proper_phrases, dehyphenate,
                     TOKEN_RE, zipf_frequency)

BENCH_RE = re.compile(r"_194[13579]_")   # original 1941-49 sample issues


def load_model(raw, model):
    out = {}
    d = raw / model
    if d.exists():
        for jf in d.rglob("p*.json"):
            r = json.loads(jf.read_text(encoding="utf-8"))
            if r.get("ok"):
                out[(r["issue_id"], str(r["page"]))] = r
    return out


def score(hyp: str, ref: str) -> dict:
    H, R = Counter(w.lower() for w in words(hyp)), \
           Counter(w.lower() for w in words(ref))
    ov = sum((H & R).values())
    p = ov / max(1, sum(H.values()))
    rc = ov / max(1, sum(R.values()))
    out = {"f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "recall": rc,
           "precision": p}
    HD, RD = Counter(digit_tokens(hyp)), Counter(digit_tokens(ref))
    out["digitR"] = (sum((HD & RD).values()) / sum(RD.values())) if RD else None
    RR = Counter({w: c for w, c in R.items() if zipf_frequency(w, "en") < 3.5})
    out["rareR"] = (sum((H & RR).values()) / sum(RR.values())) if RR else None
    ph = proper_phrases(ref)
    toks = [t.lower() for t in TOKEN_RE.findall(dehyphenate(hyp))]
    big = set(zip(toks, toks[1:]))
    out["phrase"] = (sum(1 for x in ph if x in big) / len(ph)) if ph else None
    return out


def stratum(issue_id: str) -> str:
    m = re.search(r"_(18|19|20)(\d)\d", issue_id)
    decade = f"{m.group(1)}{m.group(2)}0s" if m else "????"
    paper = issue_id.split("_")[0]
    if issue_id.startswith("anaheim_gazette"):
        paper = "gazette"
    elif issue_id.startswith("oc_plain"):
        paper = "plain-dealer"
    elif issue_id.startswith("anaheim_bulletin"):
        paper = "bulletin(int.)"
    elif issue_id.startswith("anaheim_daily"):
        paper = "daily-herald"
    elif issue_id.startswith("halloween"):
        paper = "LAT-clipping"
    return f"{decade} {paper}"


def main(model: str):
    cfg = load_config()
    raw = Path(cfg["paths"]["bench"]) / "raw"
    ref = load_model(raw, "reference-gpt")
    hyp = load_model(raw, model)
    common = sorted(set(ref) & set(hyp))
    if not common:
        print("No common pages between reference-gpt and", model)
        return

    rows = []
    for pg in common:
        s = score(hyp[pg]["text"], ref[pg]["text"])
        s["pg"] = pg
        s["bench"] = bool(BENCH_RE.search(pg[0]))
        s["dict"] = hyp[pg].get("dict_hit_rate", 0)
        s["finish"] = hyp[pg].get("finish", "")
        rows.append(s)

    fmt = lambda v: f"{v:6.3f}" if v is not None else "     —"
    for label, sel in [("PILOT (out-of-distribution)",
                        [r for r in rows if not r["bench"]]),
                       ("BASELINE (1941-49 bench sample, same scoring)",
                        [r for r in rows if r["bench"]])]:
        if not sel:
            continue
        print(f"\n=== {label}: {model} vs reference-gpt ===")
        hdr = (f"{'page':<42}{'F1':>6}{'recall':>7}{'digitR':>7}"
               f"{'rareR':>7}{'phrase':>7}{'dict':>6}")
        print(hdr); print("-" * len(hdr))
        for r in sorted(sel, key=lambda r: r["pg"]):
            tag = f"{stratum(r['pg'][0])} p{r['pg'][1]}"
            warn = " ⚠" if r["recall"] < 0.8 else ""
            print(f"{tag:<42}{r['f1']:>6.3f}{r['recall']:>7.3f}"
                  f"{fmt(r['digitR'])}{fmt(r['rareR'])}{fmt(r['phrase'])}"
                  f"{r['dict']:>6.2f}{warn}")
        print(f"{'MEAN':<42}{st.mean([r['f1'] for r in sel]):>6.3f}"
              f"{st.mean([r['recall'] for r in sel]):>7.3f}"
              f"{fmt(st.mean([r['digitR'] for r in sel if r['digitR'] is not None]))}"
              f"{fmt(st.mean([r['rareR'] for r in sel if r['rareR'] is not None]))}"
              f"{fmt(st.mean([r['phrase'] for r in sel if r['phrase'] is not None]))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-ocr")
    main(**vars(ap.parse_args()))
