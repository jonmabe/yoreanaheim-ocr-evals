"""Search-oriented spot check: which REAL queries would a model's OCR miss?

For each model, takes the proper-noun bigrams and digit tokens from the Claude
reference (the closest thing to realistic historical-search queries: names,
places, dates, addresses) and lists the ones the model's text would NOT match —
split into 'words present but not adjacent' (phrase query fails, bare AND query
still hits) vs 'words missing entirely' (no query hits).

  python scripts/search_eval.py glm-ocr [more models...]
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from metrics import proper_phrases, digit_tokens, dehyphenate, TOKEN_RE


def load_model(raw, model):
    out = {}
    d = raw / model
    if d.exists():
        for jf in d.rglob("p*.json"):
            r = json.loads(jf.read_text(encoding="utf-8"))
            if r.get("ok"):
                out[(r["issue_id"], str(r["page"]))] = r["text"]
    return out


def main(models):
    cfg = load_config()
    raw = Path(cfg["paths"]["bench"]) / "raw"
    ref = load_model(raw, "reference-claude")
    for mdl in models:
        tx = load_model(raw, mdl)
        pages = sorted(set(ref) & set(tx))
        broken, missing, dig_missing = [], [], []
        n_ph = n_dig = 0
        for pg in pages:
            toks = [t.lower() for t in TOKEN_RE.findall(dehyphenate(tx[pg]))]
            tokset, bigrams = set(toks), set(zip(toks, toks[1:]))
            for p in sorted(proper_phrases(ref[pg])):
                n_ph += 1
                if p in bigrams:
                    continue
                (broken if p[0] in tokset and p[1] in tokset
                 else missing).append((" ".join(p), pg[0][-25:]))
            hyp_d = Counter(digit_tokens(tx[pg]))
            for d, c in Counter(digit_tokens(ref[pg])).items():
                n_dig += c
                if hyp_d[d] < c:
                    dig_missing.append((d, pg[0][-25:]))
        print(f"\n=== {mdl} — {len(pages)} pages, {n_ph} phrase queries, "
              f"{n_dig} digit tokens ===")
        print(f"phrase OK {n_ph - len(broken) - len(missing)}/{n_ph}   "
              f"broken-adjacency {len(broken)}   words-missing {len(missing)}   "
              f"digit tokens short {len(dig_missing)}")
        for label, items in [("ADJACENCY BROKEN (phrase fails, AND still hits)",
                              broken),
                             ("WORDS MISSING (nothing hits)", missing),
                             ("DIGIT TOKENS MISSED (dates/addresses/prices)",
                              dig_missing)]:
            print(f"  -- {label}: {len(items)}")
            for q, pg in items[:12]:
                print(f"     '{q}'  [{pg}]")
            if len(items) > 12:
                print(f"     ... +{len(items) - 12} more")


if __name__ == "__main__":
    main(sys.argv[1:] or ["glm-ocr"])
