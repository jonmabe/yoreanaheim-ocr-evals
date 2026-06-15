"""Benchmark every reachable OCR model on the sample pages.

Resumable & incremental: each (model, page) result is cached as JSON, so you
can serve one model, run this, swap to the next model, run again — results
accumulate. Re-run with --force to recompute.

Outputs:
  data/bench/raw/<model>/<issue>/pNNN.json   per result (text, time, metrics)
  data/bench/results.json                    everything, for the web UI
  data/bench/summary.json                    per-model aggregates + ranking
Run:  python scripts/benchmark.py            (add --force to recompute)
"""
from __future__ import annotations
import argparse
import statistics as stats
from pathlib import Path

from common import load_config, parse_meta, write_json, read_json, ROOT
from render import render_page, page_png_path, page_count
from ocr_client import ocr_page, active_models
from metrics import text_metrics, consensus_scores, accuracy_vs_reference


def sample_pages(cfg, sample_key="benchmark_sample",
                 pdf_root=None) -> list[tuple[Path, object, int]]:
    """(pdf_path, meta, page_index) for the configured sample.

    Entries are either a rel-path string (first `benchmark_pages_per_pdf`
    pages) or {path: ..., pages: [...]} for explicit pages (interior pages,
    single-page clippings)."""
    pdf_root = Path(pdf_root) if pdf_root else Path(cfg["paths"]["pdf_root"])
    per = cfg.get("benchmark_pages_per_pdf", 2)
    items = []
    for entry in cfg[sample_key]:
        rel, pages = (entry, None) if isinstance(entry, str) else \
                     (entry["path"], entry.get("pages"))
        pdf = pdf_root / rel
        if not pdf.exists():
            print(f"  ! sample missing: {rel}")
            continue
        meta = parse_meta(pdf, pdf_root)
        n = page_count(pdf)
        for i in (pages if pages is not None else range(min(per, n))):
            if i < n:
                items.append((pdf, meta, i))
    return items


def run(force: bool = False, only=None, no_aggregate=False,
        sample_key="benchmark_sample", pdf_root=None):
    cfg = load_config()
    bench_root = Path(cfg["paths"]["bench"])
    raw_root = bench_root / "raw"
    models = active_models(cfg, only_reachable=True)
    if only:
        only = set(only)
        models = {n: m for n, m in models.items() if n in only}
    if not models:
        print("No reachable models. Start a vLLM server (see serve/vllm_commands.md)\n"
              "then re-run. Results accumulate across runs.")
        # still (re)build aggregates from whatever was cached before
    else:
        print(f"Reachable models: {', '.join(models)}")

    if pdf_root:
        pdf_root = str((ROOT / pdf_root).resolve())
    pages = sample_pages(cfg, sample_key=sample_key, pdf_root=pdf_root)
    print(f"Sample [{sample_key}]: {len(pages)} pages "
          f"from {len(cfg[sample_key])} issues\n")

    # 1) OCR each page with each reachable model (cached)
    for pdf, meta, pi in pages:
        png = render_page(pdf, pi, page_png_path(cfg["paths"]["pages"], meta, pi),
                          dpi=cfg["render"]["dpi"])
        for name, mcfg in models.items():
            out_json = raw_root / name / meta.issue_id / f"p{pi:03d}.json"
            if out_json.exists() and not force:
                continue
            res = ocr_page(name, mcfg, png)
            tm = text_metrics(res.text) if res.ok else {
                "chars": 0, "words": 0, "dict_hit_rate": 0.0,
                "fluency": 0.0, "alpha_ratio": 0.0}
            rec = {
                "model": name, "issue_id": meta.issue_id, "page": pi,
                "rel_path": meta.rel_path, "date": meta.date,
                "ok": res.ok, "error": res.error, "seconds": round(res.seconds, 3),
                "text": res.text, "usage": res.usage, "finish": res.finish,
                **tm,
            }
            write_json(out_json, rec)
            flag = "ok " if res.ok else "ERR"
            print(f"  [{flag}] {name:<14} {meta.issue_id} p{pi}  "
                  f"{rec['seconds']:>6.1f}s  {rec['words']:>4}w  "
                  f"dict={rec['dict_hit_rate']:.2f}" +
                  ("" if res.ok else f"  {res.error[:80]}"))

    # 2) Re-collect ALL cached results (across past runs / all models)
    results = []
    for jf in sorted(raw_root.rglob("p*.json")):
        results.append(read_json(jf))

    # 3) Consensus per page across whatever models have that page
    by_page: dict[tuple, dict] = {}
    for r in results:
        by_page.setdefault((r["issue_id"], r["page"]), {})[r["model"]] = r["text"] if r["ok"] else ""
    for (iid, pi), mt in by_page.items():
        cs = consensus_scores(mt)
        for r in results:
            if r["issue_id"] == iid and r["page"] == pi:
                r["consensus"] = cs.get(r["model"], 0.0)

    # 3b) Accuracy vs the PRIMARY ground-truth reference (first is_reference,
    # i.e. Claude). All reference models are excluded from the ranking; cross-
    # reference triangulation is done separately in scripts/triangulate.py.
    ref_names = [n for n, m in cfg["models"].items() if m.get("is_reference")]
    ref_name = ref_names[0] if ref_names else None
    if no_aggregate:
        print(f"\n[--no-aggregate] cached {len(results)} page-results; "
              f"skipping results.json/summary.json write.")
        return
    ref_text = {}
    if ref_name:
        for r in results:
            if r["model"] == ref_name and r["ok"]:
                ref_text[(r["issue_id"], r["page"])] = r["text"]
    for r in results:
        key = (r["issue_id"], r["page"])
        if ref_name and r["model"] != ref_name and r["ok"] and key in ref_text:
            r.update(accuracy_vs_reference(r["text"], ref_text[key]))

    write_json(bench_root / "results.json", results)

    # 4) Per-model aggregates + a composite ranking
    summary = {}
    model_names = sorted({r["model"] for r in results})
    for name in model_names:
        rows = [r for r in results if r["model"] == name]
        ok = [r for r in rows if r["ok"]]
        def avg(key, only_scored=False):
            src = [r for r in ok if key in r] if only_scored else ok
            # None = metric not applicable on that page (e.g. no digit tokens)
            vals = [r.get(key, 0.0) for r in src if r.get(key) is not None]
            return round(stats.mean(vals), 4) if vals else 0.0
        scored = [r for r in ok if "bow_f1" in r]   # have a ref comparison
        summary[name] = {
            "pages": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
            "is_reference": name in ref_names,
            "avg_seconds": avg("seconds"),
            "pages_per_sec": round(len(ok) / sum(r["seconds"] for r in ok), 3)
                              if ok and sum(r["seconds"] for r in ok) else 0.0,
            "avg_words": round(avg("words")),
            "dict_hit_rate": avg("dict_hit_rate"),
            "fluency": avg("fluency"),
            "consensus": avg("consensus"),
            "bow_f1": avg("bow_f1", only_scored=True),
            "bow_recall": avg("bow_recall", only_scored=True),
            "bow_precision": avg("bow_precision", only_scored=True),
            "char_sim": avg("char_sim", only_scored=True),
            "digit_recall": avg("digit_recall", only_scored=True),
            "rare_word_recall": avg("rare_word_recall", only_scored=True),
            "phrase_recovery": avg("phrase_recovery", only_scored=True),
            "scored_pages": len(scored),
        }
    # Composite quality: ground-truth bag-of-words F1 vs Claude (order-independent,
    # fair across column reading orders) dominates; fall back to proxies if no ref.
    for name, s in summary.items():
        if s["is_reference"]:
            s["quality_score"] = None
            continue
        if s["scored_pages"]:
            s["quality_score"] = round(
                0.80 * s["bow_f1"]
                + 0.10 * s["char_sim"]
                + 0.10 * s["dict_hit_rate"], 4)
        else:
            s["quality_score"] = round(
                0.55 * s["dict_hit_rate"]
                + 0.25 * s["consensus"]
                + 0.20 * min(s["fluency"] / 5.0, 1.0), 4)
    ranking = sorted([n for n in summary if not summary[n]["is_reference"]],
                     key=lambda n: summary[n]["quality_score"], reverse=True)

    write_json(bench_root / "summary.json",
               {"summary": summary, "ranking": ranking, "reference": ref_name})

    # 5) Console table
    if summary:
        have_acc = any(summary[n]["scored_pages"] for n in ranking)
        print("\n=== Benchmark summary (quality-ranked) ===")
        if ref_name:
            print(f"Reference (ground truth): {ref_name}"
                  f"  [{summary.get(ref_name,{}).get('ok',0)} pages]")
        hdr = (f"{'model':<15}{'qual':>6}{'F1':>6}{'recall':>7}{'prec':>6}"
               f"{'charS':>6}{'dict':>6}{'wpp':>6}{'s/pg':>7}{'err':>4}")
        print(hdr); print("-" * len(hdr))
        for name in ranking:
            s = summary[name]
            f1 = f"{s['bow_f1']:.2f}" if s["scored_pages"] else "  - "
            rec = f"{s['bow_recall']:.2f}" if s["scored_pages"] else "   - "
            prec = f"{s['bow_precision']:.2f}" if s["scored_pages"] else "  - "
            chs = f"{s['char_sim']:.2f}" if s["scored_pages"] else "  - "
            print(f"{name:<15}{s['quality_score']:>6.3f}{f1:>6}{rec:>7}{prec:>6}"
                  f"{chs:>6}{s['dict_hit_rate']:>6.2f}"
                  f"{s['avg_words']:>6}{s['avg_seconds']:>7.1f}{s['errors']:>4}")
        if not have_acc:
            print("\n(no ground-truth accuracy yet — run the reference model: "
                  "it scores wAcc/charS for every model)")
        print("\nWrote results.json + summary.json to", bench_root)
        print("View side-by-side at  http://localhost:5000/benchmark")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    ap.add_argument("--only", nargs="+", help="restrict to these model names")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="OCR + cache only; don't write results/summary (avoids "
                         "races with a concurrent run)")
    ap.add_argument("--sample-key", default="benchmark_sample",
                    help="config key with the sample list (e.g. pilot_sample)")
    ap.add_argument("--pdf-root", default=None,
                    help="corpus root override (e.g. dl-c6e6b6c1de78)")
    run(**vars(ap.parse_args()))
