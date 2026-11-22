"""Scan a PDF corpus: count PDFs, total PAGES (the real OCR cost driver), and
size, broken down by publication and decade. Saves JSON for the report.

  python scripts/scan_corpus.py "D:/My Files/Projects/YoreAnaheim-2026/dl-c6e6b6c1de78"
"""
from __future__ import annotations
import sys
import re
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pypdfium2 as pdfium


def page_count(p):
    try:
        doc = pdfium.PdfDocument(str(p))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return -1   # sentinel: failed to open (distinct from a real 0-page doc)


def work(p_str):
    """Top-level (picklable) worker — pypdfium2 is NOT thread-safe, so we use
    separate PROCESSES, each with its own pdfium instance."""
    from pathlib import Path as _P
    p = _P(p_str)
    return p_str, page_count(p), p.stat().st_size


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1
               else "D:/My Files/Projects/YoreAnaheim-2026/dl-c6e6b6c1de78")
    pdfs = sorted(root.rglob("*.pdf"))
    print(f"Scanning {len(pdfs)} PDFs under {root} ...")

    by_paper = defaultdict(lambda: {"pdfs": 0, "pages": 0, "bytes": 0, "failed": 0})
    by_decade = defaultdict(lambda: {"pdfs": 0, "pages": 0})
    yr = re.compile(r"(1[89]\d\d|20\d\d)")

    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work, str(p)) for p in pdfs]):
            p_str, pages, size = fut.result()
            p = Path(p_str)
            paper = p.relative_to(root).parts[0]
            by_paper[paper]["pdfs"] += 1
            by_paper[paper]["bytes"] += size
            if pages < 0:
                by_paper[paper]["failed"] += 1
            else:
                by_paper[paper]["pages"] += pages
                m = yr.search(p.as_posix())
                if m:
                    dec = f"{int(m.group(1)) // 10 * 10}s"
                    by_decade[dec]["pdfs"] += 1
                    by_decade[dec]["pages"] += pages
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(pdfs)} ...")

    total_pages = sum(v["pages"] for v in by_paper.values())
    total_bytes = sum(v["bytes"] for v in by_paper.values())
    total_pdfs = sum(v["pdfs"] for v in by_paper.values())

    out = {
        "root": str(root), "total_pdfs": total_pdfs, "total_pages": total_pages,
        "total_gb": round(total_bytes / 1e9, 1),
        "by_paper": {k: {**v, "gb": round(v["bytes"] / 1e9, 2)}
                     for k, v in sorted(by_paper.items())},
        "by_decade": dict(sorted(by_decade.items())),
    }
    Path("data/bench").mkdir(parents=True, exist_ok=True)
    Path("data/bench/corpus_scan.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== CORPUS: {total_pdfs} PDFs, {total_pages:,} pages, "
          f"{out['total_gb']} GB ===")
    print(f"{'publication':<22}{'PDFs':>7}{'pages':>9}{'GB':>7}{'pg/pdf':>8}{'fail':>6}")
    for k, v in out["by_paper"].items():
        ok = v["pdfs"] - v.get("failed", 0)
        ppi = v["pages"] / ok if ok else 0
        print(f"{k:<22}{v['pdfs']:>7}{v['pages']:>9,}{v['gb']:>7}{ppi:>8.1f}"
              f"{v.get('failed', 0):>6}")
    print("\nby decade:", {k: v["pages"] for k, v in out["by_decade"].items()})
    print("\nSaved data/bench/corpus_scan.json")


if __name__ == "__main__":
    main()
