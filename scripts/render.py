"""Render PDF pages to PNG images for OCR, with on-disk caching.

Uses pypdfium2 (bundled PDFium) — no Poppler/Ghostscript needed.
"""
from __future__ import annotations
from pathlib import Path
import pypdfium2 as pdfium

from common import IssueMeta


def page_count(pdf_path: str | Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_page(pdf_path: str | Path, page_index: int, out_path: str | Path,
                dpi: int = 300) -> Path:
    """Render one page to PNG at the given DPI. Cached: skips if out_path exists."""
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        scale = dpi / 72.0  # PDF user space is 72 dpi
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path)
    finally:
        pdf.close()
    return out_path


def page_png_path(pages_root: str | Path, meta: IssueMeta, page_index: int) -> Path:
    return Path(pages_root) / meta.issue_id / f"p{page_index:03d}.png"


def web_jpeg_path(web_root: str | Path, issue_id: str, page_index: int) -> Path:
    return Path(web_root) / issue_id / f"p{page_index:03d}.jpg"


def save_web_jpeg(src_png: str | Path, out_path: str | Path,
                  max_side: int = 2000, quality: int = 85) -> Path:
    """Downscaled JPEG for the web view (~1 MB vs ~15 MB for the 300-dpi PNG).
    Whole corpus as web JPEGs ≈ 40-50 GB — affordable; PNGs (~780 GB) are not."""
    from PIL import Image
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_png)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    img.save(out_path, format="JPEG", quality=quality)
    return out_path


def render_issue(pdf_path: str | Path, meta: IssueMeta, pages_root: str | Path,
                 dpi: int = 300, max_pages: int | None = None) -> list[Path]:
    n = page_count(pdf_path)
    if max_pages is not None:
        n = min(n, max_pages)
    out = []
    for i in range(n):
        out.append(render_page(pdf_path, i,
                               page_png_path(pages_root, meta, i), dpi=dpi))
    return out


if __name__ == "__main__":
    import sys, time
    from common import load_config, parse_meta
    cfg = load_config()
    rel = sys.argv[1] if len(sys.argv) > 1 else "anaheim-gazette/1941/ag-1941-03-27.pdf"
    pdf_root = Path(cfg["paths"]["pdf_root"])
    pdf = pdf_root / rel
    meta = parse_meta(pdf, pdf_root)
    print(f"pages: {page_count(pdf)}  rendering page 0 @ {cfg['render']['dpi']} dpi ...")
    t = time.perf_counter()
    out = render_page(pdf, 0, page_png_path(cfg["paths"]["pages"], meta, 0),
                      dpi=cfg["render"]["dpi"])
    dt = time.perf_counter() - t
    print(f"-> {out}  ({out.stat().st_size/1e6:.1f} MB, {dt:.2f}s)")
