"""Shared helpers: config loading, path handling, filename -> metadata parsing."""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | os.PathLike = None) -> None:
    """Load KEY=VALUE pairs from the project .env into os.environ (no override).

    Holds VLLM_API_KEY (local vLLM auth) and ANTHROPIC_API_KEY (Claude reference).
    Existing environment values win, so you can still override per-shell.
    """
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_config(path: str | os.PathLike = None) -> dict:
    load_env()
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve paths relative to project root
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    return cfg


DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass
class IssueMeta:
    issue_id: str      # stable id, e.g. "anaheim-gazette__1941__ag-1941-01-02"
    paper: str         # top-level folder, e.g. "anaheim-gazette"
    date: str          # ISO date "1941-01-02" or "" if unknown
    year: str          # "1941" or ""
    title: str         # human label
    rel_path: str      # path relative to pdf_root, POSIX style
    source: str        # e.g. "LAT" for clipped articles, else ""


def parse_meta(pdf_path: Path, pdf_root: Path) -> IssueMeta:
    rel = pdf_path.relative_to(pdf_root)
    rel_posix = rel.as_posix()
    parts = rel.parts
    paper = parts[0]
    stem = pdf_path.stem

    m = DATE_RE.search(rel_posix)
    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    year = m.group(1) if m else ""

    # clipped-article style: "1949-11-01 -- LAT - Some Title"
    source = ""
    title = stem
    if " -- " in stem:
        after = stem.split(" -- ", 1)[1]
        if " - " in after:
            source, title = after.split(" - ", 1)
            source = source.strip()
            title = title.strip()
        else:
            title = after.strip()
    else:
        title = f"{paper} {date}".strip()

    issue_id = re.sub(r"[^A-Za-z0-9]+", "_", rel_posix.rsplit(".", 1)[0]).strip("_")
    return IssueMeta(
        issue_id=issue_id, paper=paper, date=date, year=year,
        title=title, rel_path=rel_posix, source=source,
    )


def iter_pdfs(pdf_root: str | os.PathLike):
    """Yield (Path, IssueMeta) for every PDF under pdf_root, sorted."""
    pdf_root = Path(pdf_root)
    for p in sorted(pdf_root.rglob("*.pdf")):
        yield p, parse_meta(p, pdf_root)


def write_json(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # quick self-test: print parsed metadata for a few files
    cfg = load_config()
    root = cfg["paths"]["pdf_root"]
    n = 0
    for p, meta in iter_pdfs(root):
        print(json.dumps(asdict(meta), ensure_ascii=False))
        n += 1
        if n >= 4:
            break
    # also show one clipped article
    for p, meta in iter_pdfs(root):
        if meta.source:
            print("CLIP:", json.dumps(asdict(meta), ensure_ascii=False))
            break
    print(f"... {sum(1 for _ in iter_pdfs(root))} PDFs total")
