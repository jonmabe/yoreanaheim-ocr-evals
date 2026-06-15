"""Unified OCR client.

Two backends:
  * vlm_chat : OpenAI-compatible /chat/completions with an image (vLLM-served
               PaddleOCR-VL, GLM-OCR, HunyuanOCR, dots.ocr, MinerU raw VLM, ...)
  * mineru   : the MinerU toolchain (CLI / vlm-http-client) — optional

Every call returns an OcrResult so the benchmark and full run share one path.
"""
from __future__ import annotations
import base64
import io
import re
import time
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import os
import requests
from PIL import Image


def _auth_headers(model_cfg: dict) -> dict:
    """Bearer token for vLLM's OpenAI server. Order: model cfg -> env -> none.

    serve-any.sh launches with --api-key "$VLLM_API_KEY", so the server requires
    it. Set VLLM_API_KEY in this process's environment (don't hard-code secrets).
    """
    key = model_cfg.get("api_key") or os.environ.get("VLLM_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


@dataclass
class OcrResult:
    model: str
    text: str
    seconds: float
    ok: bool = True
    error: str = ""
    boxes: list = field(default_factory=list)   # optional [{text,bbox:[x0,y0,x1,y1]}]
    usage: dict = field(default_factory=dict)    # token usage if reported
    finish: str = ""    # finish_reason: "stop" | "length" (truncated!) |
                        # tiled: "length:2/9" = 2 of 9 tiles truncated


def encode_image(img_path: str | Path, max_side: int | None = None,
                 fmt: str = "JPEG", quality: int = 90) -> tuple[str, str]:
    """Return (data_uri, mime). Optionally downscale longest side to max_side."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    save_kw = {"quality": quality} if fmt.upper() in ("JPEG", "WEBP") else {}
    img.save(buf, format=fmt, **save_kw)
    mime = f"image/{ 'jpeg' if fmt.upper()=='JPEG' else fmt.lower() }"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}", mime


def is_up(model_cfg: dict, timeout: float = 3.0) -> bool:
    """Cheap reachability check for a model endpoint."""
    try:
        if model_cfg.get("mode") == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        if model_cfg.get("mode") == "gemini":
            return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if model_cfg.get("mode") == "anthropic":
            return bool(model_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        if model_cfg.get("mode") == "mineru":
            url = model_cfg["server_url"].rstrip("/") + "/health"
            return requests.get(url, timeout=timeout).ok
        base = model_cfg["base_url"]
        # list = multi-GPU serving; up if ANY endpoint answers /models
        bases = base if isinstance(base, list) else [base]
        for b in bases:
            try:
                if requests.get(b.rstrip("/") + "/models", timeout=timeout,
                                headers=_auth_headers(model_cfg)).ok:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _parse_dots_json(raw: str) -> str:
    """dots.ocr emits a JSON array of layout blocks (bbox/category/text) in
    reading order. Pull the text fields out, joined in order."""
    import json
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        arr = json.loads(s)
        parts = [d["text"] for d in arr if isinstance(d, dict) and d.get("text")]
        return "\n".join(parts).strip()
    except Exception:
        pass
    # Truncated/malformed JSON (dense page overflowed max_tokens): salvage every
    # "text": "..." field via regex so we don't dump bbox numbers as text.
    vals = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    if vals:
        return "\n".join(json.loads('"' + v + '"') for v in vals).strip()
    return ""  # nothing parseable — empty beats numeric garbage


_BASE_HEALTH: dict = {}   # base_url -> (ok: bool, checked_at: float)


def _base_ok(base: str, cfg: dict) -> bool:
    """Cheap cached health probe (60 s TTL) so a server can be killed
    mid-run (e.g. freeing GPU 1 for gaming) and the client just routes
    around it — and picks it back up within a minute of its return."""
    import time
    ok, t = _BASE_HEALTH.get(base, (True, 0.0))
    if time.time() - t > 60:
        try:
            ok = requests.get(base.rstrip("/") + "/models", timeout=2,
                              headers=_auth_headers(cfg)).status_code in (200, 401)
        except Exception:
            ok = False
        _BASE_HEALTH[base] = (ok, time.time())
    return ok


def _pick_base(cfg: dict) -> str:
    """base_url may be a list (same model served on several GPUs) — spread
    requests across the HEALTHY ones. Tiles of one page also spread: same
    model, no state."""
    base = cfg["base_url"]
    if isinstance(base, list):
        import random
        alive = [b for b in base if _base_ok(b, cfg)]
        base = random.choice(alive or base)   # all down -> let it fail loudly
    return base.rstrip("/")


def _ocr_vlm_chat(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    base = _pick_base(cfg)
    data_uri, _ = encode_image(
        img_path,
        max_side=cfg.get("max_side"),
        fmt=cfg.get("image_fmt", "JPEG"),
        quality=cfg.get("image_quality", 90),
    )
    payload = {
        "model": cfg["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": cfg.get("prompt", "OCR this image. Output plain text only.")},
            ],
        }],
        "max_tokens": cfg.get("max_tokens", 8192),
        "temperature": cfg.get("temperature", 0.0),
    }
    # Anti-degeneration controls. Dense newsprint pushes these OCR VLMs into
    # repetition loops under greedy decoding; repetition_penalty (vLLM extra) and
    # the OpenAI presence/frequency penalties break the loop. Only sent if set.
    for k in ("repetition_penalty", "frequency_penalty", "presence_penalty", "top_p"):
        if cfg.get(k) is not None:
            payload[k] = cfg[k]
    t = time.perf_counter()
    try:
        r = requests.post(base + "/chat/completions", json=payload,
                          headers=_auth_headers(cfg),
                          timeout=cfg.get("timeout", 300))
        dt = time.perf_counter() - t
        if not r.ok:
            return OcrResult(name, "", dt, ok=False,
                             error=f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        if cfg.get("parse") == "dots_json":
            text = _parse_dots_json(text)
        return OcrResult(name, text, dt, usage=data.get("usage", {}),
                         finish=data["choices"][0].get("finish_reason") or "")
    except Exception as e:
        return OcrResult(name, "", time.perf_counter() - t, ok=False, error=str(e))


def _ocr_mineru(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    """Run MinerU on a single image; read back the produced markdown."""
    t = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                cfg.get("bin", "mineru"),
                "-p", str(img_path), "-o", td,
                "-b", cfg.get("backend", "vlm-http-client"),
            ]
            if cfg.get("server_url"):
                cmd += ["-u", cfg["server_url"]]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=cfg.get("timeout", 600))
            dt = time.perf_counter() - t
            if proc.returncode != 0:
                return OcrResult(name, "", dt, ok=False,
                                 error=(proc.stderr or proc.stdout)[:400])
            mds = list(Path(td).rglob("*.md"))
            text = "\n\n".join(p.read_text(encoding="utf-8") for p in mds)
            return OcrResult(name, text.strip(), dt)
    except Exception as e:
        return OcrResult(name, "", time.perf_counter() - t, ok=False, error=str(e))


def _encode_raw(img_path, max_side, fmt="JPEG", quality=90):
    """Return (base64_str, media_type) for the Anthropic image block."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _ocr_anthropic(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    """Ground-truth reference via Claude vision (official anthropic SDK)."""
    import anthropic
    t = time.perf_counter()
    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        # Opus 4.7+ high-res vision tops out ~2576px long edge; 2400 is the sweet spot.
        b64, media = _encode_raw(img_path, cfg.get("max_side", 2400),
                                 quality=cfg.get("image_quality", 90))
        prompt = cfg.get("prompt", "Transcribe all text from this scanned "
                         "newspaper page in natural reading order. Output only "
                         "the transcription, no commentary.")
        text_parts = []
        with client.messages.stream(
            model=cfg.get("model", "claude-opus-4-8"),
            max_tokens=cfg.get("max_tokens", 16000),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media, "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        ) as stream:
            for chunk in stream.text_stream:
                text_parts.append(chunk)
            final = stream.get_final_message()
        usage = {"input_tokens": final.usage.input_tokens,
                 "output_tokens": final.usage.output_tokens}
        return OcrResult(name, "".join(text_parts).strip(),
                         time.perf_counter() - t, usage=usage)
    except Exception as e:
        return OcrResult(name, "", time.perf_counter() - t, ok=False, error=str(e))


def _grid_tiles(img, cols, rows, overlap):
    """Yield (col, row, PIL tile) in column-major order (down each column band,
    then the next column) — approximates newspaper reading order."""
    W, H = img.size
    for c in range(cols):
        for r in range(rows):
            x0 = max(0, int(W * (c / cols - overlap)))
            x1 = min(W, int(W * ((c + 1) / cols + overlap)))
            y0 = max(0, int(H * (r / rows - overlap)))
            y1 = min(H, int(H * ((r + 1) / rows + overlap)))
            yield c, r, img.crop((x0, y0, x1, y1))


def _ocr_tiled(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    """Tile a dense page into a grid, OCR each tile, concatenate.

    Dense broadsheets overwhelm small OCR VLMs (they loop); per-tile regions are
    within their competence. Concatenated column-major for rough reading order;
    optimized for word recall (search), not perfect layout.
    """
    import tempfile, os
    t = cfg["tiling"]
    cols, rows = t.get("cols", 3), t.get("rows", 3)
    overlap, tmax = t.get("overlap", 0.04), t.get("max_side", 1600)
    tile_cfg = dict(cfg); tile_cfg["max_side"] = tmax
    img = Image.open(img_path).convert("RGB")
    parts, ok_any, errs = [], False, []
    n_tiles, n_trunc = 0, 0
    usage = {}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        for c, r, tile in _grid_tiles(img, cols, rows, overlap):
            tf = os.path.join(td, f"t{c}{r}.png")
            tile.save(tf)
            res = _ocr_vlm_chat(name, tile_cfg, tf)
            n_tiles += 1
            if res.finish == "length":
                n_trunc += 1
            for k, v in (res.usage or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            if res.ok and res.text.strip():
                parts.append(res.text.strip()); ok_any = True
            elif not res.ok:
                errs.append(res.error)
    finish = f"length:{n_trunc}/{n_tiles}" if n_trunc else "stop"
    return OcrResult(name, "\n\n".join(parts), round(time.perf_counter() - start, 3),
                     ok=ok_any, error="" if ok_any else ("; ".join(errs)[:300]),
                     usage=usage, finish=finish)


def _ocr_openai(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    """Secondary ground-truth reference via OpenAI vision (GPT-5.x).

    GPT-5 reasoning models use max_completion_tokens and reject a custom
    temperature, so this is a distinct path from the vLLM OpenAI-compat one.
    """
    from openai import OpenAI
    t = time.perf_counter()
    try:
        client = OpenAI()  # reads OPENAI_API_KEY from env
        data_uri, _ = encode_image(img_path, max_side=cfg.get("max_side", 2048),
                                   fmt="JPEG", quality=cfg.get("image_quality", 90))
        prompt = cfg.get("prompt", "Transcribe all text from this scanned "
                         "newspaper page in reading order. Output only the text.")
        kwargs = dict(
            model=cfg.get("model", "gpt-5.5"),
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": data_uri, "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}],
            max_completion_tokens=cfg.get("max_tokens", 16000),
        )
        if cfg.get("reasoning_effort"):
            kwargs["reasoning_effort"] = cfg["reasoning_effort"]
        r = client.chat.completions.create(**kwargs)
        text = r.choices[0].message.content or ""
        usage = {"input_tokens": r.usage.prompt_tokens,
                 "output_tokens": r.usage.completion_tokens}
        return OcrResult(name, text.strip(), time.perf_counter() - t, usage=usage)
    except Exception as e:
        return OcrResult(name, "", time.perf_counter() - t, ok=False, error=str(e))


def _ocr_gemini(name: str, cfg: dict, img_path: str | Path) -> OcrResult:
    """Tertiary ground-truth reference via Google Gemini vision."""
    from google import genai
    from google.genai import types
    t = time.perf_counter()
    try:
        client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        ms = cfg.get("max_side", 2048)
        w, h = img.size
        if max(w, h) > ms:
            s = ms / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, format="JPEG",
                                     quality=cfg.get("image_quality", 90))
        prompt = cfg.get("prompt", "Transcribe all text from this scanned "
                         "newspaper page in reading order. Output only the text.")
        resp = client.models.generate_content(
            model=cfg.get("model", "gemini-3.1-pro-preview"),
            contents=[types.Part.from_bytes(data=buf.getvalue(),
                                            mime_type="image/jpeg"), prompt],
        )
        text = resp.text or ""
        usage = {}
        if getattr(resp, "usage_metadata", None):
            usage = {"input_tokens": resp.usage_metadata.prompt_token_count,
                     "output_tokens": resp.usage_metadata.candidates_token_count}
        return OcrResult(name, text.strip(), time.perf_counter() - t, usage=usage)
    except Exception as e:
        return OcrResult(name, "", time.perf_counter() - t, ok=False, error=str(e))


def ocr_page(name: str, cfg: dict, img_path: str | Path,
             overrides: dict | None = None) -> OcrResult:
    """OCR one page image. `overrides` merges over the model cfg for this call
    only — used by the degeneration-guard retry (stronger penalties) and by
    per-paper prompt routing."""
    if overrides:
        cfg = {**cfg, **overrides}
    mode = cfg.get("mode", "vlm_chat")
    if mode == "openai":
        return _ocr_openai(name, cfg, img_path)
    if mode == "gemini":
        return _ocr_gemini(name, cfg, img_path)
    if mode == "vlm_chat":
        if cfg.get("tiling", {}).get("enabled"):
            return _ocr_tiled(name, cfg, img_path)
        return _ocr_vlm_chat(name, cfg, img_path)
    if mode == "mineru":
        return _ocr_mineru(name, cfg, img_path)
    if mode == "anthropic":
        return _ocr_anthropic(name, cfg, img_path)
    return OcrResult(name, "", 0.0, ok=False, error=f"unknown mode {mode!r}")


def active_models(cfg: dict, only_reachable: bool = True) -> dict:
    """Models that are enabled and (optionally) currently reachable."""
    out = {}
    for name, m in cfg.get("models", {}).items():
        if m.get("enabled") is False:
            continue
        if only_reachable and not is_up(m):
            continue
        out[name] = m
    return out


if __name__ == "__main__":
    import sys
    from common import load_config
    cfg = load_config()
    print("Endpoint reachability:")
    for name, m in cfg["models"].items():
        if m.get("enabled") is False:
            print(f"  {name:<14} disabled")
            continue
        print(f"  {name:<14} {'UP' if is_up(m) else 'down':<5} "
              f"{m.get('base_url') or m.get('server_url')}")
