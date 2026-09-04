#!/usr/bin/env python3
"""
vlm_test_client.py — single-file Gradio test client for gpu_api_server_v6.py

Run this on any machine that can reach the GPU box on port 5432:

    pip install gradio requests pillow
    python vlm_test_client.py --gpu-url http://<gpu-host>:5432

Then open http://<this-host>:7860 in a browser.

What it does
------------
  * Uploads one or more local images, encodes them as base64 data URIs and
    POSTs them to /infer (or /infer/stream) exactly as the server expects.
  * Region tool: drag a rectangle on the image, then move or resize it by its
    handles. The crop is taken from the ORIGINAL file at full resolution and
    can be sent instead of — or alongside — the whole image, so you can ask
    the VLM about one table, chart or paragraph without it wandering off.
  * Reads the live model registry from /models so the dropdown always matches
    the server's MODEL_CONFIGS (no hardcoded model list to drift).
  * Enforces the per-model image cap (limit_mm_per_prompt) client-side so you
    get a useful error instead of a 400 from the GPU box.
  * Manual load / unload buttons — handy because the two VLMs share an
    eviction group and loading one drops the other.
  * /debug/prompt viewer, so when a VLM answers as if it never saw the image
    you can check the vision placeholders are actually in the templated prompt.
  * Health (VRAM, loaded models) and /metrics panels.

Styling follows the Vodafone Idea Design System V.01 (DM Sans; #F4F1EC ground,
white surfaces, red→yellow primary gradient, and its documented type scale).

Everything is stdlib + requests + gradio; no vLLM or torch needed here.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import time
import uuid
from typing import Any, Iterator, Optional

import requests

try:
    import gradio as gr
except ImportError:  # pragma: no cover - friendlier than a bare traceback
    raise SystemExit("gradio is not installed.  pip install gradio requests pillow")


# ─────────────────────────────── Config ──────────────────────────────────────

DEFAULT_GPU_URL = os.environ.get("GPU_API_URL", "http://localhost:5432")
CLIENT_ID       = os.environ.get("GPU_API_CLIENT_ID", "vlm-test-client")

# Client-side downscale before base64. The server also caps at 2048 px, but
# shrinking here keeps the POST body small over the wire.
MAX_UPLOAD_SIDE_PX = 2048

# Images shown in the region tool are downscaled to keep the page light; the
# box is stored in normalized (0-1) coordinates, so crops always come off the
# original file at full resolution.
PREVIEW_SIDE_PX = 1400

# A crop smaller than this (in original pixels) is almost certainly a stray
# click rather than a region the user meant to ask about.
MIN_CROP_PX = 8

# Generous: a cold VLM load (30B/38B, TP=2) can take several minutes.
CONNECT_TIMEOUT_S  = 10
READ_TIMEOUT_S     = 900

SEND_FULL   = "Full image"
SEND_REGION = "Selected region only"
SEND_BOTH   = "Full image + region"


# ─────────────────────────────── HTTP helpers ────────────────────────────────

def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _get(base: str, path: str) -> Any:
    r = requests.get(_url(base, path), timeout=(CONNECT_TIMEOUT_S, 60))
    r.raise_for_status()
    return r.json()


def _post(base: str, path: str, payload: Optional[dict] = None,
          read_timeout: int = READ_TIMEOUT_S) -> Any:
    r = requests.post(
        _url(base, path),
        json=payload if payload is not None else {},
        timeout=(CONNECT_TIMEOUT_S, read_timeout),
    )
    if r.status_code >= 400:
        # FastAPI puts the message in {"detail": ...}
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code} from {path}: {detail}")
    return r.json()


def _err(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return ("Could not reach the GPU server. Check the URL, that uvicorn is "
                f"listening on 0.0.0.0:5432, and that the port is open.\n\n{exc}")
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return (f"Timed out after {READ_TIMEOUT_S}s. A first-time VLM load can be "
                f"slow — check the server log, then retry.\n\n{exc}")
    return f"{type(exc).__name__}: {exc}"


# ─────────────────────────────── Image encoding ──────────────────────────────

def _data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _to_data_uri(img, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return _data_uri(buf.getvalue(), f"image/{fmt.lower()}")


def _crop_box(img, box: list[float]):
    """Crop a PIL image by a normalized [x, y, w, h] box."""
    from PIL import Image  # noqa: F401  (import kept local, mirrors module style)

    W, H = img.size
    x, y, w, h = box
    left   = max(0, min(W - 1, int(round(x * W))))
    top    = max(0, min(H - 1, int(round(y * H))))
    right  = max(left + 1, min(W, int(round((x + w) * W))))
    bottom = max(top + 1, min(H, int(round((y + h) * H))))

    if (right - left) < MIN_CROP_PX or (bottom - top) < MIN_CROP_PX:
        raise ValueError(
            f"The selected region is only {right - left}x{bottom - top} px in the "
            f"original image — too small to be useful. Draw a larger box."
        )
    return img.crop((left, top, right, bottom))


def _prepare_image(path: str, box: Optional[list[float]] = None):
    """
    Local file (optionally cropped to a normalized box) → (data URI, PIL preview).

    With no box and a modestly sized file the original bytes are shipped as-is:
    no re-encode, no quality loss. Anything larger is downscaled first.
    """
    from PIL import Image

    with Image.open(path) as src:
        src.load()
        img = _crop_box(src, box) if box else src.copy()

    w, h = img.size
    if not box and max(w, h) <= MAX_UPLOAD_SIDE_PX:
        with open(path, "rb") as fh:
            raw = fh.read()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return _data_uri(raw, mime), img

    if max(w, h) > MAX_UPLOAD_SIDE_PX:
        scale = MAX_UPLOAD_SIDE_PX / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    return _to_data_uri(img.convert("RGB")), img


def _preview_uri(path: str) -> str:
    """Small data URI used only for on-screen display in the region tool."""
    from PIL import Image

    with Image.open(path) as img:
        img.load()
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_SIDE_PX, PREVIEW_SIDE_PX))
        return _to_data_uri(img, quality=82)


def _paths_from_files(files: Optional[list]) -> list[str]:
    """gr.File(type='filepath') gives str paths; be tolerant of dict/obj forms."""
    out: list[str] = []
    for f in files or []:
        if isinstance(f, str):
            out.append(f)
        elif isinstance(f, dict) and f.get("path"):
            out.append(f["path"])
        elif hasattr(f, "name"):
            out.append(f.name)
    return out


def _parse_boxes(boxes_json: str) -> dict[int, list[float]]:
    """Decode the region tool's state: {"boxes": {"0": [x,y,w,h]}, "active": 0}."""
    try:
        raw = json.loads(boxes_json or "{}").get("boxes", {})
    except (ValueError, AttributeError):
        return {}

    out: dict[int, list[float]] = {}
    for k, v in (raw or {}).items():
        try:
            box = [float(n) for n in v]
        except (TypeError, ValueError):
            continue
        if len(box) == 4 and box[2] > 0 and box[3] > 0:
            out[int(k)] = box
    return out


# ─────────────────────────────── Registry ────────────────────────────────────

_registry: dict[str, dict] = {}   # name → ModelInfo dict from /models


def _refresh_registry(base: str) -> tuple[list[str], str]:
    """Pull /models. Returns (model names, markdown table)."""
    global _registry
    models = _get(base, "/models")
    _registry = {m["name"]: m for m in models}

    rows = [
        "| model | modality | loaded | lazy | TP | ctx | max img | conc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        rows.append(
            f"| `{m['name']}` | {m['modality']} | {'loaded' if m['loaded'] else '—'} | "
            f"{'yes' if m['lazy'] else 'eager'} | {m['tensor_parallel_size']} | "
            f"{m['max_model_len']} | {m.get('max_images') or '—'} | {m['max_concurrent']} |"
        )
        if m.get("error"):
            rows.append(f"| | | | | | | | **load error: {m['error']}** |")

    return [m["name"] for m in models], "\n".join(rows)


def _vision_first(names: list[str]) -> list[str]:
    vis = [n for n in names if _registry.get(n, {}).get("modality") == "vision"]
    return vis + [n for n in names if n not in vis]


# ─────────────────────────────── Payload ─────────────────────────────────────

def _collect_images(paths: list[str], boxes: dict[int, list[float]], send_mode: str):
    """
    Turn the uploaded files + drawn regions into the `images` list for /infer.
    Returns (data URIs, PIL previews, per-image labels).
    """
    uris: list[str]   = []
    previews: list    = []
    labels: list[str] = []

    for i, path in enumerate(paths):
        box  = boxes.get(i)
        name = os.path.basename(path)

        want_full   = send_mode == SEND_FULL or box is None or send_mode == SEND_BOTH
        want_region = box is not None and send_mode in (SEND_REGION, SEND_BOTH)

        if want_full:
            uri, prev = _prepare_image(path)
            uris.append(uri)
            previews.append(prev)
            labels.append(f"{name} — full")

        if want_region:
            uri, prev = _prepare_image(path, box)
            uris.append(uri)
            previews.append(prev)
            labels.append(f"{name} — region {prev.size[0]}x{prev.size[1]} px")

    return uris, previews, labels


def _build_payload(model, prompt, system, image_uris,
                   max_new_tokens, temperature, top_p, top_k,
                   repetition_penalty, stop_sequences) -> dict:
    if not model:
        raise ValueError("Pick a model first (hit Refresh if the list is empty).")
    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")

    info     = _registry.get(model, {})
    modality = info.get("modality", "text")

    if image_uris and modality != "vision":
        raise ValueError(
            f"'{model}' is text-only — it will reject images. Vision models: "
            + ", ".join(n for n, i in _registry.items() if i["modality"] == "vision")
        )

    cap = info.get("max_images")
    if image_uris and cap and len(image_uris) > cap:
        raise ValueError(
            f"'{model}' accepts at most {cap} image(s) per prompt "
            f"(limit_mm_per_prompt on the server); this request would send "
            f"{len(image_uris)}. Send fewer images, or switch to "
            f"'{SEND_REGION}' so each image counts once."
        )

    payload: dict = {
        "model":              model,
        "prompt":             prompt,
        "max_new_tokens":     int(max_new_tokens),
        "temperature":        float(temperature),
        "top_p":              float(top_p),
        "top_k":              int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "request_id":         str(uuid.uuid4()),
        "client_id":          CLIENT_ID,
    }
    if (system or "").strip():
        payload["system"] = system.strip()
    if image_uris:
        payload["images"] = image_uris
    stops = [s for s in (stop_sequences or "").split("\n") if s.strip()]
    if stops:
        payload["stop_sequences"] = stops
    return payload


def _redact(payload: dict) -> dict:
    """Same payload but with base64 blobs replaced by a size note."""
    out = dict(payload)
    if "images" in out:
        out["images"] = [f"<data URI, {len(i) // 1024} KB>" for i in out["images"]]
    return out


# ─────────────────────────────── Inference ───────────────────────────────────

def _stream_infer(base: str, payload: dict) -> Iterator[str]:
    """
    Consume the server's SSE stream.

    The server emits raw `data: <delta>\\n\\n` without escaping newlines inside
    the delta, so a delta that itself contains a blank line (very common in
    markdown output) is split across two "events". Any piece that does not
    start with "data: " is therefore a continuation of the previous delta and
    is re-joined with the blank line that separated it.
    """
    with requests.post(
        _url(base, "/infer/stream"),
        json=payload,
        stream=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    ) as r:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"HTTP {r.status_code} from /infer/stream: {detail}")

        buf   = ""
        began = False   # have we seen the first "data: " frame yet?

        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buf += chunk
            while "\n\n" in buf:
                event, buf = buf.split("\n\n", 1)

                if event.startswith("data: "):
                    data  = event[len("data: "):]
                    began = True
                    if data == "[DONE]":
                        return
                    if data.startswith("[ERROR]"):
                        raise RuntimeError(data)
                    if data:
                        yield data
                elif began:
                    # Continuation of a delta that contained a blank line.
                    yield "\n\n" + event


def run_inference(gpu_url, model, prompt, system, files, boxes_json, send_mode,
                  stream, max_new_tokens, temperature, top_p, top_k,
                  repetition_penalty, stop_sequences):
    """Main generate handler. Yields (output, stats, request json, sent gallery)."""
    try:
        paths  = _paths_from_files(files)
        boxes  = _parse_boxes(boxes_json)
        uris, previews, labels = _collect_images(paths, boxes, send_mode)
        payload = _build_payload(model, prompt, system, uris, max_new_tokens,
                                 temperature, top_p, top_k, repetition_penalty,
                                 stop_sequences)
    except Exception as exc:
        yield "", f"**Error** · {_err(exc)}", "", []
        return

    sent    = json.dumps(_redact(payload), indent=2)
    gallery = list(zip(previews, labels))
    n_box   = sum(1 for i in range(len(paths)) if i in boxes)

    detail = f"{len(uris)} image(s)"
    if n_box and send_mode != SEND_FULL:
        detail += f", {n_box} region(s)"
    yield "", f"Sending to `{model}` — {detail}…", sent, gallery

    t0 = time.time()

    if stream:
        text = ""
        try:
            for delta in _stream_infer(gpu_url, payload):
                text += delta
                yield text, f"Streaming… {len(text)} chars · {time.time() - t0:.1f}s", sent, gallery
        except Exception as exc:
            yield text, f"**Error** · {_err(exc)}", sent, gallery
            return
        yield text, (f"**Done** (streamed) · {detail} · "
                     f"{time.time() - t0:.2f}s · {len(text)} chars"), sent, gallery
        return

    try:
        resp = _post(gpu_url, "/infer", payload)
    except Exception as exc:
        yield "", f"**Error** · {_err(exc)}", sent, gallery
        return

    stats = (
        f"**{resp['model']}** · {resp.get('images', 0)} image(s) · "
        f"{resp['prompt_tokens']} prompt tokens · {resp['new_tokens']} new tokens\n\n"
        f"{resp['elapsed_s']}s server · {resp['tokens_per_sec']} tok/s · "
        f"{time.time() - t0:.2f}s wall\n\n"
        f"`request_id={resp.get('request_id')}`"
    )
    yield resp["text"], stats, sent, gallery


# ─────────────────────────────── Panel actions ───────────────────────────────

def do_refresh(gpu_url):
    try:
        names, table = _refresh_registry(gpu_url)
        names   = _vision_first(names)
        default = names[0] if names else None
        vision  = sum(1 for n in names if _registry[n]["modality"] == "vision")
        return (gr.update(choices=names, value=default),
                table,
                f"Connected · {len(names)} models, {vision} vision · `{gpu_url}`")
    except Exception as exc:
        return gr.update(), "", f"**Not connected** · {_err(exc)}"


def do_health(gpu_url):
    try:
        h = _get(gpu_url, "/health")
    except Exception as exc:
        return f"**Error** · {_err(exc)}"

    lines = [
        f"**Status** {h['status']} · **CUDA** {h['cuda_available']} · "
        f"**GPUs** {h['gpu_count']}",
        "",
        f"**Loaded** {', '.join(h['loaded_models']) or '(none)'}",
        f"**Vision models** {', '.join(h['vision_models'])}",
        "",
        "| GPU | name | used | free | total | util |",
        "|---|---|---|---|---|---|",
    ]
    for g in h["vram"]:
        lines.append(
            f"| {g['gpu']} | {g['name']} | {g['used_gb']} GB | {g['free_gb']} GB | "
            f"{g['total_gb']} GB | {g['util_pct']}% |"
        )
    if h.get("failed_models"):
        lines += ["", "**Failed loads**", "```", json.dumps(h["failed_models"], indent=2), "```"]
    return "\n".join(lines)


def do_metrics(gpu_url):
    try:
        return "```json\n" + json.dumps(_get(gpu_url, "/metrics"), indent=2) + "\n```"
    except Exception as exc:
        return f"**Error** · {_err(exc)}"


def do_load(gpu_url, model):
    if not model:
        return "Pick a model first."
    try:
        r = _post(gpu_url, f"/models/{model}/load", {})
        return f"`{model}` · **{r['status']}**\n\n```json\n{json.dumps(r['vram'], indent=2)}\n```"
    except Exception as exc:
        return f"**Error** · {_err(exc)}"


def do_unload(gpu_url, model):
    if not model:
        return "Pick a model first."
    try:
        r = _post(gpu_url, f"/models/{model}/unload", {}, read_timeout=300)
        return f"`{model}` · **{r['status']}**\n\n```json\n{json.dumps(r['vram'], indent=2)}\n```"
    except Exception as exc:
        return f"**Error** · {_err(exc)}"


def do_debug_prompt(gpu_url, model, prompt, system, files, boxes_json, send_mode):
    """Render the templated prompt without touching the GPU."""
    try:
        paths = _paths_from_files(files)
        boxes = _parse_boxes(boxes_json)
        # /debug/prompt only counts the images, so send cheap placeholders
        # instead of megabytes of base64 — but count exactly what /infer would.
        n = 0
        for i in range(len(paths)):
            box = boxes.get(i)
            if send_mode == SEND_FULL or box is None:
                n += 1
            elif send_mode == SEND_BOTH:
                n += 2
            else:
                n += 1
        payload = {
            "model":  model,
            "prompt": prompt or "",
            "images": ["x" * 4 for _ in range(n)] or None,
        }
        if (system or "").strip():
            payload["system"] = system.strip()
        r = _post(gpu_url, "/debug/prompt", payload, read_timeout=300)
    except Exception as exc:
        return f"**Error** · {_err(exc)}"

    body = r.get("prompt", "")
    note = ""
    if r.get("templated"):
        has_ph = ("<|image_pad|>" in body) or ("<image>" in body) or ("IMG_CONTEXT" in body)
        if n and not has_ph:
            note = ("**No vision placeholder in the templated prompt** — the model "
                    "will not see the image. Check `prompt_style` on the server.\n")
        elif n:
            note = f"Vision placeholders present for {r.get('images')} image(s).\n"
    else:
        note = "Text model — prompt passed through untouched.\n"
    return f"{note}\n```\n{body}\n```"


def on_images_change(files, state_version):
    """Push preview data URIs into the hidden bridge the region tool polls."""
    paths    = _paths_from_files(files)
    previews = []
    for p in paths:
        try:
            previews.append({"name": os.path.basename(p), "uri": _preview_uri(p)})
        except Exception as exc:
            previews.append({"name": f"{os.path.basename(p)} (unreadable: {exc})",
                             "uri": ""})
    version = int(state_version or 0) + 1
    return json.dumps(previews), str(version), version


# ─────────────────────────────── Design system ───────────────────────────────
# Vodafone Idea Design System V.01 — DM Sans, #F4F1EC ground, white surfaces,
# red→yellow primary gradient, documented type scale mapped pt→px.

DS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --vf-bg:          #F4F1EC;
  --vf-surface:     #FFFFFF;
  --vf-grad:        linear-gradient(90deg, #EE3B2F 0%, #F0A202 100%);
  --vf-yellow:      #F0A202;
  --vf-yellow-2:    #FEF3C7;
  --vf-green:       #0DAF94;
  --vf-green-2:     #B4E6DD;
  --vf-red:         #EE3B2F;
  --vf-red-2:       #FFF1F2;
  --vf-blue:        #5F5FEF;
  --vf-blue-2:      #EEF2FF;
  --vf-black:       #1A1917;
  --vf-black-2:     #3D3B37;
  --vf-grey:        #6B685F;
  --vf-line:        #E2DDD4;
  --vf-radius:      14px;
  --vf-font: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body, .gradio-container { background: var(--vf-bg) !important; font-family: var(--vf-font) !important; }
.gradio-container { max-width: 1560px !important; color: var(--vf-black) !important; }
.gradio-container *, .gradio-container button, .gradio-container input,
.gradio-container textarea, .gradio-container select { font-family: var(--vf-font) !important; }
footer { display: none !important; }

/* ── Masthead ─────────────────────────────────────────────────────────── */
.vf-head { padding: 4px 2px 0; }
.vf-head h1 {
  font-size: 40px; line-height: 1.15; font-weight: 400;
  color: var(--vf-black); margin: 0 0 4px;
}
.vf-head h1 .vf-accent { color: var(--vf-red); font-weight: 700; }
.vf-head p { font-size: 15px; color: var(--vf-grey); margin: 0; }
.vf-head .vf-rule {
  height: 3px; width: 96px; border-radius: 2px; margin: 14px 0 2px;
  background: var(--vf-grad);
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.vf-card {
  background: var(--vf-surface); border: 1px solid var(--vf-line);
  border-radius: var(--vf-radius); padding: 20px !important; gap: 14px !important;
  box-shadow: 0 1px 2px rgba(26,25,23,.04);
}
.vf-card-title, .vf-card-title p {
  font-size: 12px !important; letter-spacing: .09em; text-transform: uppercase;
  color: var(--vf-grey) !important; font-weight: 500; margin: 0 0 2px !important;
}

/* ── Status pill ──────────────────────────────────────────────────────── */
.vf-status, .vf-status p {
  font-size: 13px !important; color: var(--vf-black-2) !important; margin: 0 !important;
}
.vf-status { background: var(--vf-surface); border: 1px solid var(--vf-line);
  border-radius: 999px; padding: 9px 16px !important; }
.vf-status strong { font-weight: 500; }
.vf-status code { background: transparent; color: var(--vf-grey); font-size: 12px; }

/* ── Controls ─────────────────────────────────────────────────────────── */
.gradio-container label, .gradio-container .label-wrap span,
span[data-testid="block-info"] {
  font-size: 13px !important; color: var(--vf-black-2) !important; font-weight: 500 !important;
}
.gradio-container input[type=text], .gradio-container textarea,
.gradio-container .wrap-inner, .gradio-container select {
  background: var(--vf-surface) !important; border-radius: 10px !important;
  border-color: var(--vf-line) !important; color: var(--vf-black) !important;
  font-size: 15px !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
  border-color: var(--vf-yellow) !important; box-shadow: 0 0 0 3px rgba(240,162,2,.16) !important;
}
.gradio-container button { border-radius: 10px !important; font-size: 15px !important; }

.vf-primary button, button.vf-primary {
  background: var(--vf-grad) !important; color: #FFFFFF !important;
  border: none !important; font-weight: 500 !important; padding: 12px 20px !important;
}
.vf-primary button:hover, button.vf-primary:hover { filter: brightness(1.06); }
.vf-ghost button, button.vf-ghost {
  background: var(--vf-surface) !important; color: var(--vf-black-2) !important;
  border: 1px solid var(--vf-line) !important;
}
.vf-ghost button:hover, button.vf-ghost:hover { border-color: var(--vf-grey) !important; }

/* ── Output ───────────────────────────────────────────────────────────── */
.vf-output textarea {
  font-size: 15px !important; line-height: 1.6 !important; background: var(--vf-bg) !important;
}
.vf-stats, .vf-stats p { font-size: 13px !important; color: var(--vf-grey) !important; }
.vf-stats strong { color: var(--vf-black); font-weight: 500; }
.vf-stats table { font-size: 13px; border-collapse: collapse; }
.vf-stats th, .vf-stats td { border: 1px solid var(--vf-line); padding: 6px 10px; }

.vf-hidden { display: none !important; }

/* Gradio 6 renders a gr.Group as two nested nodes and puts elem_classes on
   both, which draws our card frame twice. Keep the outer one only. */
.vf-card .vf-card {
  border: none !important; padding: 0 !important; box-shadow: none !important;
  border-radius: 0 !important;
}
.column > .vf-card { height: 100%; }
.vf-row-main { align-items: stretch !important; }

/* Gradio wraps every block in a .styler panel and paints columns/rows; inside a
   card that stacks a second frame on ours. Flatten it. */
.vf-card .styler { background: transparent !important; border: none !important; }
.gradio-container .column, .gradio-container .row,
.gradio-container .form, .gradio-container .panel {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
.vf-card .block { border-width: 0 !important; background: transparent !important; }
.vf-file, .vf-card .vf-file.block {
  border: 1px solid var(--vf-line) !important; border-radius: 12px !important;
  background: var(--vf-surface) !important;
}
.gradio-container .accordion, .gradio-container .label-wrap {
  background: transparent !important;
}

/* ── Region tool ──────────────────────────────────────────────────────── */
#vf_viewer { width: 100%; }
.vf-empty {
  border: 1px dashed var(--vf-line); border-radius: 12px; padding: 40px 20px;
  text-align: center; color: var(--vf-grey); font-size: 13px; background: var(--vf-bg);
}
.vf-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.vf-thumb {
  position: relative; width: 54px; height: 54px; border-radius: 9px; overflow: hidden;
  border: 2px solid var(--vf-line); cursor: pointer; background: var(--vf-bg); padding: 0;
}
.vf-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.vf-thumb.is-active { border-color: var(--vf-red); }
.vf-thumb .vf-dot {
  position: absolute; right: 3px; bottom: 3px; width: 9px; height: 9px;
  border-radius: 50%; background: var(--vf-green); border: 1.5px solid #fff;
}
.vf-stage {
  background: var(--vf-bg); border: 1px solid var(--vf-line); border-radius: 12px;
  padding: 10px; display: flex; justify-content: center;
}
.vf-wrap { position: relative; display: inline-block; line-height: 0; cursor: crosshair;
  touch-action: none; user-select: none; }
.vf-wrap img { max-width: 100%; max-height: 460px; display: block; border-radius: 6px; }
.vf-shade { position: absolute; inset: 0; background: rgba(26,25,23,.42); pointer-events: none;
  border-radius: 6px; }
.vf-box { position: absolute; border: 2px solid var(--vf-yellow);
  box-shadow: 0 0 0 9999px rgba(0,0,0,0); cursor: move; }
.vf-box::after { content: ''; position: absolute; inset: 0;
  outline: 1px solid rgba(255,255,255,.75); outline-offset: -3px; }
.vf-h { position: absolute; width: 12px; height: 12px; background: #fff;
  border: 2px solid var(--vf-red); border-radius: 3px; }
.vf-h[data-h=nw]{left:-7px;top:-7px;cursor:nwse-resize}
.vf-h[data-h=n] {left:calc(50% - 6px);top:-7px;cursor:ns-resize}
.vf-h[data-h=ne]{right:-7px;top:-7px;cursor:nesw-resize}
.vf-h[data-h=e] {right:-7px;top:calc(50% - 6px);cursor:ew-resize}
.vf-h[data-h=se]{right:-7px;bottom:-7px;cursor:nwse-resize}
.vf-h[data-h=s] {left:calc(50% - 6px);bottom:-7px;cursor:ns-resize}
.vf-h[data-h=sw]{left:-7px;bottom:-7px;cursor:nesw-resize}
.vf-h[data-h=w] {left:-7px;top:calc(50% - 6px);cursor:ew-resize}
.vf-tag {
  position: absolute; top: -30px; left: 0; background: var(--vf-black); color: #fff;
  font-size: 11px; padding: 4px 8px; border-radius: 6px; white-space: nowrap; line-height: 1.3;
}
.vf-bar { display: flex; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.vf-btn {
  font-size: 13px; padding: 7px 14px; border-radius: 9px; border: 1px solid var(--vf-line);
  background: var(--vf-surface); color: var(--vf-black-2); cursor: pointer;
  font-family: var(--vf-font);
}
.vf-btn:hover { border-color: var(--vf-grey); }
.vf-note { font-size: 12px; color: var(--vf-grey); margin-left: auto; }
"""

# The region tool. Lives in <head> so the browser actually executes it —
# scripts inside a gr.HTML value are inserted as markup and never run.
#
# It talks to Python through three CSS-hidden textboxes:
#   #vf_version  int, bumped by Python whenever the image set changes
#   #vf_images   JSON [{name, uri}] previews, read when the version changes
#   #vf_boxes    JSON {"boxes": {idx: [x,y,w,h]}, "active": n} written by JS
# Boxes are normalized 0-1, so Python can crop the full-resolution original.
DS_JS = """
<script>
(function () {
  const S = { images: [], boxes: {}, active: 0, version: null, drag: null };

  const q  = (s) => document.querySelector(s);
  const ta = (id) => q('#' + id + ' textarea') || q('#' + id + ' input');

  function pushState() {
    const el = ta('vf_boxes');
    if (!el) return;
    const setter = Object.getOwnPropertyDescriptor(
      el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype
                                : window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, JSON.stringify({ boxes: S.boxes, active: S.active }));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function pt(ev, img) {
    const r = img.getBoundingClientRect();
    return { x: clamp01((ev.clientX - r.left) / r.width),
             y: clamp01((ev.clientY - r.top) / r.height) };
  }

  function render() {
    const root = q('#vf_viewer');
    if (!root) return;

    if (!S.images.length) {
      root.innerHTML = '<div class="vf-empty">Upload an image to draw a region on it.</div>';
      return;
    }
    if (S.active >= S.images.length) S.active = 0;

    const strip = S.images.map((im, i) =>
      '<button class="vf-thumb' + (i === S.active ? ' is-active' : '') + '" data-i="' + i + '"' +
      ' title="' + (im.name || '') + '"><img src="' + im.uri + '" alt="">' +
      (S.boxes[i] ? '<span class="vf-dot"></span>' : '') + '</button>').join('');

    root.innerHTML =
      (S.images.length > 1 ? '<div class="vf-strip">' + strip + '</div>' : '') +
      '<div class="vf-stage"><div class="vf-wrap">' +
        '<img src="' + S.images[S.active].uri + '" draggable="false" alt="">' +
      '</div></div>' +
      '<div class="vf-bar">' +
        '<button type="button" class="vf-btn" data-act="clear">Clear region</button>' +
        '<button type="button" class="vf-btn" data-act="clearall">Clear all regions</button>' +
        '<span class="vf-note">Drag on the image to draw · drag inside to move · handles to resize</span>' +
      '</div>';

    root.querySelectorAll('.vf-thumb').forEach((b) => b.addEventListener('click', () => {
      S.active = +b.dataset.i; pushState(); render();
    }));
    root.querySelector('[data-act=clear]').addEventListener('click', () => {
      delete S.boxes[S.active]; pushState(); render();
    });
    root.querySelector('[data-act=clearall]').addEventListener('click', () => {
      S.boxes = {}; pushState(); render();
    });

    const img = root.querySelector('.vf-wrap img');
    if (img.complete) drawBox(); else img.addEventListener('load', drawBox);
    root.querySelector('.vf-wrap').addEventListener('pointerdown', onDown);
  }

  function drawBox() {
    const root = q('#vf_viewer');
    const wrap = root && root.querySelector('.vf-wrap');
    if (!wrap) return;
    wrap.querySelectorAll('.vf-box, .vf-shade').forEach((n) => n.remove());

    const b = S.boxes[S.active];
    if (!b) return;
    const img = wrap.querySelector('img');

    const shade = document.createElement('div');
    shade.className = 'vf-shade';
    shade.style.clipPath =
      'polygon(0% 0%, 0% 100%, ' + pct(b[0]) + ' 100%, ' + pct(b[0]) + ' ' + pct(b[1]) + ', ' +
      pct(b[0] + b[2]) + ' ' + pct(b[1]) + ', ' + pct(b[0] + b[2]) + ' ' + pct(b[1] + b[3]) + ', ' +
      pct(b[0]) + ' ' + pct(b[1] + b[3]) + ', ' + pct(b[0]) + ' 100%, 100% 100%, 100% 0%)';
    wrap.appendChild(shade);

    const box = document.createElement('div');
    box.className = 'vf-box';
    box.style.left   = pct(b[0]);
    box.style.top    = pct(b[1]);
    box.style.width  = pct(b[2]);
    box.style.height = pct(b[3]);

    const w = Math.round(b[2] * (img.naturalWidth  || 0));
    const h = Math.round(b[3] * (img.naturalHeight || 0));
    box.innerHTML = '<span class="vf-tag">' + w + ' x ' + h + ' px</span>' +
      ['nw','n','ne','e','se','s','sw','w']
        .map((k) => '<span class="vf-h" data-h="' + k + '"></span>').join('');
    wrap.appendChild(box);
  }

  const pct = (v) => (v * 100).toFixed(4) + '%';

  function onDown(ev) {
    if (ev.button !== 0) return;
    const wrap = ev.currentTarget;
    const img  = wrap.querySelector('img');
    const p    = pt(ev, img);
    const h    = ev.target.closest('.vf-h');
    const box  = ev.target.closest('.vf-box');
    const b    = S.boxes[S.active];

    if (h && b)        S.drag = { mode: 'resize', h: h.dataset.h, img: img };
    else if (box && b) S.drag = { mode: 'move', img: img, ox: p.x - b[0], oy: p.y - b[1] };
    else {
      S.drag = { mode: 'draw', img: img, sx: p.x, sy: p.y };
      S.boxes[S.active] = [p.x, p.y, 0, 0];
    }

    wrap.setPointerCapture(ev.pointerId);
    wrap.addEventListener('pointermove', onMove);
    wrap.addEventListener('pointerup', onUp);
    wrap.addEventListener('pointercancel', onUp);
    ev.preventDefault();
  }

  function onMove(ev) {
    const d = S.drag;
    if (!d) return;
    const p = pt(ev, d.img);
    let b = S.boxes[S.active];

    if (d.mode === 'draw') {
      b = [Math.min(d.sx, p.x), Math.min(d.sy, p.y),
           Math.abs(p.x - d.sx), Math.abs(p.y - d.sy)];
    } else if (d.mode === 'move') {
      b = [clamp01(Math.min(p.x - d.ox, 1 - b[2])),
           clamp01(Math.min(p.y - d.oy, 1 - b[3])), b[2], b[3]];
    } else {
      let l = b[0], t = b[1], r = b[0] + b[2], bo = b[1] + b[3];
      if (d.h.includes('w')) l = Math.min(p.x, r - 0.005);
      if (d.h.includes('e')) r = Math.max(p.x, l + 0.005);
      if (d.h.includes('n')) t = Math.min(p.y, bo - 0.005);
      if (d.h.includes('s')) bo = Math.max(p.y, t + 0.005);
      b = [l, t, r - l, bo - t];
    }
    S.boxes[S.active] = b.map((v) => +v.toFixed(6));
    drawBox();
  }

  function onUp(ev) {
    const wrap = ev.currentTarget;
    wrap.removeEventListener('pointermove', onMove);
    wrap.removeEventListener('pointerup', onUp);
    wrap.removeEventListener('pointercancel', onUp);
    S.drag = null;

    const b = S.boxes[S.active];
    if (b && (b[2] < 0.004 || b[3] < 0.004)) delete S.boxes[S.active];  // stray click
    pushState();
    render();
  }

  // Poll the version bridge: cheap, and immune to Gradio's event wiring.
  setInterval(function () {
    const v = ta('vf_version');
    if (!v || v.value === S.version) return;
    S.version = v.value;
    try { S.images = JSON.parse(ta('vf_images').value || '[]'); }
    catch (e) { S.images = []; }
    Object.keys(S.boxes).forEach((k) => { if (+k >= S.images.length) delete S.boxes[k]; });
    if (S.active >= S.images.length) S.active = 0;
    pushState();
    render();
  }, 250);
})();
</script>
"""


# ─────────────────────────────── UI ──────────────────────────────────────────

def _theme_kwargs() -> dict:
    """
    Gradio 6 moved `css` and `head` from the Blocks constructor to launch();
    Gradio 4 and 5 only accept them on Blocks. Send them wherever this
    installation actually reads them.
    """
    try:
        major = int(str(gr.__version__).split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return {"blocks": {}, "launch": {"css": DS_CSS, "head": DS_JS}} if major >= 6 \
        else {"blocks": {"css": DS_CSS, "head": DS_JS}, "launch": {}}


def build_ui(gpu_url_default: str) -> "gr.Blocks":
    with gr.Blocks(title="VLM Test Client", **_theme_kwargs()["blocks"]) as demo:

        gr.HTML(
            '<div class="vf-head"><h1><span class="vf-accent">Vision</span> Model '
            'Test Client</h1><p>Upload an image, draw a region, and question it '
            'through the GPU inference API.</p><div class="vf-rule"></div></div>'
        )

        with gr.Row(equal_height=True):
            gpu_url = gr.Textbox(label="GPU server URL", value=gpu_url_default,
                                 scale=5, container=True,
                                 placeholder="http://10.0.0.5:5432")
            model   = gr.Dropdown(label="Model", choices=[], value=None,
                                  interactive=True, scale=3)
            refresh_btn = gr.Button("Refresh", scale=1, elem_classes="vf-ghost")

        status = gr.Markdown("Not connected · press **Refresh** to load the model registry.",
                             elem_classes="vf-status")

        # Hidden bridge between Python and the region tool (see DS_JS).
        state_version = gr.State(0)
        vf_version = gr.Textbox(value="0", elem_id="vf_version", elem_classes="vf-hidden")
        vf_images  = gr.Textbox(value="[]", elem_id="vf_images",  elem_classes="vf-hidden")
        vf_boxes   = gr.Textbox(value="{}", elem_id="vf_boxes",   elem_classes="vf-hidden")

        with gr.Row(elem_classes="vf-row-main"):
            # ── Left: image + region ────────────────────────────────────────
            with gr.Column(scale=5):
                with gr.Group(elem_classes="vf-card"):
                    gr.Markdown("Image & region", elem_classes="vf-card-title")
                    images = gr.File(label="Images (max 4 per prompt for both VLMs)",
                                     file_count="multiple", file_types=["image"],
                                     type="filepath", elem_classes="vf-file")
                    gr.HTML('<div id="vf_viewer"></div>')
                    send_mode = gr.Radio(
                        [SEND_FULL, SEND_REGION, SEND_BOTH],
                        value=SEND_REGION, label="What to send",
                        info="Regions are cropped from the original file at full "
                             "resolution. Images with no region always send whole.",
                    )

            # ── Middle: prompt ──────────────────────────────────────────────
            with gr.Column(scale=4):
                with gr.Group(elem_classes="vf-card"):
                    gr.Markdown("Prompt", elem_classes="vf-card-title")
                    prompt = gr.Textbox(label="Question", lines=5,
                                        value="Describe this image in detail.")
                    system = gr.Textbox(
                        label="System prompt (optional)", lines=2,
                        placeholder="You are a precise document-understanding assistant.")
                    stream = gr.Checkbox(label="Stream tokens (SSE)", value=True)

                    with gr.Accordion("Sampling parameters", open=False):
                        max_new_tokens = gr.Slider(1, 8192, value=1024, step=1,
                                                   label="max_new_tokens")
                        temperature    = gr.Slider(0.0, 2.0, value=0.2, step=0.05,
                                                   label="temperature")
                        top_p          = gr.Slider(0.0, 1.0, value=0.9, step=0.05,
                                                   label="top_p")
                        top_k          = gr.Slider(0, 200, value=50, step=1,
                                                   label="top_k (0 = disabled)")
                        repetition_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.05,
                                                       label="repetition_penalty")
                        stop_sequences = gr.Textbox(
                            label="Stop sequences (one per line)", lines=2)

                    with gr.Row():
                        run_btn   = gr.Button("Generate", scale=3, elem_classes="vf-primary")
                        debug_btn = gr.Button("Debug prompt", scale=2, elem_classes="vf-ghost")

            # ── Right: output ───────────────────────────────────────────────
            with gr.Column(scale=5):
                with gr.Group(elem_classes="vf-card"):
                    gr.Markdown("Response", elem_classes="vf-card-title")
                    output = gr.Textbox(label=None, show_label=False, lines=18,
                                        elem_classes="vf-output",
                                        placeholder="The model's answer appears here.")
                    stats  = gr.Markdown(elem_classes="vf-stats")

                    with gr.Accordion("Images actually sent", open=False):
                        sent_preview = gr.Gallery(show_label=False, columns=4,
                                                  height=170, object_fit="contain")
                    with gr.Accordion("Templated prompt (/debug/prompt)", open=False):
                        debug_out = gr.Markdown(elem_classes="vf-stats")
                    with gr.Accordion("Request sent (base64 redacted)", open=False):
                        sent_json = gr.Code(language="json")

        with gr.Group(elem_classes="vf-card"):
            gr.Markdown("Server", elem_classes="vf-card-title")
            with gr.Row():
                health_btn  = gr.Button("Health / VRAM", elem_classes="vf-ghost")
                metrics_btn = gr.Button("Metrics", elem_classes="vf-ghost")
                load_btn    = gr.Button("Load selected", elem_classes="vf-ghost")
                unload_btn  = gr.Button("Unload selected", elem_classes="vf-ghost")
            server_out  = gr.Markdown(elem_classes="vf-stats")
            registry_md = gr.Markdown(elem_classes="vf-stats")
            gr.Markdown(
                "`qwen3-vl` and `internvl` share an eviction group on the server: "
                "loading one automatically unloads the other. `mistral` stays resident.",
                elem_classes="vf-stats",
            )

        # ── Wiring ──────────────────────────────────────────────────────────
        refresh_btn.click(do_refresh, [gpu_url], [model, registry_md, status])
        demo.load(do_refresh, [gpu_url], [model, registry_md, status])

        images.change(on_images_change, [images, state_version],
                      [vf_images, vf_version, state_version])

        run_btn.click(
            run_inference,
            [gpu_url, model, prompt, system, images, vf_boxes, send_mode, stream,
             max_new_tokens, temperature, top_p, top_k, repetition_penalty,
             stop_sequences],
            [output, stats, sent_json, sent_preview],
        )
        debug_btn.click(do_debug_prompt,
                        [gpu_url, model, prompt, system, images, vf_boxes, send_mode],
                        [debug_out])

        health_btn.click(do_health,   [gpu_url],        [server_out])
        metrics_btn.click(do_metrics, [gpu_url],        [server_out])
        load_btn.click(do_load,       [gpu_url, model], [server_out])
        unload_btn.click(do_unload,   [gpu_url, model], [server_out])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradio test client for gpu_api_server_v6")
    ap.add_argument("--gpu-url", default=DEFAULT_GPU_URL,
                    help=f"Base URL of the GPU API server (default: {DEFAULT_GPU_URL})")
    ap.add_argument("--host", default="0.0.0.0", help="Bind address for this UI")
    ap.add_argument("--port", type=int, default=7860, help="Port for this UI")
    ap.add_argument("--share", action="store_true",
                    help="Expose a temporary public gradio.live URL")
    args = ap.parse_args()

    print(f"GPU API server : {args.gpu_url}")
    print(f"UI             : http://{args.host}:{args.port}")

    build_ui(args.gpu_url).queue().launch(
        server_name=args.host, server_port=args.port, share=args.share,
        **_theme_kwargs()["launch"],
    )


if __name__ == "__main__":
    main()
