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
  * Reads the live model registry from /models so the dropdown always matches
    the server's MODEL_CONFIGS (no hardcoded model list to drift).
  * Enforces the per-model image cap (limit_mm_per_prompt) client-side so you
    get a useful error instead of a 400 from the GPU box.
  * Manual load / unload buttons — handy because the two VLMs share an
    eviction group and loading one drops the other.
  * /debug/prompt viewer, so when a VLM answers as if it never saw the image
    you can check the vision placeholders are actually in the templated prompt.
  * Health (VRAM, loaded models) and /metrics panels.

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

# Generous: a cold VLM load (30B/38B, TP=2) can take several minutes.
CONNECT_TIMEOUT_S  = 10
READ_TIMEOUT_S     = 900


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

def _encode_image(path: str) -> str:
    """Local file → 'data:image/...;base64,...' data URI, downscaled if huge."""
    from PIL import Image

    with Image.open(path) as img:
        img.load()
        w, h = img.size
        needs_resize = max(w, h) > MAX_UPLOAD_SIDE_PX

        if not needs_resize:
            # Ship the original bytes — no re-encode, no quality loss.
            with open(path, "rb") as fh:
                raw = fh.read()
            mime = mimetypes.guess_type(path)[0] or "image/png"
            return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

        scale = MAX_UPLOAD_SIDE_PX / max(w, h)
        img = img.convert("RGB").resize(
            (max(1, int(w * scale)), max(1, int(h * scale)))
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
            f"| `{m['name']}` | {m['modality']} | {'✅' if m['loaded'] else '—'} | "
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

def _build_payload(model, prompt, system, image_paths,
                   max_new_tokens, temperature, top_p, top_k,
                   repetition_penalty, stop_sequences) -> dict:
    if not model:
        raise ValueError("Pick a model first (hit 'Refresh' if the list is empty).")
    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")

    info      = _registry.get(model, {})
    modality  = info.get("modality", "text")

    if image_paths and modality != "vision":
        raise ValueError(
            f"'{model}' is text-only — it will reject images. Vision models: "
            + ", ".join(n for n, i in _registry.items() if i["modality"] == "vision")
        )

    cap = info.get("max_images")
    if image_paths and cap and len(image_paths) > cap:
        raise ValueError(
            f"'{model}' accepts at most {cap} image(s) per prompt "
            f"(limit_mm_per_prompt on the server); you attached {len(image_paths)}."
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
    if image_paths:
        payload["images"] = [_encode_image(p) for p in image_paths]
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


def run_inference(gpu_url, model, prompt, system, files, stream,
                  max_new_tokens, temperature, top_p, top_k,
                  repetition_penalty, stop_sequences):
    """Main generate handler. Yields (output_text, stats_markdown, request_json)."""
    try:
        paths   = _paths_from_files(files)
        payload = _build_payload(model, prompt, system, paths, max_new_tokens,
                                 temperature, top_p, top_k, repetition_penalty,
                                 stop_sequences)
    except Exception as exc:
        yield "", f"**Error:** {_err(exc)}", ""
        return

    sent = json.dumps(_redact(payload), indent=2)
    n_img = len(paths)
    yield "", f"Sending to `{model}` ({n_img} image(s))…", sent

    t0 = time.time()

    if stream:
        text = ""
        try:
            for delta in _stream_infer(gpu_url, payload):
                text += delta
                yield text, f"streaming… {len(text)} chars, {time.time() - t0:.1f}s", sent
        except Exception as exc:
            yield text, f"**Error:** {_err(exc)}", sent
            return
        yield text, (f"**done** (streamed) · {n_img} image(s) · "
                     f"{time.time() - t0:.2f}s · {len(text)} chars"), sent
        return

    try:
        resp = _post(gpu_url, "/infer", payload)
    except Exception as exc:
        yield "", f"**Error:** {_err(exc)}", sent
        return

    stats = (
        f"**model** `{resp['model']}` · **images** {resp.get('images', 0)}\n\n"
        f"**prompt tokens** {resp['prompt_tokens']} · "
        f"**new tokens** {resp['new_tokens']}\n\n"
        f"**elapsed** {resp['elapsed_s']}s · **{resp['tokens_per_sec']} tok/s** "
        f"(wall {time.time() - t0:.2f}s)\n\n"
        f"`request_id={resp.get('request_id')}`"
    )
    yield resp["text"], stats, sent


# ─────────────────────────────── Panel actions ───────────────────────────────

def do_refresh(gpu_url):
    try:
        names, table = _refresh_registry(gpu_url)
        names = _vision_first(names)
        default = names[0] if names else None
        return (gr.update(choices=names, value=default),
                table,
                f"Connected to `{gpu_url}` — {len(names)} model(s).")
    except Exception as exc:
        return gr.update(), "", f"**Error:** {_err(exc)}"


def do_health(gpu_url):
    try:
        h = _get(gpu_url, "/health")
    except Exception as exc:
        return f"**Error:** {_err(exc)}"

    lines = [
        f"**status** {h['status']} · **CUDA** {h['cuda_available']} · "
        f"**GPUs** {h['gpu_count']}",
        "",
        f"**loaded** {', '.join(h['loaded_models']) or '(none)'}",
        f"**vision models** {', '.join(h['vision_models'])}",
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
        lines += ["", "**failed loads**", "```", json.dumps(h["failed_models"], indent=2), "```"]
    return "\n".join(lines)


def do_metrics(gpu_url):
    try:
        return "```json\n" + json.dumps(_get(gpu_url, "/metrics"), indent=2) + "\n```"
    except Exception as exc:
        return f"**Error:** {_err(exc)}"


def do_load(gpu_url, model):
    if not model:
        return "Pick a model first."
    try:
        r = _post(gpu_url, f"/models/{model}/load", {})
        return f"`{model}`: **{r['status']}**\n\n```json\n{json.dumps(r['vram'], indent=2)}\n```"
    except Exception as exc:
        return f"**Error:** {_err(exc)}"


def do_unload(gpu_url, model):
    if not model:
        return "Pick a model first."
    try:
        r = _post(gpu_url, f"/models/{model}/unload", {}, read_timeout=300)
        return f"`{model}`: **{r['status']}**\n\n```json\n{json.dumps(r['vram'], indent=2)}\n```"
    except Exception as exc:
        return f"**Error:** {_err(exc)}"


def do_debug_prompt(gpu_url, model, prompt, system, files):
    """Render the templated prompt without touching the GPU."""
    try:
        paths = _paths_from_files(files)
        # /debug/prompt only counts the images, so send cheap placeholders
        # instead of megabytes of base64.
        payload = {
            "model":  model,
            "prompt": prompt or "",
            "images": ["x" * 4 for _ in paths] or None,
        }
        if (system or "").strip():
            payload["system"] = system.strip()
        r = _post(gpu_url, "/debug/prompt", payload, read_timeout=300)
    except Exception as exc:
        return f"**Error:** {_err(exc)}"

    body = r.get("prompt", "")
    note = ""
    if r.get("templated"):
        has_ph = ("<|image_pad|>" in body) or ("<image>" in body) or ("IMG_CONTEXT" in body)
        if paths and not has_ph:
            note = ("\n\n⚠️ No vision placeholder in the templated prompt — the model "
                    "will not see the image. Check `prompt_style` on the server.\n")
        elif paths:
            note = f"\n\n✅ Vision placeholders present for {r.get('images')} image(s).\n"
    else:
        note = "\n\n(text model — prompt passed through untouched)\n"
    return f"{note}\n```\n{body}\n```"


# ─────────────────────────────── UI ──────────────────────────────────────────

def build_ui(gpu_url_default: str) -> "gr.Blocks":
    with gr.Blocks(title="VLM Test Client") as demo:
        gr.Markdown(
            "# VLM Test Client\n"
            "Upload an image, pick a vision model, and test it through "
            "`gpu_api_server_v6` on port 5432."
        )

        with gr.Row():
            gpu_url = gr.Textbox(
                label="GPU server URL", value=gpu_url_default, scale=4,
                placeholder="http://10.0.0.5:5432",
            )
            refresh_btn = gr.Button("🔄 Refresh models", scale=1)

        status = gr.Markdown("Hit **Refresh models** to connect.")

        with gr.Row():
            # ── Left: inputs ────────────────────────────────────────────────
            with gr.Column(scale=1):
                model = gr.Dropdown(label="Model", choices=[], value=None,
                                    interactive=True)

                images = gr.File(
                    label="Images (uploaded as base64 — max 4 for both VLMs)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                gallery = gr.Gallery(label="Attached", columns=4, height=160,
                                     show_label=True)

                prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    value="Describe this image in detail.",
                )
                system = gr.Textbox(
                    label="System prompt (optional)", lines=2,
                    placeholder="You are a precise document-understanding assistant.",
                )

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
                        label="Stop sequences (one per line)", lines=2,
                    )

                with gr.Row():
                    run_btn   = gr.Button("▶ Generate", variant="primary", scale=2)
                    debug_btn = gr.Button("🔍 Debug prompt", scale=1)

            # ── Right: outputs ──────────────────────────────────────────────
            with gr.Column(scale=1):
                output = gr.Textbox(label="Model output", lines=22)
                stats  = gr.Markdown()

                with gr.Accordion("Templated prompt (/debug/prompt)", open=False):
                    debug_out = gr.Markdown()

                with gr.Accordion("Request sent (base64 redacted)", open=False):
                    sent_json = gr.Code(language="json")

        with gr.Accordion("Server panel", open=False):
            with gr.Row():
                health_btn  = gr.Button("❤️ Health / VRAM")
                metrics_btn = gr.Button("📊 Metrics")
                load_btn    = gr.Button("⬆️ Load selected")
                unload_btn  = gr.Button("⬇️ Unload selected")
            server_out = gr.Markdown()
            gr.Markdown(
                "_`qwen3-vl` and `internvl` share an eviction group on the server: "
                "loading one automatically unloads the other. `mistral` stays resident._"
            )
            registry_md = gr.Markdown()

        # ── Wiring ──────────────────────────────────────────────────────────
        refresh_btn.click(do_refresh, [gpu_url], [model, registry_md, status])
        demo.load(do_refresh, [gpu_url], [model, registry_md, status])

        images.change(lambda f: _paths_from_files(f), [images], [gallery])

        run_btn.click(
            run_inference,
            [gpu_url, model, prompt, system, images, stream, max_new_tokens,
             temperature, top_p, top_k, repetition_penalty, stop_sequences],
            [output, stats, sent_json],
        )
        debug_btn.click(do_debug_prompt, [gpu_url, model, prompt, system, images],
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
    )


if __name__ == "__main__":
    main()
