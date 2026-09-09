#!/usr/bin/env python3
"""Demo-machine preflight. Run this on the ACTUAL demo laptop, on the demo
room's network, before the demo - not on a dev box.

    python tools/preflight.py
    python tools/preflight.py --gpu-url http://10.66.98.137:5432 --offline-check

Every check prints OK / WARN / FAIL and the script exits non-zero if anything
FAILed, so it can gate a rehearsal. Nothing here imports the pipeline's heavy
modules unless the corresponding check needs them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# sha256 of the u2netp.onnx supplied for this demo (4,574,861 bytes).
# If the file on the demo machine differs, it is a different build of the model
# and the scores it produces will not match what was calibrated.
EXPECTED_U2NETP_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
EXPECTED_U2NETP_BYTES = 4574861

# Ports the four UIs bind, per plan section 8 trap 7. review_ui.py's own default
# (7861) is included - it is missing from the plan's list.
PORTS = {
    8056: "batch_ui.py (Anu, quality gate)",
    8050: "app.py (Sudh, Dash detection demo)",
    7860: "vlm_test_client_v2.py (Adnaan, VLM client)",
    7861: "review_ui.py (manual review)",
    7870: "demo_app.py (NEW integrated demo UI)",
}

_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")
    _warnings.append(msg)


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ─────────────────────────────── Checks ──────────────────────────────────────

def check_python() -> None:
    section("Python")
    v = sys.version_info
    ok(f"python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if v < (3, 9):
        fail("the codebase uses PEP 585 / 604 syntax (list[str], str | None); needs python >= 3.9")


def check_imports() -> None:
    section("Dependencies")
    required = ["cv2", "numpy", "PIL", "pandas", "requests", "gradio"]
    optional = ["rembg", "onnxruntime", "torch", "ultralytics", "dash"]
    missing_required = []
    for mod in required:
        try:
            m = __import__(mod)
            ok(f"{mod} {getattr(m, '__version__', '(no __version__)')}")
        except ImportError as exc:
            fail(f"{mod} is missing - {exc}")
            missing_required.append(mod)

    # requests is a dependency of almost everything here. If even it is absent,
    # this is an empty virtualenv rather than a set of individual gaps - one
    # cause, not six, and the fix is different.
    if "requests" in missing_required and len(missing_required) >= 4:
        print(f"\n  NOTE  {len(missing_required)} of {len(required)} core packages are missing, "
              f"including requests.\n"
              f"        This looks like an EMPTY virtualenv ({sys.prefix}),\n"
              f"        not a machine missing individual packages.\n"
              f"        Prefer cloning the environment that already runs the existing tools\n"
              f"        (pip freeze from it) over resolving fresh versions - that also pins\n"
              f"        the one Gradio version both UI patterns are known to work under.")
    for mod in optional:
        try:
            m = __import__(mod)
            ok(f"{mod} {getattr(m, '__version__', '(no __version__)')} (optional)")
        except ImportError:
            if mod in ("rembg", "onnxruntime"):
                fail(f"{mod} is missing - stage 1 cannot segment without it "
                     f"(pip install rembg onnxruntime)")
            else:
                warn(f"{mod} not installed - only needed for use_model=True (real detector)")


def check_gradio_version() -> None:
    section("Gradio version")
    try:
        import gradio as gr
    except ImportError:
        fail("gradio is missing")
        return
    version = str(getattr(gr, "__version__", "unknown"))
    try:
        major = int(version.split(".")[0])
    except ValueError:
        warn(f"could not parse gradio version {version!r}; _theme_kwargs() will assume 4")
        return
    ok(f"gradio {version} (major {major})")
    # batch_ui.py uses gr.skip() and generator yields; vlm_test_client_v2.py has
    # a 4/5-vs-6 shim for where css=/head= are accepted. Both patterns must work
    # under whatever single version is pinned here.
    if not hasattr(gr, "skip"):
        fail("gr.skip() is unavailable - batch_ui.py's _emit() depends on it")
    else:
        ok("gr.skip() available (batch_ui.py's _emit)")
    if major >= 6:
        ok("gradio 6 - _theme_kwargs() will pass css/head to launch()")
    else:
        ok(f"gradio {major} - _theme_kwargs() will pass css/head to Blocks()")


def check_u2netp(offline_check: bool) -> None:
    section("u2netp model file")
    models_dir = PROJECT_ROOT / "models"
    path = models_dir / "u2netp.onnx"
    if not path.exists():
        fail(f"{path} not found - the segmenter would try to DOWNLOAD it on first "
             f"use. The demo must never depend on that.")
        return
    size = path.stat().st_size
    ok(f"{path} exists ({size:,} bytes)")
    if size != EXPECTED_U2NETP_BYTES:
        warn(f"size {size:,} != expected {EXPECTED_U2NETP_BYTES:,}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == EXPECTED_U2NETP_SHA256:
        ok(f"sha256 matches the supplied model ({digest[:16]}...)")
    else:
        warn(f"sha256 {digest[:16]}... != expected {EXPECTED_U2NETP_SHA256[:16]}... "
             f"- a different build of u2netp; scores may not match calibration")

    # rembg reads U2NET_HOME fresh on every call and expects a FLAT directory
    # holding the .onnx directly. foreground_segmentation.py sets it via
    # os.environ.setdefault at import time, so an operator-set value wins.
    home = os.environ.get("U2NET_HOME")
    if home:
        resolved = Path(home).resolve()
        if resolved == models_dir.resolve():
            ok(f"U2NET_HOME already points at {resolved}")
        else:
            warn(f"U2NET_HOME is set to {resolved}, NOT {models_dir.resolve()} - "
                 f"setdefault means this wins; confirm u2netp.onnx is in there too")
    else:
        ok(f"U2NET_HOME unset - foreground_segmentation.py will set it to {models_dir}")

    if offline_check:
        section("u2netp offline load (this is the one that matters)")
        # Only meaningful once rembg is importable. Asking someone to pull the
        # network and then failing with "No module named 'rembg'" tests nothing
        # and wastes a step - check that first.
        try:
            import rembg  # noqa: F401
        except ImportError:
            warn("skipping the offline load check - rembg is not installed, so this "
                 "would only re-report the missing dependency. Install the deps, "
                 "then re-run with --offline-check.")
            return
        print("  Disconnect the network NOW, then press Enter to load the model...")
        try:
            input()
        except EOFError:
            warn("not a TTY - skipping the interactive offline check")
            return
        try:
            os.environ.setdefault("U2NET_HOME", str(models_dir))
            from rembg import new_session
            new_session("u2netp")
            ok("u2netp loaded with the network down - no first-run download")
        except Exception as exc:
            fail(f"u2netp failed to load offline: {exc}")


def check_gpu(gpu_url: str) -> None:
    section(f"GPU server {gpu_url}")
    try:
        import requests
    except ImportError:
        fail("requests is missing - cannot check the GPU server")
        return

    try:
        r = requests.get(f"{gpu_url.rstrip('/')}/health", timeout=(5, 30))
        r.raise_for_status()
        h = r.json()
    except Exception as exc:
        fail(f"/health unreachable: {type(exc).__name__}: {exc}\n"
             f"        The pipeline will fall back to mock mode. Test this from the "
             f"DEMO ROOM's network, not a dev box.")
        return

    ok(f"/health ok - cuda={h.get('cuda_available')} gpus={h.get('gpu_count')}")
    loaded = h.get("loaded_models") or []
    vision = h.get("vision_models") or []
    ok(f"loaded models: {', '.join(loaded) or '(none)'}")
    ok(f"vision models: {', '.join(vision) or '(none)'}")
    if "qwen3-vl" not in loaded:
        warn("qwen3-vl is NOT resident - the first question will pay a cold load "
             "(minutes for a 30B MoE). Load it before the demo and leave it resident: "
             f"curl -XPOST {gpu_url.rstrip('/')}/models/qwen3-vl/load")
    else:
        ok("qwen3-vl is resident - no cold-load stall on the first question")
    if h.get("failed_models"):
        warn(f"failed loads reported: {json.dumps(h['failed_models'])}")

    try:
        r = requests.get(f"{gpu_url.rstrip('/')}/models", timeout=(5, 30))
        r.raise_for_status()
        models = r.json()
    except Exception as exc:
        fail(f"/models unreachable: {exc}")
        return

    # stage3_vlm MUST populate its registry from /models before building any
    # payload: _build_payload() reads a module-global _registry and defaults
    # modality to "text" when it is empty, which makes EVERY image request raise
    # "'<model>' is text-only". Confirm the fields it depends on are present.
    by_name = {m["name"]: m for m in models}
    ok(f"/models returned {len(models)} entries: {', '.join(sorted(by_name))}")
    demo_model = by_name.get("qwen3-vl")
    if not demo_model:
        fail("qwen3-vl is not in /models - the demo model is missing from the registry")
        return
    if demo_model.get("modality") != "vision":
        fail(f"qwen3-vl reports modality={demo_model.get('modality')!r}, expected 'vision' "
             f"- _build_payload() will reject every image request")
    else:
        ok("qwen3-vl modality=vision")
    cap = demo_model.get("max_images")
    if cap:
        ok(f"qwen3-vl max_images={cap} (full+crop sends 2)")
        if cap < 2:
            warn(f"max_images={cap} - 'full+crop' will be downgraded to 'full'")
    else:
        warn("qwen3-vl reports no max_images - the client-side cap will not apply")


def check_ports() -> None:
    section("Ports")
    for port, owner in sorted(PORTS.items()):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.4)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
        s.close()
        if port == 7870:
            if in_use:
                fail(f"{port} ({owner}) is already in use - the demo UI cannot bind")
            else:
                ok(f"{port} free for {owner}")
        else:
            state = "running" if in_use else "not running"
            ok(f"{port} {state} - {owner}")


def check_pipeline_imports() -> None:
    section("Pipeline modules")
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    try:
        from pipeline import config as pcfg
        from pipeline import questions as pq
    except Exception as exc:
        fail(f"could not import the pipeline package: {exc}")
        return
    ok(f"pipeline.schemas / config / questions import cleanly")

    cfg = pcfg.default_config()
    ok(f"project_root resolves to {cfg.project_root}")
    for q in pq.QUESTIONS.values():
        if not q.system_prompt.strip():
            fail(f"question {q.id!r} has an empty system_prompt")
        else:
            ok(f"question {q.id!r}: system prompt {len(q.system_prompt)} chars, "
               f"{len(q.relevant_classes)} relevant classes")
    for problem in cfg.validate():
        warn(problem)


def check_annotations() -> None:
    section("Annotation files (stub detector)")
    labels = PROJECT_ROOT / "data" / "labels"
    if not labels.exists():
        warn(f"{labels} does not exist - every image will report "
             f"'no annotation file found'")
        return
    txts = sorted(p for p in labels.glob("*.txt") if p.name != "classes.txt")
    ok(f"{labels} has {len(txts)} label file(s)")
    if not txts:
        warn("no .txt label files found - the stub detector will return zero detections")
    classes = labels / "classes.txt"
    if classes.exists():
        names = [n for n in classes.read_text().splitlines() if n.strip()]
        ok(f"classes.txt found: {names}")
    else:
        warn("no classes.txt - detections will be labelled 'class_<id>', which is "
             "what the demo audience and the VLM prompt will both see")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-url", default="http://10.66.98.137:5432")
    ap.add_argument("--offline-check", action="store_true",
                    help="interactively verify u2netp loads with the network down")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    print(f"Preflight for {PROJECT_ROOT}")
    check_python()
    check_imports()
    check_gradio_version()
    check_u2netp(args.offline_check)
    check_pipeline_imports()
    check_annotations()
    if not args.skip_gpu:
        check_gpu(args.gpu_url)
    check_ports()

    section("Summary")
    if _failures:
        print(f"  {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"    - {f.splitlines()[0]}")
    if _warnings:
        print(f"  {len(_warnings)} warning(s):")
        for w in _warnings:
            print(f"    - {w.splitlines()[0]}")
    if not _failures and not _warnings:
        print("  All checks passed.")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
