#!/usr/bin/env python3
"""Stage 3 smoke test - ask the real GPU box a real question about a real photo.

    python tools/smoke_stage3.py /data/adnaan/fieldops/demo/photos --limit 2
    python tools/smoke_stage3.py <photos> --question gps_antenna --send full+crop
    python tools/smoke_stage3.py <photos> --mock          # no GPU needed

Runs stage 1 -> stage 2 -> stage 3 for each photo, so the VLM sees the same
EXIF-corrected array everything else did and the detection block carries the
real annotation.

Before any inference it verifies via /debug/prompt that the per-question system
prompt actually reaches the templated prompt. That check exists because trap 9's
failure is SILENT: payload["system"] is only set when non-empty and the server's
prompt builders skip a falsy system turn, so a mis-wired prompt produces an
unframed answer that reads as a model-quality problem.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--question", default="hazard_warning",
                    choices=["hazard_warning", "gps_antenna"])
    ap.add_argument("--send", default="full", choices=["full", "full+crop"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-url", default=None)
    ap.add_argument("--mock", action="store_true", help="skip the GPU entirely")
    ap.add_argument("--skip-quality", action="store_true",
                    help="do not run stage 1 (faster; skips the u2netp load)")
    args = ap.parse_args()

    from pipeline.config import default_config, VLM_MODE_MOCK
    from pipeline.questions import build_detection_block, get_question, select_relevant
    from pipeline.stage2_detect import get_detector
    from pipeline.stage3_vlm import VLMClient
    from quality_check import load_image_bgr

    cfg = default_config(vlm_send_mode=args.send)
    if args.gpu_url:
        cfg.gpu_url = args.gpu_url
    if args.model:
        cfg.vlm_model = args.model
    if args.mock:
        cfg.vlm_mode = VLM_MODE_MOCK

    question = get_question(args.question)
    client = VLMClient(cfg)

    print(f"GPU        : {cfg.gpu_url}")
    print(f"model      : {cfg.vlm_model}")
    print(f"question   : {question.id}  ({question.label})")
    print(f"semantics  : {question.answer_semantics}")
    print(f"send mode  : {cfg.vlm_send_mode}")
    print(f"sampling   : temp={cfg.temperature} top_p={cfg.top_p} "
          f"max_new_tokens={cfg.max_new_tokens}\n")

    if not args.mock:
        names, error = client.refresh_registry()
        if error:
            print(f"  registry unavailable: {error}")
            print("  Every image will fall back to a MOCK answer. Re-run with --mock "
                  "to make that explicit, or fix connectivity first.\n")
        else:
            print(f"  registry   : {len(names)} model(s); vision: {client.vision_models()}")
            cap = client.max_images(cfg.vlm_model)
            print(f"  max_images : {cap}")
            if cfg.vlm_send_mode == "full+crop" and cap and cap < 2:
                print("  NOTE: full+crop needs 2 image slots; it will downgrade to full.")

            # Trap 9 verification - does the system prompt actually land?
            info, error = client.debug_prompt(question, n_images=1)
            if error:
                print(f"  /debug/prompt unavailable: {error}")
            else:
                body = info.get("prompt", "")
                head = question.system_prompt[:60]
                placeholder = any(t in body for t in
                                  ("<|image_pad|>", "<image>", "IMG_CONTEXT"))
                print(f"  system prompt in templated prompt : "
                      f"{'YES' if head in body else 'NO  <-- TRAP 9, investigate'}")
                print(f"  vision placeholder present        : "
                      f"{'YES' if placeholder else 'NO  <-- the model will not see the image'}")
            print()

    paths = ([args.target] if args.target.is_file() else
             sorted(p for p in args.target.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:args.limit])
    if not paths:
        raise SystemExit(f"No images under {args.target}")

    detector = get_detector(cfg, question=question)
    quality = None
    if not args.skip_quality:
        from pipeline.stage1_quality import build_quality_config, preload_segmenter, score_image
        preload_segmenter(cfg)
        quality = build_quality_config(cfg)

    counts: dict[str, int] = {}
    for path in paths:
        image_bgr = load_image_bgr(path)
        print(f"{path.name}")

        if quality is not None:
            q = score_image(image_bgr, config=quality, model_name=cfg.segmentation_model,
                            margin_trim=cfg.margin_trim, min_area_frac=cfg.min_area_frac,
                            max_area_frac=cfg.max_area_frac,
                            ignore_resolution=cfg.ignore_resolution)
            print(f"  quality   {q.headline}")

        det = detector.detect(image_bgr, path.stem, image_path=path)
        relevant = select_relevant(det.detections, question)
        if det.detections:
            print(f"  detect    " + ", ".join(
                f"{d.label} {d.confidence:.2f}" for d in det.detections))
        else:
            print(f"  detect    none")
        # Always show the note, not only on a miss. When names are inferred from
        # the selected question rather than read from a classes.txt, that is
        # exactly the run where you need to see it - a GPS antenna box labelled
        # "hazard_sign" looks authoritative and is not.
        if det.note:
            print(f"  note      {det.note}")

        block = build_detection_block(relevant)
        print(f"  block     {block or '(none - the VLM answers from the image alone)'}")

        t0 = time.time()
        ans = client.ask(image_bgr, question, relevant)
        wall = time.time() - t0
        counts[ans.answer] = counts.get(ans.answer, 0) + 1

        flag = "  [MOCK]" if ans.is_mock else ""
        print(f"  ANSWER    {ans.chip}{flag}")
        print(f"  reasoning {ans.reasoning[:300]}")
        print(f"  model     {ans.provenance}  server {ans.elapsed_s:.2f}s / wall {wall:.2f}s")
        if ans.error:
            print(f"  error     {ans.error}")
        if ans.raw_text and ans.raw_text.strip() != ans.reasoning.strip():
            print(f"  raw       {ans.raw_text[:200]}")
        print()

    print(f"{len(paths)} image(s): " + ", ".join(f"{k}={n}" for k, n in sorted(counts.items())))
    print(f"\nSemantics reminder - {question.answer_semantics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
