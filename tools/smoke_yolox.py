#!/usr/bin/env python3
"""YOLOX detector smoke test - real weights, real photos.

    python tools/smoke_yolox.py <photos> --ckpt /path/to/best_ckpt.pth --limit 3

Loads the checkpoint with strict=True (a partial load would produce confident
meaningless boxes, so it must fail loudly instead), runs the detector over real
images, and writes the annotated copies.

Open those images. Box position is the one thing no test can verify.
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
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=480)
    # No hardcoded default. A stale literal here silently overrides
    # cfg.yolox_class_names and reports the wrong labels while the config is
    # correct - which is exactly what happened, and it cost a round of
    # "the model has the classes backwards" that it did not.
    ap.add_argument("--classes", default=None,
                    help="comma-separated names in training index order "
                         "(default: cfg.yolox_class_names)")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo_runs" / "smoke_yolox")
    args = ap.parse_args()

    from pipeline.config import default_config
    from pipeline.stage2_detect import get_detector, render
    from quality_check import load_image_bgr

    names = (tuple(n.strip() for n in args.classes.split(",") if n.strip())
             if args.classes else default_config().yolox_class_names)
    cfg = default_config(
        use_model=True, yolox_checkpoint=str(args.ckpt), yolox_class_names=names,
        yolox_input_size=(args.height, args.width), conf_thresh=args.conf,
        yolox_nms_threshold=args.nms, yolox_fuse=not args.no_fuse,
    )
    for problem in cfg.validate():
        print(f"  config: {problem}")

    t0 = time.time()
    try:
        detector = get_detector(cfg)
    except Exception as exc:
        print(f"\nFAILED to load the detector:\n{exc}")
        return 1
    print(f"backend    : {detector.name}")
    print(f"classes    : {list(names)}"
          f"   ({'from --classes' if args.classes else 'from config'}; "
          f"index order matters)")
    print(f"input size : {args.height}x{args.width}  (height x width)")
    print(f"conf / nms : {args.conf} / {args.nms}")
    print(f"loaded in  : {time.time() - t0:.1f}s\n")

    paths = ([args.target] if args.target.is_file() else
             sorted(p for p in args.target.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:args.limit])
    if not paths:
        raise SystemExit(f"No images under {args.target}")

    args.out.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    times = []
    for path in paths:
        image_bgr = load_image_bgr(path)
        h, w = image_bgr.shape[:2]
        t = time.time()
        result = detector.detect(image_bgr, path.stem, image_path=path)
        elapsed = time.time() - t
        times.append(elapsed)

        print(f"{path.name}  ({w}x{h})  {elapsed:.2f}s")
        if result.detections:
            render(image_bgr, result, args.out / f"{path.stem}_yolox.jpg")
            for d in sorted(result.detections, key=lambda d: -d.confidence):
                totals[d.label] = totals.get(d.label, 0) + 1
                x1, y1, x2, y2 = (int(round(v)) for v in d.box)
                frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
                print(f"    {d.label:<14} {d.confidence:.3f}  "
                      f"[{x1}, {y1}, {x2}, {y2}]  {frac * 100:.2f}% of frame")
            print(f"    -> {result.annotated_path}")
        else:
            print("    no detections above threshold")
        if result.note:
            print(f"    note: {result.note}")
        print()

    print(f"{len(paths)} image(s), {sum(totals.values())} detection(s): "
          + (", ".join(f"{k}={n}" for k, n in sorted(totals.items())) or "none"))
    if times:
        print(f"inference: {sum(times) / len(times):.2f}s average, "
              f"{min(times):.2f}-{max(times):.2f}s range")
    print(f"\nNow OPEN the images in {args.out} and confirm each box sits on the "
          f"object it names. Box position cannot be verified any other way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
