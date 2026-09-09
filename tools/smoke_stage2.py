#!/usr/bin/env python3
"""Stage 2 smoke test - runs the detector and writes the annotated image.

    python tools/smoke_stage2.py /path/to/photos --limit 3

The output image is the point. The annotation was drawn in whatever orientation
the labelling tool saw, while this pipeline scores and draws on the
EXIF-corrected array from quality_check.load_image_bgr(). If the labelling tool
did not apply EXIF and the photo carries a rotation, the box will land in the
wrong place - and the ONLY reliable way to tell is to look at it.

Open the files this writes and confirm each box sits on the object it names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--question", default="hazard_warning",
                    help="which question's class names to resolve ids with "
                         "(hazard_warning | gps_antenna)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo_runs" / "smoke_detect")
    args = ap.parse_args()

    from pipeline.config import default_config
    from pipeline.questions import get_question
    from pipeline.stage2_detect import get_detector, render
    from quality_check import load_image_bgr

    cfg = default_config()
    if args.labels:
        cfg.annotation_dir = args.labels
    else:
        # Labels commonly sit beside the photos (the YOLO/CVAT sidecar layout).
        source_dir = args.target if args.target.is_dir() else args.target.parent
        cfg.annotation_dir = cfg.resolve_annotation_dir(source_dir)

    question = get_question(args.question)
    detector = get_detector(cfg, question=question)
    print(f"backend    : {detector.name}")
    print(f"is_stub    : {detector.is_stub}")
    print(f"question   : {question.id}  ({question.label})")
    print(f"labels dir : {detector.annotation_dir}")
    print(f"classes    : {getattr(detector, 'class_names', None) or 'NONE (class_<id>)'}"
          f"  [from {getattr(detector, 'class_names_source', '?')}]\n")

    paths = ([args.target] if args.target.is_file() else
             sorted(p for p in args.target.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:args.limit])
    if not paths:
        raise SystemExit(f"No images under {args.target}")

    args.out.mkdir(parents=True, exist_ok=True)
    n_with = 0
    for path in paths:
        image_bgr = load_image_bgr(path)
        h, w = image_bgr.shape[:2]
        result = detector.detect(image_bgr, path.stem, image_path=path)
        rel = path.parent.name if path.parent != args.target else ""
        print(f"{path.name}  ({w}x{h})" + (f"  [{rel}/]" if rel else ""))
        if result.detections:
            n_with += 1
            render(image_bgr, result, args.out / f"{path.stem}_detect.jpg")
            for d in result.detections:
                x1, y1, x2, y2 = (int(round(v)) for v in d.box)
                frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
                print(f"    {d.label}  {d.confidence:.2f}  "
                      f"[{x1}, {y1}, {x2}, {y2}]  {frac * 100:.2f}% of frame")
            print(f"    -> {result.annotated_path}")
        else:
            print(f"    no detections")
        if result.note:
            print(f"    note: {result.note}")
        print()

    print(f"{n_with}/{len(paths)} image(s) had detections.")
    if n_with:
        print(f"\nNow OPEN the images in {args.out} and confirm each box sits on the "
              f"object it claims. A box in the wrong place means the labelling tool "
              f"and quality_check.load_image_bgr() disagree about EXIF rotation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
