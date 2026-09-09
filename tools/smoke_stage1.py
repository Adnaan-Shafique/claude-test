#!/usr/bin/env python3
"""Stage 1 smoke test - the Phase 1 acceptance check.

Runs one real photo (or the first N in a folder) through the full stage-1 path:
EXIF-corrected load -> whole-frame score -> u2netp foreground box -> cropped
score -> verdict, and writes the green-box annotated copy the UI will show.

    python tools/smoke_stage1.py /path/to/photo.jpg
    python tools/smoke_stage1.py /path/to/folder --limit 5
    python tools/smoke_stage1.py /path/to/folder --compare quality_review_foreground.csv

--compare reads a CSV written by run_quality_batch_foreground.py (from whichever
machine that script runs on) and checks this pipeline produces the same numbers
for the same filenames. That is the plan's stated Phase 1 test: "one image in ->
QualityStageResult out, matching what the CLI produces for the same file."

This is the first thing that actually exercises rembg + onnxruntime + the
vendored modules together, so run it before anything else.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(target: Path, limit: int) -> list[Path]:
    if target.is_file():
        return [target]
    found = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not found:
        raise SystemExit(f"No images under {target} (looked for {sorted(IMAGE_EXTENSIONS)})")
    return found[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="an image file, or a folder of them")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ignore-resolution", action="store_true")
    ap.add_argument("--compare", type=Path, default=None,
                    help="quality_review_foreground.csv to check these numbers against")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo_runs" / "smoke")
    args = ap.parse_args()

    from pipeline.config import default_config
    from pipeline.stage1_quality import (annotate_for_gallery, build_quality_config,
                                         preload_segmenter, score_image)
    from quality_check import load_image_bgr

    cfg = default_config()
    if args.threshold is not None:
        cfg.quality_threshold = args.threshold
    cfg.ignore_resolution = args.ignore_resolution
    qc = build_quality_config(cfg)

    print(f"U2NET_HOME -> {cfg.ensure_u2net_home()}")
    print(f"model file -> {cfg.segmentation_model_path()} "
          f"({'present' if cfg.segmentation_model_path().exists() else 'MISSING'})")

    # Loading u2netp is the expensive part and the most likely thing to break -
    # do it once, up front, and time it, so a slow first image on stage is not a
    # surprise.
    t0 = time.time()
    try:
        preload_segmenter(cfg)
    except Exception as exc:
        print(f"\nFAILED to preload the segmenter: {type(exc).__name__}: {exc}")
        print("If this is a TypeError about sess_opts, this rembg version's "
              "new_session() does not take that keyword - report it and it will "
              "be made version-tolerant.")
        return 1
    print(f"u2netp loaded in {time.time() - t0:.1f}s\n")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in collect(args.target, args.limit):
        t_load = time.time()
        image_bgr = load_image_bgr(path)
        load_s = time.time() - t_load

        t_score = time.time()
        r = score_image(
            image_bgr, config=qc,
            model_name=cfg.segmentation_model,
            margin_trim=cfg.margin_trim,
            min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac,
            ignore_resolution=cfg.ignore_resolution,
        )
        score_s = time.time() - t_score

        annotate_for_gallery(image_bgr, r, args.out / f"{path.stem}_annotated.jpg")

        print(f"{path.name}")
        print(f"  {r.headline}")
        if r.error:
            print(f"  ERROR: {r.error}")
            rows.append((path.name, r))
            continue
        print(f"  resolution     {r.width}x{r.height}  resolution_ok={r.resolution_ok}")
        print(f"  whole frame    {r.whole_frame_score:.1f}")
        if r.segmentation_used:
            print(f"  foreground     {r.foreground_score:.1f}   box={r.foreground_box} "
                  f"area={r.mask_area_frac} alpha={r.mask_mean_alpha}")
            print(f"  delta          {r.foreground_score - r.whole_frame_score:+.1f} "
                  f"(cropping {'helped' if r.foreground_score > r.whole_frame_score else 'hurt'})")
        else:
            print(f"  foreground     no plausible box - fell back to whole-frame")
        if r.failure_reasons:
            print(f"  reasons        {', '.join(r.failure_reasons)}")
        for instruction in r.retake_instructions:
            print(f"                 -> {instruction}")
        print(f"  timing         load {load_s:.2f}s + score {score_s:.2f}s")
        print(f"  annotated      {r.annotated_path}")
        print()
        rows.append((path.name, r))

    if args.compare:
        compare(rows, args.compare)

    n_seg = sum(1 for _, r in rows if r.segmentation_used)
    n_err = sum(1 for _, r in rows if r.error)
    print(f"{len(rows)} image(s): {sum(1 for _, r in rows if r.passed)} PASS, "
          f"{sum(1 for _, r in rows if not r.passed and not r.error)} FAIL, {n_err} ERROR; "
          f"segmentation used on {n_seg}/{len(rows)}")
    return 1 if n_err else 0


def compare(rows, csv_path: Path) -> None:
    """Check these results against a run_quality_batch_foreground.py CSV."""
    import pandas as pd

    print(f"--- comparing against {csv_path} ---")
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    by_name = {str(r["filename"]): r for _, r in df.iterrows()}

    mismatches = 0
    for name, r in rows:
        ref = by_name.get(name)
        if ref is None:
            print(f"  {name}: not in the CSV, skipped")
            continue
        # The CLI rounds to 1dp on the way into the CSV, so compare at that
        # precision rather than demanding bit-identical floats.
        checks = [
            ("score", round(r.score, 1), _num(ref.get("score"))),
            ("whole_frame_score", round(r.whole_frame_score, 1), _num(ref.get("whole_frame_score"))),
            ("segmentation_used", bool(r.segmentation_used), _bool(ref.get("segmentation_used"))),
            ("resolution_ok", bool(r.resolution_ok), _bool(ref.get("resolution_ok"))),
            ("width", int(r.width), _num(ref.get("width"))),
            ("height", int(r.height), _num(ref.get("height"))),
        ]
        bad = [(f, mine, theirs) for f, mine, theirs in checks
               if theirs is not None and mine != theirs]
        if bad:
            mismatches += 1
            print(f"  {name}: MISMATCH")
            for f, mine, theirs in bad:
                print(f"      {f}: pipeline={mine!r}  CLI={theirs!r}")
        else:
            print(f"  {name}: matches the CLI")
    print(f"  {mismatches} mismatch(es)\n")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else round(f, 1)


def _bool(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return True if s == "true" else False if s == "false" else None


if __name__ == "__main__":
    raise SystemExit(main())
