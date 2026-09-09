#!/usr/bin/env python3
"""Annotation-file helper for the stub detector.

    # Which photos have labels and which do not?
    python tools/make_stub_labels.py --check --images <photo dir>

    # Write a skeleton <stem>.txt for every photo that lacks one
    python tools/make_stub_labels.py --emit --images <photo dir>

    # Write classes.txt so labels stop rendering as class_<id>
    python tools/make_stub_labels.py --classes gps_antenna,hazard_sign

--check is the one to run before the demo. The stub detector keys on the photo's
stem, so a label whose name does not match its photo is invisible: the detector
works perfectly and reports "no annotation file found" for every image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# A centred box covering ~20% of the frame - obviously a placeholder, not a
# plausible detection anyone could mistake for a real annotation.
SKELETON_LINE = "0 0.500000 0.500000 0.200000 0.200000"


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def labels_in(directory: Path) -> list[Path]:
    skip = {"classes.txt", "obj.names"}
    return sorted(p for p in directory.glob("*.txt") if p.name not in skip)


def do_check(images_dir: Path, labels_dir: Path) -> int:
    """Coverage, matching the way the detector actually looks things up: the
    sidecar beside each photo first, then the central labels folder."""
    photos = images_in(images_dir)
    print(f"photos: {len(photos)} under {images_dir}")

    matched, missing = [], []
    folders: dict[Path, int] = {}
    for photo in photos:
        sidecar = photo.with_suffix(".txt")
        central = labels_dir / f"{photo.stem}.txt"
        found = sidecar if sidecar.exists() else (central if central.exists() else None)
        if found is None:
            missing.append(photo)
            continue
        matched.append((photo, found))
        folders[found.parent] = folders.get(found.parent, 0) + 1

    for photo, label in matched:
        lines = [ln for ln in label.read_text(errors="replace").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        where = "beside" if label.parent == photo.parent else str(label.parent)
        print(f"  OK      {photo.stem}  ({len(lines)} box(es), {where})")
    for photo in missing:
        print(f"  NO TXT  {photo.stem}  -> will report 'no annotation file found'")

    # Orphans only make sense per folder that actually holds labels.
    photo_stems = {p.stem for p in photos}
    orphans = []
    for folder in folders:
        for label in labels_in(folder):
            if label.stem not in photo_stems:
                orphans.append(label)
    for label in sorted(orphans):
        print(f"  ORPHAN  {label} has no matching photo")

    from pipeline.stage2_detect import load_class_names
    print()
    for folder in sorted(folders):
        names = load_class_names(folder)
        if names:
            print(f"  {folder}: classes {names}")
        else:
            print(f"  {folder}: no classes.txt - class ids resolve from the "
                  f"selected question's defaults (hazard_warning -> hazard_sign, "
                  f"gps_antenna -> gps_antenna), else class_<id>")

    print(f"\n{len(matched)} matched, {len(missing)} photo(s) without a label, "
          f"{len(orphans)} orphaned label(s)")
    if missing:
        print("\nEvery demo photo needs a label file, or its detection panel will be "
              "empty with a note. Use --emit to scaffold the missing ones, then "
              "replace the placeholder boxes with real ones.")
    return 1 if missing or orphans else 0


def do_emit(images_dir: Path, labels_dir: Path, force: bool) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for photo in images_in(images_dir):
        # Beside the photo, matching the sidecar layout the exports use.
        dest = photo.with_suffix(".txt")
        if dest.exists() and not force:
            skipped += 1
            continue
        dest.write_text(
            f"# PLACEHOLDER for {photo.name} - replace with the real box.\n"
            f"# YOLO normalized: class_id cx cy w h  (values 0-1, optional 6th conf)\n"
            f"# or absolute px:  class_id x1 y1 x2 y2 [conf]\n"
            f"{SKELETON_LINE}\n"
        )
        written += 1
    print(f"{written} skeleton file(s) written, {skipped} left alone "
          f"(use --force to overwrite)")
    if written:
        print("These are PLACEHOLDER boxes at the centre of the frame. Replace them "
              "before the demo - a placeholder box drawn confidently on screen is "
              "worse than no detection at all.")
    return 0


def do_classes(labels_dir: Path, names: str) -> int:
    parsed = [n.strip() for n in names.split(",") if n.strip()]
    if not parsed:
        print("No class names given.")
        return 1
    labels_dir.mkdir(parents=True, exist_ok=True)
    dest = labels_dir / "classes.txt"
    dest.write_text("\n".join(parsed) + "\n")
    print(f"wrote {dest}:")
    for i, n in enumerate(parsed):
        print(f"  {i} = {n}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="folder of demo photos")
    ap.add_argument("--labels", type=Path, default=None,
                    help="label folder (default: the photo folder if it holds .txt "
                         "files, else data/labels)")
    ap.add_argument("--check", action="store_true", help="report label coverage")
    ap.add_argument("--emit", action="store_true", help="scaffold missing label files")
    ap.add_argument("--force", action="store_true", help="overwrite existing labels on --emit")
    ap.add_argument("--classes", type=str, help="comma-separated names -> classes.txt")
    args = ap.parse_args()

    if args.images and not args.images.is_dir():
        raise SystemExit(f"Not a directory: {args.images}")

    if args.labels is None:
        from pipeline.config import default_config
        args.labels = default_config().resolve_annotation_dir(args.images)
        print(f"labels dir: {args.labels}  (auto-detected)\n")

    if args.classes:
        return do_classes(args.labels, args.classes)
    if not args.images:
        ap.error("--images is required for --check and --emit")
    if args.emit:
        return do_emit(args.images, args.labels, args.force)
    if args.check:
        return do_check(args.images, args.labels)
    ap.error("pick one of --check, --emit or --classes")


if __name__ == "__main__":
    raise SystemExit(main())
