# `data/labels/` — annotation files for the stub detector

One `<stem>.txt` per image, where `<stem>` is the **original photo filename
stem**. `pipeline/stage2_detect.py`'s `AnnotationFileDetector` reads these when
`use_model=False` (the default).

## Format

Auto-detected per file:

- **YOLO normalized** — `class_id cx cy w h`, values in `[0,1]`, optional 6th
  confidence column.
- **Absolute pixel xyxy** — `class_id x1 y1 x2 y2 [conf]`, detected by any
  coordinate `> 1.5`.

Blank lines and `#` comments are ignored. A malformed line is skipped with a
note rather than crashing the image.

Column 0 may also be a **string label** (`hazard_sign 0.5 0.5 0.2 0.2`) — if it
does not parse as an int it is treated as the label directly.

## Class names

Looked up in order: `classes.txt` (one name per line, index = class_id), then
`dataset.yaml` (`names:` key), then `class_<id>`.

**There is no `classes.txt` here yet.** The sample annotation carries a bare
`class_id 0`, so detections currently render as `class_0` — in the UI panel and
in the prompt sent to the VLM. Adding a one-line `classes.txt` fixes both with
no code change.

## Filename join key

`00002_task_328989.txt` is a CVAT-style export name (frame index + task id).
Unless the photos carry matching names, every image reports *"no annotation
file found"* — the detector works and finds nothing. Rename the `.txt` files to
match the photo stems, then verify with:

    python tools/make_stub_labels.py --check --images <photo dir> --labels data/labels

(that tool lands in Phase 6.)
