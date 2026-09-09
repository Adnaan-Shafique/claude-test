# Data layout — photos and annotations

## Recommended: labels beside their photos, one folder per class

```
/data/adnaan/fieldops/demo/photos/
  hv_hazard/                          <- HV Hazardous Radiations export
    00002_task_192416.jpg
    00002_task_192416.txt             -> "0 ..." means a hazard sign
    classes.txt                       (optional: "hazard_sign")
  gps_antenna/                        <- GPS Antenna export
    00101_task_331002.jpg
    00101_task_331002.txt             -> "0 ..." means a GPS antenna
    classes.txt                       (optional: "gps_antenna")
```

**Keep labels next to their photos.** That is how the CVAT/YOLO exports already
arrive, and it is what the pipeline prefers. Splitting images and labels into
parallel trees buys nothing and adds a way to get them out of sync.

The pipeline looks for a label in this order, per image:

1. `<the image's own folder>/<stem>.txt` — works at any folder depth
2. `<cfg.annotation_dir>/<stem>.txt` — a central folder, if you prefer one

So a nested layout needs no configuration: point the pipeline at the photo root
and every sidecar is found where it sits.

## Why one folder per class matters

Each export is a separate single-class CVAT task, so **each numbers its only
class `0`**. The same id means a hazard sign in one export and a GPS antenna in
another. Keeping them in separate folders is what lets each resolve correctly.

Class names resolve per folder, in this order:

1. `classes.txt` in **the folder the label was found in** (one name per line, index = class_id)
2. `dataset.yaml` / `data.yaml` `names:`, or `obj.names`
3. The selected question's `default_class_names` — `hazard_warning` → `hazard_sign`, `gps_antenna` → `gps_antenna`
4. `class_<id>`

Step 3 makes a bare single-class export readable with no extra files. The
detection note records which source supplied the names, so the UI can show that
they were inferred from the question rather than read from the data.

## What NOT to do

**Do not merge both exports into one folder.** Every `0` would take whichever
name that folder's `classes.txt` gives it, so half the images get the wrong
label — and that label is injected into `{detection_block}`, which the prompt
tells the model to weigh as evidence. A wrong detection label is worse than
none: it yields a confident answer that is wrong for a reason nobody watching
can see.

If the two sets genuinely must share a folder, re-export them as a real
two-class dataset (hazard = 0, antenna = 1) with a matching `classes.txt`. Then
step 1 applies and the question defaults are never consulted.

## Checking a layout

```bash
python tools/make_stub_labels.py --check --images /data/adnaan/fieldops/demo/photos
```

Reports coverage the same way the detector resolves it — sidecar first, central
folder second — lists which folder each label came from, and exits non-zero on
any gap.
