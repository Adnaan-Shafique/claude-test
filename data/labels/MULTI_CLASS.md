# Adding a second annotated class

The HV Hazardous Radiations export and a GPS Antenna export are separate CVAT
tasks, and **each numbers its only class `0`**. The same id means two different
things depending on which export a file came from.

## The layout that stays correct

One folder per export, labels beside their photos:

```
photos/
  hv/                       <- HV Hazardous Radiations
    00002_task_328989.jpg
    00002_task_328989.txt   -> "0 ..." means a hazard sign
  gps/                      <- GPS Antenna
    00101_task_331002.jpg
    00101_task_331002.txt   -> "0 ..." means a GPS antenna
```

Point each question at its own folder:

```python
cfg.annotation_dirs = {
    "hazard_warning": Path("/data/adnaan/fieldops/demo/photos/hv"),
    "gps_antenna":    Path("/data/adnaan/fieldops/demo/photos/gps"),
}
```

The demo UI exposes this as one label-folder field per question.

## How a class id becomes a name

In order:

1. `classes.txt` in the label folder (one name per line, index = class_id)
2. `dataset.yaml` / `data.yaml` `names:`, or `obj.names`
3. **The selected question's `default_class_names`** — `hazard_warning` → `hazard_sign`, `gps_antenna` → `gps_antenna`
4. `class_<id>`

Step 3 is what makes a bare single-class export readable without any extra
files, and why one shared `classes.txt` across both exports is the wrong fix:
it can only give id 0 a single meaning.

The detection note records which of these supplied the names, so the UI can
show that they were inferred from the question rather than read from the data.

## What NOT to do

**Do not merge both exports into one folder.** Every `0` would resolve to
whichever name `classes.txt` gives it, so half the images get the wrong label —
and that label is injected into the VLM prompt as evidence the model is told to
weigh. A wrong detection label is worse than none: it produces a confident,
wrong-for-the-right-reason answer.

If the two sets genuinely must share a folder, re-export them as a real
two-class dataset (hazard = 0, antenna = 1) with a matching `classes.txt`. Then
step 1 applies and the question defaults are never consulted.
