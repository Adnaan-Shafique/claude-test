# Patch: make the CLI call `stage1_quality.score_image()`

`run_quality_batch_foreground.py` is not in this checkout, so this patch is
written out rather than applied. Apply it in the real repo so the CLI and the
demo can never drift.

**The CSV columns must stay byte-identical** — `batch_ui.py` reads them
(`algo_result`, `score`, `segmentation_used`, `box`, `mask_area_frac`,
`mask_mean_alpha`, `whole_frame_score`, `foreground_score`, `resolution_ok`,
`width`, `height`, `resolution_check_ignored`, `failure_reasons`,
`actual_result`, `filename`, `source_path`, `output_path`).

Replace the body of `_process_one()` between `image_bgr = load_image_bgr(path)`
and the `return {...}` with:

```python
from pipeline.stage1_quality import score_image   # at module scope

def _process_one(path_str, pass_dir_str, fail_dir_str, ignore_resolution,
                 config, model_name, margin_trim, min_area_frac, max_area_frac):
    path = Path(path_str)
    pass_dir, fail_dir = Path(pass_dir_str), Path(fail_dir_str)

    image_bgr = load_image_bgr(path)
    r = score_image(
        image_bgr,
        config=config,
        model_name=model_name,
        margin_trim=margin_trim,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        ignore_resolution=ignore_resolution,
    )

    if r.error:
        return {
            "filename": path.name, "source_path": str(path), "output_path": "",
            "algo_result": "ERROR", "score": None, "segmentation_used": None,
            "box": "", "mask_area_frac": None, "mask_mean_alpha": None,
            "whole_frame_score": None, "foreground_score": None,
            "resolution_ok": None, "width": None, "height": None,
            "resolution_check_ignored": ignore_resolution,
            "failure_reasons": f"could not process: {r.error}", "actual_result": "",
        }

    dest_dir = pass_dir if r.passed else fail_dir
    dest_path = unique_destination(dest_dir, path.name)
    shutil.copy2(path, dest_path)

    return {
        "filename": path.name,
        "source_path": str(path),
        "output_path": str(dest_path),
        "algo_result": "PASS" if r.passed else "FAIL",
        "score": round(r.score, 1),
        "segmentation_used": r.segmentation_used,
        "box": str(r.foreground_box) if r.foreground_box is not None else "",
        "mask_area_frac": r.mask_area_frac,
        "mask_mean_alpha": r.mask_mean_alpha,
        "whole_frame_score": round(r.whole_frame_score, 1),
        "foreground_score": round(r.foreground_score, 1) if r.foreground_score is not None else None,
        "resolution_ok": r.resolution_ok,
        "width": r.width,
        "height": r.height,
        "resolution_check_ignored": ignore_resolution,
        "failure_reasons": "; ".join(r.failure_reasons),
        "actual_result": "",
    }
```

Two behaviours preserved exactly:

- `box` is written as `str(tuple)` — `review_ui.parse_box()` reads it back with
  `re.findall(r"-?\d+")`, so the formatting must not change.
- The `--ignore-resolution` effective-verdict rule (`score >= threshold`,
  ignoring the resolution floor, with `resolution_too_low` filtered out of
  `failure_reasons`) now lives in `score_image()` and behaves identically.

**Verification after applying:** run the CLI over a folder before and after the
patch and `diff` the two `quality_review_foreground.csv` files. They must be
identical.
