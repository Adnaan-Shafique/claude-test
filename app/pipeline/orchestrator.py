"""Sequential per-image orchestration of the three legs.

Sequential is the right shape here, and the timings say so: on the demo host a
quality pass costs ~2.0s per image (u2netp plus two assess_quality runs), the
stub detector is free, and a VLM answer comes back in ~0.35s. Stage 1 is the
bottleneck by roughly 5x, so parallelising the VLM would buy nothing, and a
worker pool would reintroduce the ~20s cold-start the CLI already pays for.

ONE LOAD PER IMAGE. quality_check.load_image_bgr() applies EXIF transposition;
cv2.imread and a bare PIL open do not. Every stage here reuses that single
array, so the quality box, the detection box and the VLM crop are all in the
same frame. Nothing downstream is ever handed a path to re-open.
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from .questions import get_question, select_relevant
from .schemas import (FLAT_ROW_COLUMNS, PipelineRecord, STOPPED_COMPLETE,
                      STOPPED_QUALITY)

ProgressCb = Optional[Callable[[int, int, str], None]]


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def run_pipeline(image_paths: Iterable, question_id: str, cfg,
                 progress_cb: ProgressCb = None) -> list[PipelineRecord]:
    """Run every image through quality -> detection -> VLM.

    progress_cb(index, total, stage_name) fires after each STAGE rather than
    each image: 20 images x 3 stages with an uneven cost distribution makes a
    per-image bar look stalled during the quality pass.
    """
    from .stage1_quality import (annotate_for_gallery, build_quality_config,
                                 preload_segmenter, score_image)
    from .stage2_detect import get_detector
    from .stage3_vlm import VLMClient
    from quality_check import load_image_bgr

    paths = [Path(p) for p in image_paths]
    total = len(paths)
    question = get_question(question_id)

    if cfg.run_id is None:
        cfg.run_id = new_run_id()
    run_dir = cfg.run_dir
    (run_dir / "quality").mkdir(parents=True, exist_ok=True)
    (run_dir / "detection").mkdir(parents=True, exist_ok=True)

    def report(i: int, stage: str) -> None:
        if progress_cb:
            progress_cb(i, total, stage)

    report(0, "loading the segmenter")
    preload_segmenter(cfg)
    quality_config = build_quality_config(cfg)
    detector = get_detector(cfg, question=question)

    client = VLMClient(cfg)
    report(0, "checking the GPU server")
    client.ensure_registry()

    records: list[PipelineRecord] = []
    for index, path in enumerate(paths, start=1):
        # The ORIGINAL stem, captured here and never re-derived. Both renaming
        # paths in the existing code destroy it, and one of them (a random
        # uuid) is not even stable across two runs of the same file.
        stem = path.stem
        record_started = time.time()

        report(index, f"quality {index}/{total}")
        try:
            image_bgr = load_image_bgr(path)
        except Exception as exc:
            records.append(_unreadable_record(path, stem, question_id, str(exc)))
            continue

        quality = score_image(
            image_bgr, config=quality_config, model_name=cfg.segmentation_model,
            margin_trim=cfg.margin_trim, min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac, ignore_resolution=cfg.ignore_resolution,
        )
        annotate_for_gallery(image_bgr, quality, run_dir / "quality" / f"{stem}.jpg")

        record = PipelineRecord(
            filename=path.name, source_path=str(path), stem=stem,
            question_id=question_id, quality=quality,
        )

        # The gate. The toggle exists so a stage mishap - one bad photo, a
        # threshold that turns out wrong under the room's lighting - cannot
        # dead-end the demo.
        if not quality.passed and not cfg.run_downstream_on_fail:
            record.stopped_at = STOPPED_QUALITY
            records.append(record)
            record.extra["elapsed_s"] = round(time.time() - record_started, 2)
            continue

        report(index, f"detection {index}/{total}")
        detection = detector.detect(image_bgr, stem, image_path=path)
        if detection.detections:
            from .stage2_detect import render
            render(image_bgr, detection, run_dir / "detection" / f"{stem}.jpg")
        record.detection = detection

        report(index, f"asking the model {index}/{total}")
        relevant = select_relevant(detection.detections, question)
        record.vlm = client.ask(image_bgr, question, relevant)
        record.stopped_at = STOPPED_COMPLETE
        record.extra["elapsed_s"] = round(time.time() - record_started, 2)
        records.append(record)

    report(total, "writing results")
    write_results(records, run_dir)
    return records


def _unreadable_record(path: Path, stem: str, question_id: str,
                       error: str) -> PipelineRecord:
    from .schemas import QualityStageResult
    return PipelineRecord(
        filename=path.name, source_path=str(path), stem=stem, question_id=question_id,
        quality=QualityStageResult(
            passed=False, score=0.0, threshold=0.0, resolution_ok=False,
            ignore_resolution_used=False,
            failure_reasons=[f"could not read the file: {error}"],
            retake_instructions=[], foreground_box=None, segmentation_used=False,
            whole_frame_score=0.0, width=0, height=0, error=error,
        ),
        stopped_at=STOPPED_QUALITY,
    )


def sort_for_display(records: list[PipelineRecord]) -> list[PipelineRecord]:
    """Completed images first, then quality stops - the order the story is told
    in. Stable within each group so the upload order survives."""
    order = {STOPPED_COMPLETE: 0}
    return sorted(records, key=lambda r: order.get(r.stopped_at, 1))


def summarise(records: list[PipelineRecord]) -> dict:
    completed = [r for r in records if r.vlm is not None]
    answers: dict[str, int] = {}
    for r in completed:
        answers[r.vlm.answer] = answers.get(r.vlm.answer, 0) + 1
    return {
        "total": len(records),
        "passed": sum(1 for r in records if r.quality.passed),
        "failed": sum(1 for r in records if not r.quality.passed and not r.quality.error),
        "errored": sum(1 for r in records if r.quality.error),
        "stopped_at_quality": sum(1 for r in records if r.stopped_at == STOPPED_QUALITY),
        "answers": answers,
        "mocked": sum(1 for r in completed if r.vlm.is_mock),
        "segmentation_used": sum(1 for r in records if r.quality.segmentation_used),
    }


def write_results(records: list[PipelineRecord], run_dir: Path) -> tuple[Path, Path]:
    """One flat CSV row per image plus the full nested records as JSON.

    Deliberately not the same schema as quality_review_foreground.csv - that
    file's columns are read by batch_ui.py and must not move.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "pipeline_results.csv"
    json_path = run_dir / "results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FLAT_ROW_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_flat_row())

    payload = []
    for record in records:
        q = record.quality
        d = record.detection
        v = record.vlm
        payload.append({
            "filename": record.filename, "stem": record.stem,
            "source_path": record.source_path, "question_id": record.question_id,
            "stopped_at": record.stopped_at, "extra": record.extra,
            "quality": {
                "passed": q.passed, "score": q.score, "threshold": q.threshold,
                "verdict": q.verdict, "fail_kind": q.fail_kind, "headline": q.headline,
                "resolution_ok": q.resolution_ok, "width": q.width, "height": q.height,
                "whole_frame_score": q.whole_frame_score,
                "foreground_score": q.foreground_score,
                "segmentation_used": q.segmentation_used,
                "foreground_box": list(q.foreground_box) if q.foreground_box else None,
                "mask_area_frac": q.mask_area_frac, "mask_mean_alpha": q.mask_mean_alpha,
                "failure_reasons": q.failure_reasons,
                "retake_instructions": q.retake_instructions,
                "annotated_path": q.annotated_path, "error": q.error,
            },
            "detection": None if d is None else {
                "model_name": d.model_name, "is_stub": d.is_stub, "note": d.note,
                "annotated_path": d.annotated_path,
                "detections": [{"label": x.label, "confidence": x.confidence,
                                "box": x.box, "source": x.source} for x in d.detections],
            },
            "vlm": None if v is None else {
                "answer": v.answer, "reasoning": v.reasoning, "raw_text": v.raw_text,
                "model": v.model, "provenance": v.provenance, "elapsed_s": v.elapsed_s,
                "is_mock": v.is_mock, "error": v.error,
            },
        })
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def collect_images(folder, extensions=None) -> list[Path]:
    """Every image under a folder, recursively - the demo photos live in
    per-class subfolders."""
    extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)
