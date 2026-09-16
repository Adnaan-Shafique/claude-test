"""Three ways to reach the same verdict, run over the same photographs.

    MODE_CLASSIC  quality gate -> detector -> model
                  The quality gate is a hard stop: a photo that fails is never
                  looked at again.

    MODE_OR_GATE  (quality OR detector) -> model
                  Quality and detection both always run. The photo proceeds if
                  EITHER the quality gate passes or the detector found the
                  target - so a soft-focus photo where the sign is plainly
                  visible is not thrown away for a low cue score.

    MODE_VLM_ONLY everything by the model
                  One call returns quality, subject presence and the inspection
                  answer. No u2netp, no YOLOX.

WORK IS SHARED, NOT REPEATED. Running all three naively would cost three model
calls and two quality passes per image. Instead each image is loaded once,
scored once and detected once; modes 1 and 2 read the same results and differ
only in what they do with them. And when both gates agree to proceed, the
inputs to the model are identical, so the answer is computed once and reused -
which is the common case. Typical cost is two model calls per image, not three.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .orchestrator import new_run_id, write_results
from .questions import get_question, select_relevant
from .schemas import (DetectionStageResult, PipelineRecord, QualityStageResult,
                      STOPPED_COMPLETE, STOPPED_QUALITY, VLMAnswer)

MODE_CLASSIC = "classic"
MODE_OR_GATE = "or_gate"
MODE_VLM_ONLY = "vlm_only"

MODES = {
    MODE_CLASSIC: {
        "id": MODE_CLASSIC,
        "label": "1 · Quality gate → Detector → Model",
        "short": "Quality gate",
        "blurb": ("The quality gate decides first. A photograph that fails it is "
                  "never detected on or asked about."),
    },
    MODE_OR_GATE: {
        "id": MODE_OR_GATE,
        "label": "2 · Quality OR Detector → Model",
        "short": "Quality OR detector",
        "blurb": ("Quality and detection both run. The photograph proceeds if "
                  "either one is satisfied, so a soft-focus shot where the "
                  "subject is plainly detected still gets answered."),
    },
    MODE_VLM_ONLY: {
        "id": MODE_VLM_ONLY,
        "label": "3 · Everything by the model",
        "short": "Model only",
        "blurb": ("One call to the vision model judges quality, whether the "
                  "subject is visible, and the inspection question. No "
                  "segmentation model, no trained detector."),
    },
}
MODE_ORDER = [MODE_CLASSIC, MODE_OR_GATE, MODE_VLM_ONLY]

ProgressCb = Optional[Callable[[int, int, str], None]]


def run_all_modes(image_paths, question_id: str, cfg,
                  progress_cb: ProgressCb = None, prompts=None) -> dict:
    """Every mode over every image. Returns {mode_id: [PipelineRecord]}.

    `prompts` is a PromptStore supplying mode 3's per-leg system prompts; None
    means the built-in defaults.
    """
    from .stage1_quality import (annotate_for_gallery, build_quality_config,
                                 preload_segmenter, score_image)
    from .stage2_detect import get_detector, render as render_detection
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
    report(0, "checking the model server")
    client.ensure_registry()

    results = {mode: [] for mode in MODE_ORDER}

    for index, path in enumerate(paths, start=1):
        stem = path.stem
        started = time.time()

        report(index, f"reading {index}/{total}")
        try:
            image_bgr = load_image_bgr(path)
        except Exception as exc:
            for mode in MODE_ORDER:
                results[mode].append(_unreadable(path, stem, question_id, str(exc)))
            continue

        # ── Shared: one quality pass, one detection pass ─────────────────────
        report(index, f"quality {index}/{total}")
        quality = score_image(
            image_bgr, config=quality_config, model_name=cfg.segmentation_model,
            margin_trim=cfg.margin_trim, min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac, ignore_resolution=cfg.ignore_resolution)
        annotate_for_gallery(image_bgr, quality, run_dir / "quality" / f"{stem}.jpg")

        report(index, f"detection {index}/{total}")
        detection = detector.detect(image_bgr, stem, image_path=path)
        if detection.detections:
            render_detection(image_bgr, detection, run_dir / "detection" / f"{stem}.jpg")

        relevant = select_relevant(detection.detections, question)
        classic_proceeds = quality.passed or cfg.run_downstream_on_fail
        or_proceeds = (quality.passed or bool(detection.detections)
                       or cfg.run_downstream_on_fail)

        # ── Modes 1 and 2 ────────────────────────────────────────────────────
        # When both gates open, the model sees identical inputs, so the answer
        # is computed once and shared. Only a photo that mode 1 stops and mode 2
        # lets through costs a second call.
        answer = None
        if classic_proceeds or or_proceeds:
            report(index, f"asking the model {index}/{total}")
            answer = client.ask(image_bgr, question, relevant)

        results[MODE_CLASSIC].append(_record(
            path, stem, question_id, quality,
            detection if classic_proceeds else None,
            answer if classic_proceeds else None,
            STOPPED_COMPLETE if classic_proceeds else STOPPED_QUALITY, started))

        results[MODE_OR_GATE].append(_record(
            path, stem, question_id, quality,
            detection if or_proceeds else None,
            answer if or_proceeds else None,
            STOPPED_COMPLETE if or_proceeds else STOPPED_QUALITY, started,
            extra={"gate": _or_gate_reason(quality, detection)}))

        # ── Mode 3 ───────────────────────────────────────────────────────────
        report(index, f"model-only pass {index}/{total}")
        combined = client.ask_vlm_only(image_bgr, question, prompts=prompts)
        # Mode 3 annotates nothing - no foreground box, no detector boxes - so
        # its card would otherwise show a placeholder where the other two modes
        # show the photograph. Write a plain EXIF-corrected copy instead. NOT
        # the green u2netp box: that component did not run in this mode, and
        # drawing its output here would credit it for work it did not do.
        plain = run_dir / MODE_VLM_ONLY / "images" / f"{stem}.jpg"
        results[MODE_VLM_ONLY].append(_vlm_only_record(
            path, stem, question_id, combined, quality, started,
            image_bgr=image_bgr, image_path=plain))

    report(total, "writing results")
    for mode in MODE_ORDER:
        write_results(results[mode], run_dir / mode)
    return results


def _or_gate_reason(quality: QualityStageResult,
                    detection: DetectionStageResult) -> str:
    """Why mode 2 let this photograph through, in words, so the card can say so
    rather than leaving the operator to infer it."""
    if quality.passed and detection.detections:
        return "quality passed and the detector found the subject"
    if quality.passed:
        return "quality passed; the detector found nothing"
    if detection.detections:
        return (f"quality failed ({quality.score:.1f} / {quality.threshold:.0f}) but "
                f"the detector found the subject - proceeding on the OR")
    return "neither the quality gate nor the detector was satisfied"


def _record(path, stem, question_id, quality, detection, vlm, stopped, started,
            extra=None) -> PipelineRecord:
    record = PipelineRecord(
        filename=path.name, source_path=str(path), stem=stem,
        question_id=question_id, quality=quality, detection=detection, vlm=vlm,
        stopped_at=stopped)
    record.extra.update(extra or {})
    record.extra["elapsed_s"] = round(time.time() - started, 2)
    return record


def _write_plain(image_bgr, dest: Path) -> Optional[str]:
    """The photograph as mode 3 saw it, unannotated."""
    import cv2

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), image_bgr):
            return None
    except Exception:
        return None
    return str(dest)


def _vlm_only_record(path, stem, question_id, combined, classical_quality,
                     started, image_bgr=None, image_path=None) -> PipelineRecord:
    """Mode 3's result, mapped onto the same three panels the other modes fill.

    The quality and detection panels are filled from the model's own judgement
    rather than from u2netp and YOLOX, and both say so: assessed_by="vlm" and a
    presence string instead of boxes. The frame size is carried over from the
    classical pass because it is a property of the file, not a judgement.
    """
    annotated = None
    if image_bgr is not None and image_path is not None:
        annotated = _write_plain(image_bgr, Path(image_path))

    quality = QualityStageResult(
        passed=(combined["quality"] == "good"),
        score=0.0, threshold=0.0, resolution_ok=True, ignore_resolution_used=False,
        failure_reasons=([] if combined["quality"] == "good"
                         else [combined.get("quality_reasoning") or "judged poor"]),
        retake_instructions=[], foreground_box=None, segmentation_used=False,
        whole_frame_score=0.0, width=classical_quality.width,
        height=classical_quality.height, assessed_by="vlm",
        annotated_path=annotated)

    detection = DetectionStageResult(
        detections=[], annotated_path=None,
        model_name=f"{combined['model']} (presence only, no boxes)",
        is_stub=False, note="", presence=combined["subject_present"],
        presence_reasoning=combined.get("subject_reasoning", ""))

    vlm = VLMAnswer(
        answer=combined["answer"], reasoning=combined["reasoning"],
        raw_text=combined.get("raw_text", ""), model=combined["model"],
        elapsed_s=combined.get("elapsed_s", 0.0), error=combined.get("error"),
        is_mock=combined.get("is_mock", False))

    return _record(path, stem, question_id, quality, detection, vlm,
                   STOPPED_COMPLETE, started,
                   extra={"quality_reasoning": combined.get("quality_reasoning", ""),
                          "legs": combined.get("legs", {})})


def _unreadable(path, stem, question_id, error) -> PipelineRecord:
    return PipelineRecord(
        filename=path.name, source_path=str(path), stem=stem,
        question_id=question_id,
        quality=QualityStageResult(
            passed=False, score=0.0, threshold=0.0, resolution_ok=False,
            ignore_resolution_used=False,
            failure_reasons=[f"could not read the file: {error}"],
            retake_instructions=[], foreground_box=None, segmentation_used=False,
            whole_frame_score=0.0, width=0, height=0, error=error),
        stopped_at=STOPPED_QUALITY)


def compare_rows(results: dict) -> list[dict]:
    """One row per image, one column per mode - the comparison the three modes
    exist to support."""
    by_stem: dict = {}
    for mode in MODE_ORDER:
        for record in results.get(mode, []):
            row = by_stem.setdefault(record.stem, {"file": record.filename})
            if record.vlm is None:
                row[mode] = "— stopped"
            else:
                row[mode] = record.vlm.answer.upper() + (" (mock)" if record.vlm.is_mock
                                                         else "")
    return list(by_stem.values())


def agreement(results: dict) -> dict:
    """How often the three modes reached the same answer. The headline number
    for a comparison demo."""
    rows = compare_rows(results)
    if not rows:
        return {"total": 0, "unanimous": 0, "split": 0}
    unanimous = 0
    for row in rows:
        answers = {row.get(m) for m in MODE_ORDER if row.get(m)}
        if len(answers) == 1:
            unanimous += 1
    return {"total": len(rows), "unanimous": unanimous,
            "split": len(rows) - unanimous}


def run_vlm_only(image_paths, question_id: str, cfg, progress_cb: ProgressCb = None,
                 prompts=None) -> list:
    """Mode 3 alone, for iterating on prompts.

    Re-running all three after a wording change would spend the quality and
    detection passes again to produce identical results - and those are the
    expensive stages. This runs only what the prompt actually affects, so modes
    1 and 2 keep their existing results and the comparison stays meaningful.
    """
    from .stage1_quality import build_quality_config, score_image
    from .stage3_vlm import VLMClient
    from quality_check import load_image_bgr

    paths = [Path(p) for p in image_paths]
    total = len(paths)
    question = get_question(question_id)
    if cfg.run_id is None:
        cfg.run_id = new_run_id()
    run_dir = cfg.run_dir

    quality_config = build_quality_config(cfg)
    client = VLMClient(cfg)
    client.ensure_registry()

    records = []
    for index, path in enumerate(paths, start=1):
        if progress_cb:
            progress_cb(index, total, f"mode 3 · {index}/{total}")
        started = time.time()
        try:
            image_bgr = load_image_bgr(path)
        except Exception as exc:
            records.append(_unreadable(path, path.stem, question_id, str(exc)))
            continue
        # The frame size is a property of the file, so it still comes from a
        # classical read - not from a judgement.
        classical = score_image(
            image_bgr, config=quality_config, model_name=cfg.segmentation_model,
            margin_trim=cfg.margin_trim, min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac, ignore_resolution=cfg.ignore_resolution)
        combined = client.ask_vlm_only(image_bgr, question, prompts=prompts)
        plain = run_dir / MODE_VLM_ONLY / "images" / f"{path.stem}.jpg"
        records.append(_vlm_only_record(path, path.stem, question_id, combined,
                                        classical, started, image_bgr=image_bgr,
                                        image_path=plain))

    write_results(records, run_dir / MODE_VLM_ONLY)
    return records
