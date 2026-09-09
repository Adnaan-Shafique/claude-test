"""Data contracts passed between pipeline stages.

Pure stdlib on purpose - importing this module must never pull in cv2, torch,
onnxruntime or gradio, so the orchestrator's shapes can be unit-tested on a
machine with none of the heavy dependencies installed.

Two fields here are NOT in the original integration plan and are load-bearing:

  QualityStageResult.resolution_ok
  QualityStageResult.ignore_resolution_used
      quality_check.assess_quality() computes
          passed = resolution_ok and score >= threshold
      so an image can score 82 against a threshold of 65 and still FAIL, purely
      on the resolution floor. Without these fields the UI renders
      "FAIL 82.1 / 65", which reads to an audience as a scoring bug rather than
      a resolution rejection. See .fail_kind below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Verdict strings, matching run_quality_batch_foreground.py's CSV vocabulary
# exactly so a pipeline row and a CLI row can be compared without translation.
PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

# PipelineRecord.stopped_at values.
STOPPED_QUALITY = "quality"
STOPPED_DETECTION = "detection"
STOPPED_VLM = "vlm"
STOPPED_COMPLETE = "complete"

# Detection.source values. The stub path must be distinguishable from the real
# one everywhere downstream - see the provenance requirement in plan section 3.3.
SOURCE_MODEL = "model"
SOURCE_ANNOTATION = "annotation_file"


@dataclass
class QualityStageResult:
    """Stage 1 output. Mirrors what run_quality_batch_foreground._process_one()
    computes per image, minus the file-copying and CSV-row concerns."""

    passed: bool
    score: float                 # the score the verdict was made on (cropped if segmented)
    threshold: float
    resolution_ok: bool
    ignore_resolution_used: bool
    failure_reasons: list[str]
    retake_instructions: list[str]
    foreground_box: Optional[tuple[int, int, int, int]]   # x1, y1, x2, y2
    segmentation_used: bool
    whole_frame_score: float
    width: int
    height: int
    foreground_score: Optional[float] = None   # None when the segmenter found no plausible box
    mask_area_frac: Optional[float] = None
    mask_mean_alpha: Optional[float] = None
    annotated_path: Optional[str] = None       # green foreground box drawn, for the UI
    error: Optional[str] = None                # set when scoring itself raised

    @property
    def verdict(self) -> str:
        if self.error:
            return ERROR
        return PASS if self.passed else FAIL

    @property
    def fail_kind(self) -> Optional[str]:
        """Why this image failed, so the UI can say the right thing.

        "resolution" - below the min_width/min_height floor, regardless of score.
        "score"      - scored under the threshold.
        "both"       - both of the above.
        None         - it passed, or it errored.
        """
        if self.error or self.passed:
            return None
        below_floor = not self.resolution_ok and not self.ignore_resolution_used
        below_score = self.score < self.threshold
        if below_floor and below_score:
            return "both"
        if below_floor:
            return "resolution"
        return "score"

    @property
    def headline(self) -> str:
        """One line for the UI card, honest about which gate rejected the image."""
        if self.error:
            return f"{ERROR} - {self.error}"
        base = f"{self.verdict} {self.score:.1f} / {self.threshold:.0f}"
        if self.fail_kind == "resolution":
            return f"{base}  (score is fine; failed the {self.width}x{self.height} resolution floor)"
        if self.fail_kind == "both":
            return f"{base}  (also below the resolution floor)"
        return base


@dataclass
class Detection:
    """One detected object. Absolute pixel coordinates, in the same EXIF-corrected
    frame every other stage sees - see the orchestrator's load-once contract."""

    label: str
    confidence: float
    box: list[float]        # x1, y1, x2, y2 absolute px
    source: str             # SOURCE_MODEL | SOURCE_ANNOTATION

    @classmethod
    def from_model_dict(cls, d: dict, source: str = SOURCE_MODEL) -> "Detection":
        """inference.run_inference() returns {"class", "confidence", "box"}; this
        dataclass calls it `label`. Doing the rename here, in one place, is what
        keeps the real and stub detector backends interchangeable."""
        return cls(
            label=str(d["class"]),
            confidence=float(d["confidence"]),
            box=[float(v) for v in d["box"]],
            source=source,
        )

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionStageResult:
    """Stage 2 output. `model_name` is what the UI prints as provenance and must
    never imply a model ran when one did not."""

    detections: list[Detection]
    annotated_path: Optional[str]
    model_name: str          # e.g. "human annotation (no model loaded)"
    note: str = ""           # e.g. "no annotation file found for IMG_0042.txt"

    # Explicit, never inferred. An earlier version derived this by looking for
    # "model" inside model_name, which is a substring of "no model loaded" and
    # therefore reported stub results as real ones. Provenance is an honesty
    # requirement (plan section 3.3), so it is set by the detector, not guessed.
    is_stub: bool = False

    def top(self) -> Optional[Detection]:
        return max(self.detections, key=lambda d: d.confidence, default=None)


@dataclass
class VLMAnswer:
    """Stage 3 output. `answer` is always one of yes / no / unknown - the parser
    guarantees it, so the UI chip never has to handle a surprise value."""

    answer: str              # "yes" | "no" | "unknown"
    reasoning: str
    raw_text: str
    model: str
    elapsed_s: float
    error: Optional[str] = None
    is_mock: bool = False

    @property
    def chip(self) -> str:
        return self.answer.upper()

    @property
    def provenance(self) -> str:
        if self.is_mock:
            return "MOCK - GPU server unreachable"
        return self.model


@dataclass
class PipelineRecord:
    """One image's journey through all three legs.

    `stem` is the join key across every leg and is captured at upload time from
    the ORIGINAL filename. Both of the renaming paths in the existing code -
    run_quality_batch.unique_destination() ("name__1.jpg", an integer counter)
    and batch_ui._stage_uploaded_files() ("name__a1b2c3d4.jpg", a random uuid)
    - destroy it, and the uuid one is not even stable across two runs of the
    same file. Never re-derive this from an output path.
    """

    filename: str
    source_path: str
    stem: str
    question_id: str
    quality: QualityStageResult
    detection: Optional[DetectionStageResult] = None
    vlm: Optional[VLMAnswer] = None
    stopped_at: str = STOPPED_COMPLETE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat_row(self) -> dict:
        """One flat CSV row per image, per plan section 5.

        Deliberately NOT the same schema as quality_review_foreground.csv - that
        file's columns are read by batch_ui.py and must stay byte-identical, so
        this is a separate artefact with a superset of the interesting fields.
        """
        q = self.quality
        d = self.detection
        v = self.vlm
        return {
            "filename": self.filename,
            "stem": self.stem,
            "question": self.question_id,
            "quality_verdict": q.verdict,
            "quality_score": round(q.score, 1),
            "quality_threshold": q.threshold,
            "quality_fail_kind": q.fail_kind or "",
            "resolution_ok": q.resolution_ok,
            "width": q.width,
            "height": q.height,
            "whole_frame_score": round(q.whole_frame_score, 1),
            "foreground_score": round(q.foreground_score, 1) if q.foreground_score is not None else None,
            "segmentation_used": q.segmentation_used,
            "failure_reasons": "; ".join(q.failure_reasons),
            "retake_instructions": " | ".join(q.retake_instructions),
            "detection_source": (d.model_name if d else ""),
            "detections": "; ".join(f"{x.label}:{x.confidence:.2f}" for x in d.detections) if d else "",
            "detection_note": (d.note if d else ""),
            "vlm_answer": (v.answer if v else ""),
            "vlm_reasoning": (v.reasoning if v else ""),
            "vlm_model": (v.provenance if v else ""),
            "vlm_elapsed_s": (round(v.elapsed_s, 2) if v else None),
            "vlm_error": (v.error or "" if v else ""),
            "stopped_at": self.stopped_at,
            "source_path": self.source_path,
        }


FLAT_ROW_COLUMNS = [
    "filename", "stem", "question",
    "quality_verdict", "quality_score", "quality_threshold", "quality_fail_kind",
    "resolution_ok", "width", "height",
    "whole_frame_score", "foreground_score", "segmentation_used",
    "failure_reasons", "retake_instructions",
    "detection_source", "detections", "detection_note",
    "vlm_answer", "vlm_reasoning", "vlm_model", "vlm_elapsed_s", "vlm_error",
    "stopped_at", "source_path",
]
