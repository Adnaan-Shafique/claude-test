"""Stage 1 - the quality gate, as an in-process single-image function.

Extracted from run_quality_batch_foreground._process_one() so the CLI and the
demo can never drift. _process_one() mixes three concerns - scoring, copying
files into PASS/FAIL, and building a CSV row; only the first belongs here.
See PATCH_process_one.md for the drop-in replacement that makes the CLI call
this function, keeping its CSV columns byte-identical (batch_ui.py reads them).

Why in-process rather than reusing the CLI's ProcessPoolExecutor: the demo
uploads 5-20 images, and batch_ui.py's own comment records ~20s of cold-start
for 15 workers each importing cv2/onnxruntime/rembg and building an ONNX
session. At this batch size that startup cost is pure loss.

THREADING NOTE. _worker_init() in the CLI sets cv2.setNumThreads(1) and
onnxruntime intra_op_num_threads=1. That cap exists only because N worker
PROCESSES were each trying to parallelise across every core and oversubscribing
the machine. In single-process mode there is no such contention, so OpenCV is
left at its default here - capping it would make the demo slower for no reason.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .schemas import QualityStageResult

# The existing modules live in app/, one level up from app/pipeline/.
_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

try:
    from quality_check import (            # noqa: E402
        CUE_MESSAGES,
        QualityConfig,
        assess_quality,
        load_image_bgr,
    )
    from foreground_segmentation import (  # noqa: E402
        DEFAULT_MODEL_NAME,
        draw_box,
        get_foreground_box,
        preload,
    )
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    raise ImportError(
        f"Stage 1 needs quality_check.py and foreground_segmentation.py on the "
        f"path (looked in {_APP_DIR}), plus their dependencies:\n"
        f"  pip install opencv-python numpy pillow rembg onnxruntime\n"
        f"Original error: {exc}"
    ) from exc

RESOLUTION_REASON = "resolution_too_low"


def build_quality_config(cfg) -> QualityConfig:
    """A QualityConfig from a PipelineConfig. Only overrides what the demo
    actually exposes; every cue weight keeps quality_check's own default, whose
    total is 1.10 by design (the paper's 8 cues sum to 1.00 and the occlusion
    cue was added on top at its own 0.10). assess_quality() renormalises the
    vector to sum to 1 internally regardless."""
    qc = QualityConfig()
    if cfg.quality_threshold is not None:
        qc.pass_threshold = float(cfg.quality_threshold)
    if cfg.min_width is not None:
        qc.min_width = int(cfg.min_width)
    if cfg.min_height is not None:
        qc.min_height = int(cfg.min_height)
    return qc


def preload_segmenter(cfg) -> None:
    """Load u2netp once, up front, so a missing dependency or a missing model
    file surfaces immediately at UI startup rather than as a per-image error
    mid-demo. Sets U2NET_HOME first so rembg looks in <project_root>/models and
    never reaches for the network."""
    cfg.ensure_u2net_home()
    preload(model_name=cfg.segmentation_model or DEFAULT_MODEL_NAME)


def score_image(
    image_bgr,
    config: QualityConfig,
    model_name: str = DEFAULT_MODEL_NAME,
    margin_trim: float = 0.05,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.95,
    ignore_resolution: bool = False,
) -> QualityStageResult:
    """Score one already-loaded BGR array, whole-frame AND foreground-cropped,
    preferring the cropped score when the segmenter proposes a plausible box.

    image_bgr must come from quality_check.load_image_bgr() - it applies
    ImageOps.exif_transpose, and every downstream stage reuses this same array.
    Re-opening the file per stage makes the quality box and the detection box
    disagree on any photo a phone rotated via EXIF.
    """
    try:
        whole = assess_quality(image_bgr, config=config)

        box_result = get_foreground_box(
            image_bgr,
            model_name=model_name,
            margin_trim_frac=margin_trim,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
        )
        if box_result is not None:
            # roi_box MUST be passed by keyword. The real signature is
            #   assess_quality(image, config=None, return_intermediates=False, roi_box=None)
            # so a positional third argument binds to return_intermediates and
            # the ROI is silently ignored - no error, just quietly wrong scores.
            used = assess_quality(image_bgr, config=config, roi_box=box_result.box)
            segmentation_used = True
        else:
            used = whole
            segmentation_used = False
    except Exception as exc:
        height, width = (image_bgr.shape[:2] if getattr(image_bgr, "shape", None) else (0, 0))
        return QualityStageResult(
            passed=False, score=0.0, threshold=float(config.pass_threshold),
            resolution_ok=False, ignore_resolution_used=ignore_resolution,
            failure_reasons=[f"could not process: {exc}"], retake_instructions=[],
            foreground_box=None, segmentation_used=False, whole_frame_score=0.0,
            width=int(width), height=int(height), error=str(exc),
        )

    # Same effective-verdict logic as _process_one(), so a demo verdict and a CLI
    # verdict for the same file always agree.
    reasons = list(used.failure_reasons)
    instructions = list(used.retake_instructions)
    if ignore_resolution:
        reasons = [r for r in reasons if r != RESOLUTION_REASON]
        resolution_message = CUE_MESSAGES[RESOLUTION_REASON]
        instructions = [i for i in instructions if i != resolution_message]
        effective_passed = used.score >= config.pass_threshold
    else:
        effective_passed = used.passed

    return QualityStageResult(
        passed=bool(effective_passed),
        score=float(used.score),
        threshold=float(config.pass_threshold),
        resolution_ok=bool(used.resolution_ok),
        ignore_resolution_used=bool(ignore_resolution),
        failure_reasons=reasons,
        retake_instructions=instructions,
        foreground_box=tuple(box_result.box) if box_result is not None else None,
        segmentation_used=segmentation_used,
        whole_frame_score=float(whole.score),
        width=int(used.width),
        height=int(used.height),
        foreground_score=float(used.score) if segmentation_used else None,
        mask_area_frac=round(box_result.mask_area_frac, 4) if box_result is not None else None,
        mask_mean_alpha=round(box_result.mask_mean_alpha, 4) if box_result is not None else None,
    )


def annotate_for_gallery(image_bgr, result: QualityStageResult, dest_path) -> Optional[str]:
    """Write a copy with the green foreground box drawn on it, mirroring
    batch_ui._annotate_for_gallery / review_ui's display convention, so the UI
    shows exactly the region that was scored. Falls back to an unannotated copy
    on any write error rather than failing the image."""
    import cv2

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = image_bgr
        if result.foreground_box is not None:
            out = draw_box(image_bgr, result.foreground_box)
        if not cv2.imwrite(str(dest_path), out):
            raise IOError("cv2.imwrite returned False")
    except Exception:
        try:
            if not cv2.imwrite(str(dest_path), image_bgr):
                return None
        except Exception:
            return None
    result.annotated_path = str(dest_path)
    return result.annotated_path


def score_path(path, cfg, config: Optional[QualityConfig] = None):
    """Convenience for tests and the CLI: load once, score once.
    Returns (image_bgr, QualityStageResult) so the caller can reuse the array
    for the downstream stages rather than re-opening the file."""
    config = config or build_quality_config(cfg)
    image_bgr = load_image_bgr(Path(path))
    result = score_image(
        image_bgr,
        config=config,
        model_name=cfg.segmentation_model,
        margin_trim=cfg.margin_trim,
        min_area_frac=cfg.min_area_frac,
        max_area_frac=cfg.max_area_frac,
        ignore_resolution=cfg.ignore_resolution,
    )
    return image_bgr, result
