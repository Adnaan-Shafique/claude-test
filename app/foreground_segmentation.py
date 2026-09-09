"""
VENDORED, MODIFIED. This is a copy of the original foreground_segmentation.py
carried by the integrated demo pipeline, because the original toolset does not
run on the demo host. One function differs from upstream: _get_session() is now
tolerant of rembg versions that no longer accept a sess_opts keyword (see the
comment there). Do not merge this file back over the original without carrying
that change with it.

Automatic foreground/background separation - a training-data curation tool,
NOT part of the runtime edge/server pipeline.

Historical PM photos predate any fine-tuned detector, so there's no box yet
to crop quality_check.py's cues to (see README.md §5 on whole-frame Blur%
averaging getting fooled by sky-dominated shots, e.g. a lightning arrestor
shot from below against mostly sky). This module bootstraps that.

Uses U2-Net (via the `rembg` package), not SAM2: this task only needs
automatic salient-object/background separation, not promptable/interactive
segmentation, so a dedicated single-pass saliency network is a much better
fit than a large promptable transformer - u2netp's checkpoint is ~4.7MB vs.
SAM2's ~150MB+ even at its smallest, no prompt point/box needed, and it runs
fast on CPU, which matters for a batch pass over a large historical-photo
corpus. Apache-2.0 weights, MIT-licensed wrapper.

The resulting mask is converted to a padded, sanity-bounded bounding BOX,
never used as a raw mask - quality_check.assess_quality() crops to the box,
because running edge-sensitive cues (Laplacian, Tenengrad, Canny, FFT) on
pixels bounded by an irregular mask edge would read the mask boundary itself
as a fake sharp edge and corrupt the cue.

Once RF-DETR-Large / NanoDet-Plus are fine-tuned on real classes, their boxes
replace this entirely for both threshold recalibration and runtime inference
- this module is retired at that point, not folded into the shipped pipeline.

Requires (NOT in requirements.txt - curation-only dependency, never imported
by the runtime quality_check module):
    pip install rembg onnxruntime
First call downloads the model into <project_root>/models/ (see MODELS_DIR
below) - project-local rather than the user's home directory, so the weights
travel with the repo (git-ignored, not committed - see .gitignore) instead
of being scattered per-machine, and a fresh clone works with zero manual
environment-variable setup on any teammate's machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_MODEL_NAME = "u2netp"  # ~4.7MB, lightest/fastest; use "u2net" (~176MB) for higher quality

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# rembg's U2NET_HOME points at a flat directory holding the .onnx file(s)
# directly (no extra nesting) - read fresh on every call, not cached at
# import time, so this is safe to set here even though other modules may
# import rembg/onnxruntime first. setdefault() respects a value already set
# in the environment (e.g. a shared team-wide model cache), only falling
# back to the project-local folder when nothing else was configured.
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))


@dataclass
class ForegroundBox:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels, clipped to image bounds
    mask_area_frac: float           # box area / frame area, post-trim
    mask_mean_alpha: float          # mean soft-mask value inside the box, as a rough confidence proxy


_session = None  # lazy singleton - loading the ONNX model is the expensive part
_session_model_name = None


def _get_session(model_name: str, sess_opts=None):
    global _session, _session_model_name
    if _session is not None and _session_model_name == model_name:
        return _session
    try:
        from rembg import new_session
    except ImportError as exc:
        raise ImportError(
            "rembg is not installed. This is a one-time data-curation dependency, "
            "not part of the runtime pipeline - install with:\n"
            "  pip install rembg onnxruntime\n"
            "The model weights download automatically on first use."
        ) from exc

    # MODIFIED FROM THE ORIGINAL - see the "Vendored, modified" note at the top
    # of this file.
    #
    # rembg >= ~2.0.6x builds its own ort.SessionOptions inside new_session()
    # and hands it to the session class POSITIONALLY:
    #
    #     return session_class(model_name, sess_opts, providers, *args, **kwargs)
    #
    # so passing sess_opts= as a keyword makes it a second value for the same
    # parameter: "BaseSession.__init__() got multiple values for argument
    # 'sess_opts'". That fired even for sess_opts=None, which made the segmenter
    # unloadable on rembg 2.0.69 regardless of what the caller asked for.
    if sess_opts is None:
        _session = new_session(model_name)
    else:
        try:
            _session = new_session(model_name, sess_opts=sess_opts)
        except TypeError:
            # This rembg no longer accepts the keyword. It does read
            # OMP_NUM_THREADS for inter_op_num_threads, so translate the caller's
            # intent into the lever this version actually exposes rather than
            # silently dropping the thread cap - run_quality_batch_foreground.py
            # relies on it to stop N worker processes oversubscribing the box.
            requested = getattr(sess_opts, "intra_op_num_threads", 0) or \
                getattr(sess_opts, "inter_op_num_threads", 0)
            if requested:
                os.environ["OMP_NUM_THREADS"] = str(int(requested))
            _session = new_session(model_name)
    _session_model_name = model_name
    return _session


def preload(model_name: str = DEFAULT_MODEL_NAME, sess_opts=None) -> None:
    """Load (and, on first call, download) the segmentation model once, up front.
    Call this before a batch loop so a missing-dependency error or the one-time
    model download surfaces immediately, instead of being caught and logged as a
    per-image ERROR thousands of times over a large historical-photo corpus.

    sess_opts: an onnxruntime.SessionOptions, e.g. with intra_op_num_threads=1 -
    used by run_quality_batch_foreground.py's multiprocess mode so each worker
    process's ONNX Runtime session doesn't ALSO try to parallelize across every
    CPU core internally, which would oversubscribe the machine once multiple
    worker processes are already running one per core."""
    _get_session(model_name, sess_opts=sess_opts)


def get_foreground_box(
    image_bgr: np.ndarray,
    model_name: str = DEFAULT_MODEL_NAME,
    margin_trim_frac: float = 0.05,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.95,
    mask_threshold: int = 128,
    sess_opts=None,
) -> Optional[ForegroundBox]:
    """Automatic: no prompt needed, no fine-tuning, no training data - u2netp
    segments the single most salient foreground object directly. Returns None
    (caller should fall back to a whole-frame quality check) if the resulting
    box is implausible - near-empty or near-full-frame - which is the sanity
    bound against the segmenter guessing wrong on an unfamiliar telecom-
    equipment photo."""
    from rembg import remove

    session = _get_session(model_name, sess_opts=sess_opts)
    height, width = image_bgr.shape[:2]

    soft_mask = remove(image_bgr[:, :, ::-1], session=session, only_mask=True)  # rembg expects RGB
    if soft_mask.ndim == 3:
        soft_mask = soft_mask[:, :, 0]

    mask = soft_mask >= mask_threshold
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    # Trim inward before handing off a *box* - shaves off any background
    # fringe/halo the segmenter included right at its own boundary, rather
    # than passing it through to the rectangular crop quality_check.py takes.
    box_w, box_h = x2 - x1, y2 - y1
    trim_x, trim_y = int(box_w * margin_trim_frac), int(box_h * margin_trim_frac)
    x1, x2 = x1 + trim_x, x2 - trim_x
    y1, y2 = y1 + trim_y, y2 - trim_y
    if x2 <= x1 or y2 <= y1:
        return None

    area_frac = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if not (min_area_frac <= area_frac <= max_area_frac):
        return None

    mean_alpha = float(np.mean(soft_mask[y1:y2, x1:x2]) / 255.0)
    return ForegroundBox(box=(x1, y1, x2, y2), mask_area_frac=area_frac, mask_mean_alpha=mean_alpha)


def draw_box(image_bgr: np.ndarray, box: Tuple[int, int, int, int],
             color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> np.ndarray:
    """Returns a copy of image_bgr with the box drawn on it - for visually verifying the
    segmenter's crop region in gradio_app.py / review_ui.py, not used by the scoring path."""
    annotated = image_bgr.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated
