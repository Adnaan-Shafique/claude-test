"""PipelineConfig - every path, threshold and flag the demo pipeline reads.

Why the root is defined here and nowhere else
---------------------------------------------
app/detection_demo/inference.py and app/detection_demo/model_registry.py both do

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

which silently points somewhere wrong the moment either file moves. This module
resolves the root once, from a known anchor, and every pipeline stage takes it
from the config object rather than recomputing it. Override `project_root` on
the config to run against a different checkout without editing code.

Pure stdlib - no cv2, torch or gradio - so it can be imported and asserted
against on any machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# app/pipeline/config.py -> app/pipeline -> app -> <project root>
_THIS = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = _THIS.parents[2]

# The demo VLM. gpu_api_server_v6.py eager-loads this one
# (EAGER_LOAD = ["qwen3-vl", "mistral"]), so there is no cold-load stall on the
# first question of the demo. Both VLMs cap at 4 images per prompt via
# limit_mm_per_prompt, which is well clear of the 2 that "full+crop" sends.
DEFAULT_VLM_MODEL = "qwen3-vl"
DEFAULT_GPU_URL = "http://10.66.98.137:5432"

VLM_MODE_LIVE = "live"
VLM_MODE_MOCK = "mock"

SEND_FULL = "full"
SEND_FULL_CROP = "full+crop"


@dataclass
class PipelineConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    project_root: Path = DEFAULT_PROJECT_ROOT

    # ── Stage 1: quality ─────────────────────────────────────────────────────
    quality_threshold: float = 65.0        # QualityConfig.pass_threshold default
    ignore_resolution: bool = False
    min_width: Optional[int] = None        # None = QualityConfig's own default (640)
    min_height: Optional[int] = None       # None = QualityConfig's own default (480)
    segmentation_model: str = "u2netp"     # foreground_segmentation.DEFAULT_MODEL_NAME
    margin_trim: float = 0.05
    min_area_frac: float = 0.02
    max_area_frac: float = 0.95
    run_downstream_on_fail: bool = False   # the "don't dead-end the demo" toggle

    # ── Stage 2: detection ───────────────────────────────────────────────────
    use_model: bool = False                # False = AnnotationFileDetector (the stub)
    conf_thresh: float = 0.3
    # Deterministic dummy-confidence band for stub detections that carry no
    # confidence column. Re-running the same image on stage must show the same
    # number, so this is a seeded hash of "<stem>:<index>", never random().
    stub_conf_range: tuple[float, float] = (0.88, 0.97)

    # ── Stage 3: VLM ─────────────────────────────────────────────────────────
    gpu_url: str = DEFAULT_GPU_URL
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_mode: str = VLM_MODE_LIVE
    vlm_send_mode: str = SEND_FULL
    # Plan section 4.4 pads a detection crop by ~15% of the box. The real
    # annotation we have is 0.0505 x 0.0543 of the frame (0.27% by area), so
    # 15% of that box is still a ~230x190px sliver on a 4000x3000 photo - very
    # little context for a VLM to judge "is the sky above this blocked?".
    # Pad to at least this fraction of the frame's shorter side instead.
    crop_min_frame_frac: float = 0.20
    # Repeatability beats flair when running live. Note the server validates
    # repetition_penalty with ge=1.0, so 1.0 is the floor, not 0.
    max_new_tokens: int = 300
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0                          # server maps 0 -> -1 (disabled)
    repetition_penalty: float = 1.0
    request_timeout_s: int = 180

    # ── Output ───────────────────────────────────────────────────────────────
    run_id: Optional[str] = None
    _annotation_dir: Optional[Path] = field(default=None, repr=False)
    _runs_dir: Optional[Path] = field(default=None, repr=False)
    _models_dir: Optional[Path] = field(default=None, repr=False)

    # ── Derived paths ────────────────────────────────────────────────────────
    @property
    def annotation_dir(self) -> Path:
        return self._annotation_dir or (self.project_root / "data" / "labels")

    @annotation_dir.setter
    def annotation_dir(self, value) -> None:
        self._annotation_dir = Path(value) if value else None

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir or (self.project_root / "demo_runs")

    @runs_dir.setter
    def runs_dir(self, value) -> None:
        self._runs_dir = Path(value) if value else None

    @property
    def models_dir(self) -> Path:
        """Where u2netp.onnx lives. foreground_segmentation.py already points
        rembg's U2NET_HOME at <project_root>/models via os.environ.setdefault,
        so this must agree with it or the segmenter will try to download."""
        return self._models_dir or (self.project_root / "models")

    @models_dir.setter
    def models_dir(self, value) -> None:
        self._models_dir = Path(value) if value else None

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / (self.run_id or "current")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def ensure_u2net_home(self) -> str:
        """Point rembg at the repo-local model folder before anything imports it.

        foreground_segmentation.py does this itself with os.environ.setdefault at
        import time; calling this first makes the demo process independent of
        import order, and setdefault means an operator-provided U2NET_HOME
        (a shared team cache) still wins.
        """
        os.environ.setdefault("U2NET_HOME", str(self.models_dir))
        return os.environ["U2NET_HOME"]

    def resolve_annotation_dir(self, images_dir=None) -> Path:
        """Where the label files actually are.

        The YOLO/CVAT sidecar convention puts <stem>.txt next to <stem>.jpg in
        the same folder, which is how the demo photos are laid out. An explicit
        annotation_dir always wins; otherwise, if the photo folder carries .txt
        files, that is the answer. Falling through to an empty data/labels would
        report every image as "no annotation file found" - the detector working
        perfectly and finding nothing, which is the confusing failure.
        """
        if self._annotation_dir is not None:
            return self._annotation_dir
        if images_dir is not None:
            images_dir = Path(images_dir)
            if images_dir.is_dir() and any(
                q.name not in ("classes.txt", "obj.names") for q in images_dir.glob("*.txt")
            ):
                return images_dir
        return self.annotation_dir

    def segmentation_model_path(self) -> Path:
        return self.models_dir / f"{self.segmentation_model}.onnx"

    def validate(self) -> list[str]:
        """Non-fatal config problems, worth showing in the UI rather than raising.
        Returns a list of human-readable warnings; empty means everything checks out."""
        problems: list[str] = []
        if not self.project_root.exists():
            problems.append(f"project_root does not exist: {self.project_root}")
        if not self.segmentation_model_path().exists():
            problems.append(
                f"{self.segmentation_model}.onnx not found at {self.segmentation_model_path()} - "
                f"the segmenter would try to download it on first use, which must not happen on stage"
            )
        if not self.use_model and not self.annotation_dir.exists():
            problems.append(
                f"annotation_dir does not exist: {self.annotation_dir} - "
                f"every image will report 'no annotation file found'"
            )
        if self.vlm_mode not in (VLM_MODE_LIVE, VLM_MODE_MOCK):
            problems.append(f"vlm_mode must be '{VLM_MODE_LIVE}' or '{VLM_MODE_MOCK}', got {self.vlm_mode!r}")
        if self.vlm_send_mode not in (SEND_FULL, SEND_FULL_CROP):
            problems.append(f"vlm_send_mode must be '{SEND_FULL}' or '{SEND_FULL_CROP}', got {self.vlm_send_mode!r}")
        if self.repetition_penalty < 1.0:
            problems.append("repetition_penalty must be >= 1.0 (the GPU server validates this and will 422)")
        if not (0.0 <= self.temperature <= 2.0):
            problems.append("temperature must be within [0.0, 2.0] (server-side validation)")
        lo, hi = self.stub_conf_range
        if not (0.0 < lo <= hi <= 1.0):
            problems.append(f"stub_conf_range must satisfy 0 < lo <= hi <= 1, got {self.stub_conf_range}")
        return problems


def default_config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    for key, value in overrides.items():
        # The three path fields are properties backed by private attrs; setattr
        # routes through their setters, so this works for them too.
        if not hasattr(cfg, key):
            raise AttributeError(f"PipelineConfig has no field {key!r}")
        setattr(cfg, key, value)
    return cfg
