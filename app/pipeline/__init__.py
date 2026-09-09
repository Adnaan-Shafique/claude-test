"""Field Ops demo pipeline - three legs behind one orchestrator.

Only the stdlib-pure modules are re-exported here. stage1_quality (and later
stage2_detect / stage3_vlm) are NOT imported eagerly, because importing them
pulls in cv2 / onnxruntime / rembg - so `from pipeline import schemas` stays
usable on a machine with none of that installed.
"""
from . import config, questions, schemas  # noqa: F401
from .config import PipelineConfig, default_config  # noqa: F401
from .questions import QUESTION_CHOICES, QUESTIONS, Question, get_question  # noqa: F401
from .schemas import (  # noqa: F401
    Detection,
    DetectionStageResult,
    PipelineRecord,
    QualityStageResult,
    VLMAnswer,
)

__all__ = [
    "config", "questions", "schemas",
    "PipelineConfig", "default_config",
    "Question", "QUESTIONS", "QUESTION_CHOICES", "get_question",
    "QualityStageResult", "Detection", "DetectionStageResult", "VLMAnswer",
    "PipelineRecord",
]
