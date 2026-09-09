"""Stage 2 - object detection, with the no-model stub as a swappable backend.

The stub is a BACKEND, not a special case in the UI: both detectors return the
same DetectionStageResult and draw through the same helper, so when a real
checkpoint lands, flipping cfg.use_model changes nothing else. The UI only ever
reads .model_name and .is_stub for provenance.

No torch, YOLOX or ultralytics is imported at module scope. In stub mode
nothing torch-shaped is imported at all.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from .detect_draw import draw_detections
from .schemas import SOURCE_ANNOTATION, SOURCE_MODEL, Detection, DetectionStageResult

# A coordinate above this cannot be a normalized YOLO value, so the file is
# absolute pixel xyxy. 1.5 rather than 1.0 leaves room for a rounding artefact
# in a normalized file without misreading it as pixels.
ABSOLUTE_COORD_THRESHOLD = 1.5

STUB_MODEL_NAME = "human annotation (no model loaded)"


# ─────────────────────────────── Class names ─────────────────────────────────

def load_class_names(annotation_dir: Path) -> list[str]:
    """classes.txt, then dataset.yaml's `names:`, then nothing (callers fall
    back to class_<id>). Order per plan section 3.3."""
    annotation_dir = Path(annotation_dir)

    classes_txt = annotation_dir / "classes.txt"
    if classes_txt.exists():
        names = [ln.strip() for ln in classes_txt.read_text(errors="replace").splitlines()]
        names = [n for n in names if n and not n.startswith("#")]
        if names:
            return names

    for yaml_name in ("dataset.yaml", "data.yaml", "obj.names"):
        path = annotation_dir / yaml_name
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        if yaml_name == "obj.names":
            # Darknet/CVAT convention: bare newline-separated names.
            names = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if names:
                return names
            continue
        names = _names_from_yaml(text)
        if names:
            return names
    return []


def _names_from_yaml(text: str) -> list[str]:
    """Read a `names:` key without hard-depending on pyyaml.

    pyyaml is present transitively today (gradio pulls it) but is not a declared
    dependency, and a missing class list must degrade to class_<id> rather than
    crash the detector.
    """
    try:
        import yaml  # noqa: F401
        data = yaml.safe_load(text) or {}
        names = data.get("names")
        if isinstance(names, dict):
            # {0: 'a', 1: 'b'} - order by index, not by insertion.
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list):
            return [str(n) for n in names]
        return []
    except Exception:
        pass

    # Fallback: inline list `names: [a, b]` or a block of `  - a` entries.
    inline = re.search(r"^names:\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    if inline:
        return [p.strip().strip("'\"") for p in inline.group(1).split(",") if p.strip()]
    block = re.search(r"^names:\s*\n((?:\s*-\s*.+\n?)+)", text, re.MULTILINE)
    if block:
        return [ln.strip().lstrip("-").strip().strip("'\"")
                for ln in block.group(1).splitlines() if ln.strip()]
    return []


# ─────────────────────────────── Line parsing ────────────────────────────────

def _stub_confidence(stem: str, index: int, conf_range: tuple[float, float]) -> float:
    """Deterministic dummy confidence.

    Seeded on "<stem>:<index>", never random() - re-running the same image live
    and getting a different number is a credibility problem, not a cosmetic one.
    """
    lo, hi = conf_range
    digest = hashlib.md5(f"{stem}:{index}".encode("utf-8")).hexdigest()
    return round(lo + (int(digest[:4], 16) % 900) / 900.0 * (hi - lo), 4)


def parse_annotation_line(line: str, width: int, height: int) -> Optional[dict]:
    """One label line -> {label_token, class_id, box, confidence} or None.

    Raises ValueError with a human-readable reason for a malformed line, so the
    caller can note it and carry on rather than losing the whole image.
    """
    tokens = line.split()
    if len(tokens) < 5:
        raise ValueError(f"expected at least 5 fields, got {len(tokens)}")

    head, rest = tokens[0], tokens[1:]
    try:
        class_id: Optional[int] = int(head)
        label_token = None
    except ValueError:
        # Field annotators often write names rather than ids.
        class_id, label_token = None, head

    try:
        coords = [float(v) for v in rest[:4]]
    except ValueError:
        raise ValueError(f"coordinates are not numeric: {rest[:4]}") from None

    confidence = None
    if len(rest) >= 5:
        try:
            confidence = float(rest[4])
        except ValueError:
            raise ValueError(f"confidence is not numeric: {rest[4]!r}") from None

    # Format auto-detect, per file line rather than per file: a mixed file is
    # pathological but should still yield what it can.
    if any(abs(c) > ABSOLUTE_COORD_THRESHOLD for c in coords):
        x1, y1, x2, y2 = coords
    else:
        cx, cy, bw, bh = coords
        if bw <= 0 or bh <= 0:
            raise ValueError(f"non-positive box size w={bw} h={bh}")
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    return {
        "class_id": class_id,
        "label_token": label_token,
        "box": [x1, y1, x2, y2],
        "confidence": confidence,
    }


# ─────────────────────────────── Backends ────────────────────────────────────

class AnnotationFileDetector:
    """Reads human annotations from <annotation_dir>/<stem>.txt.

    Returns the identical shape the real detector does, so nothing downstream
    can tell them apart - except .is_stub and .model_name, which exist precisely
    so the UI can say so out loud.
    """

    is_stub = True
    name = STUB_MODEL_NAME

    def __init__(self, cfg, question=None, annotation_dir=None):
        self.cfg = cfg
        self.question = question
        self.annotation_dir = Path(
            annotation_dir if annotation_dir is not None
            else cfg.resolve_annotation_dir(question_id=getattr(question, "id", None)))
        self.conf_range = tuple(cfg.stub_conf_range)

        # A classes.txt / dataset.yaml in the label folder is authoritative.
        # Falling back to the question's own names is what makes a bare
        # single-class CVAT export readable: every such export numbers its only
        # class 0, so "0" means a hazard sign in one export and a GPS antenna in
        # another. The active question is the only thing that disambiguates it.
        self.class_names = load_class_names(self.annotation_dir)
        self.class_names_source = "classes.txt / dataset.yaml"
        if not self.class_names and question is not None:
            self.class_names = list(getattr(question, "default_class_names", []) or [])
            self.class_names_source = f"question {question.id!r} defaults"

    def label_for(self, class_id: Optional[int], label_token: Optional[str]) -> str:
        if label_token is not None:
            return label_token
        if class_id is not None and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def detect(self, image_bgr, stem: str, dest_path=None) -> DetectionStageResult:
        height, width = image_bgr.shape[:2]
        path = self.annotation_dir / f"{stem}.txt"

        if not path.exists():
            # Zero detections, a prominent note, and NO exception - the VLM can
            # still answer from the image alone, so this must not stop the leg.
            return DetectionStageResult(
                detections=[], annotated_path=None, model_name=self.name, is_stub=True,
                note=f"no annotation file found for {stem}.txt in {self.annotation_dir}",
            )

        detections: list[Detection] = []
        notes: list[str] = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return DetectionStageResult(
                detections=[], annotated_path=None, model_name=self.name, is_stub=True,
                note=f"could not read {path.name}: {exc}",
            )

        for lineno, raw in enumerate(lines, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                parsed = parse_annotation_line(line, width, height)
            except ValueError as exc:
                notes.append(f"{path.name} line {lineno}: {exc}")
                continue
            index = len(detections)
            confidence = parsed["confidence"]
            if confidence is None:
                confidence = _stub_confidence(stem, index, self.conf_range)
            detections.append(Detection(
                label=self.label_for(parsed["class_id"], parsed["label_token"]),
                confidence=float(confidence),
                box=parsed["box"],
                source=SOURCE_ANNOTATION,
            ))

        if not detections and not notes:
            notes.append(f"{path.name} contains no usable annotation lines")
        if any(d.label.startswith("class_") for d in detections):
            notes.append("no classes.txt, dataset.yaml or question default - "
                         "labels shown as class_<id>")
        elif self.class_names_source.startswith("question"):
            # Say where the names came from. They are inferred from the selected
            # question, not read from the data, and that distinction matters if
            # the wrong question is picked for a folder.
            notes.append(f"class names from {self.class_names_source}")

        result = DetectionStageResult(
            detections=detections, annotated_path=None, model_name=self.name,
            is_stub=True, note="; ".join(notes),
        )
        if dest_path is not None and detections:
            render(image_bgr, result, dest_path)
        return result


class ModelDetector:
    """Wraps the real detector. Zero changes to the detection logic itself.

    Every heavy import happens inside __init__/detect, never at module scope -
    inference.py imports torch and injects YOLOX paths on import, and
    model_registry.py imports eval_utils, none of which should run in stub mode.
    """

    is_stub = False

    def __init__(self, cfg):
        self.cfg = cfg
        try:
            import model_registry  # noqa: F401  (heavy: eval_utils)
            import inference       # noqa: F401  (heavy: torch, YOLOX sys.path)
        except ImportError as exc:
            raise ImportError(
                "use_model=True needs inference.py and model_registry.py on the path, "
                "plus torch, YOLOX and a trained checkpoint. None of that ships with "
                "the demo pipeline - leave cfg.use_model=False to use the annotation "
                f"stub. Original error: {exc}"
            ) from exc
        self._inference = inference
        entries = model_registry.get_available_models()
        if not entries:
            raise RuntimeError(
                "use_model=True but model_registry.get_available_models() found no "
                "checkpoints under models/yolox/runs or models/yolo/runs."
            )
        self.entry = entries[0]
        self.name = f"{self.entry['model']} ({self.entry['framework']})"

    def detect(self, image_bgr, stem: str, dest_path=None) -> DetectionStageResult:
        annotated_bgr, raw = self._inference.run_inference(
            self.entry, image_bgr, conf_thresh=self.cfg.conf_thresh)
        detections = [Detection.from_model_dict(d, source=SOURCE_MODEL) for d in raw]
        result = DetectionStageResult(
            detections=detections, annotated_path=None, model_name=self.name,
            is_stub=False, note="",
        )
        if dest_path is not None:
            # The real backend already returns a plotted image; write that
            # rather than re-drawing, so its own visual conventions survive.
            _write(annotated_bgr, dest_path, result)
        return result


def get_detector(cfg, question=None, annotation_dir=None):
    """The one switch. cfg.use_model=False (the default) gives the stub.

    `question` lets the stub resolve class ids the way that question's own
    annotation export numbered them, and selects a per-question label folder
    when cfg.annotation_dirs has one.
    """
    if cfg.use_model:
        return ModelDetector(cfg)
    return AnnotationFileDetector(cfg, question=question, annotation_dir=annotation_dir)


# ─────────────────────────────── Rendering ───────────────────────────────────

def render(image_bgr, result: DetectionStageResult, dest_path) -> Optional[str]:
    """Draw this result's boxes and write the copy the UI shows."""
    return _write(draw_detections(image_bgr, result.detections), dest_path, result)


def _write(array_bgr, dest_path, result: DetectionStageResult) -> Optional[str]:
    import cv2

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cv2.imwrite(str(dest_path), array_bgr):
            raise IOError("cv2.imwrite returned False")
    except Exception:
        return None
    result.annotated_path = str(dest_path)
    return result.annotated_path
