"""YOLOX-S inference, matched to how this checkpoint was actually trained.

Every value here is read off the training run's own logged exp table rather
than assumed (02_train.ipynb, cell 9):

    num_classes  2
    depth        0.33          (yolox_s)
    width        0.50          (yolox_s)
    act          'silu'
    input_size   (640, 480)    HEIGHT, WIDTH - NOT the stock 640x640
    test_size    (640, 480)
    test_conf    0.01          (yolox_base default)
    nmsthre      0.65          (yolox_base default)

The non-square input size is the one to get right. exps/field_ops/
yolox_s_field_ops.py reads it from configs/resolution.yaml, which
01_data_prep.ipynb computes from the ingested pool's median aspect ratio -
portrait phone photos at ~0.75 give (640, 480).

Mis-setting it is NOT always visible: for a photo taller than the canvas's own 0.75 aspect the limiting dimension is the height either way, so the rescale ratio is identical and the boxes land in the same place - the model just sees more grey padding, which changes what it predicts rather than where. For anything wider, the ratio genuinely differs (a 1600x1200 landscape photo scales 0.30 into 640x480 but 0.53 into 640x640) and every box is wrong by that factor.

The exp class is deliberately NOT reused. It opens configs/dataset.yaml and
configs/resolution.yaml at construction, and both are .gitignored, so it
raises FileNotFoundError anywhere but the training machine. The two numbers it
would have supplied are pinned above instead, from the run that produced these
weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# app/pipeline/yolox_runtime.py -> app/vendor, where the YOLOX package lives.
_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# Read off the training run's exp table. Overridable through PipelineConfig.
YOLOX_S_DEPTH = 0.33
YOLOX_S_WIDTH = 0.50
YOLOX_ACT = "silu"
DEFAULT_INPUT_SIZE = (640, 480)     # (height, width)
DEFAULT_TEST_CONF = 0.01
DEFAULT_NMS_THRESHOLD = 0.65
IN_CHANNELS = [256, 512, 1024]


def preproc(img, input_size, swap=(2, 0, 1)):
    """Verbatim from yolox/data/data_augment.py (YOLOX 0.3.0, Apache-2.0).

    Copied rather than imported: `yolox.data` pulls its dataset package, which
    needs pycocotools, and inference needs none of that. Do not "improve" this -
    it has to letterbox exactly as training did or every box shifts. Note the
    paste is top-left on a 114-grey canvas, and the scale is a single ratio
    r = min(H/h, W/w), which the caller divides the predictions back out by.
    """
    import cv2
    import numpy as np

    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def build_model(num_classes: int, depth: float = YOLOX_S_DEPTH,
                width: float = YOLOX_S_WIDTH, act: str = YOLOX_ACT):
    """Construct the network exactly as yolox_base.Exp.get_model() does, minus
    the exp's config-file reads. Head/neck initialisation is skipped because
    load_state_dict overwrites all of it."""
    try:
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    except ImportError as exc:
        raise ImportError(
            f"The YOLOX network definition is not available: {exc}\n\n"
            f"app/vendor/yolox/models/ is missing. It is absent from the field-ops "
            f"repo because that repo's root .gitignore contains a bare 'models/' "
            f"pattern, which git applies at ANY depth - so it excludes "
            f"src/YOLOX/yolox/models/ along with the checkpoint folder it was "
            f"meant for.\n\n"
            f"Copy it from a working checkout:\n"
            f"  cp -r <field-ops>/src/YOLOX/yolox/models {_VENDOR}/yolox/models\n\n"
            f"See app/vendor/README.md."
        ) from exc

    backbone = YOLOPAFPN(depth, width, in_channels=IN_CHANNELS, act=act)
    head = YOLOXHead(num_classes, width, in_channels=IN_CHANNELS, act=act)
    return YOLOX(backbone, head)


class YoloxPredictor:
    """Loads a YOLOX checkpoint once and answers with absolute-pixel boxes.

    Mirrors tools/demo.py's Predictor - same ValTransform-equivalent
    preprocessing, same postprocess(class_agnostic=True), same
    `boxes / ratio` rescale - without importing the YOLOX tools package.
    """

    def __init__(self, checkpoint, class_names: list[str],
                 input_size=DEFAULT_INPUT_SIZE, depth: float = YOLOX_S_DEPTH,
                 width: float = YOLOX_S_WIDTH, act: str = YOLOX_ACT,
                 conf_threshold: float = 0.3, nms_threshold: float = DEFAULT_NMS_THRESHOLD,
                 device: Optional[str] = None, fuse: bool = True):
        import torch

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        self.class_names = list(class_names)
        self.input_size = tuple(int(v) for v in input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = build_model(len(self.class_names), depth=depth, width=width, act=act)

        ckpt = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}

        # strict=True on purpose. A silent partial load is the worst outcome
        # here: the model would run and produce confident, meaningless boxes.
        # The message below turns the most likely cause - a class-count
        # mismatch - into something readable.
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            head_key = next((k for k in state if k.endswith("cls_preds.0.weight")), None)
            trained = state[head_key].shape[0] if head_key else "unknown"
            raise RuntimeError(
                f"The checkpoint does not match the network that was built.\n"
                f"  built for : {len(self.class_names)} class(es) "
                f"({', '.join(self.class_names)})\n"
                f"  checkpoint: {trained} class(es)\n"
                f"  depth={depth} width={width}\n"
                f"Run tools/inspect_ckpt.py against this file to see what it "
                f"actually contains.\n\nOriginal error: {exc}"
            ) from exc

        model.eval()
        if fuse:
            from yolox.utils import fuse_model
            model = fuse_model(model)
        self.model = model.to(self.device)
        self.fused = fuse

    def detect(self, image_bgr) -> list[dict]:
        """One EXIF-corrected BGR array in, a list of
        {"class", "confidence", "box"} out - the same shape
        inference.run_inference() returns, so the two backends stay
        interchangeable. Boxes are absolute pixels in the input array's frame.
        """
        import torch
        from yolox.utils import postprocess

        height, width = image_bgr.shape[:2]
        ratio = min(self.input_size[0] / height, self.input_size[1] / width)

        tensor, _ = preproc(image_bgr, self.input_size)
        tensor = torch.from_numpy(tensor).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = postprocess(outputs, len(self.class_names), self.conf_threshold,
                                  self.nms_threshold, class_agnostic=True)

        output = outputs[0]
        if output is None:
            return []

        output = output.cpu()
        # Undo the letterbox scale. The paste was top-left, so there is no
        # offset to subtract - only the single ratio to divide out.
        boxes = (output[:, 0:4] / ratio).numpy()
        scores = (output[:, 4] * output[:, 5]).numpy()
        class_ids = output[:, 6].numpy().astype(int)

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score < self.conf_threshold:
                continue
            name = (self.class_names[class_id] if 0 <= class_id < len(self.class_names)
                    else f"class_{class_id}")
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({
                "class": name,
                "confidence": float(score),
                # Clamp to the frame: a letterboxed prediction can land slightly
                # outside it, and a negative coordinate breaks the crop the VLM
                # stage takes.
                "box": [max(0.0, x1), max(0.0, y1),
                        min(float(width), x2), min(float(height), y2)],
            })
        return detections

    def describe(self) -> str:
        return (f"YOLOX-S · {len(self.class_names)} classes · "
                f"{self.input_size[0]}x{self.input_size[1]} · {self.device}"
                + (" · fused" if self.fused else ""))


# ─────────────────────────────── Predictor cache ─────────────────────────────
# Building the network and reading 60MB of weights takes seconds; a forward pass
# takes a fraction of one. Without this, every Run in the UI would pay the load
# again, which on stage reads as the model being slow.
_CACHE: dict = {}


def get_predictor(cfg) -> "YoloxPredictor":
    key = (str(cfg.yolox_checkpoint), tuple(cfg.yolox_class_names),
           tuple(cfg.yolox_input_size), cfg.yolox_depth, cfg.yolox_width,
           cfg.yolox_act, float(cfg.conf_thresh), float(cfg.yolox_nms_threshold),
           bool(cfg.yolox_fuse))
    if key not in _CACHE:
        _CACHE[key] = YoloxPredictor(
            cfg.yolox_checkpoint, list(cfg.yolox_class_names),
            input_size=cfg.yolox_input_size, depth=cfg.yolox_depth,
            width=cfg.yolox_width, act=cfg.yolox_act,
            conf_threshold=cfg.conf_thresh, nms_threshold=cfg.yolox_nms_threshold,
            fuse=cfg.yolox_fuse,
        )
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()
