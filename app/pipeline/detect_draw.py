"""Shared box drawing for both detector backends.

The real path draws boxes via Ultralytics' results.plot() or YOLOX's visual();
the stub has to produce output a demo audience cannot tell apart, so both go
through this one function. If the visual language diverges, the stub starts
looking like a different system - which is exactly the impression to avoid.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# BGR. Distinct at a glance, readable on both bright sky and dark equipment,
# and none of them the green that stage 1 uses for the foreground box - the two
# annotations must never be confused for each other.
_PALETTE = [
    (56, 56, 255),    # red
    (255, 157, 51),   # blue
    (0, 204, 255),    # amber
    (255, 112, 209),  # violet
    (51, 219, 255),   # yellow
    (204, 102, 255),  # pink
]


def color_for(label: str) -> tuple[int, int, int]:
    """Stable colour per class name, so the same label is the same colour on
    every image and across re-runs."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return _PALETTE[int(digest[:8], 16) % len(_PALETTE)]


def draw_detections(image_bgr, detections: Iterable, thickness: int | None = None):
    """Rectangle plus a filled '<label> <conf>' caption, matching what the real
    detector's own plotting produces. Returns a copy; never mutates the input,
    which every other stage is still holding."""
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    # Scale with the image so a 4000px photo does not get hairline boxes and a
    # 640px one does not get boxes that swallow the subject.
    t = thickness or max(2, round(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1200)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        # Clamp: an annotation file can legitimately describe a box that runs
        # off the frame, and cv2 will happily draw off-canvas but the caption
        # would vanish.
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        colour = color_for(det.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, t)

        caption = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, max(1, t // 2))
        # Caption sits above the box, or inside it when the box touches the top.
        cap_y = y1 - baseline - 2
        if cap_y - th < 0:
            cap_y = y1 + th + baseline + 2
        cv2.rectangle(out, (x1, cap_y - th - baseline), (x1 + tw + 4, cap_y + baseline),
                      colour, -1)
        cv2.putText(out, caption, (x1 + 2, cap_y), font, font_scale, (255, 255, 255),
                    max(1, t // 2), cv2.LINE_AA)
    return out
