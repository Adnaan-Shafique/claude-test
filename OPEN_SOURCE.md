# Open-source software used in this demo

Versions are what is actually installed on the demo host (FALCONPRD), not what
`requirements-demo.txt` declares as a lower bound.

## Runtime — the demo pipeline

| Software | Version | Licence | Used for |
|---|---|---|---|
| **Python** | 3.12.0 | PSF-2.0 | Runtime for everything below |
| **Dash** | 4.4.1 | MIT | The demo UI (`demo_dash.py`, `demo_dash_yolox.py`) |
| **Plotly** | 6.9.0 | MIT | Dash dependency (components, rendering) |
| **Flask** | 3.1.3 | BSD-3-Clause | Dash's web server |
| **NumPy** | 2.5.3 | BSD-3-Clause | Array maths across all three stages |
| **OpenCV** (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | Image I/O, quality cues, box drawing |
| **Pillow** | 11.3.0 | MIT-CMU (HPND) | EXIF-correct image loading |
| **pandas** | 2.3.3 | BSD-3-Clause | CSV export, results tables |
| **Requests** | 2.34.2 | Apache-2.0 | HTTP client to the VLM server |
| **Gradio** | 5.50.0 | Apache-2.0 | The earlier UI, kept as a fallback |

## Stage 1 — quality gate

| Software | Version | Licence | Used for |
|---|---|---|---|
| **rembg** | 2.0.69 | MIT | Wrapper around the segmentation model |
| **U²-Net** (`u2netp.onnx`) | — (4.6 MB) | Apache-2.0 | Foreground segmentation, so quality cues score the subject rather than sky |
| **ONNX Runtime** | 1.29.0 | MIT | Runs the u2netp model on CPU |

The scoring itself implements the MM-IQA framework from Aglin, Muchiri &
Nkundineza, *"A Lightweight Multi-Metric No-Reference Image Quality Assessment
Framework for UAV Imaging"* (arXiv:2604.13112) — a published method, not a
software dependency.

## Stage 2 — object detection

| Software | Version | Licence | Used for |
|---|---|---|---|
| **YOLOX** | 0.3.0 | Apache-2.0 | Detector architecture (YOLOX-S), vendored in `app/vendor/yolox/` |
| **PyTorch** | 2.14.0 | BSD-3-Clause | Runs the detector |
| **torchvision** | 0.29.0 | BSD-3-Clause | NMS (`ops.batched_nms`) in YOLOX's postprocess |
| **loguru** | 0.7.3 | MIT | Required by `yolox.utils` |
| **psutil** | 7.2.2 | BSD-3-Clause | Required by `yolox.utils` |

## Stage 3 — vision-language question answering

| Software | Version | Licence | Used for |
|---|---|---|---|
| **vLLM** | ≥ 0.11.0 | Apache-2.0 | Serves the VLM on the GPU host |
| **Qwen3-VL-30B-A3B-Instruct** | — | Apache-2.0 † | The model that answers the inspection question |
| **FastAPI** | — | MIT | The GPU server's HTTP API |
| **Uvicorn** | — | BSD-3-Clause | ASGI server for the above |
| **Transformers** | ≥ 4.57 | Apache-2.0 | Chat templating for the VLM prompt |

The GPU server (`gpu_api_server_v6.py`) runs on a separate host and predates
this demo; versions there are its own.

## Supporting tools (not part of the running demo)

| Software | Licence | Used for |
|---|---|---|
| **CVAT** | MIT | Annotating the training images |
| **Ultralytics YOLO** | AGPL-3.0 | Accuracy-ceiling reference during training only — never deployed or shipped |
| **DM Sans** | SIL OFL 1.1 | The UI typeface, per Design System V.01 |
| **pytest-free test suite** | — | Plain-Python assertions, no test framework dependency |

## Notes on licensing

- Everything in the **running demo** is permissive — Apache-2.0, MIT, BSD-3-Clause,
  PSF or HPND. Nothing copyleft is linked, bundled or distributed.
- **Ultralytics is AGPL-3.0** and is the one exception. It was used only to
  measure a local accuracy reference during training; no Ultralytics model is
  exported, deployed or shipped, and AGPL obligations attach to distribution.
- **YOLOX is vendored** (`app/vendor/yolox/`) rather than installed, with its
  Apache-2.0 `LICENSE` file kept alongside the code as that licence requires.
- **u2netp.onnx is committed** to this repo so the demo never depends on a
  first-run download. Its Apache-2.0 terms travel with it.

† Verify the Qwen3-VL model card before publishing this list externally —
model weight licences change more often than code licences, and this entry is
from the model family's stated terms rather than a file in this repo.
