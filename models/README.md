# `models/`

## `u2netp.onnx` — committed on purpose

`foreground_segmentation.py` points rembg's `U2NET_HOME` at this directory
(`os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))`), and rembg expects a
flat folder holding the `.onnx` file directly — no extra nesting.

That module's docstring says the weights are git-ignored and download on first
use. **This file is committed anyway**, deliberately, because the integration
plan requires that the demo never depend on a first-run download (§2.3). If you
want the original behaviour back, drop the `!models/u2netp.onnx` negation from
`.gitignore` and delete the file.

| | |
|---|---|
| size | 4,574,861 bytes |
| sha256 | `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8` |

`tools/preflight.py` verifies both, and `--offline-check` confirms the model
actually loads with the network down. Run that on the demo machine.
