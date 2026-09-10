# `app/vendor/yolox` — vendored YOLOX 0.3.0 (Apache-2.0)

Copied from `src/YOLOX/yolox/` in the field-ops repo. The demo host has no
route to that repo, so the pieces the detector needs travel with the pipeline —
the same reasoning that vendored `quality_check.py` and
`foreground_segmentation.py` into `app/`.

## What is here

| Path | Why |
|---|---|
| `yolox/utils/` | `postprocess` (decode + NMS), `fuse_model`, and the helpers `yolox/models` imports |
| `yolox/__init__.py` | version marker (`0.3.0`) |
| `LICENSE` | Apache-2.0, as required |

## What is MISSING and must be copied in

**`yolox/models/`** — the network definition itself (`YOLOX`, `YOLOPAFPN`,
`YOLOXHead`, `CSPDarknet`, `BaseConv`…). Without it there is nothing to load
the checkpoint's weights into.

It is absent from the field-ops repo because the root `.gitignore` contains:

```gitignore
models/
```

A bare `models/` pattern matches **at any depth**, so it excludes
`src/YOLOX/yolox/models/` along with the intended checkpoint folder at the repo
root. The files exist in any working copy that has trained successfully; they
are simply untracked.

Copy the folder here:

    cp -r <field-ops>/src/YOLOX/yolox/models app/vendor/yolox/models

Expected contents: `__init__.py`, `build.py`, `darknet.py`, `losses.py`,
`network_blocks.py`, `yolo_fpn.py`, `yolo_head.py`, `yolo_pafpn.py`,
`yolox.py`.

**Fix for the field-ops repo** (unrelated to this demo, but worth doing):
anchor the pattern to the root so it stops swallowing nested `models`
directories.

```diff
-models/
+/models/
```

## Deliberately NOT vendored

`yolox/data/`, `yolox/evaluators/`, `yolox/core/`, `yolox/exp/` and
`yolox/layers/` — they pull `pycocotools`, dataset loaders and training
machinery none of which inference needs. The one thing required from
`yolox/data` is `preproc`, copied verbatim into
`app/pipeline/yolox_runtime.py` with attribution.

`mlflow`, `wandb`, `dotenv` and `thop` are imported lazily inside functions
throughout `yolox/utils`, so none of them is needed to run inference.
