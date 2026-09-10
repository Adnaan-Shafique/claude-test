# Running the trained YOLOX-S detector

Two versions now exist. They differ in **one leg**:

| | `app/demo_dash.py` | `app/demo_dash_yolox.py` |
|---|---|---|
| Port | 7870 | 7871 |
| Stage 2 | human annotation `.txt` sidecars | trained YOLOX-S checkpoint |
| Needs | — | torch, torchvision, `yolox/models`, `best_ckpt.pth` |
| Status | frozen (see `FROZEN.md`) | new |

Different ports on purpose — run both, show either.

---

## 1. Copy in the missing YOLOX network definition

`app/vendor/yolox/` has the utils. **`yolox/models/` is missing** and has to
come from a working checkout:

```bash
rm -rf app/vendor/yolox/models   # scp nests if the target already exists
scp -r admin@10.66.98.137:/data01/sds_field/Field_Ops/src/YOLOX/yolox/models \
    app/vendor/yolox/models
rm -rf app/vendor/yolox/models/__pycache__
```

The source and the checkpoint live on AISERVER (10.66.98.137); the demo runs on
FALCONPRD. See `COPY_FROM_AISERVER.md` for both copies in one place.

Expect: `__init__.py`, `build.py`, `darknet.py`, `losses.py`,
`network_blocks.py`, `yolo_fpn.py`, `yolo_head.py`, `yolo_pafpn.py`,
`yolox.py`.

**Why it isn't in the repo.** The field-ops root `.gitignore` contains:

```gitignore
models/
```

Git applies a bare pattern at **any depth**, so it excludes
`src/YOLOX/yolox/models/` along with the checkpoint folder it was written for.
The files are on any machine that has trained; they were simply never tracked.
Worth fixing separately:

```diff
-models/
+/models/
```

## 2. Install torch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install loguru psutil
```

CPU is the right call: YOLOX-S inference costs a fraction of the ~2s the
quality stage already spends per image.

## 3. Check the checkpoint matches

```bash
python tools/inspect_ckpt.py /path/to/best_ckpt.pth
```

Should report `num_classes 2`, stem width 32 (→ width 0.50, "s"), and one
dark2 bottleneck (→ depth 0.33).

## 4. Smoke-test against real photos

```bash
python tools/smoke_yolox.py /data/adnaan/fieldops/demo/photos \
    --ckpt /path/to/best_ckpt.pth --limit 3
```

Then **open the annotated images**. Box position is the one thing no test can
check.

## 5. Run it

```bash
python app/demo_dash_yolox.py        # http://<host>:7871
```

Paste the checkpoint path into the Detector card, press **Load / check model**,
then run as before.

---

## Settings, and where they came from

Every default is read off the training run's own logged exp table
(`02_train.ipynb`, cell 9) rather than assumed:

| Setting | Value | Source |
|---|---|---|
| `num_classes` | 2 | exp table |
| classes | `hazard_sign`, `gps_antenna` | index order confirmed by you |
| `depth` | 0.33 | exp table (yolox_s) |
| `width` | 0.50 | exp table (yolox_s) |
| `act` | silu | exp table |
| `input_size` / `test_size` | **(640, 480)** | exp table — height, width |
| `nmsthre` | 0.65 | `yolox_base.py` default |
| `test_conf` | 0.01 | `yolox_base.py` default (the UI defaults to 0.30 for display) |

### On the non-square input size

`exps/field_ops/yolox_s_field_ops.py` reads `input_size` from
`configs/resolution.yaml`, which `01_data_prep.ipynb` derives from the ingested
pool's median aspect ratio. Portrait phone photos at ~0.75 produce (640, 480).

Setting it wrong never raises. On a **portrait** photo the boxes still land
correctly — height limits the scale for both 640×480 and 640×640 — but the
model sees more grey padding and predicts differently. On a **landscape** photo
the rescale ratio itself changes: 1600×1200 scales 0.30 into 640×480 but 0.40
into 640×640, so every box comes out 4/3 too large. The UI exposes the value
under Options for exactly this reason.

### Class order

YOLOX stores no class names in a checkpoint. `hazard_sign` is index 0 and
`gps_antenna` index 1, per your confirmation. Swapping them would not error —
it would put a confident wrong label on screen *and* into the model prompt as
evidence. The field is editable in the Detector card so it can be corrected
live.

## What is NOT vendored, and why

`yolox/data/`, `yolox/evaluators/`, `yolox/core/`, `yolox/exp/`,
`yolox/layers/` — all pull `pycocotools`, dataset loaders and training
machinery that inference never touches.

The `Exp` class is deliberately not reused: it opens `configs/dataset.yaml` and
`configs/resolution.yaml` at construction, both `.gitignore`d, so it raises
`FileNotFoundError` anywhere but the training machine. The two numbers it would
have supplied are pinned in `pipeline/config.py` instead.

The only thing needed from `yolox/data` is `preproc`, copied verbatim into
`pipeline/yolox_runtime.py` with attribution — it must letterbox exactly as
training did.

`mlflow`, `wandb`, `dotenv` and `thop` are imported lazily inside functions, so
none is required.
