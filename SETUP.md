# Demo pipeline setup — FALCONPRD

Target: `/data/adnaan/fieldops/demo/integrated_pipeline`, python 3.12, fresh
`demo_venv`. Detector runs in **stub mode** (`use_model=False`), so no torch,
no YOLOX, no checkpoint.

---

## 0. Can this box reach PyPI?

Do this first — it decides everything after it.

```bash
source demo_venv/bin/activate
pip download --no-deps requests -d /tmp/pipcheck && echo "PyPI OK" || echo "NO PyPI ROUTE"
```

**No route?** Skip to *Appendix: air-gapped install*.

---

## 1. Vendor the two source modules

The integrated pipeline needs two files from the original toolset. Nothing else
from those scripts ships. Copy them into `app/` next to `pipeline/`:

```
integrated_pipeline/
  app/
    quality_check.py             <- COPY IN
    foreground_segmentation.py   <- COPY IN
    pipeline/
      schemas.py  config.py  questions.py  stage1_quality.py
```

`foreground_segmentation.py` computes `MODELS_DIR` as
`Path(__file__).parent.parent / "models"`, so placing it at `app/` makes that
resolve to `integrated_pipeline/models` — exactly where `u2netp.onnx` already
is. Put it anywhere else and rembg will look in the wrong directory and try to
download.

Verify:

```bash
python -c "import sys; sys.path.insert(0,'app'); import quality_check, foreground_segmentation; print('vendored modules import OK')"
```

(That will fail until step 2 installs cv2/numpy — run it after.)

---

## 2. Install

```bash
source demo_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-demo.txt
```

Expect this to take a few minutes: rembg drags in scipy and scikit-image.

### If `import cv2` raises `libGL.so.1: cannot open shared object file`

Something pulled the non-headless wheel (rembg has been known to). Fix:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless
```

### If the rembg resolve fails or drags in too much

`get_foreground_box()` only ever calls `remove(..., only_mask=True)`, so the
alpha-matting stack is dead weight. Slim path:

```bash
pip install --no-deps rembg
pip install onnxruntime pillow numpy opencv-python-headless pooch
python -c "from rembg import new_session, remove; print('rembg usable')"
```

If that import fails, add back whatever it names, one package at a time.

---

## 3. Verify

```bash
python tools/preflight.py                    # add --skip-gpu if the GPU box is unreachable
python tests/test_phase1.py                  # expect 54 passed
python tools/preflight.py --offline-check    # ONLY once rembg is installed
```

`--offline-check` asks you to pull the network, then loads u2netp from disk.
That is the check that proves the demo never depends on a first-run download.

---

## 4. Confirm stage 1 against the CLI

The one Phase 1 acceptance test neither of us has run. Score a single image and
compare against what `run_quality_batch_foreground.py` produces for the same
file (on whichever machine that script does run):

```bash
python - <<'PY'
import sys; sys.path.insert(0, "app")
from pipeline.config import default_config
from pipeline.stage1_quality import build_quality_config, score_path
cfg = default_config()
img, r = score_path("data/demo_photos/<one_photo>.jpg", cfg)
print(r.headline)
print("whole-frame:", r.whole_frame_score, " foreground:", r.foreground_score)
print("segmented:", r.segmentation_used, " box:", r.foreground_box)
print("reasons:", r.failure_reasons)
PY
```

`score` and `whole_frame_score` must match the CLI's CSV row for that file.

---

## Appendix: air-gapped install

On any machine with PyPI access **and the same OS/python 3.12**:

```bash
pip download -r requirements-demo.txt -d wheelhouse \
    --platform manylinux2014_x86_64 --python-version 312 \
    --only-binary=:all:
tar czf wheelhouse.tgz wheelhouse
```

Copy across, then on FALCONPRD:

```bash
tar xzf wheelhouse.tgz
pip install --no-index --find-links=wheelhouse -r requirements-demo.txt
```

If `--platform` rejects a package for having no matching wheel, that package
needs building from source — flag it rather than forcing it.
