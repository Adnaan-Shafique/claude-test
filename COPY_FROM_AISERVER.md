# Copying the detector onto FALCONPRD

Two machines are involved:

| Host | Address | Holds |
|---|---|---|
| **AISERVER** | `10.66.98.137` | the Field_Ops repo, YOLOX source, `best_ckpt.pth` — and the VLM server on :5432 |
| **FALCONPRD** | `10.19.71.246` | the integrated demo pipeline (`/data/adnaan/fieldops/demo/integrated_pipeline`) |

Everything below runs **on FALCONPRD**, pulling from AISERVER.

```bash
cd /data/adnaan/fieldops/demo/integrated_pipeline
git pull

REMOTE=admin@10.66.98.137
FO=/data01/sds_field/Field_Ops
RUN=$FO/models/yolox/runs/yolox_s_field_ops_20260909_183424

# 1. The YOLOX network definition — nine .py files, the piece .gitignore ate.
#    app/vendor/yolox/models must NOT already exist, or scp nests inside it.
rm -rf app/vendor/yolox/models
scp -r $REMOTE:$FO/src/YOLOX/yolox/models app/vendor/yolox/models
rm -rf app/vendor/yolox/models/__pycache__

# 2. The trained weights (~70 MB).
mkdir -p models
scp $REMOTE:$RUN/best_ckpt.pth models/best_ckpt.pth

# 3. The class order, written by 02_train.ipynb from configs/dataset.yaml.
scp $REMOTE:$RUN/classes.json /tmp/classes.json
cat /tmp/classes.json
```

**Read step 3's output.** It should show `hazard_sign` first, `gps_antenna`
second. If the order differs, correct the Class names field in the UI (or
`yolox_class_names` in `pipeline/config.py`) — a swap does not error, it puts a
confident wrong label on screen and into the model prompt as evidence.

## Then — a separate venv, so demo_venv is never touched

`demo_venv` keeps running the frozen annotation demo on 7870. The detector gets
its own environment.

```bash
# Same base interpreter that built demo_venv - check, do not assume:
head -3 demo_venv/pyvenv.cfg

python3.12 -m venv venv_yolox
source venv_yolox/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements-demo.txt          # known to resolve here: demo_venv used it
# The proxy intercepts TLS. pypi.org is already exempted in this host's pip
# config, but download.pytorch.org is not, so it fails cert verification with
# "unable to get local issuer certificate". Extend the same exemption:
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --proxy http://10.94.147.19:8080 \
  --trusted-host download.pytorch.org \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

pip install loguru psutil

# Does the vendored package import?
python -c "import sys; sys.path.insert(0,'app/vendor'); \
  from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead; print('yolox.models OK')"

# Does the checkpoint match what we build?
python tools/inspect_ckpt.py models/best_ckpt.pth

# Real weights on real photos
python tools/smoke_yolox.py /data/adnaan/fieldops/demo/photos \
    --ckpt models/best_ckpt.pth --limit 3

python app/demo_dash_yolox.py        # http://10.19.71.246:7871
```

`inspect_ckpt.py` should report **2 classes**, stem width **32** (→ 0.50, "s"),
**1** dark2 bottleneck (→ depth 0.33). Anything else means the checkpoint and
the pinned exp values disagree — send me the output.

The checkpoint path is now prefilled in the UI from `models/best_ckpt.pth`, so
following the paths above means nothing to type on stage.

## Note on `__pycache__`

The `scp -r` brings AISERVER's compiled `.pyc` files along. Both hosts run
Python 3.12 so they would be ignored rather than mis-loaded, but they are
removed above anyway — stale bytecode is never worth debugging under time
pressure.

---

## If the torch install fails on TLS

```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate'))
```

The proxy re-signs HTTPS with a corporate CA. `pypi.org` and
`files.pythonhosted.org` are already exempted on this host (which is why
`requirements-demo.txt` installs cleanly); `download.pytorch.org` is not.

Check what is configured before working around it:

```bash
pip config list
```

- A `cert = /path/...` line means a corporate CA bundle is available. Use
  `--cert /path/...`, which verifies properly rather than skipping.
- Otherwise add `--trusted-host download.pytorch.org`, as above — the same
  exemption PyPI already has here.

**Fallback:** torch is on PyPI too, over the path that already works:

```bash
pip install torch torchvision --proxy http://10.94.147.19:8080
```

PyPI's Linux `torch` is the CUDA build, so this pulls ~3-4GB of `nvidia-*`
wheels that a CPU-only host never uses. It runs correctly -
`torch.cuda.is_available()` simply returns False - so this costs download time
and disk, not function.
