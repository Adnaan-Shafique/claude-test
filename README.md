# Field Ops inspection pipeline — demo

Answers a yes/no inspection question about a site photograph, and shows its
working: whether the photo was good enough to judge, whether the subject is
actually in frame, and what the model concluded.

Three tools that were built separately — a quality gate, an object detector and
a vision-model Q&A leg — run as one pipeline here.

```
photograph ──▶ 1. Quality gate ──▶ 2. Detection ──▶ 3. Answer ──▶  YES / NO / UNKNOWN
               MM-IQA + u2netp     YOLOX-S           Qwen3-VL        + reasoning
```

A photograph that fails the quality gate stops there — an inspection decision
made from an unusable photo is worse than no decision. Modes 2 and 3 argue with
that premise; see below.

## The two questions

| id | Question | YES means |
|---|---|---|
| `hazard_warning` | Is a hazardous-warning sign present? | a hazard/warning/danger sign or safety placard is visible |
| `gps_antenna` | Is the GPS antenna open to the sky? | the antenna's upward view is clear |

Each carries its own system prompt, answer semantics and relevant detector
classes, in `app/pipeline/questions.py`.

## Three UIs, three ports

All are Dash apps and all can run at once.

| App | Port | Stage 2 | Status |
|---|---|---|---|
| `app/demo_dash.py` | 7870 | human annotation `.txt` sidecars | **frozen** — the known-good fallback |
| `app/demo_dash_yolox.py` | 7871 | trained YOLOX-S checkpoint | superseded |
| `app/demo_dash_modes.py` | 7872 | three modes, switchable | **the current demo** |

### The three modes

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Quality | hard gate | advisory | judged by the model |
| Detection | YOLOX-S boxes | YOLOX-S boxes | presence in words, no boxes |
| A blurry photo whose sign is clearly detected | **dropped** | **answered** | answered if the model calls it usable |

One run fills all three; switching mode re-reads results rather than
re-running, so the same photographs can be argued three ways in front of an
audience. A Compare tab puts one row per photograph against one column per
mode. `MODES.md` has the reasoning and the cost.

## Quick start

Run on Python 3.11 and 3.12. Every module uses `from __future__ import
annotations`, so nothing here needs a specific minor — but those are the two
that have actually been exercised.

The interpreter must have a working `_ctypes`. One built without `libffi`
breaks pandas, rembg and gradio at once, with an error that names none of them:

```bash
python -c "import _ctypes; print('ok')"     # do this before pip install
```

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-demo.txt

# Check the machine before trusting it
python tools/preflight.py --port 7872

# Every test (459 assertions, nine suites)
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done

python app/demo_dash_modes.py      # http://<host>:7872
```

Two things are **not** in this repository and must be copied in — the trained
checkpoint (`models/best_ckpt.pth`, ~70 MB) and the YOLOX network definition
(`app/vendor/yolox/models/`, which upstream's `.gitignore` excluded). See
`COPY_FROM_AISERVER.md` and `YOLOX_SETUP.md`.

Then press **Load / check model** in the UI. It reports in one line whether
u2netp is present, whether the detector loaded with the right class names, and
whether the model server is reachable — three things worth knowing before an
audience is watching, not during.

## Reaching the model

Stage 3 talks to a Qwen3-VL server. Hosts that cannot see it go through
`llm_proxy_v3` instead:

```bash
export FIELDOPS_VLM_TRANSPORT=proxy
export FIELDOPS_GPU_URL=http://<proxy-host>:8071
export FIELDOPS_VLM_API_KEY=<key>
```

Same model, different paths and an auth header. `PROXY.md` documents the four
behaviours that genuinely differ, and the failure to expect — pointing the URL
at the proxy while leaving the transport on `direct`, which is a 404 that reads
as a dead server.

When the server cannot be reached the pipeline does not fail: it returns a
**clearly labelled mock**, always `unknown`. Asserting yes or no when no model
looked is the one thing this demo will not do.

## Where the documentation lives

| File | For |
|---|---|
| `SETUP.md` | installing on a fresh host |
| `YOLOX_SETUP.md`, `COPY_FROM_AISERVER.md` | the detector, and the two files git does not carry |
| `PROXY.md` | reaching the model through the proxy |
| `MODES.md` | what the three modes argue, and what they cost |
| `SERVICE.md` | running it under systemd |
| `FROZEN.md` | the pinned fallback, and how to restore it |
| `OPEN_SOURCE.md` | every open-source component, version and licence |
| `CHANGELOG.md` / `DECISIONS.md` | what changed when / why it is the way it is |
| `CLAUDE.md` | orientation for Claude Code sessions |

## Repository layout

```
app/
  demo_dash*.py         the three UIs
  pipeline/             schemas, config, questions, the three stages, modes
  vendor/yolox/         vendored YOLOX utils (Apache 2.0)
  quality_check.py      vendored from the original quality tool
  foreground_segmentation.py
tools/                  preflight + one smoke script per stage
tests/                  nine standalone suites - no pytest, no build step
deploy/                 systemd unit and an env template
models/                 u2netp.onnx (committed); best_ckpt.pth (copied in)
```

Run outputs land under `demo_runs/<run_id>/`:

```
quality/          u2netp foreground boxes, one per photograph
detection/        detector boxes, one per photograph with a detection
<mode>/           pipeline_results.csv + results.json, one folder per mode
vlm_only/images/  mode 3's plain EXIF-corrected copies (no boxes - u2netp
                  never ran in that mode, so drawing one would be a lie)
```

Nothing prunes them, and a rehearsal is measured in gigabytes; check `du -sh
demo_runs/` before a demo.

## Status

Demo software with a date, not a product. The bias throughout is "works on
stage" over architectural purity, and several compromises are deliberate —
`DECISIONS.md` records which, and what each one costs.

Verified end to end on the demo hosts: stage 1 ~2s per photograph, YOLOX-S
2.4–6.7s on CPU with boxes matching the human annotations at 0.89–0.94, and the
model answering in 0.21–0.48s server-side with reasoning that cites real sign
text.

`instant_graph_mcp_server_v2_4_1.py` at the repo root is unrelated earlier
work.
