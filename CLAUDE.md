# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A three-stage field-inspection demo pipeline that answers a yes/no question
about a photograph:

1. **Quality gate** — MM-IQA cues + u2netp foreground segmentation
2. **Detection** — trained YOLOX-S, or human annotation `.txt` sidecars
3. **Answer** — Qwen3-VL on a remote GPU server, reached directly or via a proxy

It integrates three previously independent tools. The bias throughout is
"works on stage" over architectural purity — this is demo software with a
date, and several deliberate compromises below exist for that reason.

`instant_graph_mcp_server_v2_4_1.py` at the repo root is unrelated earlier work
(see commits before 2026-09-09). Leave it alone.

## Commands

No build step, no linter, no pytest. Suites are standalone scripts that print
`N passed, M failed` and exit non-zero on failure.

```bash
# Every suite (459 assertions across nine files)
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done

# One suite, with its output
python tests/test_modes.py

# There is no single-test runner. Suites are linear scripts; to isolate a
# section, comment out the ones above it or read the `print("\n...")` banners
# that separate them.

# Before running anything on a demo host
python tools/preflight.py --port 7872 --transport proxy

# Smoke one stage against real data
python tools/smoke_stage1.py <photos>
python tools/smoke_yolox.py <photos> --ckpt models/best_ckpt.pth --limit 3
python tools/smoke_stage3.py <photos> --limit 2 --transport proxy
python tools/inspect_ckpt.py models/best_ckpt.pth

# The UIs (each on its own port, all runnable at once)
python app/demo_dash.py         # 7870 — FROZEN, annotation sidecars
python app/demo_dash_yolox.py   # 7871 — trained detector
python app/demo_dash_modes.py   # 7872 — three modes; the current demo
```

Tests stub `cv2`, `dash` and `requests`. Verified: all 459 pass with `cv2`,
`dash`, `requests`, `torch`, `rembg` and `pandas` all absent. Keep it that way
— a suite that needs torch cannot run where it is most needed.

**`pyyaml` is the one real test dependency.** Two suites fail without it
(`PromptStore.save`, and the `dataset.yaml` label parser). The application
degrades gracefully when it is missing; the tests that assert saving do not.

## Architecture

### Stage contracts

`app/pipeline/schemas.py` defines the dataclasses every stage fills and the UI
reads. A `PipelineRecord` holds one photograph's `quality`, `detection` and
`vlm` results plus `stopped_at`. Changing a field here touches every stage and
both renderers — read it first.

### Configuration

`app/pipeline/config.py` is a single `PipelineConfig` dataclass with
`default_config(**overrides)`. Every path derives from `project_root`, which
derives from `__file__`, so the tree relocates without edits. `validate()`
returns human-readable warnings rather than raising — the UI shows them.

`FIELDOPS_*` environment variables override config fields (see `ENV_OVERRIDES`).
Explicit keyword arguments beat the environment. **This makes
`default_config()` environment-sensitive**, which has already broken a test
that asserted defaults — see `tests/test_transport.py`, which clears those
variables before its first call and restores them at the end.

### Two transports, one GPU

Stage 3 reaches the same model either directly (`/infer`) or through
`llm_proxy_v3` (`/v1/infer` + `X-API-Key`). Every path difference lives in the
`ENDPOINTS` table in `stage3_vlm.py`; nothing else knows which route is in use.
`PROXY.md` documents the four behaviours that genuinely differ. The failure to
expect: `gpu_url` pointing at the proxy while transport is still `direct`,
which is a 404 that reads as a dead server.

### Three modes

`app/pipeline/modes.py` runs all three over one set of photographs, sharing
work: one quality pass, one detection pass, and one VLM answer when modes 1 and
2 both proceed. Mode 3 replaces the classical stages with the model itself, as
three separate calls with per-leg editable system prompts. `MODES.md` has the
reasoning.

### Questions and prompts

`app/pipeline/questions.py` is a registry — each question carries its own
system prompt, user template, answer semantics and relevant detector classes.
Mode 3's three legs have their own defaults here; `prompts.py` persists UI
overrides to `config/prompts.yaml` (gitignored, per-machine).

### Vendored code

`app/vendor/yolox/` holds YOLOX utils under its Apache licence. **`yolox/models/`
is deliberately absent from git** — the upstream repo's bare `models/` gitignore
pattern excluded it — and must be copied in per `YOLOX_SETUP.md`. `build_model()`
raises an ImportError naming that cause when it is missing.

`app/quality_check.py` and `app/foreground_segmentation.py` are vendored from
the original toolset. `foreground_segmentation.py` computes `MODELS_DIR` from
its own location, so it must stay directly in `app/`.

## Things that will bite you

**`app/demo_dash.py` is frozen** (`FROZEN.md`, commit `f978b7d`). It is the
fallback that is known to work. `demo_dash_modes.py` imports renderers from it
and re-implements only what mode 3 needs — do not "fix" the frozen file to
avoid a re-implementation.

**Arrays, never paths.** `quality_check.load_image_bgr()` applies EXIF
transposition; re-reading the file with `cv2.imread` does not. A box computed
on one and drawn on the other is silently wrong. Stage 3 only accepts the BGR
array the pipeline is already holding.

**The registry bootstrap.** `build_payload()` refuses an empty registry rather
than defaulting an unknown model to text-only — without that, every image
request fails with `'qwen3-vl' is text-only`, which looks like a broken server.
Call `refresh_registry()`/`ensure_registry()` first.

**A blank system prompt fails silently.** The server skips a falsy system turn,
so the model answers unframed and it reads as a model-quality problem.
`Question.__post_init__` and `build_payload()` both refuse one.

**Class names come from config, not the checkpoint.** YOLOX stores no names. A
wrong order does not error — it puts a confident wrong label on screen *and*
into the VLM prompt as evidence. `yolox_class_names` order is pinned from the
training run's `classes.json`: index 0 is `GPS Antenna`.

**`yolox_input_size` is `(640, 480)`, not the stock square.** For portrait
photographs the ratio is identical either way; only landscape diverges (by 4/3).

**Never let a mock assert.** Mock and parse-failure paths return `unknown`
(and `poor` for quality), always labelled. Asserting yes/no when no model
looked is the one thing this demo must not do on screen.

**Tests must not depend on their host.** Two have already failed only on the
demo VM: one read `FIELDOPS_*` from the environment, and one popped
`yolox.models` from `sys.modules` (which clears the cache but does not prevent a
re-import, so it passed only where the package was genuinely absent).

## Conventions

Comments explain *why*, especially where a line guards against a specific
failure that has actually happened. Several carry the symptom so the next
reader recognises it. Do not strip them as noise.

Commit messages are prose explaining the reasoning and what was verified, not
bullet lists.

Two companion logs are kept up to date by hand: `DECISIONS.md` (why things are
the way they are) and `CHANGELOG.md` (what changed, when). Add to them when a
change is worth explaining later.
