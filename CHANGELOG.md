# Changelog

What changed, when, and against which commit. Dates are commit dates.

Versions are milestones in the demo's development, not released artefacts —
nothing here is published or installed from a package index. `FROZEN.md`
records the one commit that is pinned as a fallback.

Reasoning lives in `DECISIONS.md`; this file is the record of *what* and
*when*. Assertion counts are the whole suite after that change.

---

## v0.6 — Deployment · 2026-09-18

| Commit | Change |
|---|---|
| `aa1146a` | systemd unit and `SERVICE.md` for the modes demo on port 7872 |

- `deploy/fieldops-demo-modes.service` — starts at boot, restarts on failure,
  logs to journald.
- `deploy/fieldops-demo.env.example` — the proxy credentials, kept out of the
  world-readable unit file. `.gitignore` refuses a filled-in copy.
- `SERVICE.md` — install, the no-root user-service path, firewall, update loop,
  and what the service does not survive (`demo_runs/` growth, ownership of the
  project directory, prompts read once at import).

No application code changed. **459 assertions, nine suites.**

---

## v0.5 — The field-ops VM · 2026-09-16 → 2026-09-17

Moved the demo to `10.19.75.122`, which cannot reach the GPU server at all.

| Commit | Date | Change |
|---|---|---|
| `48caf4f` | 09-17 | Uploads accumulate instead of replacing |
| `dc2135f` | 09-17 | Dropzone stops silently discarding dropped photographs |
| `419282c` | 09-17 | Two tests stop depending on the machine they run on |
| `f591bef` | 09-16 | Reach the VLM through `llm_proxy_v3` |

### Added
- **Second transport.** `vlm_transport` (`direct` | `proxy`), `vlm_api_key`,
  `FIELDOPS_*` environment overrides, a UI radio and key field, `--transport`
  on `preflight.py` and `smoke_stage3.py`. New `PROXY.md`.
- **Upload accumulation.** Staged-to-disk photographs in a `dcc.Store`, a
  "Clear photos" button, per-drop reporting of what was ignored and why.
- `tests/test_transport.py` — 54 assertions with `requests` stubbed by a
  recorder, so the assertions are about what is *sent*.

### Fixed
- **Dropzone lost most dropped files.** `accept="image/*"` filters on the MIME
  type the browser reports; a file from a network share arrives with none and
  react-dropzone discarded it without a word. Reproduced in Chromium — three
  dropped, one accepted. `accept` now lists extensions too, from the same set
  `collect_images()` uses. Affected `demo_dash_modes.py` and
  `demo_dash_yolox.py`; the frozen `demo_dash.py` never set `accept` and so
  never had it.
- **Two host-dependent tests.** One read `FIELDOPS_*` through
  `default_config()`, so on a proxy-configured host "a proxy with no API key"
  became a proxy *with* one. The other popped `yolox.models` from `sys.modules`,
  which clears the cache but does not stop a re-import — it passed only where
  the package was genuinely absent.
- A filename carrying a path traversal is flattened to its basename before it
  reaches disk. A staged photograph that disappears before Run is reported
  rather than vanishing from the batch.

### Changed
- `_get` raises the same `HTTP <code> from <path>` shape as `_post`, so a 401
  on the registry read gets the same explanation a POST would.
- Proxy statuses carry their meaning: 502 is "the proxy is up and the GPU
  behind it is not", 503 is load rather than breakage, 404 is almost always
  the wrong transport.

**459 assertions, nine suites** (was 379 across eight).

---

## v0.4 — Three modes · 2026-09-15 → 2026-09-16

| Commit | Date | Change |
|---|---|---|
| `61d5df5` | 09-16 | Mode 3 renders images; per-leg system prompts |
| `bf89b6d` | 09-16 | Three-mode version of the demo UI |
| `477ba5e` | 09-15 | Open-source inventory |

### Added
- `app/demo_dash_modes.py` on **port 7872** — mode 1 (quality gate → detector →
  model), mode 2 (quality **OR** detector → model), mode 3 (everything by the
  model). One run fills all three; switching mode re-reads results rather than
  re-running. A Compare tab puts one row per photograph against one column per
  mode.
- `app/pipeline/modes.py` — the shared-work runner. One quality pass, one
  detection pass, one VLM answer when modes 1 and 2 both proceed.
- `app/pipeline/prompts.py` — `PromptStore`, persisting mode-3 overrides to
  `config/prompts.yaml`.
- Mode 3 split into three calls with editable per-leg system prompts, a fixed
  user prompt shown read-only, and a "Re-run mode 3 only" button that reuses
  modes 1 and 2's results.
- Mode 3 renders the plain EXIF-corrected photograph — deliberately no
  foreground box, since u2netp never runs there.
- `MODES.md`, `OPEN_SOURCE.md`.

**379 assertions, eight suites.**

---

## v0.3 — Trained detector · 2026-09-10

| Commit | Change |
|---|---|
| `820ebc4` | Image upload in the YOLOX UI |
| `39728ce` | `smoke_yolox` stops overriding the configured class names |
| `674113c` | Class order corrected from `classes.json` |
| `c6e9338` | `preflight --port` |
| `860f7cd`, `1ce5613` | torch install: proxy TLS, and dropping `--extra-index-url` |
| `0d29e85` | `.gitignore` anchored so vendored source survives |
| `d1b9834` | Default checkpoint path; `COPY_FROM_AISERVER.md` |
| `6d96dab` | YOLOX-S detector as a second Dash version |
| `d613e80` | Checkpoint inspector |

### Added
- `app/demo_dash_yolox.py` on **port 7871** — the trained YOLOX-S checkpoint in
  place of annotation sidecars.
- `app/pipeline/yolox_runtime.py` — letterbox `preproc`, `build_model`,
  `YoloxPredictor`, all mirroring the training exp.
- `app/vendor/yolox/` — utils under Apache 2.0. `yolox/models/` is *not*
  tracked: the upstream repo's bare `models/` gitignore pattern excluded it.
- `tools/inspect_ckpt.py` — reads class count, width and depth out of a
  checkpoint and says what they imply.
- `YOLOX_SETUP.md`, `COPY_FROM_AISERVER.md`.

### Fixed
- **Class order.** Reported as the detector swapping classes; the config was
  right and `smoke_yolox.py`'s `--classes` default was overriding it. Two guard
  tests added.
- **Missing upload widget** in the YOLOX UI — the annotation version's sidecar
  constraint had been carried over, but a trained detector needs no sidecar.

**Verified on FALCONPRD:** 2.4–6.7s per photograph on CPU, boxes matching the
human annotations at 0.89–0.94 confidence.

---

## v0.2 — Dash UI, frozen · 2026-09-09

| Commit | Change |
|---|---|
| `cf23a9a` | Record the `dash-ui-v1` freeze |
| `f978b7d` | **Rebuild the UI in Dash, styled to Design System V.01** |

`app/demo_dash.py` on **port 7870**, with `app/assets/demo.css`. Verified end
to end on FALCONPRD against the live GPU server, and **frozen** at `f978b7d` as
the known-good fallback — see `FROZEN.md`. Later versions are separate files.

A local git tag `dash-ui-v1` marks it; the tag is not on the remote (the
environment's proxy rejects tag pushes with 403), so the SHA is the durable
reference.

**229 assertions, six suites.**

---

## v0.1 — The pipeline · 2026-09-09

| Commit | Change |
|---|---|
| `cd5214d` | Phase 4: orchestrator and the integrated demo UI |
| `9623abb` | Phase 3: VLM client, prompt wiring, tolerant parser, mock fallback |
| `72c13b1`, `4aa7be2`, `06d819e` | Label resolution: beside the photo, per question, at any depth |
| `a08c823` | Phase 2: annotation-file detector, shared box drawing, label tooling |
| `8491b13` | rembg 2.0.69 compatibility |
| `eb17da8` | Stage 1 smoke test |
| `c5a6764` | Vendor `quality_check` + `foreground_segmentation`; setup and requirements |
| `9e5da32` | preflight: gate the offline check on rembg |
| `c4c7ee4` | Phase 1: schemas, config, question registry, quality stage |

### Added
- `app/pipeline/` — `schemas.py` (stage contracts), `config.py`, `questions.py`
  (per-question system prompts and class names), the three stages, and
  `orchestrator.py`.
- `tools/preflight.py` — checks a demo host before the demo, not a dev box.
- `SETUP.md`.

### Fixed
- **rembg 2.0.69** builds its own `SessionOptions` and passes it positionally,
  so passing `sess_opts` — even `None` — raised `got multiple values for
  argument 'sess_opts'`. A thread cap now becomes `OMP_NUM_THREADS`.
- **Sidecar labels not found.** `resolve_annotation_dir` used non-recursive
  `glob` while images used `rglob`. Labels now resolve per image, beside the
  photograph first.
- **Two single-class CVAT exports both numbering class 0** — resolved with
  per-question `default_class_names`.

**Verified on FALCONPRD:** stage 1 ~2s per photograph, segmentation 3/3; stage 3
0.21–0.48s server-side with reasoning citing real sign text.

---

## Before this work

Commits before `c4c7ee4` (2026-07-21 and earlier) are unrelated Instant Graph
MCP server work — `instant_graph_mcp_server_v2_4_1.py` at the repo root.
