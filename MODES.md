# Three pipeline modes

`app/demo_dash_modes.py` on **port 7872**. One click runs all three modes over
the same photographs; the mode selector then switches between three sets of
results without re-running anything.

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Name | Quality gate → Detector → Model | (Quality **OR** Detector) → Model | Everything by the model |
| Quality | MM-IQA + u2netp, **hard gate** | MM-IQA + u2netp, advisory | the model judges it |
| Detection | YOLOX-S, boxes | YOLOX-S, boxes | the model reports presence, **no boxes** |
| Question | Qwen3-VL | Qwen3-VL | Qwen3-VL, a third call of its own |
| A blurry photo whose sign is clearly detected | **dropped** | **answered** | answered if the model calls it usable |

## What each mode is arguing

**Mode 1** — the pipeline as built. Quality decides first, and a photograph
that fails is never looked at again. Cheap, and defensible: you don't want an
inspection decision made from an unusable photograph.

**Mode 2** — the quality score is advisory, not final. A photo proceeds if the
gate passes **or** the detector found the subject. The argument: a soft-focus
shot where the warning sign is plainly detected at 0.94 is not a photo you
should throw away. Every card in this mode states *why* it was let through.

**Mode 3** — no u2netp, no YOLOX. The vision model makes all three judgements
itself, as **three separate calls** over the same photograph: is it usable, is
the subject there, and what is the answer. The argument: one model that sees the
whole photograph may judge those more coherently than three components that each
see a slice.

They are three calls rather than one so that each leg has its **own system
prompt**, editable in the UI, and so that a wording change to one judgement
cannot drag the other two with it. The image is encoded once and reused across
all three, so the extra calls cost tokens, not another decode.

Mode 2 is **strictly more permissive** than mode 1 — never less. The only case
where they differ is the interesting one: quality failed, the detector found
the subject anyway.

## Comparing them

The **Compare modes** tab puts one row per photograph against one column per
mode, and counts how often all three agree. A row where they disagree is the
one worth talking about: same photograph, three ways of deciding, different
conclusions.

## Cost

Running all three does **not** cost three times as much. Each photograph is
loaded once, quality-scored once and detected once; modes 1 and 2 read the same
results and differ only in what they do with them. When both gates open, the
model sees identical inputs, so that answer is computed once and shared.

Modes 1 and 2 share a single model call when both gates open; a second is only
needed when mode 1 stops a photo that mode 2 lets through. Mode 3 adds three
calls of its own — one per leg. Typical cost is therefore **four model calls per
photograph** for all three modes.

On current timings (~2s quality, ~3s detection, ~0.4s per model call) that is
roughly **6.5s per photograph** for all three modes, against ~5.5s for one.

## What mode 3 does NOT do

It draws no bounding boxes. Qwen3-VL can be asked for coordinates, but its
grounding accuracy is well below YOLOX's, and a visibly wrong box on screen is
worse than an honest "presence, not geometry". The detection panel in mode 3
shows a YES/NO/UNKNOWN presence chip and the model's reasoning instead, and
says so on the card.

Mode 3 also has no numeric quality score. The card says "judged by the model,
no numeric score" rather than printing a fabricated number next to PASS, and the
quality panel omits the MM-IQA line entirely rather than showing `0.0`.

It renders the photograph as it was judged: the plain, EXIF-corrected original,
with **no green foreground box** — u2netp never ran in this mode, so drawing one
would claim a crop that did not happen.

## Tuning mode 3's prompts

The **Mode 3 prompts** card in the sidebar holds one editable system prompt per
leg. The *user* prompt below each is shown but fixed: it carries the JSON
contract the parser keys on, and an edit there would silently turn every answer
into `unknown`.

Edits are per question — switching the question dropdown switches the whole set
— and **Save** writes only the legs you actually changed to `config/prompts.yaml`.
Text equal to the built-in default clears the override; when nothing is
overridden the file is deleted, so the repo default is the absence of a file, not
a copy of it. A malformed file falls back to the built-ins and says so in the
status line rather than failing the run.

**Re-run mode 3 only** re-runs the three legs over the photographs already
loaded, reusing modes 1 and 2's existing results. Quality scoring and detection
are the expensive stages and a prompt change cannot affect them, so re-running
everything would spend them to produce identical output. The comparison tab
still lines up row for row afterwards.

## Reaching the model

All three modes call the same GPU server, either directly or through
`llm_proxy_v3` on FALCONPRD when the host cannot see the GPU box. Mode 3 is the
most affected by the choice: it makes three calls per photograph where modes 1
and 2 share one. See `PROXY.md`.

## Files

| Path | Role |
|---|---|
| `app/pipeline/modes.py` | The three modes, the shared-work runner, comparison and agreement |
| `app/pipeline/questions.py` | The three legs, their default system prompts and their fixed user prompts |
| `app/pipeline/prompts.py` | `PromptStore` — reads and writes `config/prompts.yaml` |
| `app/pipeline/stage3_vlm.py` | `ask_leg()`, `ask_vlm_only()`, `parse_quality()`, `parse_presence()` |
| `app/demo_dash_modes.py` | The UI |
| `tests/test_modes.py` | 90 assertions |

Each mode writes its own `pipeline_results.csv` and `results.json` under
`demo_runs/<run_id>/<mode>/`.
