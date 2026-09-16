# Three pipeline modes

`app/demo_dash_modes.py` on **port 7872**. One click runs all three modes over
the same photographs; the mode selector then switches between three sets of
results without re-running anything.

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Name | Quality gate → Detector → Model | (Quality **OR** Detector) → Model | Everything by the model |
| Quality | MM-IQA + u2netp, **hard gate** | MM-IQA + u2netp, advisory | the model judges it |
| Detection | YOLOX-S, boxes | YOLOX-S, boxes | the model reports presence, **no boxes** |
| Question | Qwen3-VL | Qwen3-VL | Qwen3-VL, same call as the other two judgements |
| A blurry photo whose sign is clearly detected | **dropped** | **answered** | answered if the model calls it usable |

## What each mode is arguing

**Mode 1** — the pipeline as built. Quality decides first, and a photograph
that fails is never looked at again. Cheap, and defensible: you don't want an
inspection decision made from an unusable photograph.

**Mode 2** — the quality score is advisory, not final. A photo proceeds if the
gate passes **or** the detector found the subject. The argument: a soft-focus
shot where the warning sign is plainly detected at 0.94 is not a photo you
should throw away. Every card in this mode states *why* it was let through.

**Mode 3** — no u2netp, no YOLOX. One call to the vision model returns all
three judgements. The argument: one model that sees the whole photograph may
judge "is this usable, is the thing there, and what's the answer" more
coherently than three components that each see a slice.

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

Typical cost is **two model calls per photograph**, not three — one for
modes 1 and 2, one for mode 3. A third is only needed when mode 1 stops a photo
that mode 2 lets through.

On current timings (~2s quality, ~3s detection, ~0.4s per model call) that is
roughly **6s per photograph** for all three modes, against ~5.5s for one.

## What mode 3 does NOT do

It draws no bounding boxes. Qwen3-VL can be asked for coordinates, but its
grounding accuracy is well below YOLOX's, and a visibly wrong box on screen is
worse than an honest "presence, not geometry". The detection panel in mode 3
shows a YES/NO/UNKNOWN presence chip and the model's reasoning instead, and
says so on the card.

Mode 3 also has no numeric quality score. The card says "judged by the model,
no numeric score" rather than printing a fabricated number next to PASS.

## Files

| Path | Role |
|---|---|
| `app/pipeline/modes.py` | The three modes, the shared-work runner, comparison and agreement |
| `app/pipeline/questions.py` | Mode 3's combined prompt, built from each question's own system prompt |
| `app/pipeline/stage3_vlm.py` | `ask_combined()` and `parse_combined()` |
| `app/demo_dash_modes.py` | The UI |
| `tests/test_modes.py` | 54 assertions |

Each mode writes its own `pipeline_results.csv` and `results.json` under
`demo_runs/<run_id>/<mode>/`.
