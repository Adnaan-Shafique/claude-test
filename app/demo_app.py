#!/usr/bin/env python3
"""Field Ops integrated demo - one page driving all three legs.

    python app/demo_app.py                      -> http://<host>:7870

Port 7870 deliberately: 8056 (batch_ui), 8050 (Dash detection), 7860 (VLM
client) and 7861 (review_ui) stay free so any of those can run alongside as a
fallback.

This drives the three legs as LIBRARY CALLS through pipeline.orchestrator. It
does not shell out to, or embed, any of the existing UIs.
"""
from __future__ import annotations

import base64
import html
import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import gradio as gr  # noqa: E402

from pipeline.config import (SEND_FULL, SEND_FULL_CROP, VLM_MODE_LIVE,  # noqa: E402
                             VLM_MODE_MOCK, default_config)
from pipeline.orchestrator import (collect_images, run_pipeline,  # noqa: E402
                                   sort_for_display, summarise)
from pipeline.questions import QUESTIONS, get_question  # noqa: E402
from pipeline.schemas import STOPPED_QUALITY  # noqa: E402

CARD_IMAGE_PX = 460   # thumbnails are embedded as data URIs; keep them modest


# ─────────────────────────────── Design system ───────────────────────────────
# Vodafone Idea Design System V.01 - DM Sans, #F4F1EC ground, white surfaces,
# red-to-yellow primary gradient. Carried over from vlm_test_client_v2.py so
# the integrated demo looks like the tool the team already knows.

DS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --vf-bg:#F4F1EC; --vf-surface:#FFFFFF;
  --vf-grad:linear-gradient(90deg,#EE3B2F 0%,#F0A202 100%);
  --vf-yellow:#F0A202; --vf-yellow-2:#FEF3C7;
  --vf-green:#0DAF94; --vf-green-2:#B4E6DD;
  --vf-red:#EE3B2F; --vf-red-2:#FFF1F2;
  --vf-blue:#5F5FEF; --vf-blue-2:#EEF2FF;
  --vf-black:#1A1917; --vf-black-2:#3D3B37; --vf-grey:#6B685F;
  --vf-line:#E2DDD4; --vf-radius:14px;
  --vf-font:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}
body,.gradio-container{background:var(--vf-bg)!important;font-family:var(--vf-font)!important;}
.gradio-container{max-width:1680px!important;color:var(--vf-black)!important;}
.gradio-container *,.gradio-container button,.gradio-container input,
.gradio-container textarea,.gradio-container select{font-family:var(--vf-font)!important;}
footer{display:none!important;}

.vf-head{padding:4px 2px 0;}
.vf-head h1{font-size:40px;line-height:1.15;font-weight:400;color:var(--vf-black);margin:0 0 4px;}
.vf-head h1 .vf-accent{color:var(--vf-red);font-weight:700;}
.vf-head p{font-size:15px;color:var(--vf-grey);margin:0;}
.vf-head .vf-rule{height:3px;width:96px;border-radius:2px;margin:14px 0 2px;background:var(--vf-grad);}

.vf-card{background:var(--vf-surface);border:1px solid var(--vf-line);
  border-radius:var(--vf-radius);padding:20px!important;gap:14px!important;
  box-shadow:0 1px 2px rgba(26,25,23,.04);}
.vf-card-title,.vf-card-title p{font-size:12px!important;letter-spacing:.09em;
  text-transform:uppercase;color:var(--vf-grey)!important;font-weight:500;margin:0 0 2px!important;}
.vf-card .vf-card{border:none!important;padding:0!important;box-shadow:none!important;border-radius:0!important;}
.vf-card .styler{background:transparent!important;border:none!important;}
.gradio-container .column,.gradio-container .row,.gradio-container .form,
.gradio-container .panel{background:transparent!important;border:none!important;box-shadow:none!important;}
.vf-card .block{border-width:0!important;background:transparent!important;}

.vf-status,.vf-status p{font-size:13px!important;color:var(--vf-black-2)!important;margin:0!important;}
.vf-status{background:var(--vf-surface);border:1px solid var(--vf-line);
  border-radius:999px;padding:9px 16px!important;}

.gradio-container label,.gradio-container .label-wrap span,span[data-testid="block-info"]{
  font-size:13px!important;color:var(--vf-black-2)!important;font-weight:500!important;}
.gradio-container input[type=text],.gradio-container textarea,
.gradio-container .wrap-inner,.gradio-container select{
  background:var(--vf-surface)!important;border-radius:10px!important;
  border-color:var(--vf-line)!important;color:var(--vf-black)!important;font-size:15px!important;}
.gradio-container button{border-radius:10px!important;font-size:15px!important;}
.vf-primary button,button.vf-primary{background:var(--vf-grad)!important;color:#fff!important;
  border:none!important;font-weight:500!important;padding:12px 20px!important;}
.vf-ghost button,button.vf-ghost{background:var(--vf-surface)!important;
  color:var(--vf-black-2)!important;border:1px solid var(--vf-line)!important;}

/* ── Result cards ─────────────────────────────────────────────────────────── */
.rc{background:var(--vf-surface);border:1px solid var(--vf-line);border-radius:var(--vf-radius);
  padding:18px;margin-bottom:18px;box-shadow:0 1px 2px rgba(26,25,23,.04);}
.rc-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px;}
.rc-name{font-size:17px;font-weight:700;color:var(--vf-black);}
.rc-sub{font-size:12px;color:var(--vf-grey);}
.rc-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;}
.rc-col h4{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--vf-grey);
  font-weight:500;margin:0 0 8px;}
.rc-col img{width:100%;border-radius:9px;border:1px solid var(--vf-line);display:block;}
.rc-none{border:1px dashed var(--vf-line);border-radius:9px;padding:26px 12px;text-align:center;
  color:var(--vf-grey);font-size:13px;background:var(--vf-bg);}
.rc-line{font-size:13px;color:var(--vf-black-2);margin:7px 0;line-height:1.5;}
.rc-muted{font-size:12px;color:var(--vf-grey);}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:500;}
.pill-pass{background:var(--vf-green-2);color:#07584A;}
.pill-fail{background:var(--vf-red-2);color:#A32118;}
.pill-err{background:#EFEFEF;color:var(--vf-black-2);}
.pill-stub{background:var(--vf-blue-2);color:#3B3BB5;}
.pill-mock{background:var(--vf-yellow-2);color:#7A5200;}
.chip{display:inline-block;padding:9px 22px;border-radius:11px;font-size:23px;font-weight:700;
  letter-spacing:.03em;}
.chip-yes{background:var(--vf-green-2);color:#07584A;}
.chip-no{background:var(--vf-red-2);color:#A32118;}
.chip-unknown{background:#EFEFEF;color:var(--vf-black-2);}
.rc-reason{font-size:14px;line-height:1.6;color:var(--vf-black);margin:11px 0;}
.rc details{margin-top:9px;}
.rc summary{font-size:12px;color:var(--vf-grey);cursor:pointer;}
.rc pre{background:var(--vf-bg);border:1px solid var(--vf-line);border-radius:8px;padding:10px;
  font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:230px;overflow:auto;}
.rc-warn{background:var(--vf-yellow-2);border-radius:8px;padding:9px 12px;font-size:12.5px;
  color:#7A5200;margin:9px 0;}
.rc-stopped{background:var(--vf-red-2);border-radius:8px;padding:11px 13px;font-size:13px;
  color:#A32118;}
.rc-ul{margin:7px 0 0 17px;padding:0;font-size:13px;color:var(--vf-black-2);line-height:1.55;}
"""


def _theme_kwargs() -> dict:
    """Gradio 6 moved css/head from the Blocks constructor to launch(); 4 and 5
    only accept them on Blocks. Send them wherever this installation reads them.
    Carried over from vlm_test_client_v2.py rather than reinvented."""
    try:
        major = int(str(gr.__version__).split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return ({"blocks": {}, "launch": {"css": DS_CSS}} if major >= 6
            else {"blocks": {"css": DS_CSS}, "launch": {}})


# ─────────────────────────────── Rendering ───────────────────────────────────

def _thumb(path, max_px: int = CARD_IMAGE_PX) -> str:
    """Downscaled data URI. Embedding avoids Gradio's static-path permissions
    entirely, which is one less thing to misconfigure on the day."""
    if not path:
        return ""
    try:
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            return ""
        h, w = image.shape[:2]
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _img_or_placeholder(path, alt: str) -> str:
    uri = _thumb(path)
    if uri:
        return f'<img src="{uri}" alt="{html.escape(alt)}">'
    return f'<div class="rc-none">{html.escape(alt)}</div>'


def _quality_column(record) -> str:
    q = record.quality
    pill = ("pill-err" if q.error else "pill-pass" if q.passed else "pill-fail")
    parts = [
        '<div class="rc-col"><h4>1 &middot; Quality gate</h4>',
        _img_or_placeholder(q.annotated_path, "image could not be rendered"),
        f'<div class="rc-line"><span class="pill {pill}">{html.escape(q.headline)}</span></div>',
    ]
    if not q.error:
        parts.append(
            f'<div class="rc-muted">{q.width}&times;{q.height} &middot; '
            f'whole frame {q.whole_frame_score:.1f}'
            + (f' &rarr; cropped {q.foreground_score:.1f}' if q.segmentation_used
               else ' &middot; no foreground box, whole-frame score used')
            + '</div>')
    if q.failure_reasons:
        parts.append('<div class="rc-line">Reasons: '
                     + html.escape(", ".join(q.failure_reasons)) + '</div>')
    if q.retake_instructions:
        parts.append('<ul class="rc-ul">'
                     + "".join(f"<li>{html.escape(i)}</li>" for i in q.retake_instructions)
                     + "</ul>")
    parts.append("</div>")
    return "".join(parts)


def _detection_column(record) -> str:
    d = record.detection
    if d is None:
        return ('<div class="rc-col"><h4>2 &middot; Detection</h4>'
                '<div class="rc-none">Not run &mdash; the quality gate stopped '
                'this image.</div></div>')
    # Provenance is an honesty requirement, not a style preference: nobody
    # watching should think a model produced these boxes.
    banner = ('<span class="pill pill-stub">human annotation (no model loaded)</span>'
              if d.is_stub else f'<span class="pill pill-stub">{html.escape(d.model_name)}</span>')
    parts = ['<div class="rc-col"><h4>2 &middot; Detection</h4>',
             _img_or_placeholder(d.annotated_path, "no boxes to draw"),
             f'<div class="rc-line">{banner}</div>']
    if d.detections:
        parts.append('<ul class="rc-ul">' + "".join(
            f"<li>{html.escape(x.label)} &mdash; {x.confidence:.2f}</li>"
            for x in sorted(d.detections, key=lambda x: -x.confidence)) + "</ul>")
    else:
        parts.append('<div class="rc-line rc-muted">No detections.</div>')
    if d.note:
        parts.append(f'<div class="rc-warn">{html.escape(d.note)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _vlm_column(record, question) -> str:
    v = record.vlm
    if v is None:
        return ('<div class="rc-col"><h4>3 &middot; Model answer</h4>'
                '<div class="rc-none">Not run &mdash; the quality gate stopped '
                'this image.</div></div>')
    chip = {"yes": "chip-yes", "no": "chip-no"}.get(v.answer, "chip-unknown")
    parts = [
        '<div class="rc-col"><h4>3 &middot; Model answer</h4>',
        f'<div><span class="chip {chip}" title="{html.escape(question.answer_semantics)}">'
        f'{html.escape(v.chip)}</span></div>',
        f'<div class="rc-reason">{html.escape(v.reasoning)}</div>',
    ]
    if v.is_mock:
        parts.append('<div class="rc-warn"><b>MOCK</b> &mdash; no model examined this '
                     'image. The answer above is canned.</div>')
    parts.append(f'<div class="rc-muted">{html.escape(v.provenance)} &middot; '
                 f'{v.elapsed_s:.2f}s</div>')
    if v.error:
        parts.append(f'<div class="rc-warn">{html.escape(v.error)}</div>')
    if v.raw_text:
        parts.append('<details><summary>Raw model output</summary><pre>'
                     + html.escape(v.raw_text) + "</pre></details>")
    parts.append("</div>")
    return "".join(parts)


def render_cards(records, question) -> str:
    if not records:
        return ('<div class="rc-none">No results yet. Pick a question, point at a '
                'photo folder, and run the pipeline.</div>')
    blocks = []
    for record in sort_for_display(records):
        stopped = ""
        if record.stopped_at == STOPPED_QUALITY:
            stopped = ('<div class="rc-stopped">Stopped at the quality gate &mdash; '
                       'the downstream legs did not run. Turn on '
                       '&ldquo;Run downstream legs on FAIL images anyway&rdquo; in '
                       'Advanced to override.</div>')
        blocks.append(
            '<div class="rc">'
            f'<div class="rc-head"><span class="rc-name">{html.escape(record.filename)}</span>'
            f'<span class="rc-sub">{html.escape(record.stem)}</span></div>'
            f'{stopped}'
            '<div class="rc-cols">'
            f'{_quality_column(record)}{_detection_column(record)}'
            f'{_vlm_column(record, question)}'
            "</div></div>")
    return "".join(blocks)


def render_summary(records, question, run_dir) -> str:
    if not records:
        return ""
    s = summarise(records)
    answers = s["answers"]
    lines = [
        f"### {s['total']} photo(s) &mdash; {question.label}",
        f"**Quality:** {s['passed']} PASS &nbsp;&middot;&nbsp; {s['failed']} FAIL"
        + (f" &nbsp;&middot;&nbsp; {s['errored']} ERROR" if s["errored"] else ""),
        f"**Answers:** " + (", ".join(f"{k.upper()} {n}" for k, n in sorted(answers.items()))
                            or "none"),
        f"**Foreground segmentation used:** {s['segmentation_used']}/{s['total']}",
    ]
    if s["stopped_at_quality"]:
        lines.append(f"**Stopped at the quality gate:** {s['stopped_at_quality']} "
                     f"(downstream legs not run)")
    if s["mocked"]:
        lines.append(f"**MOCK answers:** {s['mocked']} &mdash; the GPU server was not "
                     f"reached for these; no model examined them.")
    lines.append(f"\n_{question.answer_semantics}_")
    lines.append(f"\n**Run folder:** `{run_dir}`")
    return "\n\n".join(lines)


def rows_for_table(records) -> list[list]:
    out = []
    for record in sort_for_display(records):
        row = record.to_flat_row()
        out.append([row["filename"], row["quality_verdict"], row["quality_score"],
                    row["detections"], row["vlm_answer"], row["stopped_at"]])
    return out


TABLE_HEADERS = ["file", "quality", "score", "detections", "answer", "stopped at"]


# ─────────────────────────────── Actions ─────────────────────────────────────

def check_connections(gpu_url: str) -> str:
    """u2netp on disk and the GPU server reachable, as one status line."""
    cfg = default_config(gpu_url=gpu_url)
    bits = []

    model_path = cfg.segmentation_model_path()
    bits.append(f"u2netp {'ready' if model_path.exists() else 'MISSING at ' + str(model_path)}")

    from pipeline.stage3_vlm import VLMClient
    client = VLMClient(cfg)
    health, error = client.health()
    if error:
        bits.append(f"GPU server unreachable ({error.splitlines()[0][:90]}) &mdash; "
                    f"answers will be MOCK")
    else:
        loaded = ", ".join(health.get("loaded_models") or []) or "none"
        names, reg_error = client.refresh_registry()
        cap = client.max_images(cfg.vlm_model)
        bits.append(f"GPU server ok &middot; loaded: {loaded}"
                    + (f" &middot; {cfg.vlm_model} max_images={cap}" if cap else "")
                    + (f" &middot; registry error: {reg_error}" if reg_error else ""))
    return " &nbsp;|&nbsp; ".join(bits)


def run(question_id, folder, uploads, gpu_url, vlm_model, vlm_mode, send_mode,
        threshold, ignore_resolution, run_on_fail, annotation_dir, use_model,
        progress=gr.Progress()):
    question = get_question(question_id)
    empty_table = []

    if folder and folder.strip():
        paths = collect_images(Path(folder.strip()))
        source = f"folder `{folder.strip()}`"
        if not paths:
            yield (f"No images found under `{folder.strip()}`.", "",
                   render_cards([], question), empty_table, None, None)
            return
    elif uploads:
        paths = [Path(f) for f in uploads]
        source = f"{len(paths)} uploaded file(s)"
    else:
        yield ("Point at a photo folder on this machine, or upload images.", "",
               render_cards([], question), empty_table, None, None)
        return

    cfg = default_config(
        gpu_url=gpu_url.strip(), vlm_model=vlm_model.strip() or "qwen3-vl",
        vlm_mode=vlm_mode, vlm_send_mode=send_mode,
        quality_threshold=float(threshold) if threshold else 65.0,
        ignore_resolution=bool(ignore_resolution),
        run_downstream_on_fail=bool(run_on_fail), use_model=bool(use_model),
    )
    if annotation_dir and annotation_dir.strip():
        cfg.annotation_dir = annotation_dir.strip()
    elif folder and folder.strip():
        # Sidecar labels live with the photos; uploads never carry them, which
        # is why the folder path is the more reliable input for a live demo.
        cfg.annotation_dir = cfg.resolve_annotation_dir(Path(folder.strip()),
                                                        question_id=question_id)

    warnings = [f"Config: {w}" for w in cfg.validate()]

    def on_progress(index, total, stage):
        progress((index / max(total, 1)) * 0.97, desc=stage)

    yield (f"Running {len(paths)} image(s) from {source}&hellip;", "",
           render_cards([], question), empty_table, None, None)

    try:
        records = run_pipeline(paths, question_id, cfg, progress_cb=on_progress)
    except Exception as exc:
        yield (f"**Run failed:** {html.escape(str(exc))}\n\n```\n"
               f"{html.escape(traceback.format_exc()[-1800:])}\n```",
               "", render_cards([], question), empty_table, None, None)
        return

    run_dir = cfg.run_dir
    status = f"Done &mdash; {len(records)} image(s) from {source}."
    if warnings:
        status += "  \n" + "  \n".join(html.escape(w) for w in warnings)
    yield (status, render_summary(records, question, run_dir),
           render_cards(records, question), rows_for_table(records),
           str(run_dir / "pipeline_results.csv"), str(run_dir / "results.json"))


def show_prompt(question_id: str) -> str:
    q = get_question(question_id)
    from pipeline.questions import render_user_prompt
    return (f"SYSTEM PROMPT\n{'-' * 62}\n{q.system_prompt}\n\n"
            f"USER PROMPT (no detections)\n{'-' * 62}\n{render_user_prompt(q)}")


# ─────────────────────────────── UI ──────────────────────────────────────────

def build_ui() -> gr.Blocks:
    choices = [(q.label, q.id) for q in QUESTIONS.values()]
    first = choices[0][1]

    with gr.Blocks(title="Field Ops Demo", **_theme_kwargs()["blocks"]) as demo:
        gr.HTML('<div class="vf-head"><h1><span class="vf-accent">Field Ops</span> '
                'inspection pipeline</h1><p>Quality gate &rarr; object detection '
                '&rarr; model question answering, in one pass.</p>'
                '<div class="vf-rule"></div></div>')

        status = gr.Markdown("Press **Check connections** to verify the segmenter "
                             "and the GPU server.", elem_classes="vf-status")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes="vf-card"):
                    gr.Markdown("Question", elem_classes="vf-card-title")
                    question = gr.Dropdown(choices=choices, value=first,
                                           label="What are we asking about each photo?")
                    semantics = gr.Markdown(f"_{QUESTIONS[first].answer_semantics}_")
                    with gr.Accordion("Prompt sent to the model", open=False):
                        prompt_view = gr.Textbox(value=show_prompt(first), lines=15,
                                                 interactive=False, show_label=False)

                with gr.Group(elem_classes="vf-card"):
                    gr.Markdown("Photos", elem_classes="vf-card-title")
                    folder = gr.Textbox(
                        label="Photo folder on this machine",
                        placeholder="/data/adnaan/fieldops/demo/photos/hv",
                        info="Preferred: labels sit beside the photos and travel with "
                             "them. Uploads do not carry their .txt sidecars.")
                    uploads = gr.File(label="or upload images", file_count="multiple",
                                      file_types=["image"], type="filepath",
                                      elem_classes="vf-file")

                with gr.Accordion("Advanced", open=False):
                    threshold = gr.Number(label="Quality pass threshold", value=65)
                    ignore_resolution = gr.Checkbox(
                        label="Ignore the minimum-resolution floor", value=False)
                    run_on_fail = gr.Checkbox(
                        label="Run downstream legs on FAIL images anyway", value=False)
                    use_model = gr.Checkbox(
                        label="Use a trained detector instead of annotations",
                        value=False,
                        info="Off = human annotation files. No checkpoint ships with "
                             "the demo, so turning this on will fail until one does.")
                    annotation_dir = gr.Textbox(
                        label="Annotation folder override",
                        placeholder="(blank = look beside each photo)")
                    gpu_url = gr.Textbox(label="GPU server",
                                         value=default_config().gpu_url)
                    vlm_model = gr.Textbox(label="Model", value=default_config().vlm_model)
                    vlm_mode = gr.Radio([VLM_MODE_LIVE, VLM_MODE_MOCK],
                                        value=VLM_MODE_LIVE, label="Model mode")
                    send_mode = gr.Radio([SEND_FULL, SEND_FULL_CROP], value=SEND_FULL,
                                         label="What to send the model")

                with gr.Row():
                    run_btn = gr.Button("Run pipeline", elem_classes="vf-primary", scale=3)
                    check_btn = gr.Button("Check connections", elem_classes="vf-ghost",
                                          scale=2)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Results"):
                        cards = gr.HTML(render_cards([], QUESTIONS[first]))
                    with gr.Tab("Summary"):
                        summary = gr.Markdown()
                        table = gr.Dataframe(headers=TABLE_HEADERS, wrap=True,
                                             label="Every image")
                        with gr.Row():
                            csv_out = gr.File(label="pipeline_results.csv")
                            json_out = gr.File(label="results.json")

        question.change(lambda qid: (f"_{get_question(qid).answer_semantics}_",
                                     show_prompt(qid)),
                        inputs=question, outputs=[semantics, prompt_view])
        check_btn.click(check_connections, inputs=gpu_url, outputs=status)
        run_btn.click(
            run,
            inputs=[question, folder, uploads, gpu_url, vlm_model, vlm_mode, send_mode,
                    threshold, ignore_resolution, run_on_fail, annotation_dir, use_model],
            outputs=[status, summary, cards, table, csv_out, json_out],
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7870")),
        **_theme_kwargs()["launch"],
    )


if __name__ == "__main__":
    main()
