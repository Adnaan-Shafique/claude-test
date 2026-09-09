#!/usr/bin/env python3
"""Field Ops integrated demo - Dash UI.

    python app/demo_dash.py            ->  http://<host>:7870

Styled to Vodafone Idea Design System V.01 (DM Sans, #F4F1EC ground, white
surfaces, red-to-yellow primary gradient). All of it lives in
app/assets/demo.css, which Dash serves automatically - no inline style soup, so
a token change is one edit in one file.

Ports: 7870 here, leaving 8056 (batch_ui), 8050 (Dash detection demo), 7860
(VLM client) and 7861 (review_ui) free to run alongside as fallbacks.

This drives the three legs as LIBRARY CALLS through pipeline.orchestrator. It
does not shell out to, or embed, any of the existing UIs.
"""
from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update  # noqa: E402

from pipeline.config import (SEND_FULL, SEND_FULL_CROP, VLM_MODE_LIVE,  # noqa: E402
                             VLM_MODE_MOCK, default_config)
from pipeline.orchestrator import (collect_images, run_pipeline,  # noqa: E402
                                   sort_for_display, summarise)
from pipeline.questions import QUESTIONS, get_question, render_user_prompt  # noqa: E402
from pipeline.schemas import STOPPED_QUALITY  # noqa: E402

CARD_IMAGE_PX = 460
FONTS = ["https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap"]


# ─────────────────────────────── Helpers ─────────────────────────────────────

def thumb(path, max_px: int = CARD_IMAGE_PX) -> str:
    """Downscaled data URI. Embedding sidesteps Dash's static-asset routing
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
        return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
                if ok else "")
    except Exception:
        return ""


def image_or_placeholder(path, alt: str):
    uri = thumb(path)
    return html.Img(src=uri, alt=alt) if uri else html.Div(alt, className="rc-none")


def tile(value, caption, kind=""):
    return html.Div([html.Div(str(value), className="n"),
                     html.Div(caption, className="k")],
                    className=f"tile {kind}".strip())


# ─────────────────────────────── Result cards ────────────────────────────────

def quality_column(record):
    q = record.quality
    pill = "pill-err" if q.error else "pill-pass" if q.passed else "pill-fail"
    children = [
        html.H4("1 · Quality gate"),
        image_or_placeholder(q.annotated_path, "Image could not be rendered"),
        html.Div(html.Span(q.headline, className=f"pill {pill}"), className="rc-line"),
    ]
    if not q.error:
        detail = f"{q.width}×{q.height} · whole frame {q.whole_frame_score:.1f}"
        detail += (f" → cropped {q.foreground_score:.1f}" if q.segmentation_used
                   else " · no foreground box, whole-frame score used")
        children.append(html.Div(detail, className="rc-muted"))
        if q.segmentation_used and q.foreground_score is not None:
            delta = q.foreground_score - q.whole_frame_score
            if abs(delta) >= 1.0:
                children.append(html.Div(
                    f"Cropping to the foreground moved the score {delta:+.1f}.",
                    className="rc-muted"))
    if q.failure_reasons:
        children.append(html.Div("Reasons: " + ", ".join(q.failure_reasons),
                                 className="rc-line"))
    if q.retake_instructions:
        children.append(html.Ul([html.Li(i) for i in q.retake_instructions],
                                className="rc-ul"))
    return html.Div(children, className="rc-col")


def detection_column(record):
    d = record.detection
    if d is None:
        return html.Div([html.H4("2 · Detection"),
                         html.Div("Not run — the quality gate stopped this image.",
                                  className="rc-none")], className="rc-col")
    children = [
        html.H4("2 · Detection"),
        image_or_placeholder(d.annotated_path, "No boxes to draw"),
        # Provenance is an honesty requirement, not a style preference: nobody
        # watching should think a model produced these boxes.
        html.Div(html.Span(d.model_name, className="pill pill-stub"), className="rc-line"),
    ]
    if d.detections:
        children.append(html.Ul(
            [html.Li(f"{x.label} — {x.confidence:.2f}")
             for x in sorted(d.detections, key=lambda x: -x.confidence)],
            className="rc-ul"))
    else:
        children.append(html.Div("No detections.", className="rc-muted"))
    if d.note:
        children.append(html.Div(d.note, className="banner banner-warn"))
    return html.Div(children, className="rc-col")


def vlm_column(record, question):
    v = record.vlm
    if v is None:
        return html.Div([html.H4("3 · Model answer"),
                         html.Div("Not run — the quality gate stopped this image.",
                                  className="rc-none")], className="rc-col")
    chip = {"yes": "chip-yes", "no": "chip-no"}.get(v.answer, "chip-unknown")
    children = [
        html.H4("3 · Model answer"),
        html.Div(html.Span(v.chip, className=f"chip {chip}",
                           title=question.answer_semantics)),
        html.Div(v.reasoning, className="rc-reason"),
    ]
    if v.is_mock:
        children.append(html.Div(
            [html.B("MOCK"), " — no model examined this image. The answer above is canned."],
            className="banner banner-warn"))
    children.append(html.Div(f"{v.provenance} · {v.elapsed_s:.2f}s", className="rc-muted"))
    if v.error:
        children.append(html.Div(v.error, className="banner banner-warn"))
    if v.raw_text:
        children.append(html.Details([html.Summary("Raw model output"),
                                      html.Pre(v.raw_text)]))
    return html.Div(children, className="rc-col")


def result_card(record, question):
    children = [html.Div([html.Span(record.filename, className="rc-name"),
                          html.Span(record.stem, className="rc-sub")], className="rc-head")]
    if record.stopped_at == STOPPED_QUALITY:
        children.append(html.Div(
            "Stopped at the quality gate — the downstream legs did not run. Turn on "
            "“Run downstream legs on FAIL images anyway” under Options to override.",
            className="banner banner-stop"))
    children.append(html.Div([quality_column(record), detection_column(record),
                              vlm_column(record, question)], className="rc-cols"))
    return html.Div(children, className="rc")


def results_view(records, question):
    if not records:
        return html.Div("No results yet. Choose a question, point at a photo folder, "
                        "and run the pipeline.", className="empty")
    s = summarise(records)
    answers = s["answers"]
    tiles = [tile(s["total"], "photos"),
             tile(s["passed"], "quality pass"),
             tile(s["failed"], "quality fail"),
             tile(answers.get("yes", 0), "yes", "tile-yes"),
             tile(answers.get("no", 0), "no", "tile-no"),
             tile(answers.get("unknown", 0), "unknown", "tile-unknown")]
    if s["mocked"]:
        tiles.append(tile(s["mocked"], "mock", "tile-mock"))
    return html.Div([html.Div(tiles, className="tiles")]
                    + [result_card(r, question) for r in sort_for_display(records)])


def summary_view(records, question, run_dir):
    if not records:
        return html.Div("Run the pipeline to see a summary.", className="empty")
    s = summarise(records)
    rows = []
    for record in sort_for_display(records):
        row = record.to_flat_row()
        rows.append({"file": row["filename"], "quality": row["quality_verdict"],
                     "score": row["quality_score"], "detections": row["detections"],
                     "answer": (row["vlm_answer"] or "—").upper(),
                     "latency": row["vlm_elapsed_s"] or "", "stopped at": row["stopped_at"]})
    columns = [{"name": c, "id": c} for c in
               ["file", "quality", "score", "detections", "answer", "latency", "stopped at"]]

    notes = [html.Div(question.answer_semantics, className="banner banner-info")]
    if s["stopped_at_quality"]:
        notes.append(html.Div(
            f"{s['stopped_at_quality']} image(s) stopped at the quality gate; their "
            f"detection and model legs did not run.", className="banner banner-warn"))
    if s["mocked"]:
        notes.append(html.Div(
            f"{s['mocked']} answer(s) are MOCK — the GPU server was not reached and no "
            f"model examined those images.", className="banner banner-warn"))

    return html.Div(notes + [
        dash_table.DataTable(
            data=rows, columns=columns, page_size=25,
            style_as_list_view=True,
            style_cell={"textAlign": "left", "padding": "11px 12px",
                        "whiteSpace": "normal", "height": "auto"},
            style_header={"fontWeight": "500"},
            style_data_conditional=[
                {"if": {"filter_query": '{quality} = "FAIL"', "column_id": "quality"},
                 "color": "#A32118", "fontWeight": "500"},
                {"if": {"filter_query": '{quality} = "PASS"', "column_id": "quality"},
                 "color": "#07584A", "fontWeight": "500"},
                {"if": {"filter_query": '{answer} = "YES"', "column_id": "answer"},
                 "color": "#07584A", "fontWeight": "700"},
                {"if": {"filter_query": '{answer} = "NO"', "column_id": "answer"},
                 "color": "#A32118", "fontWeight": "700"},
            ]),
        html.Div([html.Div("Run folder", className="field-label"),
                  html.Div(str(run_dir), className="mono")],
                 style={"marginTop": "20px"}),
    ])


# ─────────────────────────────── Controls ────────────────────────────────────

QUESTION_OPTIONS = [{"label": q.label, "value": q.id} for q in QUESTIONS.values()]
FIRST = QUESTION_OPTIONS[0]["value"]


def prompt_text(question_id: str) -> str:
    q = get_question(question_id)
    return (f"SYSTEM PROMPT\n{'─' * 58}\n{q.system_prompt}\n\n"
            f"USER PROMPT (shown without detections)\n{'─' * 58}\n{render_user_prompt(q)}")


def controls():
    cfg = default_config()
    return html.Div([
        html.Div([
            html.Div("Question", className="card-title"),
            html.Div([
                dcc.Dropdown(id="question", options=QUESTION_OPTIONS, value=FIRST,
                             clearable=False, searchable=False),
                html.Div(id="semantics", className="field-help"),
            ], className="field"),
            html.Details([
                html.Summary("Prompt sent to the model"),
                html.Div(id="prompt-view", className="prompt-box"),
            ], className="disclose"),
        ], className="card"),

        html.Div([
            html.Div("Photos", className="card-title"),
            html.Div([
                html.Label("Folder on this machine", htmlFor="folder"),
                dcc.Input(id="folder", type="text", debounce=True,
                          placeholder="/data/adnaan/fieldops/demo/photos/hv"),
                html.Div("Preferred. Annotation .txt files sit beside the photos and "
                         "are found automatically; browser uploads do not carry them.",
                         className="field-help"),
            ], className="field"),
            dcc.Upload(id="uploads", multiple=True, children=html.Div(
                "or drop images here", className="rc-none",
                style={"cursor": "pointer", "marginBottom": "0"})),
            html.Div(id="upload-note", className="field-help"),
        ], className="card"),

        html.Div([
            html.Div("Run", className="card-title"),
            html.Button("Run pipeline", id="run", n_clicks=0, className="btn btn-primary"),
            html.Button("Check connections", id="check", n_clicks=0,
                        className="btn btn-ghost"),

            html.Details([
                html.Summary("Options"),
                html.Div([
                    html.Label("Quality pass threshold", htmlFor="threshold"),
                    dcc.Input(id="threshold", type="number", value=65, min=0, max=100,
                              step=0.5),
                ], className="field"),
                html.Div(dcc.Checklist(
                    id="flags", className="opt",
                    options=[
                        {"label": "Ignore the minimum-resolution floor", "value": "ignore_res"},
                        {"label": "Run downstream legs on FAIL images anyway",
                         "value": "run_on_fail"},
                        {"label": "Use a trained detector instead of annotation files",
                         "value": "use_model"},
                    ], value=[]), className="field"),
                html.Div([
                    html.Label("Annotation folder override", htmlFor="annotation-dir"),
                    dcc.Input(id="annotation-dir", type="text",
                              placeholder="(blank = look beside each photo)"),
                ], className="field"),
                html.Div([
                    html.Label("GPU server", htmlFor="gpu-url"),
                    dcc.Input(id="gpu-url", type="text", value=cfg.gpu_url),
                ], className="field"),
                html.Div([
                    html.Label("Model", htmlFor="vlm-model"),
                    dcc.Input(id="vlm-model", type="text", value=cfg.vlm_model),
                ], className="field"),
                html.Div([
                    html.Label("Model mode"),
                    dcc.RadioItems(id="vlm-mode", className="opt",
                                   options=[{"label": "Live", "value": VLM_MODE_LIVE},
                                            {"label": "Mock (no GPU)", "value": VLM_MODE_MOCK}],
                                   value=VLM_MODE_LIVE),
                ], className="field"),
                html.Div([
                    html.Label("What to send the model"),
                    dcc.RadioItems(id="send-mode", className="opt",
                                   options=[{"label": "Full image", "value": SEND_FULL},
                                            {"label": "Full image + detection crop",
                                             "value": SEND_FULL_CROP}],
                                   value=SEND_FULL),
                ], className="field"),
            ], className="disclose"),
        ], className="card"),
    ])


def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.H1([html.Span("Field Ops", className="accent"),
                         " inspection pipeline"]),
                html.P("Quality gate, object detection and model question answering, "
                       "in one pass over a folder of site photographs."),
                html.Div(className="rule"),
            ], className="masthead"),

            html.Div(id="status", className="statusbar", children=[
                html.Span(className="dot"),
                html.Span("Press Check connections to verify the segmenter and the "
                          "GPU server."),
            ]),

            html.Div([
                html.Div(controls()),
                html.Div([
                    dcc.Tabs(id="tabs", value="results", parent_className="tabs-bar",
                             className="tabs-bar", children=[
                        dcc.Tab(label="Results", value="results", className="tab",
                                selected_className="tab--selected"),
                        dcc.Tab(label="Summary", value="summary", className="tab",
                                selected_className="tab--selected"),
                        dcc.Tab(label="Export", value="export", className="tab",
                                selected_className="tab--selected"),
                    ]),
                    dcc.Loading(html.Div(id="panel", style={"marginTop": "20px"}),
                                type="dot", color="#EE3B2F"),
                ]),
            ], className="cols"),

            dcc.Store(id="run-store"),
            dcc.Download(id="download"),
        ], className="shell"),
    ])


# ─────────────────────────────── App ─────────────────────────────────────────

app = Dash(__name__, external_stylesheets=FONTS, title="Field Ops Demo",
           update_title="Running…", suppress_callback_exceptions=True)
app.layout = layout()
server = app.server

_RESULTS: dict = {"records": [], "question_id": FIRST, "run_dir": None}


@app.callback(Output("semantics", "children"), Output("prompt-view", "children"),
              Input("question", "value"))
def on_question(question_id):
    return get_question(question_id).answer_semantics, prompt_text(question_id)


@app.callback(Output("upload-note", "children"), Input("uploads", "filename"))
def on_upload(filenames):
    if not filenames:
        return ""
    return (f"{len(filenames)} file(s) staged. Uploaded images arrive without their "
            f"annotation sidecars, so set an annotation folder under Options or use "
            f"the folder path above.")


@app.callback(Output("status", "children"), Input("check", "n_clicks"),
              State("gpu-url", "value"), prevent_initial_call=True)
def on_check(_clicks, gpu_url):
    cfg = default_config(gpu_url=(gpu_url or "").strip())
    parts = []

    model_path = cfg.segmentation_model_path()
    if model_path.exists():
        parts += [html.Span(className="dot dot-ok"), html.Span("u2netp ready")]
    else:
        parts += [html.Span(className="dot dot-bad"),
                  html.Span(f"u2netp MISSING at {model_path}")]
    parts.append(html.Span("|", className="status-sep"))

    from pipeline.stage3_vlm import VLMClient
    client = VLMClient(cfg)
    health, error = client.health()
    if error:
        parts += [html.Span(className="dot dot-bad"),
                  html.Span(f"GPU server unreachable — answers will be MOCK "
                            f"({error.splitlines()[0][:80]})")]
    else:
        client.refresh_registry()
        loaded = ", ".join(health.get("loaded_models") or []) or "none"
        cap = client.max_images(cfg.vlm_model)
        resident = cfg.vlm_model in (health.get("loaded_models") or [])
        parts += [html.Span(className="dot dot-ok" if resident else "dot dot-warn"),
                  html.Span(f"GPU server ok · loaded: {loaded}"
                            + (f" · {cfg.vlm_model} max_images={cap}" if cap else "")
                            + ("" if resident else
                               f" · {cfg.vlm_model} not resident, the first answer "
                               f"will pay a cold load"))]
    return parts


@app.callback(Output("panel", "children"), Output("status", "children", allow_duplicate=True),
              Output("run-store", "data"),
              Input("run", "n_clicks"), Input("tabs", "value"),
              State("question", "value"), State("folder", "value"),
              State("uploads", "contents"), State("uploads", "filename"),
              State("threshold", "value"), State("flags", "value"),
              State("annotation-dir", "value"), State("gpu-url", "value"),
              State("vlm-model", "value"), State("vlm-mode", "value"),
              State("send-mode", "value"),
              prevent_initial_call="initial_duplicate")
def on_run(n_clicks, tab, question_id, folder, upload_contents, upload_names,
           threshold, flags, annotation_dir, gpu_url, vlm_model, vlm_mode, send_mode):
    import dash
    triggered = (dash.callback_context.triggered[0]["prop_id"].split(".")[0]
                 if dash.callback_context.triggered else "")

    question = get_question(question_id or FIRST)

    # A tab switch just re-renders what is already in memory.
    if triggered != "run":
        return _panel(tab, question), no_update, no_update

    flags = flags or []
    cfg = default_config(
        gpu_url=(gpu_url or "").strip(), vlm_model=(vlm_model or "qwen3-vl").strip(),
        vlm_mode=vlm_mode, vlm_send_mode=send_mode,
        quality_threshold=float(threshold) if threshold is not None else 65.0,
        ignore_resolution="ignore_res" in flags,
        run_downstream_on_fail="run_on_fail" in flags,
        use_model="use_model" in flags,
    )

    paths, source = _resolve_inputs(folder, upload_contents, upload_names, cfg)
    if not paths:
        return (html.Div(source, className="empty"),
                [html.Span(className="dot dot-warn"), html.Span(source)], no_update)

    if annotation_dir and annotation_dir.strip():
        cfg.annotation_dir = annotation_dir.strip()
    elif folder and folder.strip():
        cfg.annotation_dir = cfg.resolve_annotation_dir(Path(folder.strip()),
                                                        question_id=question.id)

    try:
        records = run_pipeline(paths, question.id, cfg)
    except Exception as exc:
        return (html.Div([html.Div(f"Run failed: {exc}", className="banner banner-stop"),
                          html.Pre(traceback.format_exc()[-2400:])], className="card"),
                [html.Span(className="dot dot-bad"), html.Span(f"Run failed: {exc}")],
                no_update)

    _RESULTS.update(records=records, question_id=question.id, run_dir=cfg.run_dir)
    s = summarise(records)
    status = [html.Span(className="dot dot-ok"),
              html.Span(f"Done — {s['total']} photo(s) from {source} · "
                        f"{s['passed']} pass, {s['failed']} fail · "
                        + ", ".join(f"{k.upper()} {n}" for k, n in sorted(s["answers"].items()))
                        + (f" · {s['mocked']} MOCK" if s["mocked"] else ""))]
    return _panel(tab, question), status, {"run_dir": str(cfg.run_dir)}


def _resolve_inputs(folder, upload_contents, upload_names, cfg):
    """Returns (paths, source description) - or ([], message) when there is
    nothing to run."""
    if folder and folder.strip():
        root = Path(folder.strip())
        if not root.exists():
            return [], f"No such folder: {root}"
        paths = collect_images(root)
        if not paths:
            return [], f"No images found under {root}."
        return paths, f"{root}"
    if upload_contents:
        # Uploads arrive base64 in the callback; stage them so every stage sees
        # a real path, exactly as a folder run would.
        staged = cfg.runs_dir / "uploads" / (cfg.run_id or "current")
        staged.mkdir(parents=True, exist_ok=True)
        paths = []
        for content, name in zip(upload_contents, upload_names or []):
            try:
                _, b64 = content.split(",", 1)
            except ValueError:
                continue
            dest = staged / name
            dest.write_bytes(base64.b64decode(b64))
            paths.append(dest)
        if not paths:
            return [], "None of the uploaded files could be decoded."
        return paths, f"{len(paths)} uploaded file(s)"
    return [], "Point at a photo folder on this machine, or drop images above."


def _panel(tab, question):
    records = _RESULTS["records"]
    run_dir = _RESULTS["run_dir"]
    if tab == "summary":
        return summary_view(records, question, run_dir)
    if tab == "export":
        if not records:
            return html.Div("Run the pipeline to produce a CSV and JSON.", className="empty")
        return html.Div([
            html.Div("Export", className="card-title"),
            html.Div("One flat row per image, plus the full nested records.",
                     className="field-help", style={"marginBottom": "16px"}),
            html.Div([
                html.Button("Download pipeline_results.csv", id="dl-csv", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("Download results.json", id="dl-json", n_clicks=0,
                            className="btn btn-ghost"),
            ], className="btn-row"),
            html.Div([html.Div("Run folder", className="field-label"),
                      html.Div(str(run_dir), className="mono")],
                     style={"marginTop": "20px"}),
        ], className="card")
    return results_view(records, question)


@app.callback(Output("download", "data"),
              Input("dl-csv", "n_clicks"), Input("dl-json", "n_clicks"),
              prevent_initial_call=True)
def on_download(csv_clicks, json_clicks):
    import dash
    if not dash.callback_context.triggered or not _RESULTS["run_dir"]:
        return no_update
    which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    name = "pipeline_results.csv" if which == "dl-csv" else "results.json"
    path = Path(_RESULTS["run_dir"]) / name
    if not path.exists():
        return no_update
    return dcc.send_file(str(path))


def main() -> None:
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASH_PORT", "7870")))


if __name__ == "__main__":
    main()
