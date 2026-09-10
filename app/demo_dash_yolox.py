#!/usr/bin/env python3
"""Field Ops integrated demo - Dash UI, TRAINED YOLOX-S detector.

    python app/demo_dash_yolox.py       ->  http://<host>:7871

The annotation-file version is app/demo_dash.py on 7870 (frozen at the
dash-ui-v1 commit in FROZEN.md). This one differs in exactly one leg: stage 2
runs the trained YOLOX-S checkpoint instead of reading human .txt sidecars.
Different port on purpose, so both can be up at once and either can be shown.

The card, tile and summary renderers are IMPORTED from demo_dash rather than
copied. They are identical by design - a real detection and a human annotation
must look the same on screen, or the audience learns to read the styling
instead of the provenance banner - and a copy would drift.

Defaults come from the training run's own logged exp table (02_train.ipynb
cell 9): 2 classes, depth 0.33, width 0.50, input 640x480. See
pipeline/yolox_runtime.py for why the non-square input size matters.
"""
from __future__ import annotations

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
from pipeline.questions import QUESTIONS, get_question  # noqa: E402

# Shared renderers - see the module docstring on why these are imported.
from demo_dash import (FONTS, QUESTION_OPTIONS, FIRST, prompt_text,  # noqa: E402
                       results_view, summary_view, tile)

FONT_SHEET = FONTS


def _cfg_from_controls(checkpoint, classes, conf, nms, size_h, size_w, fuse,
                       gpu_url, vlm_model, vlm_mode, send_mode, threshold, flags):
    flags = flags or []
    names = tuple(n.strip() for n in (classes or "").split(",") if n.strip())
    cfg = default_config(
        use_model=True,
        yolox_checkpoint=(checkpoint or "").strip() or None,
        yolox_class_names=names or ("hazard_sign", "gps_antenna"),
        yolox_input_size=(int(size_h or 640), int(size_w or 480)),
        yolox_nms_threshold=float(nms if nms is not None else 0.65),
        yolox_fuse="fuse" in flags,
        conf_thresh=float(conf if conf is not None else 0.30),
        gpu_url=(gpu_url or "").strip(),
        vlm_model=(vlm_model or "qwen3-vl").strip(),
        vlm_mode=vlm_mode, vlm_send_mode=send_mode,
        quality_threshold=float(threshold) if threshold is not None else 65.0,
        ignore_resolution="ignore_res" in flags,
        run_downstream_on_fail="run_on_fail" in flags,
    )
    return cfg


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
            html.Div("Detector", className="card-title"),
            html.Div([
                html.Label("YOLOX checkpoint", htmlFor="ckpt"),
                dcc.Input(id="ckpt", type="text", debounce=True,
                          placeholder="/path/to/best_ckpt.pth"),
                html.Div("YOLOX-S, depth 0.33 / width 0.50, as trained.",
                         className="field-help"),
            ], className="field"),
            html.Div([
                html.Label("Class names, in training index order", htmlFor="classes"),
                dcc.Input(id="classes", type="text",
                          value=", ".join(cfg.yolox_class_names)),
                html.Div("YOLOX stores no class names in a checkpoint. Index 0 first. "
                         "A wrong order puts a wrong label on screen and into the "
                         "model prompt as evidence.", className="field-help"),
            ], className="field"),
            html.Div([
                html.Label("Confidence threshold", htmlFor="conf"),
                dcc.Input(id="conf", type="number", value=0.30, min=0.01, max=0.99,
                          step=0.01),
            ], className="field"),
            html.Button("Load / check model", id="check", n_clicks=0,
                        className="btn btn-ghost"),
        ], className="card"),

        html.Div([
            html.Div("Photos", className="card-title"),
            html.Div([
                html.Label("Folder on this machine", htmlFor="folder"),
                dcc.Input(id="folder", type="text", debounce=True,
                          placeholder="/data/adnaan/fieldops/demo/photos"),
                html.Div("The detector reads the image, so no annotation sidecars "
                         "are needed here - point at any folder of photos.",
                         className="field-help"),
            ], className="field"),
        ], className="card"),

        html.Div([
            html.Div("Run", className="card-title"),
            html.Button("Run pipeline", id="run", n_clicks=0, className="btn btn-primary"),
            html.Details([
                html.Summary("Options"),
                html.Div([
                    html.Label("Quality pass threshold", htmlFor="threshold"),
                    dcc.Input(id="threshold", type="number", value=65, min=0, max=100,
                              step=0.5),
                ], className="field"),
                html.Div([
                    html.Label("Detector input size (height, width)"),
                    html.Div([
                        dcc.Input(id="size-h", type="number", value=640, min=32,
                                  step=32, style={"width": "48%", "marginRight": "4%"}),
                        dcc.Input(id="size-w", type="number", value=480, min=32,
                                  step=32, style={"width": "48%"}),
                    ]),
                    html.Div("640x480 as trained - NOT the stock 640x640. Changing "
                             "this does not error, it shifts every box.",
                             className="field-help"),
                ], className="field"),
                html.Div([
                    html.Label("NMS IoU threshold", htmlFor="nms"),
                    dcc.Input(id="nms", type="number", value=0.65, min=0.05, max=0.95,
                              step=0.05),
                ], className="field"),
                html.Div(dcc.Checklist(
                    id="flags", className="opt",
                    options=[
                        {"label": "Fuse conv+BN (faster inference)", "value": "fuse"},
                        {"label": "Ignore the minimum-resolution floor", "value": "ignore_res"},
                        {"label": "Run downstream legs on FAIL images anyway",
                         "value": "run_on_fail"},
                    ], value=["fuse"]), className="field"),
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
                html.P("Quality gate, trained YOLOX-S detection and model question "
                       "answering, in one pass over a folder of site photographs."),
                html.Div(className="rule"),
            ], className="masthead"),

            html.Div(id="status", className="statusbar", children=[
                html.Span(className="dot"),
                html.Span("Set the checkpoint path, then press Load / check model."),
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
            dcc.Download(id="download"),
        ], className="shell"),
    ])


app = Dash(__name__, external_stylesheets=FONT_SHEET,
           title="Field Ops Demo — YOLOX", update_title="Running…",
           assets_folder=str(APP_DIR / "assets"), suppress_callback_exceptions=True)
app.layout = layout()
server = app.server

_RESULTS: dict = {"records": [], "question_id": FIRST, "run_dir": None}


@app.callback(Output("semantics", "children"), Output("prompt-view", "children"),
              Input("question", "value"))
def on_question(question_id):
    return get_question(question_id).answer_semantics, prompt_text(question_id)


@app.callback(Output("status", "children"), Input("check", "n_clicks"),
              State("ckpt", "value"), State("classes", "value"), State("conf", "value"),
              State("nms", "value"), State("size-h", "value"), State("size-w", "value"),
              State("flags", "value"), State("gpu-url", "value"),
              prevent_initial_call=True)
def on_check(_clicks, ckpt, classes, conf, nms, size_h, size_w, flags, gpu_url):
    cfg = _cfg_from_controls(ckpt, classes, conf, nms, size_h, size_w, flags,
                             gpu_url, "qwen3-vl", VLM_MODE_LIVE, SEND_FULL, 65, flags)
    parts = []

    model_path = cfg.segmentation_model_path()
    parts += [html.Span(className="dot dot-ok" if model_path.exists() else "dot dot-bad"),
              html.Span("u2netp ready" if model_path.exists()
                        else f"u2netp MISSING at {model_path}"),
              html.Span("|", className="status-sep")]

    # Loading the detector here, on demand, means a bad checkpoint path or a
    # class-count mismatch surfaces before a run rather than inside one.
    try:
        from pipeline.yolox_runtime import get_predictor
        predictor = get_predictor(cfg)
        parts += [html.Span(className="dot dot-ok"),
                  html.Span(f"{predictor.describe()} · classes: "
                            f"{', '.join(predictor.class_names)}")]
    except Exception as exc:
        first = str(exc).strip().splitlines()[0]
        parts += [html.Span(className="dot dot-bad"),
                  html.Span(f"Detector not loaded — {first}")]
    parts.append(html.Span("|", className="status-sep"))

    from pipeline.stage3_vlm import VLMClient
    client = VLMClient(cfg)
    health, error = client.health()
    if error:
        parts += [html.Span(className="dot dot-bad"),
                  html.Span("GPU server unreachable — answers will be MOCK")]
    else:
        client.refresh_registry()
        loaded = ", ".join(health.get("loaded_models") or []) or "none"
        parts += [html.Span(className="dot dot-ok"),
                  html.Span(f"GPU server ok · loaded: {loaded}")]
    return parts


@app.callback(Output("panel", "children"),
              Output("status", "children", allow_duplicate=True),
              Input("run", "n_clicks"), Input("tabs", "value"),
              State("question", "value"), State("folder", "value"),
              State("ckpt", "value"), State("classes", "value"), State("conf", "value"),
              State("nms", "value"), State("size-h", "value"), State("size-w", "value"),
              State("threshold", "value"), State("flags", "value"),
              State("gpu-url", "value"), State("vlm-model", "value"),
              State("vlm-mode", "value"), State("send-mode", "value"),
              prevent_initial_call="initial_duplicate")
def on_run(n_clicks, tab, question_id, folder, ckpt, classes, conf, nms, size_h,
           size_w, threshold, flags, gpu_url, vlm_model, vlm_mode, send_mode):
    import dash
    triggered = (dash.callback_context.triggered[0]["prop_id"].split(".")[0]
                 if dash.callback_context.triggered else "")
    question = get_question(question_id or FIRST)

    if triggered != "run":
        return _panel(tab, question), no_update

    if not folder or not folder.strip():
        msg = "Point at a photo folder on this machine."
        return (html.Div(msg, className="empty"),
                [html.Span(className="dot dot-warn"), html.Span(msg)])

    cfg = _cfg_from_controls(ckpt, classes, conf, nms, size_h, size_w, flags,
                             gpu_url, vlm_model, vlm_mode, send_mode, threshold, flags)
    problems = cfg.validate()
    blocking = [p for p in problems if "yolox_checkpoint" in p]
    if blocking:
        return (html.Div([html.Div(p, className="banner banner-stop") for p in blocking],
                         className="card"),
                [html.Span(className="dot dot-bad"), html.Span(blocking[0])])

    root = Path(folder.strip())
    if not root.exists():
        msg = f"No such folder: {root}"
        return (html.Div(msg, className="empty"),
                [html.Span(className="dot dot-bad"), html.Span(msg)])
    paths = collect_images(root)
    if not paths:
        msg = f"No images found under {root}."
        return (html.Div(msg, className="empty"),
                [html.Span(className="dot dot-warn"), html.Span(msg)])

    try:
        records = run_pipeline(paths, question.id, cfg)
    except Exception as exc:
        return (html.Div([html.Div(f"Run failed: {exc}", className="banner banner-stop"),
                          html.Pre(traceback.format_exc()[-2400:])], className="card"),
                [html.Span(className="dot dot-bad"), html.Span(f"Run failed: {exc}")])

    _RESULTS.update(records=records, question_id=question.id, run_dir=cfg.run_dir)
    s = summarise(records)
    n_boxes = sum(len(r.detection.detections) for r in records if r.detection)
    status = [html.Span(className="dot dot-ok"),
              html.Span(f"Done — {s['total']} photo(s) · {s['passed']} pass, "
                        f"{s['failed']} fail · {n_boxes} detection(s) · "
                        + ", ".join(f"{k.upper()} {n}"
                                    for k, n in sorted(s["answers"].items()))
                        + (f" · {s['mocked']} MOCK" if s["mocked"] else ""))]
    return _panel(tab, question), status


def _panel(tab, question):
    records = _RESULTS["records"]
    run_dir = _RESULTS["run_dir"]
    if tab == "summary":
        return summary_view(records, question, run_dir)
    if tab == "export":
        if not records:
            return html.Div("Run the pipeline to produce a CSV and JSON.",
                            className="empty")
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

    view = results_view(records, question)
    if not records:
        return view
    # One extra tile the annotation version cannot show: what the detector
    # actually found, per class.
    counts: dict[str, int] = {}
    for record in records:
        for det in (record.detection.detections if record.detection else []):
            counts[det.label] = counts.get(det.label, 0) + 1
    if counts:
        extra = html.Div([tile(n, f"{label} boxes") for label, n in sorted(counts.items())],
                         className="tiles")
        return html.Div([extra, view])
    return view


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
    return dcc.send_file(str(path)) if path.exists() else no_update


def main() -> None:
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASH_PORT", "7871")))


if __name__ == "__main__":
    main()
