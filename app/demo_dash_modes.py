#!/usr/bin/env python3
"""Field Ops integrated demo - Dash UI, THREE PIPELINE MODES.

    python app/demo_dash_modes.py       ->  http://<host>:7872

    1  Quality gate -> Detector -> Model     (the pipeline as built)
    2  (Quality OR Detector) -> Model        (a failed quality score is not
                                              fatal if the detector found the
                                              subject)
    3  Everything by the model               (one call judges quality, subject
                                              presence and the question)

One click runs ALL THREE over the same photographs; the mode selector then
switches between the three sets of results with no re-running. That comparison
is the point: the same photograph, three ways of deciding about it, side by
side.

Siblings, all runnable at once on their own ports: app/demo_dash.py on 7870
(annotation detector, frozen - see FROZEN.md) and app/demo_dash_yolox.py on
7871 (single-mode YOLOX).

The card, tile and summary renderers are IMPORTED from demo_dash rather than
copied. They are identical by design - a real detection and a human annotation
must look the same on screen, or the audience learns to read the styling
instead of the provenance banner - and a copy would drift.

Defaults come from the training run's own logged exp table (02_train.ipynb
cell 9): 2 classes, depth 0.33, width 0.50, input 640x480. See
pipeline/yolox_runtime.py for why the non-square input size matters.
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

from pipeline.config import (SEND_FULL, SEND_FULL_CROP,  # noqa: E402
                            TRANSPORT_DIRECT, TRANSPORT_PROXY,
                            VLM_MODE_LIVE,
                             VLM_MODE_MOCK, default_config)
from pipeline.modes import (MODE_ORDER, MODE_VLM_ONLY, MODES,  # noqa: E402
                           agreement, compare_rows, run_all_modes, run_vlm_only)
from pipeline.prompts import default_store  # noqa: E402
from pipeline.orchestrator import (collect_images, new_run_id,  # noqa: E402
                                   sort_for_display, summarise)
from pipeline.questions import (LEG_LABELS, LEGS, QUESTIONS,  # noqa: E402
                                default_leg_system, get_question, render_leg_user)
from pipeline.schemas import STOPPED_QUALITY  # noqa: E402

# Shared renderers - see the module docstring on why these are imported.
# Shared renderers. vlm_column is reused verbatim so all three modes look
# identical; quality_column and detection_column are re-implemented below
# because mode 3 judges quality in words and reports subject PRESENCE rather
# than boxes, and app/demo_dash.py is frozen (see FROZEN.md) so it cannot grow
# those cases. classical_quality_column is the frozen renderer, still used for
# modes 1 and 2.
from demo_dash import (FONTS, QUESTION_OPTIONS, FIRST,  # noqa: E402
                       image_or_placeholder, prompt_text,
                       quality_column as classical_quality_column,
                       summary_view, tile, vlm_column)

FONT_SHEET = FONTS


def _cfg_from_controls(checkpoint, classes, conf, nms, size_h, size_w, fuse,
                       gpu_url, vlm_model, vlm_mode, send_mode, threshold, flags,
                       transport=None, api_key=None):
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
        vlm_transport=(transport or TRANSPORT_DIRECT),
        vlm_api_key=(api_key or "").strip(),
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
            html.Div("Mode", className="card-title"),
            dcc.RadioItems(
                id="mode", className="opt modes",
                options=[{"label": MODES[m]["label"], "value": m} for m in MODE_ORDER],
                value=MODE_ORDER[0]),
            html.Div(id="mode-blurb", className="field-help"),
            html.Div("All three run together. Switching mode re-reads results "
                     "already computed - it never re-runs the pipeline.",
                     className="field-help", style={"marginTop": "10px"}),
        ], className="card"),

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
                          value=str(cfg.models_dir / "best_ckpt.pth"),
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
            html.Div("Mode 3 prompts", className="card-title"),
            html.Div("One system prompt per leg. Each is a separate call, so a "
                     "change here affects only that judgement.",
                     className="field-help", style={"marginBottom": "12px"}),
            html.Div([
                html.Details([
                    html.Summary(LEG_LABELS[leg]),
                    dcc.Textarea(id=f"prompt-{leg}", className="prompt-edit",
                                 value="", rows=9),
                    html.Details([
                        html.Summary("User prompt (fixed — carries the JSON contract)"),
                        html.Div(id=f"userprompt-{leg}", className="prompt-box"),
                    ], style={"marginTop": "8px"}),
                ], className="disclose", open=(leg == LEGS[0]))
                for leg in LEGS
            ]),
            html.Div([
                html.Button("Save", id="prompt-save", n_clicks=0,
                            className="btn btn-ghost"),
                html.Button("Reset to defaults", id="prompt-reset", n_clicks=0,
                            className="btn btn-ghost"),
            ], className="btn-row", style={"marginTop": "6px"}),
            html.Button("Re-run mode 3 only", id="rerun3", n_clicks=0,
                        className="btn btn-ghost"),
            html.Div(id="prompt-status", className="field-help"),
        ], className="card"),

        html.Div([
            html.Div("Photos", className="card-title"),
            html.Div([
                html.Label("Folder on this machine", htmlFor="folder"),
                dcc.Input(id="folder", type="text", debounce=True,
                          placeholder="/data/adnaan/fieldops/demo/photos"),
            ], className="field"),
            html.Div("or", className="or-rule"),
            # Upload is fully supported here, unlike the annotation version:
            # that one needed a .txt sidecar beside each photo and a browser
            # upload cannot carry one. A trained detector reads the image, so
            # dragging a photo straight off a laptop works end to end.
            dcc.Upload(id="uploads", multiple=True, className="dropzone",
                       accept="image/*", children=html.Div([
                           html.Div("Drop images here, or click to browse",
                                    className="dz-main"),
                           html.Div("JPG, PNG, TIFF · the detector reads the image "
                                    "directly, so no annotation files are needed",
                                    className="dz-sub"),
                       ])),
            html.Div(id="upload-note", className="dz-list"),
        ], className="card"),

        html.Div([
            html.Div("Run", className="card-title"),
            html.Button("Run all three modes", id="run", n_clicks=0,
                        className="btn btn-primary"),
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
                    html.Label("Route to the model"),
                    dcc.RadioItems(
                        id="transport", className="opt",
                        options=[
                            {"label": "Direct to the GPU server",
                             "value": TRANSPORT_DIRECT},
                            {"label": "Via the LLM proxy on FALCONPRD",
                             "value": TRANSPORT_PROXY},
                        ], value=cfg.vlm_transport),
                    html.Div(id="transport-help", className="field-help"),
                ], className="field"),
                html.Div([
                    html.Label("Server URL", htmlFor="gpu-url"),
                    dcc.Input(id="gpu-url", type="text", value=cfg.gpu_url),
                ], className="field"),
                html.Div([
                    html.Label("API key (proxy only)", htmlFor="api-key"),
                    # type="password" so a shoulder-surfer at the demo does not
                    # read the key off the screen. It is still sent in a header,
                    # not a URL, so it stays out of logs and history.
                    dcc.Input(id="api-key", type="password",
                              value=os.environ.get("FIELDOPS_VLM_API_KEY", ""),
                              placeholder="X-API-Key — leave blank if the proxy "
                                          "runs with auth disabled"),
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
                html.P("The same photographs judged three ways: a hard quality "
                       "gate, a quality-or-detector gate, and the vision model "
                       "deciding everything by itself."),
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
                        dcc.Tab(label="Compare modes", value="compare",
                                className="tab", selected_className="tab--selected"),
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
           title="Field Ops Demo — Modes", update_title="Running…",
           assets_folder=str(APP_DIR / "assets"), suppress_callback_exceptions=True)
app.layout = layout()
server = app.server

# {mode_id: [PipelineRecord]} for the last run, so switching modes is instant.
_RESULTS: dict = {"modes": {}, "question_id": FIRST, "run_dir": None,
                  "paths": [], "cfg": None}

# Mode 3's editable system prompts, loaded from config/prompts.yaml at startup.
PROMPTS = default_store()


@app.callback(Output("semantics", "children"), Output("prompt-view", "children"),
              Input("question", "value"))
def on_question(question_id):
    return get_question(question_id).answer_semantics, prompt_text(question_id)


@app.callback([Output(f"prompt-{leg}", "value") for leg in LEGS]
              + [Output(f"userprompt-{leg}", "children") for leg in LEGS]
              + [Output("prompt-status", "children")],
              Input("question", "value"))
def on_question_prompts(question_id):
    """Load this question's prompts into the editors. Each question has its own
    set, so switching the dropdown switches the whole pair - the same principle
    the question registry was built on."""
    question = get_question(question_id)
    systems = [PROMPTS.get(question_id, leg) for leg in LEGS]
    users = [render_leg_user(question, leg) for leg in LEGS]
    overridden = [LEG_LABELS[leg].split("·")[-1].strip()
                  for leg in LEGS if PROMPTS.is_overridden(question_id, leg)]
    if PROMPTS.load_error:
        status = f"Using built-in prompts — {PROMPTS.load_error}"
    elif overridden:
        status = f"Edited: {', '.join(overridden)} (saved to config/prompts.yaml)"
    else:
        status = "All three legs are using the built-in prompts."
    return systems + users + [status]


@app.callback(Output("mode-blurb", "children"), Input("mode", "value"))
def on_mode(mode):
    return MODES[mode]["blurb"]


@app.callback(Output("prompt-status", "children", allow_duplicate=True),
              Input("prompt-save", "n_clicks"), Input("prompt-reset", "n_clicks"),
              State("question", "value"),
              *[State(f"prompt-{leg}", "value") for leg in LEGS],
              prevent_initial_call=True)
def on_prompt_buttons(save_clicks, reset_clicks, question_id, *values):
    import dash
    which = (dash.callback_context.triggered[0]["prop_id"].split(".")[0]
             if dash.callback_context.triggered else "")
    if which == "prompt-reset":
        PROMPTS.reset(question_id)
        error = PROMPTS.save()
        return (f"Could not save: {error}" if error else
                "Reset to the built-in prompts. Reselect the question to reload "
                "the boxes.")
    for leg, value in zip(LEGS, values):
        PROMPTS.set(question_id, leg, value or "")
    error = PROMPTS.save()
    if error:
        return f"Could not save: {error}"
    changed = [LEG_LABELS[leg].split("·")[-1].strip()
               for leg in LEGS if PROMPTS.is_overridden(question_id, leg)]
    if not changed:
        return ("Saved — all three legs match the built-in prompts, so nothing "
                "is overridden.")
    return (f"Saved to config/prompts.yaml. Edited: {', '.join(changed)}. "
            f"Press \u201cRe-run mode 3 only\u201d to see the effect.")


@app.callback(Output("upload-note", "children"), Input("uploads", "filename"))
def on_upload(filenames):
    if not filenames:
        return ""
    shown = ", ".join(filenames[:4]) + (f" +{len(filenames) - 4} more"
                                        if len(filenames) > 4 else "")
    return [html.Span(f"{len(filenames)} file(s) ready", className="dz-count"),
            html.Span(f" — {shown}")]


@app.callback(Output("transport-help", "children"), Input("transport", "value"))
def on_transport(transport):
    if transport == TRANSPORT_PROXY:
        return ("POST /v1/infer with an X-API-Key header. Use this where the box "
                "cannot see the GPU server directly.")
    return "POST /infer, no auth. Use this where the GPU server is reachable."


@app.callback(Output("status", "children"), Input("check", "n_clicks"),
              State("ckpt", "value"), State("classes", "value"), State("conf", "value"),
              State("nms", "value"), State("size-h", "value"), State("size-w", "value"),
              State("flags", "value"), State("gpu-url", "value"),
              State("transport", "value"), State("api-key", "value"),
              prevent_initial_call=True)
def on_check(_clicks, ckpt, classes, conf, nms, size_h, size_w, flags, gpu_url,
             transport, api_key):
    cfg = _cfg_from_controls(ckpt, classes, conf, nms, size_h, size_w, flags,
                             gpu_url, "qwen3-vl", VLM_MODE_LIVE, SEND_FULL, 65, flags,
                             transport=transport, api_key=api_key)
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
        # The error already distinguishes "proxy down" from "proxy up, GPU
        # unreachable" - show it rather than flattening both into one line.
        parts += [html.Span(className="dot dot-bad"),
                  html.Span(f"No model via {client.via} — answers will be MOCK. "
                            f"{error}")]
    else:
        client.refresh_registry()
        loaded = ", ".join(health.get("loaded_models") or []) or "none"
        note = ""
        if client.registry_is_allowlist:
            note = (" · registry came from the proxy's allowlist, not the GPU "
                    "server's own — names and image caps only")
        parts += [html.Span(className="dot dot-ok"),
                  html.Span(f"Model reachable via {client.via} · loaded: "
                            f"{loaded}{note}")]
        if client.registry_error:
            parts += [html.Span("|", className="status-sep"),
                      html.Span(className="dot dot-warn"),
                      html.Span(client.registry_error)]
    return parts


@app.callback(Output("panel", "children"),
              Output("status", "children", allow_duplicate=True),
              Input("run", "n_clicks"), Input("tabs", "value"), Input("mode", "value"),
              State("question", "value"), State("folder", "value"),
              State("uploads", "contents"), State("uploads", "filename"),
              State("ckpt", "value"), State("classes", "value"), State("conf", "value"),
              State("nms", "value"), State("size-h", "value"), State("size-w", "value"),
              State("threshold", "value"), State("flags", "value"),
              State("gpu-url", "value"), State("vlm-model", "value"),
              State("vlm-mode", "value"), State("send-mode", "value"),
              State("transport", "value"), State("api-key", "value"),
              prevent_initial_call="initial_duplicate")
def on_run(n_clicks, tab, mode, question_id, folder, upload_contents, upload_names,
           ckpt, classes, conf, nms, size_h, size_w, threshold, flags, gpu_url,
           vlm_model, vlm_mode, send_mode, transport, api_key):
    import dash
    triggered = (dash.callback_context.triggered[0]["prop_id"].split(".")[0]
                 if dash.callback_context.triggered else "")
    question = get_question(question_id or FIRST)

    # A tab switch OR a mode switch just re-renders what is already in memory.
    if triggered != "run":
        return _panel(tab, mode, question), no_update


    cfg = _cfg_from_controls(ckpt, classes, conf, nms, size_h, size_w, flags,
                             gpu_url, vlm_model, vlm_mode, send_mode, threshold, flags,
                             transport=transport, api_key=api_key)
    problems = cfg.validate()
    blocking = [p for p in problems if "yolox_checkpoint" in p]
    if blocking:
        return (html.Div([html.Div(p, className="banner banner-stop") for p in blocking],
                         className="card"),
                [html.Span(className="dot dot-bad"), html.Span(blocking[0])])

    paths, source = _resolve_inputs(folder, upload_contents, upload_names, cfg)
    if not paths:
        return (html.Div(source, className="empty"),
                [html.Span(className="dot dot-warn"), html.Span(source)])

    try:
        by_mode = run_all_modes(paths, question.id, cfg, prompts=PROMPTS)
    except Exception as exc:
        return (html.Div([html.Div(f"Run failed: {exc}", className="banner banner-stop"),
                          html.Pre(traceback.format_exc()[-2400:])], className="card"),
                [html.Span(className="dot dot-bad"), html.Span(f"Run failed: {exc}")])

    _RESULTS.update(modes=by_mode, question_id=question.id, run_dir=cfg.run_dir,
                    paths=paths, cfg=cfg)
    agree = agreement(by_mode)
    parts = []
    for mode_id in MODE_ORDER:
        counts = summarise(by_mode[mode_id])["answers"]
        parts.append(MODES[mode_id]["short"] + ": "
                     + (", ".join(f"{k.upper()} {n}" for k, n in sorted(counts.items()))
                        or "none"))
    status = [html.Span(className="dot dot-ok"),
              html.Span(f"Done — {agree['total']} photo(s) from {source}, all three "
                        f"modes · {agree['unanimous']} unanimous, {agree['split']} "
                        f"split · " + "  |  ".join(parts))]
    return _panel(tab, mode, question), status


@app.callback(Output("panel", "children", allow_duplicate=True),
              Output("status", "children", allow_duplicate=True),
              Input("rerun3", "n_clicks"), State("tabs", "value"),
              State("mode", "value"), prevent_initial_call=True)
def on_rerun3(_clicks, tab, mode):
    """Re-run mode 3 over the same photos with the prompts as they stand now.

    Modes 1 and 2 keep their existing records, so the comparison tab still lines
    up row for row - only the leg the prompt actually governs is recomputed.
    """
    paths, cfg = _RESULTS.get("paths"), _RESULTS.get("cfg")
    question = get_question(_RESULTS["question_id"])
    if not paths or cfg is None:
        return no_update, [html.Span(className="dot dot-warn"),
                           html.Span("Nothing to re-run yet — run the pipeline first.")]

    # Deliberately the same run id: mode 3's subfolder is rewritten in place so
    # the run folder keeps describing all three modes as they currently stand,
    # and the CSV/JSON downloads for modes 1 and 2 keep resolving.
    try:
        records = run_vlm_only(paths, question.id, cfg, prompts=PROMPTS)
    except Exception as exc:
        return (html.Div([html.Div(f"Re-run failed: {exc}", className="banner banner-stop"),
                          html.Pre(traceback.format_exc()[-2400:])], className="card"),
                [html.Span(className="dot dot-bad"), html.Span(f"Re-run failed: {exc}")])

    _RESULTS["modes"][MODE_VLM_ONLY] = records
    counts = summarise(records)["answers"]
    tally = ", ".join(f"{k.upper()} {n}" for k, n in sorted(counts.items())) or "none"
    return (_panel(tab, mode, question),
            [html.Span(className="dot dot-ok"),
             html.Span(f"Mode 3 re-run on {len(records)} photo(s) with the current "
                       f"prompts — {tally}. Modes 1 and 2 are unchanged.")])


def _resolve_inputs(folder, upload_contents, upload_names, cfg):
    """Folder path wins when given; uploads otherwise. Returns (paths, source)
    or ([], message) when there is nothing to run."""
    if folder and folder.strip():
        root = Path(folder.strip())
        if not root.exists():
            return [], f"No such folder: {root}"
        paths = collect_images(root)
        if not paths:
            return [], f"No images found under {root}."
        return paths, str(root)

    if upload_contents:
        # Uploads arrive base64 in the callback. Stage them to disk so every
        # stage sees a real path, exactly as a folder run would - the quality
        # leg re-reads the file and the run folder keeps a copy of what was
        # actually processed.
        if cfg.run_id is None:
            cfg.run_id = new_run_id()
        staged = cfg.runs_dir / "uploads" / cfg.run_id
        staged.mkdir(parents=True, exist_ok=True)
        paths, skipped = [], 0
        for content, name in zip(upload_contents, upload_names or []):
            try:
                _, b64 = content.split(",", 1)
                dest = staged / Path(name).name
                dest.write_bytes(base64.b64decode(b64))
                paths.append(dest)
            except Exception:
                skipped += 1
        if not paths:
            return [], "None of the uploaded files could be decoded."
        note = f"{len(paths)} uploaded file(s)"
        return paths, note + (f" ({skipped} skipped)" if skipped else "")

    return [], "Drop images above, or point at a photo folder on this machine."


def quality_column(record):
    """The quality panel, with mode 3's case.

    Modes 1 and 2 use the frozen renderer unchanged. Mode 3 has no MM-IQA
    score, so printing the numeric fields would show "whole frame 0.0" - a
    number the model never produced. It gets the photograph, the verdict and
    the model's own reasoning instead.
    """
    q = record.quality
    if q.assessed_by != "vlm":
        return classical_quality_column(record)

    pill = "pill-err" if q.error else "pill-pass" if q.passed else "pill-fail"
    children = [
        html.H4("1 · Quality gate"),
        # The plain, EXIF-corrected photograph: mode 3 never runs u2netp, so
        # there is no foreground box to draw and claiming one would be a lie.
        image_or_placeholder(q.annotated_path, "Image could not be rendered"),
        html.Div(html.Span(q.headline, className=f"pill {pill}"), className="rc-line"),
    ]
    if q.width and q.height:
        children.append(html.Div(f"{q.width}×{q.height} · judged by the model, "
                                 "no MM-IQA score and no u2netp crop",
                                 className="rc-muted"))
    if q.failure_reasons:
        children.append(html.Div("Model's reasoning: " + " ".join(q.failure_reasons),
                                 className="rc-line"))
    elif record.extra.get("quality_reasoning"):
        children.append(html.Div("Model's reasoning: "
                                 + record.extra["quality_reasoning"],
                                 className="rc-line"))
    return html.Div(children, className="rc-col")


def detection_column(record):
    """The detection panel, with mode 3's presence case.

    Modes 1 and 2 fill this from the trained detector and it shows boxes. Mode 3
    has no detector: the model reports whether the subject is visible, in words.
    Rendering an empty box list there would read as "the detector found
    nothing", which is a different claim entirely.
    """
    d = record.detection
    if d is None:
        return html.Div([html.H4("2 · Detection"),
                         html.Div("Not run — the quality gate stopped this image.",
                                  className="rc-none")], className="rc-col")

    if d.presence is not None:
        chip = {"yes": "chip-yes", "no": "chip-no"}.get(d.presence, "chip-unknown")
        children = [
            html.H4("2 · Subject visible?"),
            html.Div(html.Span(d.presence.upper(), className=f"chip {chip}")),
            html.Div(d.presence_reasoning or "", className="rc-reason"),
            html.Div(html.Span(d.model_name, className="pill pill-stub"),
                     className="rc-line"),
            html.Div("Judged by the model. No boxes are drawn in this mode — it "
                     "reports presence, not geometry.", className="rc-muted"),
        ]
        return html.Div(children, className="rc-col")

    children = [
        html.H4("2 · Detection"),
        image_or_placeholder(d.annotated_path, "No boxes to draw"),
        html.Div(html.Span(d.model_name, className="pill pill-stub"),
                 className="rc-line"),
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


def result_card(record, question):
    children = [html.Div([html.Span(record.filename, className="rc-name"),
                          html.Span(record.stem, className="rc-sub")],
                         className="rc-head")]
    if record.stopped_at == STOPPED_QUALITY:
        children.append(html.Div(
            "Stopped at the quality gate — the downstream legs did not run.",
            className="banner banner-stop"))
    gate = record.extra.get("gate")
    if gate:
        # Mode 2 only: say WHY the OR let this photograph through, rather than
        # leaving the operator to work it out from two panels.
        children.append(html.Div(f"Gate: {gate}", className="banner banner-info"))
    children.append(html.Div([quality_column(record), detection_column(record),
                              vlm_column(record, question)], className="rc-cols"))
    return html.Div(children, className="rc")


def results_view(records, question):
    if not records:
        return html.Div("No results yet. Choose a question, point at photographs, "
                        "and run all three modes.", className="empty")
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


def _mode_banner(mode_id):
    return html.Div([html.B(MODES[mode_id]["label"]), " — ", MODES[mode_id]["blurb"]],
                    className="banner banner-info")


def _presence_note(records):
    """Mode 3 reports subject presence as text rather than boxes. Say so once,
    above the cards, so nobody reads an empty detection panel as 'found
    nothing'."""
    counts = {}
    for record in records:
        if record.detection is not None and record.detection.presence:
            counts[record.detection.presence] = counts.get(
                record.detection.presence, 0) + 1
    if not counts:
        return None
    summary = ", ".join(f"{k.upper()} {n}" for k, n in sorted(counts.items()))
    return html.Div(
        f"Subject visible, as judged by the model: {summary}. This mode draws no "
        f"boxes — it reports presence, not geometry.", className="banner banner-warn")


def _comparison(question):
    """One row per photograph, one column per mode. This is what the three modes
    exist to show."""
    by_mode = _RESULTS["modes"]
    rows = compare_rows(by_mode)
    if not rows:
        return html.Div("Run the pipeline to compare the modes.", className="empty")
    agree = agreement(by_mode)

    data = [{"file": r["file"],
             **{MODES[m]["short"]: r.get(m, "—") for m in MODE_ORDER}}
            for r in rows]
    columns = [{"name": "file", "id": "file"}] + [
        {"name": MODES[m]["short"], "id": MODES[m]["short"]} for m in MODE_ORDER]

    tiles = [tile(agree["total"], "photos"),
             tile(agree["unanimous"], "all three agree", "tile-yes"),
             tile(agree["split"], "modes disagree",
                  "tile-no" if agree["split"] else "")]

    conditional = []
    for m in MODE_ORDER:
        col = MODES[m]["short"]
        conditional += [
            {"if": {"filter_query": f'{{{col}}} = "YES"', "column_id": col},
             "color": "#07584A", "fontWeight": "700"},
            {"if": {"filter_query": f'{{{col}}} = "NO"', "column_id": col},
             "color": "#A32118", "fontWeight": "700"},
            {"if": {"filter_query": f'{{{col}}} contains "stopped"', "column_id": col},
             "color": "#6B685F"},
        ]

    return html.Div([
        html.Div(tiles, className="tiles"),
        html.Div(question.answer_semantics, className="banner banner-info"),
        dash_table.DataTable(
            data=data, columns=columns, page_size=30, style_as_list_view=True,
            style_cell={"textAlign": "left", "padding": "11px 12px",
                        "whiteSpace": "normal", "height": "auto"},
            style_header={"fontWeight": "500"},
            style_data_conditional=conditional),
        html.Div("A row where the three disagree is the interesting one: same "
                 "photograph, three ways of deciding, different conclusions.",
                 className="field-help", style={"marginTop": "14px"}),
    ])


def _panel(tab, mode, question):
    records = _RESULTS["modes"].get(mode, [])
    run_dir = _RESULTS["run_dir"]

    if tab == "compare":
        return _comparison(question)
    if tab == "summary":
        if not records:
            return html.Div("Run the pipeline to see a summary.", className="empty")
        return html.Div([_mode_banner(mode),
                         summary_view(records, question, run_dir)])
    if tab == "export":
        if not records:
            return html.Div("Run the pipeline to produce a CSV and JSON.",
                            className="empty")
        return html.Div([
            html.Div("Export", className="card-title"),
            html.Div(f"Each mode writes its own pipeline_results.csv and "
                     f"results.json, under demo_runs/<run>/<mode>/. These buttons "
                     f"serve the currently selected mode "
                     f"({MODES[mode]['short']}).",
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

    # Results
    if not records:
        return html.Div([_mode_banner(mode),
                         results_view([], question)])
    blocks = [_mode_banner(mode)]
    if mode == MODE_VLM_ONLY:
        note = _presence_note(records)
        if note is not None:
            blocks.append(note)
    else:
        counts = {}
        for record in records:
            for det in (record.detection.detections if record.detection else []):
                counts[det.label] = counts.get(det.label, 0) + 1
        if counts:
            blocks.append(html.Div([tile(n, f"{label} boxes")
                                    for label, n in sorted(counts.items())],
                                   className="tiles"))
    blocks.append(results_view(records, question))
    return html.Div(blocks)


@app.callback(Output("download", "data"),
              Input("dl-csv", "n_clicks"), Input("dl-json", "n_clicks"),
              State("mode", "value"), prevent_initial_call=True)
def on_download(csv_clicks, json_clicks, mode):
    import dash
    if not dash.callback_context.triggered or not _RESULTS["run_dir"]:
        return no_update
    which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    name = "pipeline_results.csv" if which == "dl-csv" else "results.json"
    # Each mode writes into its own subfolder, so serve the selected one.
    path = Path(_RESULTS["run_dir"]) / mode / name
    return dcc.send_file(str(path)) if path.exists() else no_update


def main() -> None:
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASH_PORT", "7872")))


if __name__ == "__main__":
    main()
