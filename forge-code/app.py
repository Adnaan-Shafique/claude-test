"""
Forge Code — Dash UI.

A coding agent for Python and SQL: upload a .py or .sql file and ask questions about it,
request edit suggestions, or ask for code from scratch. Backed by Mistral.

All database, auth and LLM logic lives in backend.py. This file is UI and callbacks only.

Run:
    python app.py                      # start the server
    python app.py create-admin         # create the first administrator account

See README.md for the environment variables that must be set first.
"""

from __future__ import annotations

import getpass
import os
import sys
from datetime import datetime, timedelta

from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import g, jsonify
from flask import request as flask_request

import backend
from backend import AppError

# ============================================================
# THEME
# ============================================================

CONTENT_WIDTH = "760px"

COLORS = {
    "bg": "#FFFFFF",
    "bg_secondary": "#FAFAFA",
    "surface": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#111827",
    "text_dim": "#667085",
    "red": "#E8442C",
    "red_light": "#FF6B52",
    "red_soft": "#FFF1F2",
    "green": "#1FA971",
    "amber": "#B54708",
}

PORT = int(os.environ.get("PORT", "8054"))
DEBUG = os.environ.get("FORGE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

app = Dash(__name__, suppress_callback_exceptions=True, update_title=None)
app.title = "Forge Code"
server = app.server

# ============================================================
# SESSION — httpOnly cookie, verified server-side on every callback
# ============================================================

# Dash callbacks run inside a normal Flask request, so the session cookie is readable
# here and the user id never has to be trusted from the browser. Cookies are set by
# stashing the intent on flask.g during a callback and attaching it in after_request,
# since a callback's own return value cannot carry headers.


def current_user() -> dict | None:
    """The authenticated user for the in-flight request, or None."""
    token = flask_request.cookies.get(backend.COOKIE_NAME)
    if not token:
        return None
    try:
        return backend.resolve_user(token)
    except AppError:
        return None


def require_user() -> dict:
    user = current_user()
    if user is None:
        raise AppError("Your session has expired. Please log in again.")
    return user


def require_admin() -> dict:
    user = require_user()
    if not user["is_admin"]:
        raise AppError("Administrator privileges are required.")
    return user


def client_ip() -> str:
    """Caller's address, honouring one layer of reverse proxy."""
    forwarded = flask_request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return flask_request.remote_addr or "unknown"


@server.after_request
def _apply_session_cookie(response):
    token = getattr(g, "forge_set_token", None)
    if token:
        response.set_cookie(
            backend.COOKIE_NAME,
            token,
            max_age=backend.TOKEN_LIFETIME_HOURS * 3600,
            httponly=True,
            secure=backend.COOKIE_SECURE,
            samesite="Strict",
            path="/",
        )
    if getattr(g, "forge_clear_token", False):
        response.delete_cookie(backend.COOKIE_NAME, path="/", samesite="Strict")
    return response


# ============================================================
# PAGE SHELL — global CSS and the small vanilla-JS helpers
# ============================================================

# DM Sans is self-hosted from assets/fonts/ (one variable font covering weight 100-900)
# so the app has zero external dependencies at runtime — required for an air-gapped VM.
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        @font-face {
            font-family: 'DM Sans';
            src: url('/assets/fonts/DMSans-variable.ttf') format('truetype-variations');
            font-weight: 100 900;
            font-style: normal;
            font-display: swap;
        }
        * { box-sizing: border-box; }
        html, body {
            margin: 0; padding: 0;
            background-color: #FAFAFA;
            background-image: radial-gradient(circle, #ECECEC 1px, transparent 1px);
            background-size: 22px 22px;
            font-family: 'DM Sans', sans-serif;
            font-weight: 400;
            color: #111827;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 4px; }

        .vi-shell {
            display: flex; margin: 20px; height: calc(100vh - 40px);
            background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 20px;
            overflow: hidden; box-shadow: 0 2px 24px rgba(17, 24, 39, 0.05);
        }

        .vi-btn {
            background: #E8442C; color: #ffffff; border: none; border-radius: 10px;
            font-family: inherit; font-weight: 400; cursor: pointer;
            transition: background 0.15s ease, transform 0.1s ease;
        }
        .vi-btn:hover { background: #D13B24; }
        .vi-btn:active { transform: scale(0.97); }
        .vi-btn:disabled { background: #F3A99A; cursor: not-allowed; }

        .vi-btn-outline {
            background: #FFFFFF; color: #E8442C; border: 1px solid #E8442C; border-radius: 8px;
            font-family: inherit; font-weight: 400; cursor: pointer;
            transition: background 0.15s ease;
        }
        .vi-btn-outline:hover { background: #FFF1F2; }

        .vi-btn-secondary {
            background: #FFFFFF; color: #111827; border: 1px solid #E5E7EB; border-radius: 8px;
            font-family: inherit; font-weight: 400; cursor: pointer;
            transition: background 0.15s ease;
        }
        .vi-btn-secondary:hover { background: #FAFAFA; }

        .vi-icon-btn {
            background: transparent; border: none; cursor: pointer; font-size: 15px; color: #667085;
            transition: color 0.15s ease;
        }
        .vi-icon-btn:hover { color: #E8442C; }

        .vi-input {
            background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; color: #111827;
            font-family: inherit; font-weight: 400; outline: none;
            transition: border-color 0.15s ease;
        }
        .vi-input:focus-within, .vi-input:focus { border-color: #E8442C; }
        .vi-input::placeholder { color: #98A2B3; }

        .vi-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; }
        .vi-link {
            color: #E8442C; cursor: pointer; text-decoration: none; font-size: 13px;
        }
        .vi-link:hover { text-decoration: underline; }

        /* Sidebar */
        .vi-sidebar {
            width: 300px; min-width: 300px; height: 100%;
            background: #FAFAFA; border-right: 1px solid #E5E7EB;
            display: flex; flex-direction: column; padding: 20px 14px; overflow: hidden;
        }
        .vi-recent-list { flex: 1; overflow-y: auto; margin: 0 -6px; padding: 0 6px; }
        .vi-sidebar-item {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 12px; margin-bottom: 2px; border-radius: 10px;
            color: #111827; cursor: pointer; font-size: 14px;
            transition: background 0.15s ease;
        }
        .vi-sidebar-item:hover { background: #F1F1F1; }
        .vi-sidebar-item.active { background: #FFF1F2; color: #E8442C; }
        .vi-sidebar-row-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .vi-sidebar-row-time { font-size: 12px; color: #98A2B3; flex-shrink: 0; }
        .vi-sidebar-item.active .vi-sidebar-row-time { color: #E8442C; opacity: 0.7; }

        /* Empty state */
        .vi-empty-icon {
            width: 64px; height: 64px; border-radius: 16px; background: #FFF1F2;
            display: flex; align-items: center; justify-content: center;
            font-size: 26px; color: #E8442C; font-family: inherit; font-weight: 400;
            margin: 0 auto 20px;
        }
        .vi-info-card {
            background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
            padding: 18px; flex: 1; min-width: 200px;
        }
        .vi-info-card-icon {
            width: 36px; height: 36px; border-radius: 10px; background: #FFF1F2;
            display: flex; align-items: center; justify-content: center; font-size: 16px;
            margin-bottom: 12px;
        }

        /* Composer */
        .vi-composer {
            border: 1px solid #E5E7EB; border-radius: 16px; background: #FFFFFF;
            padding: 12px 14px; transition: border-color 0.15s ease;
        }
        .vi-composer:focus-within { border-color: #E8442C; }
        .vi-composer textarea {
            width: 100%; border: none; outline: none; resize: none; background: transparent;
            font-family: inherit; font-size: 14px; color: #111827; line-height: 1.5;
            min-height: 24px; max-height: 160px;
        }
        .vi-composer textarea::placeholder { color: #98A2B3; }
        .vi-attach-btn {
            display: inline-flex; align-items: center; gap: 6px;
            background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
            padding: 6px 12px; font-size: 13px; color: #344054; cursor: pointer;
            transition: background 0.15s ease;
        }
        .vi-attach-btn:hover { background: #FAFAFA; }
        .vi-scope-hint { font-size: 12px; color: #98A2B3; }
        .vi-send-btn {
            width: 34px; height: 34px; border-radius: 50%; padding: 0; font-size: 15px;
            display: flex; align-items: center; justify-content: center;
        }

        /* Chat bubbles */
        .vi-msg-user {
            background: #F3F4F6; color: #111827; padding: 10px 16px; border-radius: 16px;
            max-width: 70%; white-space: pre-wrap;
        }
        .vi-file-badge {
            display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #667085;
            background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 999px; padding: 3px 10px;
            margin-top: 6px;
        }

        /* Code block */
        .vi-code-wrap { position: relative; border-radius: 12px; overflow: hidden; margin-bottom: 12px; }
        .vi-code-header {
            display: flex; justify-content: space-between; align-items: center;
            background: #1F2430; padding: 8px 14px; font-size: 12px; color: #9CA3AF;
            font-family: ui-monospace, monospace;
        }
        .vi-code-block {
            background: #16181F; color: #E7E9EE; margin: 0; padding: 14px;
            overflow-x: auto; font-family: ui-monospace, SFMono-Regular, 'SF Mono', Consolas, monospace;
            font-size: 13px; line-height: 1.55; white-space: pre;
        }
        .vi-copy-btn {
            background: transparent; border: none; color: #9CA3AF; cursor: pointer; font-size: 12px;
            transition: color 0.15s ease;
        }
        .vi-copy-btn:hover { color: #FFFFFF; }
        .vi-truncated-note {
            background: #FFFBEB; color: #B54708; font-size: 12px; padding: 6px 14px;
        }
        .vi-instructions-heading {
            font-size: 12px; font-weight: 400; color: #667085; text-transform: uppercase;
            letter-spacing: 0.04em; margin: 14px 0 6px;
        }

        /* Admin table */
        .vi-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .vi-table th {
            text-align: left; padding: 10px 12px; font-weight: 400; color: #667085;
            border-bottom: 1px solid #E5E7EB; font-size: 12px; text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .vi-table td { padding: 12px; border-bottom: 1px solid #F3F4F6; vertical-align: middle; }
        .vi-pill {
            display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
        }
        .vi-pill-active { background: #ECFDF3; color: #1FA971; }
        .vi-pill-pending { background: #FFFBEB; color: #B54708; }
        .vi-pill-admin { background: #FFF1F2; color: #E8442C; }

        /* Loading indicator */
        .vi-spinner {
            width: 15px; height: 15px; border-radius: 50%;
            border: 2px solid #E5E7EB; border-top-color: #E8442C;
            animation: vi-spin 0.7s linear infinite;
        }
        @keyframes vi-spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <script>
        // Enter (without Shift) submits; Shift+Enter inserts a newline. Delegated from
        // document so it survives Dash re-rendering the textarea.
        document.addEventListener('keydown', function (e) {
            if (e.target && e.target.id === 'chat-input' && e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const btn = document.getElementById('send-button');
                if (btn && !btn.disabled) { btn.click(); }
            }
            // Enter submits the login and register forms too.
            if (e.target && e.key === 'Enter') {
                const formBtn = {
                    'login-username': 'login-button', 'login-password': 'login-button',
                    'register-username': 'register-button', 'register-email': 'register-button',
                    'register-password': 'register-button', 'register-password2': 'register-button',
                }[e.target.id];
                if (formBtn) {
                    const btn = document.getElementById(formBtn);
                    if (btn) { btn.click(); }
                }
            }
        });

        // File attach: read the picked file with plain FileReader and push it straight into
        // the attached-file-store, bypassing dcc.Upload/react-dropzone's own base64-and-setProps
        // pipeline — that internal handling is unreliable in some privacy-hardened browsers
        // (confirmed: no request at all fires there on file selection). dcc.Upload still renders
        // the real native <input type=file> we hook into here.
        //
        // Both checks below are mirrored server-side in backend.validate_upload(), since this
        // store is client-side and can be set to anything.
        document.addEventListener('change', function (e) {
            if (!e.target || e.target.type !== 'file' || !e.target.closest('#file-upload') || !e.target.files || !e.target.files[0]) { return; }
            const file = e.target.files[0];
            const lower = file.name.toLowerCase();
            // The cap lives in one place (backend.MAX_UPLOAD_BYTES) and is rendered into
            // a hidden node by the chat layout so the two checks cannot drift apart.
            const capNode = document.getElementById('max-upload-bytes');
            const maxBytes = parseInt((capNode && capNode.textContent) || '200000', 10);
            const reject = function (msg) {
                dash_clientside.set_props('upload-error', {children: msg});
                e.target.value = '';
            };
            if (!lower.endsWith('.py') && !lower.endsWith('.sql')) {
                return reject('Only .py and .sql files are supported.');
            }
            if (file.size > maxBytes) {
                return reject('That file is ' + Math.round(file.size / 1024) + ' KB, over the ' + Math.round(maxBytes / 1024) + ' KB limit.');
            }
            const reader = new FileReader();
            reader.onload = function () {
                dash_clientside.set_props('upload-error', {children: ''});
                dash_clientside.set_props('attached-file-store', {data: {name: file.name, content: reader.result}});
                e.target.value = '';
            };
            reader.readAsText(file);
        });
    </script>
</body>
</html>
"""


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Bumping this store is how a callback asks the browser to reload after the
        # session cookie has been set or cleared.
        dcc.Store(id="reload-signal", data=0),
        html.Div(id="page-content"),
    ]
)

THINKING_PHRASES = ["Thinking...", "Still thinking...", "Pondering...", "Coding...", "Working on it...", "Almost there..."]


# --- Clientside helpers --------------------------------------------------

app.clientside_callback(
    """
    function(signal) {
        if (signal) { window.location.replace('/'); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url", "search"),
    Input("reload-signal", "data"),
)

# Cycle the "thinking" text purely client-side so it keeps animating while the server
# is blocked on the LLM call.
app.clientside_callback(
    f"""
    function(n_intervals) {{
        const phrases = {THINKING_PHRASES!r};
        return phrases[n_intervals % phrases.length];
    }}
    """,
    Output("thinking-text", "children"),
    Input("thinking-interval", "n_intervals"),
)

# Auto-scroll so the newest question jumps to the top of the viewport, leaving room
# below for the answer to appear.
app.clientside_callback(
    """
    function(chatHistoryData) {
        const container = document.getElementById('chat-window');
        if (container && container.lastElementChild) {
            container.lastElementChild.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-window", "title"),
    Input("chat-history", "data"),
)

# Copy-to-clipboard with temporary "Copied" feedback, self-contained per code block.
app.clientside_callback(
    """
    function(n_clicks, id, codeText) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        navigator.clipboard.writeText(codeText || "");
        const selfId = JSON.stringify({index: id.index, type: id.type});
        setTimeout(function () {
            const el = document.getElementById(selfId);
            if (el) { el.textContent = "Copy"; }
        }, 1400);
        return "Copied";
    }
    """,
    Output({"type": "copy-btn", "index": MATCH}, "children"),
    Input({"type": "copy-btn", "index": MATCH}, "n_clicks"),
    State({"type": "copy-btn", "index": MATCH}, "id"),
    State({"type": "code-block", "index": MATCH}, "children"),
)


# ============================================================
# LAYOUT BUILDERS
# ============================================================

def icon(name, size=16, extra_style=None):
    """Render one of the assets/*.svg icons."""
    style = {"width": f"{size}px", "height": f"{size}px", "display": "block"}
    if extra_style:
        style.update(extra_style)
    return html.Img(src=f"/assets/{name}.svg", style=style)


def brand(size=26):
    return html.Div(
        [
            html.Span("Forge", style={"fontWeight": 100, "fontSize": f"{size}px", "color": COLORS["text"]}),
            html.Span(" Code", style={"fontWeight": 400, "fontSize": f"{size}px", "color": COLORS["red"]}),
        ]
    )


def _auth_card(children):
    return html.Div(
        style={
            "display": "flex", "flexDirection": "column", "alignItems": "center",
            "justifyContent": "center", "minHeight": "100vh", "padding": "20px",
        },
        children=html.Div(
            className="vi-card",
            style={"padding": "40px", "width": "340px", "textAlign": "center", "boxShadow": "0 2px 24px rgba(17,24,39,0.06)"},
            children=children,
        ),
    )


def _auth_input(input_id, placeholder, input_type="text"):
    return dcc.Input(
        id=input_id, type=input_type, placeholder=placeholder, className="vi-input",
        debounce=False, autoComplete="off",
        style={"marginBottom": "12px", "padding": "10px 12px", "width": "100%"},
    )


def login_layout():
    return _auth_card(
        [
            html.Div(brand(), style={"marginBottom": "22px"}),
            _auth_input("login-username", "Username"),
            _auth_input("login-password", "Password", "password"),
            html.Button("Log In", id="login-button", n_clicks=0, className="vi-btn", style={"padding": "10px 20px", "width": "100%", "marginTop": "6px"}),
            html.Div(id="login-error", style={"color": COLORS["red_light"], "marginTop": "12px", "fontSize": "13px", "lineHeight": "1.5"}),
            html.Div(
                style={"marginTop": "18px", "fontSize": "13px", "color": COLORS["text_dim"]},
                children=["New here? ", dcc.Link("Create an account", href="/register", className="vi-link")],
            ),
        ]
    )


def register_layout():
    return _auth_card(
        [
            html.Div(brand(), style={"marginBottom": "8px"}),
            html.Div(
                "Create an account. An administrator approves it before you can log in.",
                style={"fontSize": "13px", "color": COLORS["text_dim"], "marginBottom": "20px", "lineHeight": "1.5"},
            ),
            _auth_input("register-username", "Username"),
            _auth_input("register-email", "Email (optional)", "email"),
            _auth_input("register-password", f"Password ({backend.MIN_PASSWORD_LENGTH}+ characters)", "password"),
            _auth_input("register-password2", "Confirm password", "password"),
            html.Button("Register", id="register-button", n_clicks=0, className="vi-btn", style={"padding": "10px 20px", "width": "100%", "marginTop": "6px"}),
            html.Div(id="register-message", style={"marginTop": "12px", "fontSize": "13px", "lineHeight": "1.5"}),
            html.Div(
                style={"marginTop": "18px", "fontSize": "13px", "color": COLORS["text_dim"]},
                children=["Already approved? ", dcc.Link("Log in", href="/", className="vi-link")],
            ),
        ]
    )


def build_header(user, pending_count=0, show_sidebar_toggle=True):
    """Top bar: sidebar-expand toggle + brand on the left; admin link, user, log out right."""
    right_children = []
    if user["is_admin"]:
        label = "Admin" if not pending_count else f"Admin ({pending_count})"
        right_children.append(
            dcc.Link(label, href="/admin", className="vi-btn-secondary", style={"padding": "8px 14px", "fontSize": "13px", "textDecoration": "none"})
        )
    right_children.append(
        html.Span(user["username"], style={"fontSize": "13px", "color": COLORS["text_dim"]})
    )
    right_children.append(
        html.Button(
            ["Log out ", icon("log_out_icon", 14, {"display": "inline-block", "marginLeft": "6px", "verticalAlign": "text-bottom"})],
            id="logout-button", n_clicks=0, className="vi-btn-outline",
            style={"padding": "8px 16px", "fontSize": "13px", "display": "flex", "alignItems": "center"},
        )
    )

    return html.Div(
        style={
            "padding": "18px 28px", "display": "flex", "alignItems": "center", "justifyContent": "space-between",
            "borderBottom": f"1px solid {COLORS['border']}", "backgroundColor": COLORS["surface"], "minHeight": "68px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "14px"},
                children=[
                    html.Button(
                        icon("back_arrow", 16, {"transform": "scaleX(-1)"}), id="sidebar-expand-btn", n_clicks=0,
                        className="vi-icon-btn", style={"padding": "4px 8px", "display": "none"},
                    ) if show_sidebar_toggle else html.Div(),
                    brand(20),
                ],
            ),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=right_children),
        ],
    )


def fetch_conversations(user_id):
    try:
        return backend.list_conversations_for_user(user_id)
    except AppError:
        return []


def format_relative_time(created_at):
    """'10:30 AM' for today, 'Yesterday' for yesterday, else a short date."""
    if not created_at:
        return ""
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return ""
    dt = created_at
    today = datetime.now(dt.tzinfo).date() if dt.tzinfo else datetime.now().date()
    if dt.date() == today:
        return dt.strftime("%I:%M %p").lstrip("0")
    if dt.date() == today - timedelta(days=1):
        return "Yesterday"
    # strftime has no portable no-pad day, so strip the zero by hand.
    return f"{dt.strftime('%b')} {dt.day}"


def render_sidebar_items(conversations, active_id):
    items = [
        html.Div(
            style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "marginBottom": "10px"},
            children=[
                html.Button(icon("back_arrow", 16), id="sidebar-collapse-btn", n_clicks=0, className="vi-icon-btn", style={"padding": "4px 8px"}),
            ],
        ),
        html.Div(
            id="new-chat-button",
            n_clicks=0,
            className="vi-btn",
            style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "8px", "padding": "12px", "marginBottom": "18px", "fontSize": "14px"},
            children=[icon("new_chat_icon", 16), html.Span("New Chat")],
        ),
        html.Div("RECENT CONVERSATIONS", style={"padding": "6px 12px", "fontSize": "11px", "fontWeight": 400, "letterSpacing": "0.04em", "color": "#98A2B3"}),
    ]

    rows = []
    for conv in conversations or []:
        rows.append(
            html.Div(
                id={"type": "conv-item", "index": conv["id"]},
                n_clicks=0,
                className="vi-sidebar-item" + (" active" if conv["id"] == active_id else ""),
                children=[
                    html.Span(conv["title"], className="vi-sidebar-row-title"),
                    html.Span(format_relative_time(conv.get("created_at")), className="vi-sidebar-row-time"),
                ],
            )
        )
    if not rows:
        rows.append(
            html.Div(
                f"Nothing yet. Conversations are kept for {backend.CONVERSATION_RETENTION_DAYS} days.",
                style={"padding": "10px 12px", "fontSize": "13px", "color": "#98A2B3", "lineHeight": "1.5"},
            )
        )

    items.append(html.Div(rows, className="vi-recent-list"))
    return items


def render_feedback_controls(idx, feedback):
    if feedback == "up":
        return html.Div("Marked helpful", style={"color": COLORS["green"], "fontSize": "12px", "marginTop": "8px"})
    if feedback == "down":
        return html.Div("Marked unhelpful", style={"color": COLORS["red_light"], "fontSize": "12px", "marginTop": "8px"})
    return html.Div(
        style={"marginTop": "8px", "opacity": "0.7"},
        children=[
            html.Button("\U0001F44D", id={"type": "upvote-btn", "index": idx}, n_clicks=0, className="vi-icon-btn", style={"marginRight": "10px"}),
            html.Button("\U0001F44E", id={"type": "downvote-btn", "index": idx}, n_clicks=0, className="vi-icon-btn"),
        ],
    )


def render_answer_body(idx, item):
    """Render a plain-text answer, or every code block in the answer plus its instructions."""
    blocks = item.get("blocks") or []
    if not blocks and item.get("code"):
        # A history entry stored before `blocks` existed.
        blocks = [{"language": backend.guess_language(item["code"]), "code": item["code"], "truncated": False}]

    if not blocks:
        return html.Div(
            item.get("answer") or item.get("instructions") or "",
            style={"color": COLORS["text"], "whiteSpace": "pre-wrap", "lineHeight": "1.6", "fontSize": "14px"},
        )

    children = []
    for block_idx, block in enumerate(blocks):
        # The copy button and its code block are matched by a shared composite index, so
        # a multi-block answer gets one working copy button per block.
        pair_id = f"{idx}-{block_idx}"
        header = [
            html.Span(block.get("language") or "Code"),
            html.Button("Copy", id={"type": "copy-btn", "index": pair_id}, n_clicks=0, className="vi-copy-btn"),
        ]
        wrap_children = [
            html.Div(header, className="vi-code-header"),
            html.Pre(block["code"], id={"type": "code-block", "index": pair_id}, className="vi-code-block"),
        ]
        if block.get("truncated"):
            wrap_children.append(
                html.Div(
                    "This block looks cut off — the model hit its output limit. Ask for the rest if you need it.",
                    className="vi-truncated-note",
                )
            )
        children.append(html.Div(wrap_children, className="vi-code-wrap"))

    if item.get("instructions"):
        children.append(html.Div("Instructions", className="vi-instructions-heading"))
        children.append(
            html.Div(
                item["instructions"],
                style={"color": COLORS["text"], "fontSize": "14px", "lineHeight": "1.6", "whiteSpace": "pre-wrap"},
            )
        )
    return html.Div(children)


def render_bubbles(exchanges):
    """Render the chat as question/answer pairs.

    An item with pending=True is a just-sent message still waiting on the model: only its
    question bubble renders, and the separate loading row fills the answer area.
    """
    bubbles = []
    for idx, item in enumerate(exchanges or []):
        question_children = [html.Div(item["question"], style={"whiteSpace": "pre-wrap"})]
        if item.get("file_name"):
            question_children.append(html.Div([html.Span("\U0001F4C4 "), item["file_name"]], className="vi-file-badge"))

        question_bubble = html.Div(
            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "14px"},
            children=html.Div(question_children, className="vi-msg-user"),
        )
        if item.get("pending"):
            bubbles.append(html.Div(style={"marginBottom": "8px"}, children=[question_bubble]))
            continue
        if item.get("error"):
            bubbles.append(
                html.Div(
                    style={"marginBottom": "36px"},
                    children=[
                        question_bubble,
                        html.Div(
                            item["answer"],
                            style={"color": COLORS["amber"], "background": "#FFFBEB", "border": "1px solid #FDE68A",
                                   "borderRadius": "10px", "padding": "10px 14px", "fontSize": "14px", "lineHeight": "1.6"},
                        ),
                    ],
                )
            )
            continue
        bubbles.append(
            html.Div(
                style={"marginBottom": "36px"},
                children=[question_bubble, render_answer_body(idx, item), render_feedback_controls(idx, item.get("feedback"))],
            )
        )
    return bubbles


def render_empty_state():
    def info_card(card_icon, title, desc):
        return html.Div(
            className="vi-info-card",
            children=[
                html.Div(card_icon, className="vi-info-card-icon"),
                html.Div(title, style={"fontWeight": 400, "fontSize": "14px", "marginBottom": "4px"}),
                html.Div(desc, style={"fontSize": "13px", "color": COLORS["text_dim"], "lineHeight": "1.5"}),
            ],
        )

    return html.Div(
        style={"maxWidth": "700px", "margin": "60px auto 0", "textAlign": "center", "padding": "0 20px"},
        children=[
            html.Div("</>", className="vi-empty-icon"),
            html.Div(
                [html.Span("Welcome to "), html.Span("Forge Code", style={"color": COLORS["red"]})],
                style={"fontSize": "30px", "fontWeight": 400, "marginBottom": "10px"},
            ),
            html.Div(
                "Upload a Python or SQL file and ask about the code in it, request edit suggestions, "
                "or describe something you want written from scratch.",
                style={"fontSize": "15px", "color": COLORS["text_dim"], "marginBottom": "32px", "lineHeight": "1.6"},
            ),
            html.Div(
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                children=[
                    info_card("\U0001F4AC", "Ask a question", "Get Python or SQL with clear instructions"),
                    info_card(icon("upload_docs_icon", 18), "Upload a file", "Explain, review or edit your own code"),
                    info_card("\U0001F6E1️", "Private by design", "Your files and chats stay on your own infrastructure"),
                ],
            ),
        ],
    )


def chat_layout(user):
    """Main chat page: sidebar, message history, feedback reason bar, composer."""
    conversations = fetch_conversations(user["user_id"])
    pending_count = 0
    if user["is_admin"]:
        try:
            pending_count = backend.count_pending_users()
        except AppError:
            pending_count = 0

    return html.Div(
        className="vi-shell",
        children=[
            html.Div(id="sidebar", className="vi-sidebar", children=render_sidebar_items(conversations, None)),
            html.Div(
                style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column", "height": "100%"},
                children=[
                    build_header(user, pending_count),
                    dcc.Store(id="chat-history", data=[]),
                    dcc.Store(id="active-conversation-id", data=None),
                    dcc.Store(id="sidebar-conversations", data=conversations),
                    dcc.Store(id="active-downvote-index", data=None),
                    dcc.Store(id="sidebar-collapsed", data=False),
                    dcc.Store(id="attached-file-store", data=None),
                    dcc.Store(id="pending-request", data=None),
                    dcc.Store(id="is-loading", data=False),
                    dcc.Interval(id="thinking-interval", interval=5000, disabled=True),
                    # Read by the file-picker JS so the client-side size cap matches the server's.
                    html.Div(str(backend.MAX_UPLOAD_BYTES), id="max-upload-bytes", style={"display": "none"}),
                    html.Div(
                        style={"flex": "1", "overflowY": "auto", "display": "flex", "justifyContent": "center"},
                        children=html.Div(
                            style={"width": "100%", "maxWidth": CONTENT_WIDTH, "padding": "32px 20px"},
                            children=[
                                html.Div(id="empty-state", children=render_empty_state()),
                                html.Div(id="chat-window", children=[]),
                                html.Div(
                                    id="loading-row",
                                    style={"display": "none", "alignItems": "center", "gap": "10px", "marginTop": "4px"},
                                    children=[html.Div(className="vi-spinner"), html.Span(id="thinking-text", style={"color": COLORS["text_dim"], "fontSize": "14px"})],
                                ),
                            ],
                        ),
                    ),
                    html.Div(
                        style={"display": "flex", "justifyContent": "center"},
                        children=html.Div(
                            style={"width": "100%", "maxWidth": CONTENT_WIDTH, "padding": "0 20px"},
                            children=[
                                html.Div(
                                    id="reason-bar",
                                    style={"display": "none", "padding": "10px 0", "alignItems": "center"},
                                    children=[
                                        html.Span("Why was this unhelpful?", style={"marginRight": "10px", "fontSize": "14px", "color": COLORS["text_dim"]}),
                                        dcc.Input(id="reason-input", type="text", value="", placeholder="Optional reason...", className="vi-input", style={"flex": "1", "padding": "8px", "marginRight": "10px"}),
                                        html.Button("Submit", id="reason-submit", n_clicks=0, className="vi-btn", style={"padding": "8px 16px", "marginRight": "6px"}),
                                        html.Button("Cancel", id="reason-cancel", n_clicks=0, className="vi-btn-secondary", style={"padding": "8px 16px"}),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    html.Div(
                        style={"display": "flex", "justifyContent": "center", "padding": "10px 20px 20px"},
                        children=html.Div(
                            style={"width": "100%", "maxWidth": CONTENT_WIDTH},
                            children=[
                                html.Div(id="upload-error", style={"color": COLORS["red_light"], "fontSize": "13px", "marginBottom": "8px"}),
                                html.Div(
                                    className="vi-composer",
                                    children=[
                                        html.Div(id="attached-file-chip"),
                                        dcc.Textarea(
                                            id="chat-input", value="",
                                            placeholder="Ask about your code, request an edit, or describe what you want written...",
                                            style={"minHeight": "24px"},
                                        ),
                                        html.Div(
                                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginTop": "10px"},
                                            children=[
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                                    children=[
                                                        dcc.Upload(
                                                            id="file-upload",
                                                            accept=".py,.sql",
                                                            children=html.Div([icon("upload_docs_icon", 14), html.Span("Attach File")], className="vi-attach-btn"),
                                                        ),
                                                        html.Span("Python and SQL only", className="vi-scope-hint"),
                                                    ],
                                                ),
                                                html.Button("↑", id="send-button", n_clicks=0, className="vi-btn vi-send-btn", disabled=True),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    "AI may make mistakes. Please verify important information.",
                                    style={"textAlign": "center", "fontSize": "12px", "color": "#98A2B3", "marginTop": "10px"},
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )


def render_user_rows(users, current_user_id):
    rows = []
    for user in users:
        status = (
            html.Span("Active", className="vi-pill vi-pill-active")
            if user["is_active"]
            else html.Span("Pending approval", className="vi-pill vi-pill-pending")
        )
        role = html.Span("Admin", className="vi-pill vi-pill-admin") if user["is_admin"] else html.Span("User", style={"color": COLORS["text_dim"]})

        actions = []
        if user["is_active"]:
            actions.append(
                html.Button(
                    "Suspend", id={"type": "suspend-user", "index": user["id"]}, n_clicks=0,
                    className="vi-btn-secondary", style={"padding": "6px 12px", "fontSize": "12px"},
                    disabled=user["id"] == current_user_id,
                )
            )
        else:
            actions.append(
                html.Button(
                    "Approve", id={"type": "approve-user", "index": user["id"]}, n_clicks=0,
                    className="vi-btn", style={"padding": "6px 12px", "fontSize": "12px"},
                )
            )
        if not user["is_admin"]:
            actions.append(
                html.Button(
                    "Make admin", id={"type": "promote-user", "index": user["id"]}, n_clicks=0,
                    className="vi-btn-secondary", style={"padding": "6px 12px", "fontSize": "12px"},
                )
            )

        rows.append(
            html.Tr(
                [
                    html.Td(user["username"]),
                    html.Td(user["email"] or html.Span("—", style={"color": "#98A2B3"})),
                    html.Td(status),
                    html.Td(role),
                    html.Td(format_relative_time(user["created_at"]), style={"color": COLORS["text_dim"]}),
                    html.Td(html.Div(actions, style={"display": "flex", "gap": "8px"})),
                ]
            )
        )
    return rows


def admin_layout(user):
    """Account approval screen. Only reachable by an admin."""
    try:
        users = backend.list_users(user["user_id"])
        error = None
    except AppError as exc:
        users, error = [], exc.message

    return html.Div(
        className="vi-shell",
        style={"flexDirection": "column"},
        children=[
            build_header(user, show_sidebar_toggle=False),
            dcc.Store(id="admin-refresh", data=0),
            html.Div(
                style={"flex": "1", "overflowY": "auto", "padding": "28px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "6px"},
                        children=[
                            html.Div("User accounts", style={"fontSize": "20px"}),
                            dcc.Link("← Back to chat", href="/", className="vi-link"),
                        ],
                    ),
                    html.Div(
                        "New registrations stay inactive until approved here.",
                        style={"fontSize": "13px", "color": COLORS["text_dim"], "marginBottom": "20px"},
                    ),
                    html.Div(id="admin-message", style={"fontSize": "13px", "marginBottom": "14px", "color": COLORS["red_light"]}, children=error or ""),
                    html.Div(
                        className="vi-card",
                        style={"padding": "8px 16px 4px"},
                        children=html.Table(
                            className="vi-table",
                            children=[
                                html.Thead(html.Tr([html.Th("Username"), html.Th("Email"), html.Th("Status"), html.Th("Role"), html.Th("Registered"), html.Th("Actions")])),
                                html.Tbody(id="admin-user-rows", children=render_user_rows(users, user["user_id"])),
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )


def message_page(title, body, link_label="Back to log in", link_href="/"):
    return _auth_card(
        [
            html.Div(brand(), style={"marginBottom": "16px"}),
            html.Div(title, style={"fontSize": "16px", "marginBottom": "8px"}),
            html.Div(body, style={"fontSize": "13px", "color": COLORS["text_dim"], "lineHeight": "1.6", "marginBottom": "18px"}),
            dcc.Link(link_label, href=link_href, className="vi-link"),
        ]
    )


# ============================================================
# ROUTING
# ============================================================

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    user = current_user()

    if pathname == "/register":
        return chat_layout(user) if user else register_layout()

    if not user:
        return login_layout()

    if pathname == "/admin":
        if not user["is_admin"]:
            return message_page(
                "Not available",
                "That page is for administrators only.",
                link_label="Back to chat",
            )
        return admin_layout(user)

    return chat_layout(user)


# ============================================================
# AUTH CALLBACKS
# ============================================================

@app.callback(
    Output("login-error", "children"),
    Output("reload-signal", "data"),
    Input("login-button", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    State("reload-signal", "data"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, username, password, signal):
    if not n_clicks:
        return no_update, no_update
    try:
        result = backend.login_user(username or "", password or "", client_ip=client_ip())
    except AppError as exc:
        return exc.message, no_update

    # after_request turns this into the Set-Cookie header; the reload then arrives with
    # the cookie already in place, so render_page sees an authenticated user.
    g.forge_set_token = result["token"]
    return "", (signal or 0) + 1


@app.callback(
    Output("reload-signal", "data", allow_duplicate=True),
    Input("logout-button", "n_clicks"),
    State("reload-signal", "data"),
    prevent_initial_call=True,
)
def handle_logout(n_clicks, signal):
    if not n_clicks:
        return no_update
    g.forge_clear_token = True
    return (signal or 0) + 1


@app.callback(
    Output("register-message", "children"),
    Output("register-message", "style"),
    Input("register-button", "n_clicks"),
    State("register-username", "value"),
    State("register-email", "value"),
    State("register-password", "value"),
    State("register-password2", "value"),
    prevent_initial_call=True,
)
def handle_register(n_clicks, username, email, password, password2):
    error_style = {"marginTop": "12px", "fontSize": "13px", "lineHeight": "1.5", "color": COLORS["red_light"]}
    ok_style = {"marginTop": "12px", "fontSize": "13px", "lineHeight": "1.5", "color": COLORS["green"]}

    if not n_clicks:
        return no_update, no_update
    if (password or "") != (password2 or ""):
        return "The two passwords do not match.", error_style

    try:
        backend.register_user(username or "", password or "", email, client_ip=client_ip())
    except AppError as exc:
        return exc.message, error_style

    return (
        "Account created. An administrator has to approve it before you can log in.",
        ok_style,
    )


# ============================================================
# SIDEBAR CALLBACKS
# ============================================================

@app.callback(
    Output("active-conversation-id", "data", allow_duplicate=True),
    Input("new-chat-button", "n_clicks"),
    prevent_initial_call=True,
)
def handle_new_chat(n_clicks):
    if not n_clicks:
        return no_update
    return None


@app.callback(
    Output("active-conversation-id", "data", allow_duplicate=True),
    Input({"type": "conv-item", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_conversation_click(n_clicks_list):
    if not ctx.triggered_id or not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    return ctx.triggered_id["index"]


@app.callback(
    Output("chat-history", "data", allow_duplicate=True),
    Input("active-conversation-id", "data"),
    prevent_initial_call=True,
)
def load_conversation(active_id):
    if active_id is None:
        return []
    try:
        user = require_user()
        # Ownership is checked in the backend, so a guessed or stale id yields nothing
        # rather than another user's chat.
        messages = backend.get_conversation_messages(user["user_id"], active_id)
    except AppError:
        return []
    return [{**m, "feedback": None} for m in messages]


@app.callback(
    Output("sidebar", "children"),
    Input("sidebar-conversations", "data"),
    Input("active-conversation-id", "data"),
)
def update_sidebar(conversations, active_id):
    return render_sidebar_items(conversations or [], active_id)


@app.callback(
    Output("sidebar-collapsed", "data"),
    Input("sidebar-collapse-btn", "n_clicks"),
    Input("sidebar-expand-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_sidebar(collapse_clicks, expand_clicks):
    if ctx.triggered_id == "sidebar-collapse-btn":
        return True
    if ctx.triggered_id == "sidebar-expand-btn":
        return False
    return no_update


@app.callback(
    Output("sidebar", "style"),
    Output("sidebar-expand-btn", "style"),
    Input("sidebar-collapsed", "data"),
)
def apply_sidebar_collapsed(collapsed):
    if collapsed:
        return (
            {"width": "0", "minWidth": "0", "padding": "0", "border": "none", "overflow": "hidden"},
            {"padding": "4px 8px", "display": "inline-block"},
        )
    return {}, {"padding": "4px 8px", "display": "none"}


# ============================================================
# CHAT CALLBACKS
# ============================================================

@app.callback(
    Output("attached-file-chip", "children"),
    Input("attached-file-store", "data"),
)
def render_attached_file_chip(file_data):
    """The native <input type=file> is read by plain JS (see index_string), which writes
    straight into attached-file-store. This callback only renders the resulting chip."""
    if not file_data:
        return None
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px", "padding": "6px 12px", "marginBottom": "8px",
               "width": "fit-content", "background": COLORS["bg_secondary"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "999px", "fontSize": "13px"},
        children=[
            html.Span("\U0001F4C4"),
            html.Span(file_data.get("name", "file")),
            html.Button("✕", id={"type": "remove-attachment", "index": 0}, n_clicks=0, className="vi-icon-btn", style={"fontSize": "12px"}),
        ],
    )


@app.callback(
    Output("attached-file-store", "data", allow_duplicate=True),
    Input({"type": "remove-attachment", "index": 0}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_attached_file(n_clicks):
    if not n_clicks:
        return no_update
    return None


@app.callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Output("attached-file-store", "data", allow_duplicate=True),
    Output("upload-error", "children", allow_duplicate=True),
    Output("pending-request", "data"),
    Output("is-loading", "data"),
    Input("send-button", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("attached-file-store", "data"),
    State("active-conversation-id", "data"),
    prevent_initial_call=True,
)
def handle_send(n_clicks, question, history, attached_file, conversation_id):
    """Fast leg: show the question and flip on the loading state before the model runs."""
    if not question or not question.strip():
        return (no_update,) * 6

    file_name = attached_file.get("name") if attached_file else None
    placeholder = {"question": question, "answer": None, "pending": True, "file_name": file_name}
    history = (history or []) + [placeholder]

    pending = {"question": question, "conversation_id": conversation_id}
    if attached_file:
        pending["file_name"] = file_name
        pending["file_content"] = attached_file.get("content")

    return history, "", None, "", pending, True


@app.callback(
    Output("chat-history", "data", allow_duplicate=True),
    Output("active-conversation-id", "data", allow_duplicate=True),
    Output("sidebar-conversations", "data"),
    Output("is-loading", "data", allow_duplicate=True),
    Output("pending-request", "data", allow_duplicate=True),
    Input("pending-request", "data"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def resolve_send(pending, history):
    """Slow leg: calls into the backend, then swaps the placeholder for the real answer."""
    if not pending or not history:
        return (no_update,) * 5

    file_name = pending.get("file_name")

    def failed(message):
        history[-1] = {
            "question": pending["question"], "answer": message, "response_type": "message",
            "blocks": [], "feedback": None, "file_name": file_name, "error": True,
        }
        return history, no_update, no_update, False, None

    try:
        user = require_user()
    except AppError as exc:
        return failed(exc.message)

    try:
        data = backend.ask_logic(
            user_id=user["user_id"],
            question=pending["question"],
            # The conversation id comes from the browser, so the backend re-checks that
            # this user owns it before writing anything to it.
            conversation_id=pending.get("conversation_id"),
            file_name=file_name,
            file_content=pending.get("file_content"),
        )
    except AppError as exc:
        return failed(exc.message)
    except Exception:
        # Never leak an internal traceback into the chat window.
        app.logger.exception("ask_logic failed")
        return failed("Something went wrong handling that request. Try again shortly.")

    history[-1] = {
        "question": data["question"],
        "answer": data["answer"],
        "response_type": data.get("response_type"),
        "instructions": data.get("instructions"),
        "code": data.get("code"),
        "blocks": data.get("blocks") or [],
        "feedback": None,
        "file_name": data.get("file_name"),
    }
    return history, data["conversation_id"], fetch_conversations(user["user_id"]), False, None


@app.callback(
    Output("chat-window", "children"),
    Output("empty-state", "style"),
    Input("chat-history", "data"),
)
def update_chat_window(history):
    return render_bubbles(history or []), {"display": "block"} if not history else {"display": "none"}


@app.callback(
    Output("loading-row", "style"),
    Output("thinking-interval", "disabled"),
    Input("is-loading", "data"),
)
def toggle_loading_row(is_loading):
    style = {"alignItems": "center", "gap": "10px", "marginTop": "4px", "display": "flex" if is_loading else "none"}
    return style, not is_loading


@app.callback(
    Output("send-button", "disabled"),
    Input("chat-input", "value"),
    Input("is-loading", "data"),
)
def toggle_send_disabled(value, is_loading):
    """Disabled when the input is empty, and while a request is in flight."""
    return bool(is_loading) or not (value and value.strip())


# ============================================================
# FEEDBACK CALLBACKS
# ============================================================

def _rated_item(history, index):
    """The history entry a feedback click refers to, if it can be rated."""
    if index is None or not history or index >= len(history):
        return None
    item = history[index]
    if item.get("pending") or item.get("error") or not item.get("answer"):
        return None
    return item


@app.callback(
    Output("chat-history", "data", allow_duplicate=True),
    Input({"type": "upvote-btn", "index": ALL}, "n_clicks"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def handle_upvote(n_clicks_list, history):
    if not ctx.triggered_id or not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update

    idx = ctx.triggered_id["index"]
    item = _rated_item(history, idx)
    if item is None:
        return no_update

    try:
        user = require_user()
        backend.submit_feedback(user["user_id"], item["question"], item["answer"], "up", None)
    except AppError:
        return no_update

    history[idx]["feedback"] = "up"
    return history


@app.callback(
    Output("active-downvote-index", "data"),
    Input({"type": "downvote-btn", "index": ALL}, "n_clicks"),
    Input("reason-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def handle_downvote_click(n_clicks_list, cancel_clicks):
    if ctx.triggered_id == "reason-cancel":
        return None
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    return ctx.triggered_id["index"]


@app.callback(
    Output("reason-bar", "style"),
    Input("active-downvote-index", "data"),
)
def toggle_reason_bar(active_index):
    return {"padding": "10px 0", "alignItems": "center", "display": "flex" if active_index is not None else "none"}


@app.callback(
    Output("chat-history", "data", allow_duplicate=True),
    Output("active-downvote-index", "data", allow_duplicate=True),
    Output("reason-input", "value"),
    Input("reason-submit", "n_clicks"),
    State("reason-input", "value"),
    State("active-downvote-index", "data"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def handle_reason_submit(n_clicks, reason, active_index, history):
    if not n_clicks:
        return no_update, no_update, no_update

    item = _rated_item(history, active_index)
    if item is None:
        return no_update, None, ""

    try:
        user = require_user()
        backend.submit_feedback(user["user_id"], item["question"], item["answer"], "down", reason or None)
    except AppError:
        return no_update, no_update, no_update

    history[active_index]["feedback"] = "down"
    return history, None, ""


# ============================================================
# ADMIN CALLBACKS
# ============================================================

@app.callback(
    Output("admin-user-rows", "children"),
    Output("admin-message", "children"),
    Input({"type": "approve-user", "index": ALL}, "n_clicks"),
    Input({"type": "suspend-user", "index": ALL}, "n_clicks"),
    Input({"type": "promote-user", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_admin_action(approve_clicks, suspend_clicks, promote_clicks):
    if not ctx.triggered_id or not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update, no_update

    action = ctx.triggered_id["type"]
    target_id = ctx.triggered_id["index"]

    try:
        admin = require_admin()
        if action == "approve-user":
            backend.set_user_active(admin["user_id"], target_id, True)
        elif action == "suspend-user":
            backend.set_user_active(admin["user_id"], target_id, False)
        elif action == "promote-user":
            backend.set_user_admin(admin["user_id"], target_id, True)
        users = backend.list_users(admin["user_id"])
    except AppError as exc:
        return no_update, exc.message

    return render_user_rows(users, admin["user_id"]), ""


# ============================================================
# OPS ROUTES
# ============================================================

@server.route("/health")
def health_route():
    status = backend.health_check()
    code = 200 if status["db"] == "ok" and status["mistral"] == "ok" else 503
    return jsonify(status), code


# ============================================================
# ENTRYPOINT
# ============================================================

def create_admin_cli() -> int:
    """`python app.py create-admin` — create the first administrator interactively."""
    print("Creating an administrator account.\n")
    username = input("Username: ").strip()
    email = input("Email (optional): ").strip() or None
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("\nThe two passwords do not match.", file=sys.stderr)
        return 1
    try:
        result = backend.bootstrap_admin(username, password, email)
    except AppError as exc:
        print(f"\n{exc.message}", file=sys.stderr)
        return 1

    print(f"\nCreated administrator {result['username']!r} (id {result['user_id']}). You can log in now.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        sys.exit(create_admin_cli())

    if len(sys.argv) > 1:
        print(f"Unknown argument {sys.argv[1]!r}. Usage: python app.py [create-admin]", file=sys.stderr)
        sys.exit(2)

    # debug defaults off: the reloader and the interactive traceback console have no
    # business on a multi-user host. Set FORGE_DEBUG=true for local development.
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
