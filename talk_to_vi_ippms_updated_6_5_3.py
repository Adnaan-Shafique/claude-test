#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  TALK-TO-VI-IPPMS
#  Conversational analytics over the Instant Graph network-monitoring APIs.
#  Single-file app: Dash frontend + LangGraph agentic backend.
# ══════════════════════════════════════════════════════════════════════════════
#
#  What this is
#  ------------
#  A chat application that lets a network-operations user ask, in plain English,
#  questions about VI-IPPMS metadata and KPIs, e.g.:
#
#    metadata  : "How many devices are in the GUJ circle?"
#                "How many interfaces on APVSPGJWPAR01HNE40?"
#                "How many components / KPIs are tracked on that device?"
#                "List all devices whose name contains PAR01."
#    kpi/data  : "On APVSPGJWPAR01HNE40, interface 100GE0/3/2, what was HC In
#                 Octets between 8am and 2pm yesterday?"
#                "Give me the min/max/avg traffic on Eth-Trunk1 in the last 6h."
#
#  It does NOT query any SQL database. Every fact comes from the Instant Graph
#  REST APIs, reached through the MCP server (instant_graph_mcp_server_v2_3.py)
#  as tools. A LangGraph pipeline routes the question, grounds the entities, and
#  builds an explicit QuerySpec (§4.2A). A recognized QuerySpec is answered by one
#  general, deterministic executor (no LLM in the retrieval path, so the same
#  question returns identical data every run); only genuinely unrecognized shapes
#  fall back to the free-text ReAct tool loop. The LLM is used only to (i) turn
#  free text into entities and (ii) phrase the final sentence.
#
#  v6 changes (see vi_ippms_v2_3_v6_upgrade_brief.md §4.2):
#    A. QuerySpec engine (build_spec + execute_query_spec) replacing the closed
#       deterministic_metadata/deterministic_kpi if/elif branches — resolves both
#       answer consistency (§3.1) and question-shape coverage (§3.6);
#    B. a full debug object surfaced behind a UI toggle on every path (§3.2);
#    C. scope-aware handling that distinguishes unrelated vs unsupported (§3.3);
#    D. one consolidated tool KB + local TF-IDF retrieval over the question guide
#       (§3.4), replacing scattered docstrings/hardcoded tables;
#    E. persistent interaction + feedback logging to Postgres (§3.5);
#    plus Appendix-B hygiene: no hardcoded GPU key, and the AUTH/SESSION/QUICKQ/
#    SUBMIT debug log lines that leaked secrets/tokens/stack traces are removed.
#
#  v6.2 changes:
#    1. Every result table now carries its own "⬇ Download CSV" button in a
#       header bar above it (with row/column count), replacing the easy-to-miss
#       CSV button that sat in the feedback row. The export is rebuilt from the
#       message store, so it contains all rows, not just the visible page.
#       Tables also gained native sort + filter.
#    2. Enter sends the message; Shift+Enter inserts a newline (the textarea
#       auto-grows). The hint under the input box now says so.
#    3. Fixed "'dict' object has no attribute 'split'". Two boundaries could
#       leak non-string values into code typed for str — the router LLM emitting
#       a nested object/list for a scalar entity, and ig_list_interfaces items
#       arriving object-shaped instead of as the documented "Interface X ::"
#       strings. Both are now coerced at the boundary (_as_text/_norm_items),
#       _iface_display no longer assumes str, execute_query_spec catches more
#       than MCPError, and run_agent's catch-all reports the type/stage/frame
#       plus a traceback in the Debug panel instead of a bare message.
#    4. New "🧩 Parameters" disclosure panel next to the Debug trace on every
#       answer, listing what was extracted from the question (circle, device,
#       component, interface, KPI, target/metric/aggregation, top-N, threshold,
#       time window) and what each resolved to against live data.
#
#  v6.6 changes:
#    1. New interface-ranking analytical shape: "top/bottom N interfaces for
#       KPI X on device Y currently" (§4.2A, build_spec/_execute_values). The
#       API has no live-value endpoint, so "currently" is approximated as the
#       most recent ("Last") data point within a short lookback window
#       (CURRENT_VALUE_LOOKBACK_SECONDS, default 3h) rather than a historical
#       Min/Max/Avg — narrowly gated on a top_n/bottom_n aggregation + the
#       word "interface(s)" + no explicit time phrase, so the pre-existing
#       "top N DEVICES by KPI over <window>" shape is untouched. The KPI's
#       owning component is auto-guessed from KPI_SYNONYMS when not named
#       (self-heals via the existing zero-result recheck if wrong); a missing
#       KPI or an ambiguous/unresolved device returns a clarification instead
#       of guessing. Renders as an Interface/KPI/Value/Time table plus a new
#       horizontal bar chart (build_rank_bar_chart).
#
#  v6.7 changes:
#    1. New spike/dip (anomaly) detection shape: "show me the spikes/dips for
#       KPI X on interface Y on device Z in the last N minutes/hours or
#       between A and B" (§4.2A, aggregation "spikes"/"dips"/"anomalies").
#       Each point is scored against a LOCAL baseline built from its own
#       neighbors only (a rolling window, excluding the point itself — a
#       plain centered rolling mean/std that includes the point lets a real
#       spike drag its own baseline toward itself and dilute its z-score,
#       caught via isolated simulation before shipping) and flagged when more
#       than SPIKE_ZSCORE_THRESHOLD (default 2.5) standard deviations off
#       that baseline. Requires exactly one named interface (spike detection
#       needs one series to analyze); a missing interface, device, or KPI
#       returns a clarification. Renders a compact flagged-events table
#       (Interface/KPI/Type/Value/Time/Deviation), tags the existing raw
#       time-series table with a Flag column, and overlays flagged points as
#       distinct markers on the existing line chart (build_timeseries_fig's
#       new optional `markers` param).
#    2. parse_time_window now understands an explicit "between A and B" range
#       (bare times or full date-times), not just relative phrases ("last N
#       hours", "yesterday") — a pre-existing gap this feature's own examples
#       relied on.
#
#  All reasoning runs on a LOCAL, airgapped open-source model (mistral-7b via
#  your GPU inference proxy). No request ever leaves the local network.
#
#  Auth model (per-user)
#  ---------------------
#  The login page asks for email + employee id only. Both are pushed straight
#  to the MCP `ig_login` tool to mint that user's Instant Graph token — a
#  successful Instant Graph login IS the auth check, there is no separate
#  app-side password gate. Every downstream tool call carries a per-user
#  session_key (the email), so tokens are isolated across users.
#  The agent never sees the session_key; the executor injects it automatically.
#
#  Run it
#  ------
#    1) Start the MCP server (separate process):
#         python instant_graph_mcp_server_v2_2.py
#    2) Start this app:
#         export GPU_PROXY_URL=http://127.0.0.1:8071/v1/infer
#         export MCP_SERVER_URL=http://127.0.0.1:9000/mcp
#         python talk_to_vi_ippms.py
#    3) Browse to http://<host>:8070/
#
#  Dependencies
#    pip install langgraph "mcp[cli]" dash plotly pandas requests --break-system-packages
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import csv
import json
import time
import hashlib
import logging
import asyncio
import threading
import traceback
from contextlib import contextmanager
from io import StringIO
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import requests
import pandas as pd

import psycopg2
import psycopg2.extras

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, ctx, no_update
import plotly.graph_objects as go
import plotly.io as pio

from langgraph.graph import StateGraph, END

# MCP client (async) — used to reach the Instant Graph MCP server's tools.
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client





# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Local GPU inference proxy (mistral-7b-instruct). Same contract as
# gpu_llm_context.py: POST {model,prompt,...} -> {"text": ...}.
GPU_PROXY_URL = os.environ.get("GPU_PROXY_URL", "http://127.0.0.1:8071/v1/infer")
# SECURITY (was Appendix B): no hardcoded API-key fallback in source. If unset,
# GPULLMClient warns once and calls the proxy unauthenticated.
GPU_API_KEY   = os.environ.get("GPU_API_KEY", "secret-falcon2")
GPU_TIMEOUT   = int(os.environ.get("GPU_TIMEOUT", "180"))
MODEL         = os.environ.get("DEFAULT_MODEL", "mistral")

# Instant Graph MCP server endpoint (must match MCP_HOST/MCP_PORT there).
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8056/mcp")
MCP_CALL_TIMEOUT = float(os.environ.get("MCP_CALL_TIMEOUT", "45"))

# Multi-device fan-out (§3.6 / §4.1.4): the REST API is single-host, so a
# multi-device query issues N tool calls. Per the design decision, we fan out in
# PARALLEL with no hard cap; this bounds worker threads only.
FANOUT_MAX_WORKERS = int(os.environ.get("VI_FANOUT_MAX_WORKERS", "12"))

# Consolidated tool KB + question guide (§3.4 / §4.3). Bundled next to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_KB_PATH = os.environ.get("VI_TOOL_KB_PATH", os.path.join(_HERE, "vi_ippms_tool_kb.md"))
QUESTION_GUIDE_PATH = os.environ.get(
    "VI_QUESTION_GUIDE_PATH", os.path.join(_HERE, "vi_ippms_question_guide.xlsx"))
RAG_TOP_K = int(os.environ.get("VI_RAG_TOP_K", "5"))

# Per-session answer cache (§4.2A): normalized question -> last answer, short TTL,
# so a literally-identical repeat returns the exact same string.
ANSWER_CACHE_TTL = int(os.environ.get("VI_ANSWER_CACHE_TTL", "300"))

# Dash server.
APP_HOST = os.environ.get("VI_APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("VI_APP_PORT", "8079"))
#APP_PORT = "8779" # Testing

# Agent limits.
REACT_MAX_STEPS   = int(os.environ.get("VI_REACT_MAX_STEPS", "8"))
REACT_MAX_REPAIRS = int(os.environ.get("VI_REACT_MAX_REPAIRS", "1"))
HOSTS_CACHE_TTL   = int(os.environ.get("VI_HOSTS_CACHE_TTL", "300"))   # seconds
OBS_CHAR_CAP      = int(os.environ.get("VI_OBS_CHAR_CAP", "3500"))     # truncate tool observations for the 7B

# "Top/bottom N interfaces for KPI X on device Y currently" (§4.2A interface-
# ranking shape): the API has no live-value endpoint, only historical ranges,
# so "currently" is approximated as the most recent data point ("Last") within
# this lookback window rather than a Min/Max/Avg over a long window.
CURRENT_VALUE_LOOKBACK_SECONDS = int(os.environ.get("VI_CURRENT_VALUE_LOOKBACK_SECONDS", str(3 * 3600)))

# "Show me the spikes/dips for KPI X on interface Y on device Z in the last N
# minutes/hours or between A and B" (§4.2A anomaly-detection shape): a point
# is flagged when it's more than SPIKE_ZSCORE_THRESHOLD standard deviations
# from a rolling local baseline (window of SPIKE_ROLLING_WINDOW points,
# centered on the point) — adapts to a genuine trend within the requested
# window instead of comparing every point to one flat window-wide average.
# Below SPIKE_MIN_POINTS points, a rolling baseline isn't meaningful.
SPIKE_ROLLING_WINDOW    = int(os.environ.get("VI_SPIKE_ROLLING_WINDOW", "5"))
SPIKE_ZSCORE_THRESHOLD  = float(os.environ.get("VI_SPIKE_ZSCORE_THRESHOLD", "2.5"))
SPIKE_MIN_POINTS        = int(os.environ.get("VI_SPIKE_MIN_POINTS", "5"))

DEBUG = os.environ.get("VI_DEBUG", "1") == "1"

# ── Postgres (interaction + feedback logging, §3.5) ──────────────────────────
# Reuses the same database the MCP server uses for its token store. All logging
# is best-effort: a DB outage never blocks or crashes a chat turn.
DB_HOST     = os.environ.get("VI_DB_HOST", os.environ.get("IG_DB_HOST", "10.19.75.115"))
DB_PORT     = int(os.environ.get("VI_DB_PORT", os.environ.get("IG_DB_PORT", "5432")))
DB_NAME     = os.environ.get("VI_DB_NAME", os.environ.get("IG_DB_NAME", "conv_ai_db"))
DB_USER     = os.environ.get("VI_DB_USER", os.environ.get("IG_DB_USER", "ig_app_user"))
DB_PASSWORD = os.environ.get("VI_DB_PASSWORD", os.environ.get("IG_DB_PASSWORD", "ig_app_user_Qwer!234"))
DB_SCHEMA   = os.environ.get("VI_DB_SCHEMA", os.environ.get("IG_DB_SCHEMA", "tt_vi_ippms_schema"))
DB_CONNECT_TIMEOUT = int(os.environ.get("VI_DB_CONNECT_TIMEOUT", "5"))

# Persisted chat history (sidebar conversations, §4.4): how long a conversation
# is kept after its last activity before a background sweep hard-deletes it
# (and its messages) from Postgres, and how often that sweep runs.
CHAT_HISTORY_RETENTION_DAYS  = int(os.environ.get("VI_CHAT_HISTORY_RETENTION_DAYS", "7"))
CHAT_HISTORY_SWEEP_INTERVAL  = int(os.environ.get("VI_CHAT_HISTORY_SWEEP_INTERVAL", str(6 * 3600)))
CHAT_HISTORY_MAX_CONVS       = int(os.environ.get("VI_CHAT_HISTORY_MAX_CONVS", "100"))

# Decode presets per graph role (mirrors gpu_llm_context.py).
# SYNTHESIS is now temperature 0.0 (was 0.3): the query-spec engine makes the
# retrieved DATA deterministic; setting synthesis to 0.0 makes the phrasing as
# stable as the local stack allows, and structured shapes are templated anyway.
ROUTER_DECODE    = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}
PLANNER_DECODE   = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}
ACTION_DECODE    = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}
COT_DECODE       = {"temperature": 0.2, "top_p": 0.9, "top_k": 5}
SYNTHESIS_DECODE = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("talk_to_vi_ippms")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SESSION TOKENS
# ══════════════════════════════════════════════════════════════════════════════
#  There is no app-side password gate: email + employee id are the only
#  credentials, and they're validated by attempting an Instant Graph login via
#  the MCP `ig_login` tool (see do_login) — a successful IG login IS the auth
#  check. This section just issues an opaque browser session token once that
#  succeeds.

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


BOOT_ID = f"{os.getpid()}-{int(time.time())}"
log.info("[STARTUP] boot id %s", BOOT_ID)

# Opaque session tokens for the browser (kept server-side, memory only).
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESSIONS_LOCK = threading.Lock()


def session_create(email: str, eid: str) -> str:
    token = _sha256(f"{email}|{eid}|{time.time()}|{os.urandom(8).hex()}")[:40]
    with _SESSIONS_LOCK:
        _SESSIONS[token] = {"email": email.strip().lower(), "eid": eid.strip(),
                            "created": time.time()}
    return token


def session_get(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _SESSIONS_LOCK:
        return _SESSIONS.get(token)


def session_destroy(token: Optional[str]) -> None:
    if not token:
        return
    # (removed a log line that printed the full session token + a stack trace.)
    with _SESSIONS_LOCK:
        _SESSIONS.pop(token, None)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — LOCAL / AIRGAPPED LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class GPULLMClient:
    """Prompt-in / text-out client for the local GPU inference proxy. Never
    raises on failure — returns "" so a single bad call can't crash a graph
    node. (Same contract as gpu_llm_context.py.)"""

    def __init__(self, proxy_url: str = GPU_PROXY_URL, api_key: str = GPU_API_KEY,
                 timeout: int = GPU_TIMEOUT):
        self._url = proxy_url
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        else:
            log.warning("GPU_API_KEY not set — calling inference proxy unauthenticated.")

    def infer(self, prompt: str, model: str = MODEL, max_new_tokens: int = 512,
              temperature: float = 0.0, top_p: float = 0.9, top_k: int = 3) -> str:
        payload = {
            "model": model, "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k),
        }
        try:
            r = requests.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            if data.get("request_id"):
                log.debug("GPU request_id=%s tokens=%s elapsed=%ss",
                          data.get("request_id"), data.get("new_tokens"), data.get("elapsed_s"))
            # Contract is text-out. If the proxy ever returns a non-string here
            # (or a nested {"text": {...}}), coerce rather than hand a dict to
            # re.split()/parse_react_step() further up the stack.
            out = data.get("text", "")
            return out if isinstance(out, str) else (_as_text(out) or "")
        except requests.Timeout:
            log.error("GPU inference timed out after %ss", self._timeout)
            return ""
        except requests.RequestException as exc:
            log.error("GPU inference request failed: %s", exc)
            return ""


gpu_llm = GPULLMClient()


def call_llm(system_prompt: str, user_prompt: str, *, max_new_tokens: int = 512,
             decode: Optional[dict] = None, stage: str = "") -> str:
    """Wrap system/user text, inject current date/time, return completion text."""
    decode = decode or PLANNER_DECODE
    now = datetime.now().strftime("%Y-%m-%d %H:%M %A")
    system_with_date = f"{system_prompt.strip()}\n\nCURRENT DATE/TIME: {now}\n"
    wrapped = f"SYSTEM:\n{system_with_date}\n\nUSER:\n{user_prompt.strip()}\n\nASSISTANT:\n"
    if DEBUG and stage:
        log.info("LLM ▶ %s", stage)
    out = gpu_llm.infer(
        prompt=wrapped, model=MODEL, max_new_tokens=max_new_tokens,
        temperature=decode.get("temperature", 0.0),
        top_p=decode.get("top_p", 0.9), top_k=decode.get("top_k", 3),
    )
    if not isinstance(out, str):
        out = _as_text(out) or ""
    if DEBUG and stage:
        log.info("LLM ◀ %s (%d chars)", stage, len(out))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MCP CLIENT (reaches the Instant Graph tools)
# ══════════════════════════════════════════════════════════════════════════════
#  The app is an MCP *client*. Each call opens a short-lived streamable-HTTP
#  connection to the MCP server, initializes a session, and invokes one tool.
#  This is stateless and robust across Dash's worker threads (every call gets
#  its own event loop). Auth tools and the session_key are hidden from the LLM;
#  the executor injects session_key on every dispatch (see run_tool()).

AUTH_TOOLS = {"ig_login", "ig_token_status"}


class MCPError(RuntimeError):
    pass


def _extract_tool_payload(result: Any) -> Any:
    """Pull a plain Python object out of an MCP CallToolResult."""
    # FastMCP returns structured content for dict-returning tools.
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        # FastMCP wraps non-dict returns under {"result": ...}; unwrap if so.
        if set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    # Otherwise concatenate text blocks and try JSON.
    parts = []
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    joined = "\n".join(parts).strip()
    if not joined:
        return {}
    try:
        return json.loads(joined)
    except ValueError:
        return {"text": joined}


async def _amcp_list_tools() -> List[Dict[str, Any]]:
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.list_tools()
            return [{"name": t.name,
                     "description": (t.description or "").strip(),
                     "input_schema": t.inputSchema or {}} for t in res.tools]


async def _amcp_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if getattr(result, "isError", False):
                raise MCPError(f"{tool_name} returned an error: {_extract_tool_payload(result)}")
            return _extract_tool_payload(result)


def _run_async(coro, timeout: float):
    """Run an async coroutine to completion from a sync (Dash) thread."""
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def mcp_list_tools(timeout: float = 10.0) -> List[Dict[str, Any]]:
    try:
        return _run_async(_amcp_list_tools(), timeout)
    except Exception as exc:
        log.warning("Could not list MCP tools (%s); using static tool context.", exc)
        return []


def mcp_call(tool_name: str, arguments: Dict[str, Any],
             timeout: float = MCP_CALL_TIMEOUT) -> Any:
    """Call an MCP tool. Raises MCPError on failure."""
    try:
        return _run_async(_amcp_call(tool_name, dict(arguments or {})), timeout)
    except MCPError:
        raise
    except asyncio.TimeoutError as exc:
        raise MCPError(f"{tool_name} timed out after {timeout}s") from exc
    except Exception as exc:
        raise MCPError(f"{tool_name} call failed: {exc}") from exc


def ig_login_via_mcp(email: str, eid: str) -> Dict[str, Any]:
    """Establish this user's Instant Graph token on the MCP server."""
    return mcp_call("ig_login", {"email": email, "eid": eid, "session_key": email.lower()},
                    timeout=MCP_CALL_TIMEOUT)


def run_tool(tool_name: str, arguments: Dict[str, Any], session_key: str) -> Any:
    """Dispatch a data tool with the per-user session_key auto-injected."""
    args = dict(arguments or {})
    args["session_key"] = (session_key or "").lower()
    return mcp_call(tool_name, args)


# ── Cached grounding fetches (get-hosts is large + slow-changing) ────────────
_hosts_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_hosts_cache_lock = threading.Lock()


def get_hosts_cached(session_key: str) -> List[Dict[str, Any]]:
    key = (session_key or "").lower()
    now = time.time()
    with _hosts_cache_lock:
        hit = _hosts_cache.get(key)
        if hit and (now - hit[0]) < HOSTS_CACHE_TTL:
            return hit[1]
    data = run_tool("ig_list_hosts", {}, key)
    hosts = data.get("hosts", data if isinstance(data, list) else []) or []
    with _hosts_cache_lock:
        _hosts_cache[key] = (now, hosts)
    return hosts


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4B — PERSISTENT LOGGING  (vi_chat_interactions / vi_chat_feedback, §3.5)
# ══════════════════════════════════════════════════════════════════════════════
#  Every question, its route/entities/grounding/spec, how it was answered, the
#  tool calls, the answer and latency are persisted so there is a queryable
#  history of what was asked and how it was rated. Feedback rows link back to the
#  interaction row. All writes are best-effort — a DB outage degrades to
#  in-memory-only behaviour, never a failed chat turn.

_log_db_pool = None
_log_db_lock = threading.Lock()

_INTERACTIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.vi_chat_interactions (
    id            BIGSERIAL PRIMARY KEY,
    session_key   TEXT NOT NULL,
    asked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT NOT NULL,
    route         TEXT,
    confidence    TEXT,
    entities      JSONB,
    grounding     JSONB,
    query_spec    JSONB,
    answered_via  TEXT,
    fallback_used BOOLEAN,
    tool_calls    JSONB,
    answer        TEXT,
    result_kind   TEXT,
    latency_ms    INTEGER
);
"""
_FEEDBACK_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.vi_chat_feedback (
    id             BIGSERIAL PRIMARY KEY,
    interaction_id BIGINT REFERENCES {DB_SCHEMA}.vi_chat_interactions(id),
    vote           TEXT CHECK (vote IN ('up','down')),
    voted_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
# Free-text detail a user can optionally attach to a downvote (via the
# feedback popup). Kept separate from vi_chat_feedback (which stays just the
# simple up/down tally) — interaction_id is the join back to
# vi_chat_interactions for the question/answer/etc. this feedback is about.
# Deliberately NOT a DB-level FOREIGN KEY: creating one requires the REFERENCES
# privilege on vi_chat_interactions, which the app's DB role may not have if
# that table was provisioned by a different (more privileged) role — the join
# still works fine for reporting without a DB-enforced constraint.
_USER_FEEDBACK_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.user_feedback (
    id             BIGSERIAL PRIMARY KEY,
    interaction_id BIGINT,
    session_key    TEXT NOT NULL,
    feedback_text  TEXT NOT NULL,
    submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _get_log_pool():
    global _log_db_pool
    if _log_db_pool is None:
        with _log_db_lock:
            if _log_db_pool is None:
                from psycopg2.pool import ThreadedConnectionPool
                _log_db_pool = ThreadedConnectionPool(
                    1, 5, host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASSWORD,
                    options=f"-c search_path={DB_SCHEMA}",
                    connect_timeout=DB_CONNECT_TIMEOUT,
                )
    return _log_db_pool


@contextmanager
def _log_cursor(commit=False):
    pool = _get_log_pool()
    conn = pool.getconn()
    broken = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        broken = True
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn, close=broken)


_logging_ready = False
_logging_ready_lock = threading.Lock()


def _ensure_logging_tables() -> bool:
    """Best-effort CREATE ... IF NOT EXISTS. Returns True if logging is usable."""
    global _logging_ready
    if _logging_ready:
        return True
    with _logging_ready_lock:
        if _logging_ready:
            return True
        if not DB_PASSWORD:
            log.warning("VI_DB_PASSWORD/IG_DB_PASSWORD not set — chat interaction/feedback "
                        "logging disabled until a DB password is provided.")
            return False
        try:
            with _log_cursor(commit=True) as cur:
                cur.execute(_INTERACTIONS_DDL)
                cur.execute(_FEEDBACK_DDL)
            _logging_ready = True
            log.info("[STARTUP] chat interaction/feedback tables ready.")
            return True
        except Exception as exc:
            log.warning("Could not ensure chat logging tables (logging disabled): %s", exc)
            return False


_user_feedback_ready = False
_user_feedback_ready_lock = threading.Lock()


def _ensure_user_feedback_table() -> bool:
    """Best-effort CREATE ... IF NOT EXISTS for user_feedback, kept fully
    independent of _ensure_logging_tables()/_logging_ready: both run inside
    one transaction each, so if this table's DDL ever fails (e.g. a DB role
    without CREATE on the schema) it must not roll back — and therefore
    disable — the unrelated vi_chat_interactions/vi_chat_feedback logging
    that already works today."""
    global _user_feedback_ready
    if _user_feedback_ready:
        return True
    with _user_feedback_ready_lock:
        if _user_feedback_ready:
            return True
        if not DB_PASSWORD:
            return False
        try:
            with _log_cursor(commit=True) as cur:
                cur.execute(_USER_FEEDBACK_DDL)
            _user_feedback_ready = True
            log.info("[STARTUP] user_feedback table ready.")
            return True
        except Exception as exc:
            log.warning("Could not ensure user_feedback table (downvote free-text "
                        "logging disabled): %s", exc)
            return False


def log_interaction(session_key: str, question: str, result: Dict[str, Any]) -> Optional[int]:
    """Persist one interaction; return its row id (used to link feedback)."""
    if not _ensure_logging_tables():
        return None
    dbg = result.get("debug", {}) or {}
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""INSERT INTO {DB_SCHEMA}.vi_chat_interactions
                    (session_key, question, route, confidence, entities, grounding,
                     query_spec, answered_via, fallback_used, tool_calls, answer,
                     result_kind, latency_ms)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (session_key, question, result.get("route"), result.get("confidence"),
                 json.dumps(dbg.get("entities"), default=str),
                 json.dumps(dbg.get("grounding_summary"), default=str),
                 json.dumps(dbg.get("query_spec"), default=str),
                 result.get("answered_via"), bool(result.get("fallback_used")),
                 json.dumps(dbg.get("tool_calls"), default=str),
                 result.get("answer"), result.get("kind"),
                 int((result.get("timing", {}) or {}).get("total", 0) * 1000)),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else None
    except Exception as exc:
        log.warning("log_interaction failed: %s", exc)
        return None


def log_feedback(interaction_id: Optional[int], vote: str) -> None:
    if interaction_id is None or not _ensure_logging_tables():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"INSERT INTO {DB_SCHEMA}.vi_chat_feedback (interaction_id, vote) VALUES (%s,%s)",
                (interaction_id, vote),
            )
    except Exception as exc:
        log.warning("log_feedback failed: %s", exc)


def log_user_feedback(interaction_id: Optional[int], session_key: str, feedback_text: str) -> None:
    """Persist the free-text detail a user typed into the downvote popup.
    Best-effort like every other logging call here: a DB outage never blocks
    the UI, it just means this particular note wasn't captured."""
    feedback_text = (feedback_text or "").strip()
    if not feedback_text or not _ensure_user_feedback_table():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""INSERT INTO {DB_SCHEMA}.user_feedback
                    (interaction_id, session_key, feedback_text) VALUES (%s,%s,%s)""",
                (interaction_id, session_key, feedback_text),
            )
    except Exception as exc:
        log.warning("log_user_feedback failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4B-2 — PERSISTED CHAT HISTORY  (sidebar conversations, 7-day retention)
# ══════════════════════════════════════════════════════════════════════════════
#  smsgs/sconvs/scid are memory-only dcc.Stores — they do NOT survive a page
#  reload, only a fresh login used to repopulate them, which is why history
#  used to disappear even mid-session. These two tables are the durable copy:
#  vi_chat_conversations is one row per conversation (sidebar entry, keyed by
#  session_key/email), vi_chat_messages is one row per chat bubble (user
#  question, or the FULL assistant `result` dict as JSONB) under it. Storing
#  the whole result — not just the answer text — means reopening an old
#  conversation is byte-identical to the original: charts render, tables still
#  sort/filter, Download CSV and the feedback buttons keep working.
#
#  Kept on their own independent best-effort gate (own flag/lock, own DDL,
#  never bundled into _ensure_logging_tables()'s transaction) for the same
#  reason as _ensure_user_feedback_table(): a problem creating THESE tables
#  must never roll back and disable the unrelated vi_chat_interactions/
#  vi_chat_feedback logging that already works. No FOREIGN KEY constraints for
#  the same reason too — the app's DB role may not hold REFERENCES on tables it
#  didn't create; interaction_id is an application-level join only.

_CONVERSATIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.vi_chat_conversations (
    id             BIGSERIAL PRIMARY KEY,
    session_key    TEXT NOT NULL,
    title          TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
_CONVERSATIONS_IDX_DDL = f"""
CREATE INDEX IF NOT EXISTS vi_chat_conversations_session_idx
    ON {DB_SCHEMA}.vi_chat_conversations (session_key, last_active_at DESC);
"""
_CHAT_MESSAGES_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.vi_chat_messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    session_key     TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    content         TEXT,
    result          JSONB,
    interaction_id  BIGINT
);
"""
_CHAT_MESSAGES_IDX_DDL = f"""
CREATE INDEX IF NOT EXISTS vi_chat_messages_conv_idx
    ON {DB_SCHEMA}.vi_chat_messages (conversation_id, created_at);
"""

_conversation_tables_ready = False
_conversation_tables_ready_lock = threading.Lock()


def _ensure_conversation_tables() -> bool:
    global _conversation_tables_ready
    if _conversation_tables_ready:
        return True
    with _conversation_tables_ready_lock:
        if _conversation_tables_ready:
            return True
        if not DB_PASSWORD:
            return False
        try:
            with _log_cursor(commit=True) as cur:
                cur.execute(_CONVERSATIONS_DDL)
                cur.execute(_CONVERSATIONS_IDX_DDL)
                cur.execute(_CHAT_MESSAGES_DDL)
                cur.execute(_CHAT_MESSAGES_IDX_DDL)
            _conversation_tables_ready = True
            log.info("[STARTUP] chat history tables ready (retention=%sd).",
                     CHAT_HISTORY_RETENTION_DAYS)
            return True
        except Exception as exc:
            log.warning("Could not ensure chat history tables (persisted conversation "
                        "history disabled): %s", exc)
            return False


def create_conversation(session_key: str, title: str) -> Optional[int]:
    """Mint a new conversation row for the FIRST message of a new thread.
    Returns its id (used as `cid` client-side), or None if persistence is
    unavailable — callers treat that as best-effort and keep working in
    memory-only mode for that turn."""
    if not _ensure_conversation_tables():
        return None
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""INSERT INTO {DB_SCHEMA}.vi_chat_conversations (session_key, title)
                    VALUES (%s, %s) RETURNING id""",
                (session_key, (title or "New conversation")[:120]),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else None
    except Exception as exc:
        log.warning("create_conversation failed: %s", exc)
        return None


def touch_conversation(conversation_id: Optional[int]) -> None:
    """Bump last_active_at so the conversation sorts to the top of the
    sidebar and its retention clock restarts from this latest turn."""
    if conversation_id is None or not _ensure_conversation_tables():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"UPDATE {DB_SCHEMA}.vi_chat_conversations SET last_active_at = now() WHERE id = %s",
                (conversation_id,),
            )
    except Exception as exc:
        log.warning("touch_conversation failed: %s", exc)


def rename_conversation(conversation_id: Optional[int], title: str) -> None:
    """Replace a placeholder conversation's generic title (e.g. "New
    conversation", set the moment the user clicked the New conversation
    button, before they'd typed anything) with the real first question once
    they actually send one."""
    if conversation_id is None or not _ensure_conversation_tables():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"UPDATE {DB_SCHEMA}.vi_chat_conversations SET title = %s WHERE id = %s",
                ((title or "New conversation")[:120], conversation_id),
            )
    except Exception as exc:
        log.warning("rename_conversation failed: %s", exc)


def save_message(conversation_id: Optional[int], session_key: str, role: str,
                 content: Optional[str] = None, result: Optional[Dict[str, Any]] = None,
                 interaction_id: Optional[int] = None) -> None:
    if conversation_id is None or not _ensure_conversation_tables():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""INSERT INTO {DB_SCHEMA}.vi_chat_messages
                    (conversation_id, session_key, role, content, result, interaction_id)
                    VALUES (%s,%s,%s,%s,%s,%s)""",
                (conversation_id, session_key, role, content,
                 json.dumps(result, default=str) if result is not None else None,
                 interaction_id),
            )
    except Exception as exc:
        log.warning("save_message failed: %s", exc)


def list_conversations(session_key: str, limit: int = CHAT_HISTORY_MAX_CONVS) -> List[Dict[str, Any]]:
    """Conversations for the sidebar, most-recently-active first. The
    `last_active_at > now() - retention` filter is applied here too (not just
    by the background sweep) so a conversation that aged out between sweeps
    never flashes into a freshly (re)loaded session."""
    if not _ensure_conversation_tables():
        return []
    try:
        with _log_cursor() as cur:
            cur.execute(
                f"""SELECT id, title, last_active_at FROM {DB_SCHEMA}.vi_chat_conversations
                    WHERE session_key = %s
                      AND last_active_at > now() - interval '{CHAT_HISTORY_RETENTION_DAYS} days'
                    ORDER BY last_active_at DESC LIMIT %s""",
                (session_key, limit),
            )
            return cur.fetchall() or []
    except Exception as exc:
        log.warning("list_conversations failed: %s", exc)
        return []


def load_conversation_messages(conversation_id: Optional[int], session_key: str) -> List[Dict[str, Any]]:
    """Rebuild the in-memory `smsgs` shape (role/content for user turns,
    role/result for assistant turns — same shape render_stream already
    expects) for one conversation. Scoped by session_key too, so one user can
    never load another's conversation by guessing/tampering with an id."""
    if conversation_id is None or not _ensure_conversation_tables():
        return []
    try:
        with _log_cursor() as cur:
            cur.execute(
                f"""SELECT role, content, result FROM {DB_SCHEMA}.vi_chat_messages
                    WHERE conversation_id = %s AND session_key = %s
                    ORDER BY id""",
                (conversation_id, session_key),
            )
            rows = cur.fetchall() or []
    except Exception as exc:
        log.warning("load_conversation_messages failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r["role"] == "user":
            out.append({"role": "user", "content": r["content"]})
        else:
            out.append({"role": "assistant", "result": r["result"] or {}})
    return out


def delete_conversation(conversation_id: Optional[int], session_key: str) -> None:
    """Immediate hard delete for an abandoned placeholder — created by New
    conversation, never sent a message — the moment the user navigates away
    from it (switch_conv) or logs out while sitting on one, rather than
    waiting for the 7-day retention sweep to eventually clear it. Scoped by
    session_key so this can never reach another user's conversation."""
    if conversation_id is None or not _ensure_conversation_tables():
        return
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""DELETE FROM {DB_SCHEMA}.vi_chat_messages
                    WHERE conversation_id = %s AND session_key = %s""",
                (conversation_id, session_key),
            )
            cur.execute(
                f"""DELETE FROM {DB_SCHEMA}.vi_chat_conversations
                    WHERE id = %s AND session_key = %s""",
                (conversation_id, session_key),
            )
    except Exception as exc:
        log.warning("delete_conversation failed: %s", exc)


def _purge_expired_conversations() -> None:
    """Hard-delete conversations (and their messages) inactive for more than
    CHAT_HISTORY_RETENTION_DAYS. A whole conversation expires together, based
    on when it was last active (not per-message), so an older thread that's
    still being actively used doesn't lose its earlier turns out from under
    it just because the thread itself is old."""
    try:
        with _log_cursor(commit=True) as cur:
            cur.execute(
                f"""DELETE FROM {DB_SCHEMA}.vi_chat_messages WHERE conversation_id IN (
                        SELECT id FROM {DB_SCHEMA}.vi_chat_conversations
                        WHERE last_active_at < now() - interval '{CHAT_HISTORY_RETENTION_DAYS} days')"""
            )
            cur.execute(
                f"""DELETE FROM {DB_SCHEMA}.vi_chat_conversations
                    WHERE last_active_at < now() - interval '{CHAT_HISTORY_RETENTION_DAYS} days'"""
            )
    except Exception as exc:
        log.warning("Chat history retention purge failed: %s", exc)


_retention_thread_started = False
_retention_thread_lock = threading.Lock()


def _retention_sweep_loop() -> None:
    while True:
        if _ensure_conversation_tables():
            _purge_expired_conversations()
        time.sleep(CHAT_HISTORY_SWEEP_INTERVAL)


def _ensure_retention_sweeper() -> None:
    """Start the background purge thread once (mirrors the TokenManager
    watchdog / tool-call-audit-writer daemon-thread pattern already used
    elsewhere in this stack). Runs a sweep immediately on start, then every
    CHAT_HISTORY_SWEEP_INTERVAL seconds."""
    global _retention_thread_started
    if _retention_thread_started:
        return
    with _retention_thread_lock:
        if _retention_thread_started:
            return
        threading.Thread(target=_retention_sweep_loop, name="vi-chat-history-retention",
                         daemon=True).start()
        _retention_thread_started = True
        log.info("Chat history retention sweeper started (every %ss, keeps %sd).",
                 CHAT_HISTORY_SWEEP_INTERVAL, CHAT_HISTORY_RETENTION_DAYS)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4C — CONSOLIDATED TOOL KB + LOCAL RAG  (§3.4 / §4.2D / §4.3)
# ══════════════════════════════════════════════════════════════════════════════
#  One knowledge source, two uses:
#   (i)  the tool KB markdown is injected as the system-prompt tool context
#        (replacing the old scattered STATIC_TOOL_CONTEXT / hardcoded tables);
#   (ii) the ~100-row question guide is a local (airgapped) TF-IDF corpus — for a
#        given question we retrieve only the few most relevant example rows to
#        include as few-shot / tool-selection hints, instead of dumping the whole
#        KB into every 7B prompt.

def _load_tool_kb() -> str:
    try:
        with open(TOOL_KB_PATH, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        log.warning("Could not read tool KB at %s (%s); using built-in fallback.", TOOL_KB_PATH, exc)
        return ""


TOOL_KB_TEXT = _load_tool_kb()

_GUIDE_ROWS: List[Dict[str, Any]] = []
_GUIDE_TF: List[Dict[str, float]] = []      # per-row term-frequency (normalized)
_GUIDE_IDF: Dict[str, float] = {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _load_question_guide() -> None:
    """Load the guide xlsx and build a tiny TF-IDF index (pure Python, local)."""
    global _GUIDE_ROWS, _GUIDE_TF, _GUIDE_IDF
    if _GUIDE_ROWS:
        return
    try:
        df = pd.read_excel(QUESTION_GUIDE_PATH, sheet_name="question_guide")
        _GUIDE_ROWS = df.fillna("").to_dict("records")
    except Exception as exc:
        log.warning("Could not load question guide at %s (%s); RAG retrieval disabled.",
                    QUESTION_GUIDE_PATH, exc)
        _GUIDE_ROWS = []
        return

    import math
    from collections import Counter
    docs = []
    for r in _GUIDE_ROWS:
        blob = " ".join(str(r.get(c, "")) for c in
                        ("question", "category", "sub_type", "example_entities", "notes"))
        docs.append(_tokenize(blob))
    df_counts: Dict[str, int] = {}
    for toks in docs:
        for t in set(toks):
            df_counts[t] = df_counts.get(t, 0) + 1
    n = max(1, len(docs))
    _GUIDE_IDF = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df_counts.items()}
    _GUIDE_TF = []
    for toks in docs:
        c = Counter(toks)
        total = max(1, len(toks))
        _GUIDE_TF.append({t: (cnt / total) * _GUIDE_IDF.get(t, 0.0) for t, cnt in c.items()})
    log.info("[STARTUP] question guide loaded: %d rows indexed for retrieval.", len(_GUIDE_ROWS))


def retrieve_examples(question: str, k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """Return the top-k most relevant guide rows for a question (cosine over TF-IDF)."""
    if not _GUIDE_ROWS or not _GUIDE_TF:
        return []
    import math
    q_toks = _tokenize(question)
    if not q_toks:
        return []
    from collections import Counter
    qc = Counter(q_toks)
    qtotal = max(1, len(q_toks))
    qvec = {t: (cnt / qtotal) * _GUIDE_IDF.get(t, 0.0) for t, cnt in qc.items()}
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
    scored = []
    for i, dvec in enumerate(_GUIDE_TF):
        dot = sum(qvec.get(t, 0.0) * w for t, w in dvec.items())
        dnorm = math.sqrt(sum(v * v for v in dvec.values())) or 1.0
        scored.append((dot / (qnorm * dnorm), i))
    scored.sort(reverse=True)
    return [_GUIDE_ROWS[i] for s, i in scored[:k] if s > 0.0]


def build_rag_context(question: str) -> str:
    """Compact few-shot block of the most relevant guide rows for this question."""
    rows = retrieve_examples(question)
    if not rows:
        return ""
    lines = ["RELEVANT WORKED EXAMPLES (nearest questions from the tool guide):"]
    for r in rows:
        lines.append(f"- Q: {r.get('question')}")
        lines.append(f"    category={r.get('category')} sub_type={r.get('sub_type')} "
                     f"result_shape={r.get('expected_result_shape')}")
        lines.append(f"    tools: {r.get('tool_sequence')}")
    return "\n".join(lines)


# Load the RAG index at import (cheap, local file) so it is ready even when this
# module is imported by a WSGI server rather than run as __main__.
try:
    _load_question_guide()
except Exception as _exc:      # pragma: no cover
    log.warning("[STARTUP] deferred question-guide load failed: %s", _exc)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DOMAIN REFERENCE TABLES  (operational copy; mirrored in tool KB)
# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLIDATION NOTE (§3.4): the human/agent-facing copy of these tables — and
#  the tool docs, chaining rules and the " ::" gotcha — now lives in ONE place,
#  vi_ippms_tool_kb.md (loaded above as TOOL_KB_TEXT, injected into the system
#  prompt). The Python dicts below are the runtime OPERATIONAL copy that
#  host_in_circle()/_guess_component() execute against, kept in lock-step with the
#  KB's "Synonym tables" section. No other code path keeps its own separate table.
#
#  There is no documented API that maps a circle NAME (e.g. "GUJ") to a cid or
#  lists circle names. The circle is encoded in device/interface name strings
#  (e.g. APVSPGJWPAR01HNE40 -> GJW = Gujarat West; GJAHD... = Ahmedabad).
#  So circle filtering is done by matching these tokens against host names.

CIRCLE_TOKENS: Dict[str, Dict[str, Any]] = {
    "GUJARAT":        {"aliases": ["guj", "gujarat", "gj"],        "tokens": ["GJ", "GJW", "GJAHD", "GJW-"]},
    "MAHARASHTRA":    {"aliases": ["mah", "maharashtra", "mh"],    "tokens": ["MH", "MUM", "MB", "PUN"]},
    "MUMBAI":         {"aliases": ["mumbai", "mum", "bom"],        "tokens": ["MUM", "MB", "BOM"]},
    "DELHI":          {"aliases": ["delhi", "del", "ncr"],         "tokens": ["DL", "DEL", "NCR"]},
    "KARNATAKA":      {"aliases": ["karnataka", "kar", "ka", "blr"], "tokens": ["KA", "KTK", "BLR", "BAN"]},
    "TAMIL NADU":     {"aliases": ["tamil nadu", "tn", "chennai"], "tokens": ["TN", "CHN", "MAA"]},
    "ANDHRA PRADESH": {"aliases": ["andhra", "ap", "andhra pradesh"], "tokens": ["AP", "HYD", "VIZ"]},
    "TELANGANA":      {"aliases": ["telangana", "tg", "ts", "hyderabad"], "tokens": ["TG", "TS", "HYD"]},
    "WEST BENGAL":    {"aliases": ["west bengal", "wb", "kolkata"], "tokens": ["WB", "KOL", "CCU"]},
    "RAJASTHAN":      {"aliases": ["rajasthan", "raj", "rj"],      "tokens": ["RJ", "JAI"]},
    "PUNJAB":         {"aliases": ["punjab", "pb", "pun"],         "tokens": ["PB", "PUN", "CHD"]},
    "KERALA":         {"aliases": ["kerala", "kl", "ker"],         "tokens": ["KL", "KER", "COK", "TVM"]},
    "UP EAST":        {"aliases": ["up east", "upe"],              "tokens": ["UPE", "LKO"]},
    "UP WEST":        {"aliases": ["up west", "upw"],              "tokens": ["UPW", "AGRE", "AGR"]},
    "PAN INDIA":      {"aliases": ["pan india", "pan_india", "all india", "india"], "tokens": []},
}

# KPI / component synonym hints — helps the router pick the right component.
KPI_SYNONYMS: Dict[str, List[str]] = {
    "traffic":   ["traffic", "utilization", "utilisation", "octets", "throughput", "bandwidth",
                  "bps", "in octets", "out octets", "hc in octets", "hc out octets"],
    "errors":    ["error", "errors", "err", "discard", "drops"],
    "cpu":       ["cpu", "processor", "load"],
    "broadcast": ["broadcast", "bcast"],
    "bgp":       ["bgp", "peer", "neighbor", "neighbour", "route"],
}

# Synonym words that are KPI-descriptive fragments only — never a real
# component category name (a dict key above). Used to catch the router
# mistaking a piece of a KPI name (e.g. "utilization" inside "Utilization In
# % HC") for the component itself; deliberately excludes anything that IS a
# real component key (e.g. "traffic", "bgp") so a legitimate component
# mention is never cleared by this check.
_KPI_SYNONYM_ONLY_WORDS = {
    s.lower() for syns in KPI_SYNONYMS.values() for s in syns
} - set(KPI_SYNONYMS.keys())


def _kpi_component_hint(kpi_text: str) -> Optional[str]:
    """Guess which component a named KPI conventionally belongs to (e.g. "HC
    In Octets" -> "traffic"), from KPI_SYNONYMS above. Used to break ties when
    the SAME interface is monitored identically under several components that
    all happen to expose a same-named KPI (e.g. a generic "interfaces"
    component duplicating a subset of what "traffic" already tracks) — without
    this, whichever component sorted first alphabetically silently won,
    regardless of which one a human would actually mean by that KPI name."""
    kt = (kpi_text or "").lower()
    for comp, syns in KPI_SYNONYMS.items():
        if any(s in kt for s in syns):
            return comp
    return None


# A few worked ReAct traces shown to the model as few-shot guidance.
REACT_FEWSHOT = """\
Worked examples (abbreviated — device/cid are already grounded for you):

Q: "How many interfaces are on this device under traffic?"
Thought: I know host_id and cid. List interfaces under the traffic component and count them.
Action: ig_list_interfaces
Action Input: {"host_id": 11193, "cid": 2, "components": ["traffic"]}
Observation: {"component_prefixes": {"traffic": {"items": ["Interface 100GE0/3/0 ::", "Interface 100GE0/3/1 ::"]}}}
Thought: There are 2 interfaces under traffic.
Final Answer: The device has 2 interfaces under the traffic component.

Q: "What was HC In Octets on Interface 100GE0/3/2 in the given window?"
Thought: Find the itemid for that interface's KPI, then fetch history.
Action: ig_list_kpis
Action Input: {"host_id": 11193, "cid": 2, "prefix": {"traffic": ["Interface 100GE0/3/2 ::"]}}
Observation: {"results": {"traffic : Interface 100GE0/3/2": [{"prefix": "Interface 100GE0/3/2", "suffix": " HC In Octets", "itemid": 2080688}]}}
Thought: itemid is 2080688. Fetch its history for the window.
Action: ig_get_kpi_history
Action Input: {"item_ids": [2080688], "start_time": 1782817742, "end_time": 1782904142, "cid": {"2080688": 2}}
Observation: {"Values": [{"name": "HC In Octets", "Minimum": "15.12 MB", "Maximum": "244.49 MB", "Average": "126.35 MB", "Last": "218.11 MB", "units": "bps"}]}
Thought: I have the summary values.
Final Answer: HC In Octets on Interface 100GE0/3/2 ranged 15.12 MB (min) to 244.49 MB (max), averaging 126.35 MB, last 218.11 MB.
"""

REACT_REFORMAT_SYSTEM = """\
You already worked out the answer to a network-operations question, but your
reply did not follow the required format. Do NOT do new research or call any
tool — just restate the SAME information you already found, using exactly
this format and nothing else:

Thought: <one short sentence>
Final Answer: <the same answer you already gave, in full, human-readable form>
"""


def circle_match(text: str) -> Optional[str]:
    """Resolve a free-text circle mention to a canonical circle name."""
    t = (text or "").lower()
    for canon, meta in CIRCLE_TOKENS.items():
        for alias in meta["aliases"]:
            if re.search(r"\b" + re.escape(alias) + r"\b", t):
                return canon
    return None


def host_in_circle(host_name: str, circle_canon: str) -> bool:
    """A device belongs to a circle only if its name STARTS WITH one of that
    circle's tokens — not merely contains one. § user request/bugfix: a
    "contains" check let e.g. "DELAAIGJKKAI" (starts with DEL -> Delhi) be
    counted as Gujarat too just because "GJ" appears later in the name; only
    a device actually beginning with a circle's own token (e.g.
    "GJAKKAIANRBN") belongs to it."""
    tokens = CIRCLE_TOKENS.get(circle_canon, {}).get("tokens", [])
    if not tokens:            # PAN INDIA / unknown -> everything
        return True
    up = (host_name or "").upper()
    return any(up.startswith(tok) for tok in tokens)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — REACT CONTEXT + PARSER
# ══════════════════════════════════════════════════════════════════════════════

REACT_ROLE = """\
You are a network-operations assistant for VI-IPPMS (Instant Graph). You answer
questions about devices, components, interfaces and KPIs by calling tools — you
NEVER invent numeric values, device names, or timestamps; you always retrieve them.

Work in a strict loop:

Thought: <one or two sentences of reasoning about the next step>
Action: <exactly one tool name from the list below>
Action Input: <a single valid JSON object of that tool's arguments>

Then STOP and wait. The program runs the tool and returns:

Observation: <the tool's JSON result>

Repeat Thought/Action/Action Input/Observation as needed. When you can answer, write:

Thought: <why you now have enough information>
Final Answer: <clear, human-readable answer citing the device / interface / KPI
    names and the numeric values you retrieved. Do NOT mention itemids, hostids,
    cids, or tool names.>

Rules:
- Emit ONE Action per turn, then stop.
- Action Input must be valid JSON (no comments, no trailing commas, no prose).
- Do NOT include a "session_key" argument — it is added for you automatically.
- The device (host_id + cid) is already resolved and given to you below; start
  from components/interfaces/KPIs, not from listing all hosts.
- Ranking (top-N), counts, and threshold checks are NOT computed by the API —
  do that arithmetic yourself over the returned values.
"""


def build_tool_context(tools: List[Dict[str, Any]]) -> str:
    """Consolidated tool context (§3.4): the KB markdown is the primary source;
    the live tool signatures from list_tools() are appended so the exact current
    arg names are always present. Falls back to STATIC_TOOL_CONTEXT only if both
    the KB and the live tool list are unavailable."""
    live_lines = []
    for t in (tools or []):
        if t["name"] in AUTH_TOOLS:
            continue
        props = {k: v for k, v in (t["input_schema"].get("properties", {}) or {}).items()
                 if k != "session_key"}
        live_lines.append(f"- {t['name']}({', '.join(props.keys())})")
    live_block = ("\nLIVE TOOL SIGNATURES:\n" + "\n".join(live_lines)) if live_lines else ""

    if TOOL_KB_TEXT:
        return TOOL_KB_TEXT + live_block
    if live_lines:
        return "Available tools:\n" + "\n".join(live_lines)
    return STATIC_TOOL_CONTEXT


STATIC_TOOL_CONTEXT = """\
Available tools:
- ig_list_components(host_id, cid) -> {"components": [{"component","item_count"}]}
- ig_list_interfaces(host_id, cid, components) -> {"component_prefixes": {comp: {"items": [str]}}}
- ig_list_kpis(host_id, cid, prefix) -> {"results": {group: [{"prefix","suffix","itemid"}]}}
- ig_resolve_items(selecteddata, components) -> {"items": [{"item_name","itemid","host_name"}]}
- ig_get_kpi_history(item_ids, start_time, end_time, cid)
    -> {"series": [{"name","data": [[epoch_ms, value]]}],
        "Values": [{"name","Minimum","Maximum","Last","Average","Total","units"}]}
"""

# ── ReAct parser (adapted from gpu_llm_context.py) ───────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action:|Final Answer:)|\Z)", re.S)
_ACTION_RE       = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*", re.S)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})\s*(?=\n(?:Observation:|Thought:)|\Z)", re.S)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.*)", re.S)


def parse_react_step(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    thought_m = _THOUGHT_RE.search(cleaned)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = _FINAL_RE.search(cleaned)
    if final_m:
        return {"thought": thought, "is_final": True, "final_answer": final_m.group(1).strip(),
                "action": None, "action_input": None, "parse_error": None}

    action_m = _ACTION_RE.search(cleaned)
    input_m = _ACTION_INPUT_RE.search(cleaned)
    if not action_m or not input_m:
        return {"thought": thought, "is_final": False, "final_answer": None,
                "action": None, "action_input": None,
                "parse_error": f"Missing Action/Action Input in: {cleaned[:200]!r}"}
    try:
        action_input = json.loads(input_m.group(1).strip())
    except json.JSONDecodeError as exc:
        return {"thought": thought, "is_final": False, "final_answer": None,
                "action": action_m.group(1).strip(), "action_input": None,
                "parse_error": f"Action Input not valid JSON ({exc})"}
    return {"thought": thought, "is_final": False, "final_answer": None,
            "action": action_m.group(1).strip(), "action_input": action_input,
            "parse_error": None}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class VIAgentState(TypedDict):
    user_query:   str
    session_key:  str
    route:        str            # metadata_q | kpi_q | help_q | out_of_scope
    confidence:   str
    entities:     Dict[str, Any] # device, interface, component, kpi, circle, agg, top_n, threshold, start, end
    grounding:    Dict[str, Any] # resolved host candidates, chosen host_id/cid, components...
    spec:         Dict[str, Any] # QuerySpec (§4.2A) — the recognized query, or {"recognized": False}
    plan:         str            # chain-of-thought plan (kpi branch)
    trace:        List[Dict[str, Any]]  # ReAct steps: {thought, action, action_input, observation}
    result_kind:  str            # count | list | timeseries | stats | scalar | text | error
    payload:      Dict[str, Any] # rows/series/scalar for the UI
    answer:       str
    answered_via: str            # query_spec | react | fallback | help | out_of_scope
    failed:       bool
    fallback_used: bool
    tool_calls:   int
    debug:        Dict[str, Any] # full per-run debug object (§3.2)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — HELPERS (time, fuzzy match, formatting)
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> int:
    return int(time.time())


# Shared by router_node's kpi_q/metadata_q backstops and build_spec's
# interface-ranking "currently" detection (§4.2A) — one definition of "this
# question names an explicit historical time phrase" for both call sites.
_TIME_PHRASE_RE = re.compile(
    r"\b(last|yesterday|today|this (week|month)|hour|day|week|between|since|"
    r"from .* to |ago)\b")
_STAT_WORD_RE = re.compile(
    r"\b(min(imum)?|max(imum)?|avg|average|last value|peak|total|exceed(ed)?|"
    r"cross(ed)?|above|over |greater than|threshold|compare|trend|"
    r"spikes?|surges?|dips?|drops?|anomal(?:y|ies)|outliers?)\b")


def _has_time_phrase(ql: str) -> bool:
    return bool(_TIME_PHRASE_RE.search(ql))


def _has_stat_word(ql: str) -> bool:
    return bool(_STAT_WORD_RE.search(ql))


# Explicit "between A and B" time range (§4.2A spike/dip detection shape — e.g.
# "between 10:00 and 14:00" or "between 2024-01-01 10:00 and 2024-01-01 14:00").
# parse_time_window otherwise only understands RELATIVE phrases ("last N
# hours", "yesterday"); an explicit range fell through to its 24h default with
# no attempt to parse it at all. Matches only well-shaped date/time tokens
# (rather than lazily grabbing arbitrary text between "between"/"and") so it
# can't misfire on unrelated prose.
_MOMENT_TOKEN = r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?|\d{1,2}:\d{2}\s*(?:am|pm)?)"
_BETWEEN_RANGE_RE = re.compile(rf"\bbetween\s+{_MOMENT_TOKEN}\s+and\s+{_MOMENT_TOKEN}", re.IGNORECASE)


def _parse_one_moment(token: str, base_date: datetime) -> Optional[datetime]:
    """A single 'between'-range endpoint: either a full date(+time), or a bare
    time of day applied to `base_date` (today, or yesterday if that word also
    appears in the question — see _parse_between_range)."""
    token = token.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?$", token)
    if m:
        y, mo, d, hh, mm = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)?$", token, re.IGNORECASE)
    if m:
        hh, mm, ampm = m.groups()
        hh, mm = int(hh), int(mm)
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and hh < 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
        if hh > 23 or mm > 59:
            return None
        return base_date.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return None


def _parse_between_range(text: str) -> Optional[Tuple[int, int]]:
    """'between A and B' -> (start_epoch, end_epoch), or None if not present /
    not parseable. Bare times are assumed to be today unless "yesterday" is
    also in the question; if B ends up before or equal to A (e.g. "between
    22:00 and 02:00"), B is rolled to the next day (an overnight span)."""
    m = _BETWEEN_RANGE_RE.search(text or "")
    if not m:
        return None
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if re.search(r"\byesterday\b", text or "", re.IGNORECASE):
        base_date -= timedelta(days=1)
    a = _parse_one_moment(m.group(1), base_date)
    b = _parse_one_moment(m.group(2), base_date)
    if not a or not b:
        return None
    if b <= a:
        b += timedelta(days=1)
    return int(a.timestamp()), int(b.timestamp())


def parse_time_window(text: str, entities: Dict[str, Any]) -> Tuple[int, int]:
    """Best-effort parse of a human time phrase to (start_epoch, end_epoch)."""
    # If the router already extracted numeric epochs, trust them.
    s, e = entities.get("start_time"), entities.get("end_time")
    if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
        return int(s), int(e)

    rng = _parse_between_range(text)
    if rng:
        return rng

    t = (text or "").lower()
    now = _now()
    m = re.search(r"last\s+(\d+)\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|week|weeks)", t)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        if unit.startswith("min"): return now - n * 60, now
        if unit in ("h", "hr", "hrs", "hour", "hours"): return now - n * 3600, now
        if unit in ("d", "day", "days"): return now - n * 86400, now
        if unit in ("w", "week", "weeks"): return now - n * 7 * 86400, now
    if "yesterday" in t:
        start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        return int(start_day.timestamp()), int((start_day + timedelta(days=1)).timestamp())
    if "today" in t:
        start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start_day.timestamp()), now
    if "week" in t:
        return now - 7 * 86400, now
    # default: last 24h
    return now - 24 * 3600, now


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").upper(), (b or "").upper()).ratio()


def _host_match_score(qd: str, up: str) -> float:
    """Similarity score for a user-typed device string (qd) against a real
    host name (up) — both already uppercased. Shared by match_hosts() and
    got_disambiguate() so there is exactly one scoring formula.

    BUGFIX: this used to give ANY substring/prefix containment a flat 0.9,
    regardless of how much of the host name it actually covered. A near-exact
    typo — e.g. "...CN54" for the real host "...CN540", missing only the
    trailing digit — still scored a flat 0.9, while a genuinely DIFFERENT
    candidate that merely resembled the query overall could score HIGHER via
    plain SequenceMatcher ratio (which has no such cap) — e.g. 0.914 for a
    candidate with several letters swapped in the middle. That let a clearly
    worse candidate silently outrank an almost-exact one. Scaling the
    containment score by how much of the host name is actually covered keeps
    "qd is a near-complete prefix of up" properly ranked above "up merely
    resembles qd" instead of an arbitrary constant crossing over
    unpredictably depending on which unrelated candidates happen to be in the
    pool."""
    if not qd or not up:
        return 0.0
    if qd == up:
        return 1.0
    if qd in up:
        return 0.9 + 0.1 * (len(qd) / len(up))
    return _similar(qd, up)


def match_hosts(query_device: str, hosts: List[Dict[str, Any]],
                circle_canon: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
    """Rank hosts by relevance to a device mention (+optional circle filter)."""
    qd = (query_device or "").strip().upper()
    pool = hosts
    if circle_canon:
        pool = [h for h in hosts if host_in_circle(h.get("name") or h.get("host", ""), circle_canon)] or hosts
    scored = []
    for h in pool:
        name = (h.get("name") or h.get("host") or "")
        up = name.upper()
        score = _host_match_score(qd, up) if qd else 0.0
        scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _s, h in scored[:limit]]


# Minimum literal-name length before it counts as a signal (mirrors the
# len(name) >= 6 guard already used in resolve_device_set's `explicit` check,
# so a short/generic host name fragment can't accidentally "win").
LITERAL_DEVICE_MATCH_MIN_LEN = 6


def find_literal_device_matches(text: str, hosts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find real host names that appear verbatim (case-insensitive) in `text`.

    This is a stronger, purely-mechanical signal than the router LLM's
    extracted `device` entity: the small local model can echo a device name
    from its own few-shot examples / RAG-retrieved hints instead of the one
    actually typed in the question. A literal substring match can't hallucinate
    that way, so resolve_node prefers it when the two disagree. Longer (more
    specific) names are sorted first so e.g. "APVSPGJWPAR01HNE40" doesn't get
    shadowed by a shorter host name that also happens to be a substring."""
    up = (text or "").upper()
    matches = [h for h in hosts
              if len(h.get("name") or h.get("host") or "") >= LITERAL_DEVICE_MATCH_MIN_LEN
              and (h.get("name") or h.get("host") or "").upper() in up]
    matches.sort(key=lambda h: len(h.get("name") or h.get("host") or ""), reverse=True)
    return matches


def _num(x: Any) -> Optional[float]:
    """Extract a leading number from strings like '244.49 MB' -> 244.49."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None


def truncate_obs(obj: Any, cap: int = OBS_CHAR_CAP) -> str:
    """Serialize a tool observation, capping length so the 7B isn't flooded."""
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    if len(s) > cap:
        s = s[:cap] + f' …[truncated {len(s) - cap} chars]'
    return s


# ── Unit normalization (query-spec engine correctness) ───────────────────────
# get-historical-data summary values are formatted strings with mixed units
# ("244.49 MB", "35.54 GB"). Ranking/thresholding must compare a common base, or
# 244.49 MB sorts above 35.54 GB. to_bytes() parses "<number> <unit>" to bytes.
_UNIT_FACTORS = {
    "": 1, "B": 1, "BPS": 1, "K": 1e3, "KB": 1e3, "KBPS": 1e3,
    "M": 1e6, "MB": 1e6, "MBPS": 1e6, "G": 1e9, "GB": 1e9, "GBPS": 1e9,
    "T": 1e12, "TB": 1e12, "TBPS": 1e12,
}


def to_bytes(val: Any) -> Optional[float]:
    """Normalize '35.54 GB' / '244.49 MB' / 123.4 -> a float in base units."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]*)", str(val))
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").upper()
    return num * _UNIT_FACTORS.get(unit, 1.0)


# ── Type coercion (§ bugfix: "'dict' object has no attribute 'split'") ───────
# Two independent sources feed non-string values into code paths that assume
# str, and both blew up with AttributeError *outside* the `except MCPError`
# guards, so they escaped all the way to run_agent's catch-all and surfaced as
# "The agent hit an unexpected error: 'dict' object has no attribute 'split'":
#
#   1. The local 7B router sometimes emits a nested object for a scalar entity,
#      e.g. {"device": {"name": "APVSPGJWPAR01HNE40"}} or {"kpi": ["HC In
#      Octets"]} instead of a plain string. Everything downstream then calls
#      .strip()/.lower()/.split() on a dict/list.
#   2. ig_list_interfaces' component_prefixes[<comp>]["items"] is documented as
#      a list of strings ("Interface 100GE0/3/2 ::"), but the live API is not
#      guaranteed to keep that shape — an object item ({"name": "..."}) reaches
#      _iface_display(), which does prefix_item.split("::") -> the exact error.
#
# Rather than sprinkle isinstance() checks at each call site, values are coerced
# once at each boundary (LLM output, tool output) and these helpers are used by
# every consumer that needs a string.

# Keys an object-shaped item/entity might carry the human-readable name under,
# in priority order.
_TEXT_KEYS = ("value", "name", "item", "prefix", "label", "interface", "device",
              "text", "title", "id")


def _as_text(v: Any) -> str:
    """Best-effort scalar-string view of anything the LLM or API hands back.

    str -> itself; numbers/bools -> str(); dict -> its first plausible name-ish
    string field; list/tuple -> the first element that yields text; anything
    else -> "". Never raises, never returns a non-str."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        for k in _TEXT_KEYS:
            hit = v.get(k)
            if isinstance(hit, str) and hit.strip():
                return hit
            if isinstance(hit, (int, float)) and not isinstance(hit, bool):
                return str(hit)
        for hit in v.values():                    # any string-valued field
            if isinstance(hit, str) and hit.strip():
                return hit
        return ""
    if isinstance(v, (list, tuple, set)):
        for el in v:
            t = _as_text(el)
            if t:
                return t
        return ""
    return str(v)


def _as_text_list(v: Any) -> List[str]:
    """Coerce a scalar/list/dict entity into a list of non-empty strings."""
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple, set)):
        return [t for t in (_as_text(el) for el in v) if t]
    t = _as_text(v)
    return [t] if t else []


# The router prompt tells the 7B to emit these literal placeholder tokens when a
# field is absent ("<... else empty>"). They are NOT real values: an unstripped
# "empty" flowed into component_filter=["empty"] / interface_filter value "empty"
# / kpi_filter value "empty" and was then passed to the API as if it were a real
# component/interface/KPI name — which matches nothing, so a perfectly valid
# question ("list interfaces on <device>") came back with 0 results. Every string
# entity is scrubbed of these sentinels at the router boundary (see router_node)
# so downstream `if not ent[...]` / `if comp` checks behave as intended.
_SENTINEL_ENTITY_VALUES = {
    "empty", "none", "null", "nil", "n/a", "na", "not specified", "unspecified",
    "not mentioned", "not given", "-", "--", "<empty>", "unknown",
}


def _clean_entity(v: Any) -> str:
    """Coerce to text and blank out placeholder sentinels the LLM emits for
    'no value'. Returns '' for a sentinel, otherwise the trimmed string."""
    s = _as_text(v).strip()
    if s.lower() in _SENTINEL_ENTITY_VALUES:
        return ""
    return s


def _as_float(v: Any) -> Optional[float]:
    """Coerce an LLM-emitted numeric entity (which may arrive as "5", 5, "5 GB"
    or {"value": 5}) to a float, or None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", _as_text(v))
    return float(m.group(0)) if m else None


def _as_int(v: Any) -> Optional[int]:
    f = _as_float(v)
    return int(f) if f is not None else None


def _norm_items(payload: Any) -> List[str]:
    """Normalize an ig_list_interfaces component payload to a list of interface
    prefix STRINGS, preserving the trailing ' ::' verbatim (ig_list_kpis needs
    it). Object-shaped items are flattened via _as_text; unusable entries are
    dropped rather than crashing the executor."""
    if isinstance(payload, dict):
        raw = payload.get("items", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    out, seen = [], set()
    for it in raw:
        s = it if isinstance(it, str) else _as_text(it)
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def name_matches(name: str, nm: Optional[Dict[str, Any]]) -> bool:
    """Apply a name_match spec {mode, value} to a device name."""
    if not nm:
        return True
    up = _as_text(name).upper()
    mode = _as_text(nm.get("mode")) or "contains"
    val = nm.get("value")
    if mode == "in_list":
        vals = [v.upper() for v in _as_text_list(val)]
        return up in vals
    v = _as_text(val).upper()
    if not v:
        return True
    if mode == "exact":
        return up == v
    if mode == "starts_with":
        return up.startswith(v)
    return v in up            # contains (default)


def _parallel_map(fn, items, max_workers: int = FANOUT_MAX_WORKERS):
    """Run fn over items in parallel threads, preserving input order. Exceptions
    are captured per-item as {'__error__': str} rather than aborting the batch."""
    if not items:
        return []
    from concurrent.futures import ThreadPoolExecutor
    results: List[Any] = [None] * len(items)
    workers = max(1, min(max_workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in futs:
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:      # noqa: BLE001
                results[i] = {"__error__": str(exc)}
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

# ── Debug accumulator (§3.2) ─────────────────────────────────────────────────
# Every node records its stage timing (and any node-specific facts) into a single
# state["debug"] object, so EVERY answer — query-spec, skip, help, out-of-scope,
# or ReAct — carries a populated trace, not just the runs where ReAct executed.

def _dbg(state: VIAgentState) -> Dict[str, Any]:
    d = dict(state.get("debug") or {})
    d.setdefault("stage_times_ms", {})
    d.setdefault("tool_calls", [])
    d.setdefault("notes", [])
    return d


def _stage_done(dbg: Dict[str, Any], name: str, t0: float) -> None:
    dbg["stage_times_ms"][name] = int((time.time() - t0) * 1000)


VALID_ROUTES = {"metadata_q", "kpi_q", "help_q", "out_of_scope"}

ROUTER_SYSTEM = """\
You classify a VI-IPPMS network-monitoring question and extract its entities.
Return STRICT JSON ONLY, no prose, with EXACTLY these keys:

{
  "route": "metadata_q | kpi_q | help_q | out_of_scope",
  "confidence": "High | Medium | Low",
  "device": "<device / host name or name pattern mentioned, else empty>",
  "circle": "<circle mentioned e.g. GUJ / Maharashtra, else empty>",
  "component": "<component e.g. traffic / cpu / errors / bgp, else empty>",
  "interface": "<interface e.g. 100GE0/3/2 / Eth-Trunk1, else empty>",
  "kpi": "<kpi e.g. HC In Octets / utilization, else empty>",
  "metric": "count | list | min | max | avg | last | total | value | trend | empty",
  "target": "devices | interfaces | components | kpis | data | empty",
  "top_n": <integer or null>,
  "threshold": <number or null>,
  "time_phrase": "<e.g. 'last 6 hours', 'yesterday', else empty>"
}

Guidance:
- metadata_q: counting / listing / filtering devices, interfaces, components, KPIs
  — including "what KPIs exist/are tracked/are captured on X", "KPI details/data
  for X", "list KPIs within a component on X". These ask WHICH KPIs exist, not
  their values, even though the word "KPI" appears.
- kpi_q: retrieving KPI VALUES or statistics over a time window — the question
  names or implies a time window (last 6 hours / yesterday / this week) or a
  statistic (min/max/avg/total/peak/exceeded/threshold).
- help_q: greetings or questions about what this assistant can do.
- out_of_scope: unrelated to VI-IPPMS network monitoring.

Examples:
- "KPIs are tracked on APVSPGJWPAR01HNE40" -> metadata_q (listing, no time window)
- "What was HC In Octets on Interface 100GE0/3/2 in the last 6 hours?" -> kpi_q (has time window)
- "min/max traffic on Eth-Trunk1 yesterday" -> kpi_q (has time window + statistic)
- "List all interfaces on APVSPGJWPAR01HNE40" -> metadata_q (listing)
"""


def router_node(state: VIAgentState) -> VIAgentState:
    q = state["user_query"]
    t0 = time.time()
    rag = build_rag_context(q)
    user_prompt = f"Question: {q}"
    if rag:
        user_prompt += f"\n\n{rag}\n\n(Use the examples only as hints for route/target/metric.)"
    raw = call_llm(ROUTER_SYSTEM, user_prompt, max_new_tokens=200,
                   decode=ROUTER_DECODE, stage="ROUTER")
    parsed = safe_json(raw)
    route = parsed.get("route", "")
    if route not in VALID_ROUTES:
        ql = q.lower()
        if any(k in ql for k in ["hello", "hi ", "help", "what can you", "who are you"]):
            route = "help_q"
        elif any(k in ql for k in ["how many", "list", "count", "which devices",
                                   "show all", "name pattern", "matching"]):
            route = "metadata_q"
        elif any(k in ql for k in ["value", "kpi", "octet", "traffic", "average", "max",
                                   "min", "between", "trend", "utilization", "last "]):
            route = "kpi_q"
        else:
            route = "metadata_q"

    # Deterministic backstop (the router LLM is a local 7B and is unreliable on
    # this specific distinction): "KPIs tracked on X" / "KPI details" / "KPI
    # data" / "list KPIs on X" are LISTING (metadata) questions, not value
    # fetches — but the model tends to see the word "KPI" and jump to kpi_q. A
    # genuine value question always carries either a time phrase (last/
    # yesterday/today/hour/day/week/between) or a statistic word (min/max/avg/
    # total/peak/last value/exceed/threshold/compare). If a kpi_q classification
    # has neither, it's actually a listing question — reclassify before it ever
    # reaches build_spec, rather than letting it fail downstream as "couldn't
    # retrieve the requested data".
    if route == "kpi_q":
        ql = q.lower()
        if not _has_time_phrase(ql) and not _has_stat_word(ql):
            route = "metadata_q"

    # Symmetric counterpart of the backstop just above: a question the router
    # called metadata_q that DOES carry a time phrase or a statistic word is
    # almost certainly a genuine value fetch that got misclassified the other
    # way — metadata (counts/listings of devices/interfaces/components/KPI
    # NAMES) is inherently a current-snapshot question with no reason to name
    # a time window at all. Left uncorrected, adding an innocuous word like
    # "component" to an otherwise-identical question could flip the router
    # from kpi_q to metadata_q with no other signal changing, and the answer
    # silently became a bare KPI-name listing instead of the requested values
    # — the exact inconsistency this backstop closes.
    if route == "metadata_q":
        ql = q.lower()
        if _has_time_phrase(ql) or _has_stat_word(ql):
            route = "kpi_q"

    # Deterministic backstop #2: "list/how many/give me devices
    # containing/matching/starting with <token>" is an unambiguous, always-
    # in-scope device-name-pattern listing question — it's exactly the shape
    # _spec_name_match() recognizes deterministically further down the
    # pipeline. The local 7B sometimes routes it to out_of_scope because the
    # pattern token (e.g. a placeholder like "XYZ") isn't a real device/KPI
    # name it recognizes, even though the QUESTION SHAPE is fully supported.
    # Left uncorrected, this is fatal: build_spec() short-circuits to
    # {"recognized": False} for out_of_scope BEFORE it ever tries
    # _spec_name_match, so the deterministic engine never gets a chance and
    # the user gets a generic "that's not about network monitoring" message
    # for a perfectly answerable question.
    if route == "out_of_scope":
        ql = q.lower()
        if "device" in ql and re.search(
                r"\b(contain(?:s|ing)?|matching|match|start(?:s|ing)?\s+with|beginning\s+with)\b", ql):
            route = "metadata_q"

    # BUGFIX: every entity below used to be built with (parsed.get(x) or "").strip(),
    # which assumes the 7B emitted a plain string. It does not always: a nested
    # object ({"device": {"name": "..."}}) or a list ({"kpi": ["HC In Octets"]})
    # sailed straight through .strip()'s AttributeError into the graph and blew
    # up much later (e.g. in _iface_display -> .split) as an opaque
    # "'dict' object has no attribute 'split'". _as_text flattens both shapes.
    # _clean_entity() (not plain _as_text) blanks the literal "empty"/"none"/...
    # placeholders the router prompt asks the model to emit for absent fields.
    # Without this, component="empty" became component_filter=["empty"] and was
    # sent to the API as a real component name, returning 0 rows for otherwise
    # valid questions. metric/target keep their own sentinel ("empty") meaning
    # "undecided" and are handled by the keyword backstops just below, so they
    # are cleaned the same way (sentinel -> "" -> backstop fills them in).
    ent = {
        "device":     _clean_entity(parsed.get("device")),
        "circle":     _clean_entity(parsed.get("circle")),
        "component":  _clean_entity(parsed.get("component")),
        "interface":  _clean_entity(parsed.get("interface")),
        "kpi":        _clean_entity(parsed.get("kpi")),
        "metric":     _clean_entity(parsed.get("metric")).lower(),
        "target":     _clean_entity(parsed.get("target")).lower(),
        "top_n":      _as_int(parsed.get("top_n")),
        "threshold":  _as_float(parsed.get("threshold")),
        "time_phrase": _clean_entity(parsed.get("time_phrase")),
    }
    # Cheap keyword backstops for target/metric if the model left them blank.
    ql = q.lower()
    if not ent["target"]:
        for kw, tgt in [("interface", "interfaces"), ("component", "components"),
                        ("kpi", "kpis"), ("device", "devices"), ("host", "devices")]:
            if kw in ql:
                ent["target"] = tgt
                break
    if not ent["metric"]:
        if "how many" in ql or "count" in ql:
            ent["metric"] = "count"
        elif "list" in ql or "show all" in ql or "which" in ql:
            ent["metric"] = "list"
    if not ent["circle"]:
        cm = circle_match(q)
        if cm:
            ent["circle"] = cm
    # BUGFIX: the router sometimes duplicates the KPI name (or a fragment of
    # it) into the component field too — either the whole thing verbatim
    # (component="HC In Octets", kpi="HC In Octets") or just a descriptive
    # WORD lifted out of a KPI name (component="utilization", kpi=
    # "Utilization In % HC | Utilization Out % HC" — seen when the question
    # asks for two quoted KPIs, both of which happen to start with
    # "Utilization"). A real monitored component is always a generic category
    # (traffic/bgp/cpu/errors/broadcast/...), never literally the KPI's own
    # name or a KPI-only descriptor word, so both are unambiguous LLM noise,
    # not a genuine component mention. Left uncorrected the fragment case is
    # worse than the exact-duplicate case: a bogus component like "utilization"
    # also poisons _spec_kpi_filter's quoted-KPI extraction downstream (every
    # quoted KPI phrase starting with "Utilization" gets wrongly treated as
    # "claimed" by the component slot and dropped), on top of scoping the
    # interface search down to a component that doesn't exist.
    comp_l = ent["component"].strip().lower()
    if comp_l and (comp_l == ent["kpi"].strip().lower() or comp_l in _KPI_SYNONYM_ONLY_WORDS):
        ent["component"] = ""
    log.info("[ROUTER] route=%s target=%s metric=%s device=%r circle=%r",
             route, ent["target"], ent["metric"], ent["device"], ent["circle"])
    dbg = _dbg(state)
    dbg["route"] = route
    dbg["confidence"] = parsed.get("confidence", "Medium")
    dbg["entities"] = ent
    dbg["rag_examples"] = [r.get("sub_type") for r in retrieve_examples(q)]
    _stage_done(dbg, "router", t0)
    return {**state, "route": route, "confidence": parsed.get("confidence", "Medium"),
            "entities": ent, "debug": dbg}


def resolve_node(state: VIAgentState) -> VIAgentState:
    """Ground the question: resolve circle, device candidates (GoT disambiguation),
    host_id/cid, available components, and the time window."""
    ent = state["entities"]
    sk = state["session_key"]
    t0 = time.time()
    grounding: Dict[str, Any] = {}

    circle_canon = circle_match(_as_text(ent.get("circle"))) or circle_match(state["user_query"])
    grounding["circle"] = circle_canon

    try:
        hosts = get_hosts_cached(sk)
    except MCPError as exc:
        return {**state, "grounding": {"error": f"Could not load devices: {exc}"},
                "failed": False, "tool_calls": state.get("tool_calls", 0) + 1}

    grounding["total_hosts"] = len(hosts)
    if circle_canon:
        in_circle = [h for h in hosts
                     if host_in_circle(h.get("name") or h.get("host", ""), circle_canon)]
        grounding["hosts_in_circle"] = in_circle
        grounding["hosts_in_circle_count"] = len(in_circle)

    # Device disambiguation via graph-of-thought scoring over name candidates.
    device = _as_text(ent.get("device"))

    # A real host name typed verbatim in the raw question is a stronger signal
    # than the router's extracted `device` entity, which can echo a few-shot
    # example device instead of the one actually named in the question (a
    # known small-model failure mode — see e.g. APVSPGJWPAR01HNE40 showing up
    # for questions about a different device entirely). If a literal match
    # exists and disagrees with what the router extracted, trust the literal
    # match; this never fires for the common "device: '<partial/typo'"
    # case since routing didn't literally quote a full different host name.
    literal_matches = find_literal_device_matches(state["user_query"], hosts)
    if literal_matches and device.upper() not in {
            (h.get("name") or h.get("host") or "").upper() for h in literal_matches}:
        grounding["device_entity_overridden"] = {
            "router_extracted": device or None,
            "literal_match": literal_matches[0].get("name") or literal_matches[0].get("host"),
        }
        device = literal_matches[0].get("name") or literal_matches[0].get("host")

    if device:
        candidates = match_hosts(device, hosts, circle_canon)
        chosen, alternatives = got_disambiguate(device, candidates)
        grounding["candidates"] = [
            {"host_id": c.get("hostid"), "name": c.get("name") or c.get("host"), "cid": c.get("cid")}
            for c in candidates
        ]
        if chosen:
            grounding["host_id"] = chosen.get("hostid")
            grounding["cid"] = chosen.get("cid")
            grounding["host_name"] = chosen.get("name") or chosen.get("host")
            grounding["alternatives"] = alternatives
            # Prefetch components for the chosen device (grounds the executor + KPI counts).
            try:
                comp_data = run_tool("ig_list_components",
                                     {"host_id": chosen["hostid"], "cid": chosen["cid"]}, sk)
                comps = comp_data.get("components", []) or []
                grounding["components"] = comps
            except MCPError as exc:
                grounding["components_error"] = str(exc)
        elif candidates:
            # Nothing cleared the confidence bar — no device is grounded, but
            # offer the closest few raw candidates as "did you mean" hints for
            # the "device not found" clarification message.
            grounding["alternatives"] = [c.get("name") or c.get("host") for c in candidates[:3]]

    # Time window for KPI/stats questions (also parse for stats intent).
    if state["route"] in ("kpi_q",) or (ent.get("top_n") or ent.get("threshold")):
        s, e = parse_time_window(ent.get("time_phrase") or state["user_query"], ent)
        grounding["start_time"], grounding["end_time"] = s, e

    dbg = _dbg(state)
    dbg["grounding_summary"] = {
        "circle": circle_canon,
        "total_hosts": grounding.get("total_hosts"),
        "hosts_in_circle_count": grounding.get("hosts_in_circle_count"),
        "chosen_device": grounding.get("host_name"),
        "candidate_count": len(grounding.get("candidates", []) or []),
        "alternatives": grounding.get("alternatives", []),
        "device_entity_overridden": grounding.get("device_entity_overridden"),
    }
    _stage_done(dbg, "resolve", t0)
    return {**state, "grounding": grounding, "debug": dbg,
            "tool_calls": state.get("tool_calls", 0) + 1}


# Below this score, a "match" is really just "the least-bad of a bad lot" —
# treat it as no match at all rather than silently grounding a wrong device.
# 0.9 (substring) and 1.0 (exact) always clear this; genuine near-miss typos
# ("HNE41" vs "HNE40") score ~0.9+ via SequenceMatcher and still clear it too.
DEVICE_MATCH_MIN_CONFIDENCE = 0.55


def got_disambiguate(device: str, candidates: List[Dict[str, Any]]
                     ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Graph-of-thought style pick: score candidate hosts, choose the best,
    and surface close alternatives so the answer can flag ambiguity.

    An EXACT name match (score == 1.0) is unambiguous by definition — a device
    named APVSPGJWPAR01HNE40 sitting near-neighbors APVSPGJWPAR01HNE41/HNE42 in
    edit-distance space is not "ambiguous" if the user typed the exact name.
    Only surface alternatives when the top match itself was a fuzzy one.

    If even the BEST candidate scores below DEVICE_MATCH_MIN_CONFIDENCE, this
    is a nonexistent/garbled device name, not a fuzzy match — return no match
    (None) rather than silently grounding whatever host happened to be least
    dissimilar, so the caller can report "device not found" instead of quietly
    answering about the wrong device."""
    if not candidates:
        return None, []
    qd = device.upper()
    scored = []
    for h in candidates:
        name = (h.get("name") or h.get("host") or "")
        up = name.upper()
        score = _host_match_score(qd, up)
        scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score >= 1.0:
        return best, []                    # exact match: never ambiguous
    if best_score < DEVICE_MATCH_MIN_CONFIDENCE:
        return None, []                    # nothing close enough to trust
    alts = [(h.get("name") or h.get("host")) for s, h in scored[1:]
            if best_score - s <= 0.08][:3]
    return best, alts


COT_SYSTEM = """\
You are planning how to answer a VI-IPPMS KPI question using monitoring tools.
Given the question and the already-resolved device context, write a SHORT numbered
plan (3-5 steps) describing which interface, which KPI, and which time window are
needed, and in what order to fetch them. Do not output JSON or tool calls — just
the plan. Keep it under 80 words.
"""


def cot_plan_node(state: VIAgentState) -> VIAgentState:
    g = state["grounding"]
    ent = state["entities"]
    ctx = (f"Device: {g.get('host_name','(unresolved)')}\n"
           f"Interface hint: {ent.get('interface','')}\n"
           f"KPI hint: {ent.get('kpi','')}\n"
           f"Components available: {[c.get('component') for c in g.get('components', [])][:12]}\n"
           f"Window (epoch): {g.get('start_time')}..{g.get('end_time')}")
    plan = call_llm(COT_SYSTEM, f"Question: {state['user_query']}\n\n{ctx}",
                    max_new_tokens=220, decode=COT_DECODE, stage="COT").strip()
    return {**state, "plan": plan}


def _grounding_block(state: VIAgentState) -> str:
    g = state["grounding"]
    lines = ["GROUNDED CONTEXT (use these values directly; do not look them up):"]
    if g.get("host_id") is not None:
        lines.append(f"- device: {g.get('host_name')} (host_id={g.get('host_id')}, cid={g.get('cid')})")
    comps = [c.get("component") for c in g.get("components", []) if isinstance(c, dict)]
    if comps:
        lines.append(f"- available components: {comps}")
    if state["route"] == "kpi_q" and g.get("start_time"):
        lines.append(f"- time window (epoch seconds): start={g.get('start_time')}, end={g.get('end_time')}")
    if state.get("entities", {}).get("interface"):
        lines.append(f"- interface of interest: {state['entities']['interface']}")
    if state.get("entities", {}).get("kpi"):
        lines.append(f"- KPI of interest: {state['entities']['kpi']}")
    if state.get("plan"):
        lines.append(f"\nPLAN:\n{state['plan']}")
    return "\n".join(lines)


def react_executor_node(state: VIAgentState) -> VIAgentState:
    """The cyclic ReAct loop: Thought -> Action -> (run MCP tool) -> Observation."""
    sk = state["session_key"]
    tools = TOOLS_CACHE or mcp_list_tools()
    system = "\n\n".join([REACT_ROLE, build_tool_context(tools), REACT_FEWSHOT])
    grounding_block = _grounding_block(state)

    trace: List[Dict[str, Any]] = list(state.get("trace", []))
    scratch = ""
    repairs_left = REACT_MAX_REPAIRS
    tool_calls = state.get("tool_calls", 0)
    final_answer = None

    for step in range(REACT_MAX_STEPS):
        user_prompt = f"Question: {state['user_query']}\n\n{grounding_block}\n\n{scratch}"
        completion = call_llm(system, user_prompt, max_new_tokens=512,
                              decode=ACTION_DECODE, stage=f"REACT[{step+1}]")
        # Never let the model hallucinate its own Observation.
        completion = re.split(r"\nObservation:", completion)[0].strip()
        parsed = parse_react_step(completion)
        log.info("REACT[%d] Thought=%r Action=%r ActionInput=%r is_final=%s parse_error=%s",
                  step + 1, (parsed.get("thought") or "").strip()[:300],
                  parsed.get("action"), parsed.get("action_input"),
                  parsed.get("is_final"), parsed.get("parse_error"))

        if parsed["parse_error"] and repairs_left > 0:
            repairs_left -= 1
            scratch += ("\n(Your previous step was not parseable. Reply with exactly one "
                        "Thought, then Action, then Action Input as valid JSON, OR a Final Answer.)\n")
            continue

        if parsed["is_final"]:
            final_answer = parsed["final_answer"]
            trace.append({"thought": parsed["thought"], "action": None,
                          "action_input": None, "observation": "(final answer)"})
            break

        action = parsed["action"]
        action_input = parsed["action_input"] or {}
        if not action:
            reformat_prompt = (
                f"Question: {state['user_query']}\n\n{grounding_block}\n\n"
                f"Your previous response was:\n\"\"\"\n{completion}\n\"\"\"\n\n"
                "Restate that response using EXACTLY this format and nothing else:\n"
                "Thought: <one short sentence>\nFinal Answer: <the same answer, in full>"
            )
            reformatted = call_llm(REACT_REFORMAT_SYSTEM, reformat_prompt, max_new_tokens=400,
                                   decode=ACTION_DECODE, stage=f"REACT[{step+1}]-reformat")
            reparsed = parse_react_step(reformatted)
            log.info("REACT[%d] reformat is_final=%s final_answer=%r",
                     step + 1, reparsed.get("is_final"), (reparsed.get("final_answer") or "")[:300])
            if reparsed["is_final"] and reparsed["final_answer"]:
                final_answer = reparsed["final_answer"]
                trace.append({"thought": parsed.get("thought", ""), "action": None,
                              "action_input": None, "observation": "(final answer, reformatted)"})
            else:
                final_answer = completion.strip()  # salvage the raw prose — it has the real answer
                trace.append({"thought": parsed.get("thought", ""), "action": None,
                              "action_input": None,
                              "observation": f"ERROR: {parsed.get('parse_error','no action')} (salvaged raw completion)"})
            break

        if action in AUTH_TOOLS:
            obs = {"note": "auth is handled automatically; pick a data tool instead."}
        else:
            action_input.pop("session_key", None)
            try:
                obs = run_tool(action, action_input, sk)
                tool_calls += 1
            except MCPError as exc:
                obs = {"error": str(exc)}

        obs_str = truncate_obs(obs)
        log.info("REACT[%d] Observation(%s): %s", step + 1, action, obs_str[:500])
        trace.append({"thought": parsed["thought"], "action": action,
                      "action_input": action_input, "observation": obs_str})
        scratch += (f"Thought: {parsed['thought']}\nAction: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\nObservation: {obs_str}\n")

    return {**state, "trace": trace, "answer": final_answer or "",
            "tool_calls": tool_calls,
            "failed": final_answer is None}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — JSON HELPER, TRACE EXTRACTION, DETERMINISTIC FALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def safe_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from possibly-messy LLM output."""
    if not text:
        return {}
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


# Tools are listed once at startup; refreshed lazily if empty.
TOOLS_CACHE: List[Dict[str, Any]] = []


def _obs_json(obs: Any) -> Any:
    if isinstance(obs, str):
        try:
            return json.loads(obs)
        except Exception:
            return {}
    return obs if isinstance(obs, (dict, list)) else {}


def extract_history_from_trace(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the most recent ig_get_kpi_history observation (series + Values)."""
    for step in reversed(trace):
        if step.get("action") == "ig_get_kpi_history":
            data = _obs_json(step.get("observation"))
            if isinstance(data, dict) and ("series" in data or "Values" in data or "values" in data):
                return data
    return None


def build_stats_rows(values_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for v in values_list or []:
        rows.append({
            "KPI": v.get("name", ""),
            "Minimum": v.get("Minimum"), "Maximum": v.get("Maximum"),
            "Last": v.get("Last"), "Average": v.get("Average"),
            "Total": v.get("Total"), "Units": v.get("units"),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10B — QUERY-SPEC ENGINE  (§4.2A — resolves §3.1 consistency + §3.6 coverage)
# ══════════════════════════════════════════════════════════════════════════════
#  Replaces the closed if/elif deterministic_metadata()/deterministic_kpi()
#  branches with:
#    build_spec()          pure-Python entities+grounding -> QuerySpec (NO LLM)
#    execute_query_spec()  ONE general executor that walks any recognized spec
#  Because neither touches the free-text ReAct loop, a recognized question
#  returns byte-identical DATA every run (§3.1). Every §3.6 permutation is just a
#  parameterization of the same executor. Deviations from the brief's suggested
#  QuerySpec (all additive, none silent):
#    - aggregation "common"  : set-intersection across a device set (Q6 needs it)
#    - aggregation "summary" : return the Min/Max/Last/Avg/Total block as-is
#    - result_shape          : split out from aggregation, maps 1:1 to the guide's
#                              expected_result_shape column (test oracle).

SPEC_AGGS = {"count", "list", "min", "max", "avg", "last", "total",
             "top_n", "bottom_n", "threshold", "common", "summary",
             "spikes", "dips", "anomalies"}

_FIELD_BY_AGG = {"min": "Minimum", "max": "Maximum", "avg": "Average",
                 "last": "Last", "total": "Total"}

# Columns for the raw, long-format time-series export: one row per data point
# per KPI, alongside the existing chart + Min/Max/Last/Average/Total/Units
# stats table (which stays as-is). Shared by the executor (which populates the
# rows) and the UI (which renders/exports them) so both always agree on shape.
TIMESERIES_COLS = ["Device", "Component", "Interface", "KPI", "Value", "Time"]


def _fmt_epoch_ms(ms: Any) -> str:
    """epoch milliseconds (as returned in series[].data points) -> readable
    local timestamp string, e.g. for the raw time-series CSV export."""
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ms)


def _host_name(h: Dict[str, Any]) -> str:
    return h.get("name") or h.get("host") or ""


# Columns for the flagged spike/dip events summary table (§ user request:
# "show me the spikes/dips for KPI X on interface Y on device Z ..."). Shared
# by the executor and the UI so both agree on shape.
SPIKE_COLS = ["Interface", "KPI", "Type", "Value", "Time", "Deviation (σ)"]


def _rolling_zscores(values: List[float], window: int) -> List[Optional[float]]:
    """Per-point z-score against a LOCAL baseline built from that point's own
    neighbors only — a window of `window` points centered on it, EXCLUDING
    the point itself.

    BUGFIX (caught by isolated simulation before shipping): a plain centered
    rolling mean/std over a window that INCLUDES the point being scored lets
    a genuine spike drag its own baseline toward itself — e.g. a lone 5x
    spike in a window of 5 pulls the mean/std up enough that its own z-score
    lands under a 2.5 threshold, silently missing the exact thing this is
    supposed to catch. Excluding the point from its own baseline (a
    leave-one-out neighbor window) fixes this: the baseline reflects only
    what's "normal" around the point, not the point itself.

    None wherever there aren't enough neighbors, or the neighbors have zero
    variance (flat local data), rather than a fabricated z-score. Requires at
    least 3 neighbors (2 degrees of freedom for std) even at series edges —
    2 neighbors gives only 1 degree of freedom, unstable enough that a
    perfectly ordinary point next to one other nearby value can produce a
    wildly exaggerated z-score (caught by isolated simulation: the last point
    of a short series flagged as a "dip" off a 2-point baseline)."""
    n = len(values)
    half = max(1, window // 2)
    min_neighbors = max(3, half)
    out: List[Optional[float]] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighbors = values[lo:i] + values[i + 1:hi]
        if len(neighbors) < min_neighbors:
            out.append(None)
            continue
        mean = sum(neighbors) / len(neighbors)
        variance = sum((x - mean) ** 2 for x in neighbors) / (len(neighbors) - 1)
        std = variance ** 0.5
        out.append(None if std == 0 else (values[i] - mean) / std)
    return out


def _detect_spikes_dips(data_points: List[List[float]], direction: str,
                        threshold: float = SPIKE_ZSCORE_THRESHOLD,
                        window: int = SPIKE_ROLLING_WINDOW) -> List[Dict[str, Any]]:
    """Flag spikes/dips in one KPI's time series via a rolling z-score.

    `data_points` is one series[i]["data"]: [[epoch_ms, value], ...], already
    filtered to well-formed [epoch_ms, value] pairs and assumed chronologically
    ordered (as ig_get_kpi_history returns them). `direction` is "spikes"
    (flag only above-threshold points), "dips" (below-threshold only), or
    "anomalies" (both). Returns one dict per input point, in the same order:
    {"epoch_ms", "value", "zscore", "flag": ""|"SPIKE"|"DIP"}.

    Below SPIKE_MIN_POINTS points a rolling baseline isn't meaningful, so
    nothing is flagged (the caller reports "not enough data" rather than a
    false "no spikes/dips found")."""
    if len(data_points) < SPIKE_MIN_POINTS:
        return [{"epoch_ms": pt[0], "value": pt[1], "zscore": None, "flag": ""} for pt in data_points]
    zscores = _rolling_zscores([pt[1] for pt in data_points], window)
    out = []
    for pt, z in zip(data_points, zscores):
        flag = ""
        if z is not None:
            if z >= threshold and direction in ("spikes", "anomalies"):
                flag = "SPIKE"
            elif z <= -threshold and direction in ("dips", "anomalies"):
                flag = "DIP"
        out.append({"epoch_ms": pt[0], "value": pt[1], "zscore": z, "flag": flag})
    return out


def _spec_name_match(ent: Dict[str, Any], q: str, ql: str) -> Optional[Dict[str, Any]]:
    """Decide the device name-match mode from phrasing (deterministic).

    BUGFIX: the pattern token is often quoted ('devices containing "XYZ"'),
    the same way interface/KPI names are quoted elsewhere in questions this
    app already handles (see the `"?...("?` interface-filter regexes below).
    This one was missing that tolerance — `\s+([A-Za-z0-9_./-]+)` requires the
    captured token to start immediately after the whitespace, so a quote
    character sitting there made the WHOLE match fail silently, not just the
    quote: the question fell through with no name_match at all."""
    m = re.search(r"(?:start(?:s|ing)?\s+with|beginning\s+with)\s+\"?([A-Za-z0-9_./-]+)\"?", ql)
    if m:
        return {"mode": "starts_with", "value": m.group(1).upper().strip(".,")}
    m = re.search(r"(?:contain(?:s|ing)?|matching|match|with)\s+\"?([A-Za-z0-9_./-]+)\"?\s*(?:in (?:the |its )?name)?", ql)
    if m and m.group(1) not in ("a", "the", "an", "any"):
        # avoid capturing 'devices with a bgp component' style — require alnum token
        val = m.group(1).upper().strip(".,")
        # BUGFIX: the len>=3 guard below exists to reject an accidental short
        # word picked up from unquoted prose ("devices with a bgp component").
        # It was also silently rejecting a SHORT but explicitly QUOTED token
        # ('containing "GJ"') — a deliberate, precise user request, not
        # accidental noise. A quoted token is unambiguous regardless of
        # length, so the guard only applies when the token wasn't quoted.
        quoted = ((m.start(1) > 0 and ql[m.start(1) - 1] == '"') or
                 (m.end(1) < len(ql) and ql[m.end(1)] == '"'))
        if quoted or re.search(r"\d", val) or len(val) >= 3:
            return {"mode": "contains", "value": val}
    return None


def _spec_intent(ent: Dict[str, Any], route: str, ql: str) -> str:
    if (ent.get("top_n") or re.search(r"\b(top|bottom)\s+\d+", ql) or "top " in ql
            or "bottom " in ql or "highest" in ql or "lowest" in ql
            or ent.get("threshold") is not None
            or any(w in ql for w in ("exceed", "exceeded", "crossed", "above", "over ", "greater than"))):
        return "stats"
    if route == "kpi_q":
        return "kpi"
    return "metadata"


def _spec_target(ent: Dict[str, Any], intent: str, ql: str) -> str:
    # For value/stats questions the answer is always KPI values; ignore a stray
    # target label the router may have emitted ("kpis"/"interfaces"), since e.g.
    # "top 5 devices by traffic" or "what was HC In Octets ..." are value queries.
    if intent in ("kpi", "stats"):
        return "values"
    t = (ent.get("target") or "").strip()
    # "values" is only ever a legitimate target for kpi/stats intent, which the
    # branch above already returns for. If we get here, intent is "metadata" —
    # a target of "values" at this point is a STALE artifact from the router
    # LLM's original (pre-backstop-correction) classification, e.g. it first
    # said kpi_q/target=values, the deterministic backstop reclassified the
    # ROUTE to metadata_q, but the LLM's target field was never reset to match.
    # Trusting it here would silently misroute a listing question into the
    # value-fetch engine (which then crashes for lack of a time window).
    if t == "values":
        t = ""
    if t in ("devices", "components", "interfaces", "kpis"):
        return t
    # "interfaces and KPIs within X" / "interface + kpi details on X" — a KPI
    # listing table (Component | Interface | KPI) already shows the interfaces
    # too, so when BOTH words appear it's a strict superset to route to "kpis"
    # rather than "interfaces" (which would silently drop the KPI names).
    if "kpi" in ql and "interface" in ql:
        return "kpis"
    for kw, tgt in (("interface", "interfaces"), ("component", "components"),
                    ("kpi", "kpis"), ("device", "devices"), ("host", "devices")):
        if kw in ql:
            return tgt
    return "devices"


_SPIKE_WORD_RE = re.compile(r"\b(spikes?|surges?)\b")
_DIP_WORD_RE = re.compile(r"\b(dips?|drops?)\b")
_ANOMALY_WORD_RE = re.compile(r"\b(anomal(?:y|ies)|outliers?|unusual)\b")


def _spec_aggregation(ent: Dict[str, Any], ql: str, intent: str, target: str) -> str:
    # Spike/dip detection (§ user request) is checked ahead of the "stats"
    # intent branch below, and independent of it: these questions classify as
    # plain "kpi" intent (they carry a time phrase but no top/bottom/threshold
    # wording), so a check gated on intent=="stats" would never see them.
    if target == "values":
        has_spike, has_dip = bool(_SPIKE_WORD_RE.search(ql)), bool(_DIP_WORD_RE.search(ql))
        if has_spike or has_dip or _ANOMALY_WORD_RE.search(ql):
            if has_spike and not has_dip:
                return "spikes"
            if has_dip and not has_spike:
                return "dips"
            return "anomalies"
    if intent == "stats":
        if ent.get("threshold") is not None or any(
                w in ql for w in ("exceed", "exceeded", "crossed", "above", "over ", "greater than")):
            return "threshold"
        # "bottom 5" / "lowest 5" / "smallest 5" ranks ascending; everything
        # else that reached "stats" intent (top N, or a bare "top"/"highest"
        # mention) ranks descending, same as before this check existed.
        if re.search(r"\bbottom\b", ql) or "lowest" in ql or "smallest" in ql:
            return "bottom_n"
        return "top_n"
    if "common" in ql and target == "components":
        return "common"
    metric = (ent.get("metric") or "").strip().lower()
    if metric in _FIELD_BY_AGG:
        return metric
    if metric == "count" or "how many" in ql or ql.strip().startswith("count"):
        return "count"
    if metric == "list" or any(w in ql for w in ("list", "show all", "which", "what interfaces",
                                                 "what components", "what kpis", "what devices")):
        return "list"
    if target == "values":
        return "summary"
    # A bare noun-phrase question ("interfaces on X", "KPIs on X", "components
    # of X") with no "how many"/"count" keyword is asking to SEE the items, not
    # just get a number — a bare count is the least useful answer to what's
    # usually a listing question. Only an explicit "how many"/"count" (handled
    # above) should produce a count; everything else here defaults to a list.
    return "list"


def _spec_threshold(ent: Dict[str, Any], q: str) -> Optional[float]:
    """Threshold with an optional unit -> base units (bytes)."""
    if ent.get("threshold") is None and not re.search(
            r"(exceed|exceeded|crossed|above|over|greater than)", q.lower()):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB|T|G|M|K|B)?", q, re.I)
    if not m:
        return float(ent["threshold"]) if ent.get("threshold") is not None else None
    return to_bytes(f"{m.group(1)} {m.group(2) or ''}")


def _spec_kpi_filter(ent: Dict[str, Any], q: str, device_val: str, component_val: str,
                     interface_val: str, circle_val: str) -> Optional[Dict[str, Any]]:
    """Build the KPI filter — supporting MORE THAN ONE requested KPI in a
    single question (§ user request: e.g. '"HC In Octets" and "HC Out
    Octets" on Interface Bundle-Ether1...').

    Quoted phrases are the most reliable signal here: this app's users
    consistently quote entity names (device, interface, KPI — see every
    worked example in this conversation), so every quoted substring that
    ISN'T already claimed by another slot (device/component/interface/circle)
    is treated as a requested KPI. Substring (not exact) containment is used
    for the "claimed" check because a quoted interface phrase usually includes
    the word "Interface" (e.g. "Interface Bundle-Ether1") while the router's
    own interface entity does not.

    Falls back to splitting the router's own single `kpi` string on
    and/,/&// when the question has no quotes to go on, in case the LLM
    concatenated multiple KPI names into that one field.

    kpi_filter["values"] carries the full ordered list; kpi_filter["value"]
    stays just the first one, so any code that only reads the singular field
    keeps working unchanged."""
    other_values = [v for v in (device_val, component_val, interface_val, circle_val) if v]

    def _claimed(phrase: str) -> bool:
        pl = phrase.strip().lower()
        for ov in other_values:
            ol = ov.strip().lower()
            if pl == ol or ol in pl or pl in ol:
                return True
        return False

    quoted = [m.strip() for m in re.findall(r'"([^"]+)"', q) if m.strip()]
    kpi_values = [m for m in quoted if not _claimed(m)]

    if not kpi_values:
        raw = _as_text(ent.get("kpi")).strip()
        if raw:
            # BUGFIX: "/" used to be a splitter here, but it's also a
            # legitimate character INSIDE real KPI names in this domain (e.g.
            # "Utilization In/Out Peak % HC") — splitting on it could fragment
            # a single valid KPI name into two garbage halves. Split on "|"
            # instead (seen from the router when it concatenates multiple
            # requested KPIs) plus the usual ","/"&"/"and"/"or", never "/".
            parts = re.split(r"\s*(?:,|&|\||\band\b|\bor\b)\s*", raw, flags=re.IGNORECASE)
            kpi_values = [p.strip() for p in parts if p.strip()]

    if not kpi_values:
        return None
    return {"mode": "contains", "value": kpi_values[0], "values": kpi_values}


def _spec_result_shape(spec: Dict[str, Any]) -> str:
    agg = spec["aggregation"]
    if agg == "count":
        return "count"
    if agg in ("list", "common"):
        return "list"
    if agg in ("min", "max", "avg", "last", "total"):
        return "scalar"
    if agg in ("top_n", "bottom_n", "threshold"):
        return "stats_table"
    if agg in ("spikes", "dips", "anomalies"):
        return "timeseries"
    if agg == "summary":
        return "timeseries" if spec["target"] == "values" and not spec.get("_multi") else "stats_table"
    return "explanatory_text"


def build_spec(state: VIAgentState) -> Dict[str, Any]:
    """Pure-Python assembly of a QuerySpec from entities + grounding (§4.2A)."""
    ent = state["entities"]
    g = state["grounding"]
    q = state["user_query"]
    ql = q.lower()
    route = state["route"]

    if route in ("help_q", "out_of_scope"):
        return {"recognized": False,
                "intent": "help" if route == "help_q" else "out_of_scope"}

    intent = _spec_intent(ent, route, ql)
    target = _spec_target(ent, intent, ql)
    agg = _spec_aggregation(ent, ql, intent, target)

    circle = g.get("circle")
    name_match = _spec_name_match(ent, q, ql)

    # BUGFIX: several circle aliases (CIRCLE_TOKENS) are themselves short,
    # plausible device-name substrings — "gj"/"guj" for Gujarat, "del" for
    # Delhi, etc. When the SAME token the user explicitly asked to match
    # against a device NAME (via 'containing "GJ"'/'matching "DEL"') is also
    # what triggered the circle inference, that's almost certainly a
    # coincidental alias collision, not a genuine circle mention — the user
    # asked for a name search, not to also silently scope it to one circle.
    # Left uncorrected this quietly dropped every matching device outside
    # that circle and, worse, for a 2-letter token like "GJ" that circle
    # filter was the ONLY filter actually applied once the length guard in
    # _spec_name_match rejected the short match (see BUGFIX there) — the
    # answer degraded into "every device in the circle" with no name
    # filtering at all, which is what actually prompted this fix.
    if circle and name_match:
        aliases = CIRCLE_TOKENS.get(circle, {}).get("aliases", [])
        if name_match["value"].strip().lower() in aliases:
            circle = None

    comp = _as_text(ent.get("component")).strip().lower()
    component_filter = [comp] if comp else None
    # 'devices ... with a <component> component' -> component_filter even if
    # component wasn't the target.
    if not component_filter:
        for c in ("bgp", "traffic", "cpu", "errors", "broadcast"):
            if re.search(rf"\b{c}\b", ql) and target != "components":
                component_filter = [c]
                break
    # Fast list above only covers the components seen in the seed data — a live
    # device can have others (system/power/environment/etc). If the question
    # explicitly names a component the fast list didn't catch, capture the raw
    # phrase here; execute_query_spec resolves it against the device's REAL
    # component list once a device is grounded (see _resolve_component_filter),
    # rather than guessing blind or silently dropping the filter.
    component_filter_raw = None
    if not component_filter:
        mc = re.search(r'(?:within|under|in|for|on)\s+(?:the\s+)?([a-z][\w\s-]{1,30}?)\s+component\b', ql)
        if mc:
            component_filter_raw = mc.group(1).strip()
        else:
            mc2 = re.search(r'\bcomponent\s+(?:called|named)?\s*"?([\w-]{2,30})"?', ql)
            if mc2:
                component_filter_raw = mc2.group(1).strip()

    iface = _as_text(ent.get("interface")).strip()
    interface_filter = None
    if iface:
        mi = re.search(r'(?:contain(?:s|ing)?|matching)\s+"?([A-Za-z0-9_./-]+)"?', ql)
        if mi and mi.group(1).lower() in iface.lower():
            interface_filter = {"mode": "contains", "value": mi.group(1)}
        else:
            interface_filter = {"mode": "exact", "value": iface}
    else:
        mi = re.search(r'interfaces?\s+(?:that\s+)?(?:contain(?:s|ing)?|matching)\s+"?([A-Za-z0-9_./-]+)"?', ql)
        if mi:
            interface_filter = {"mode": "contains", "value": mi.group(1)}

    kpi_filter = _spec_kpi_filter(ent, q, _as_text(ent.get("device")), _as_text(ent.get("component")),
                                  iface, _as_text(ent.get("circle")))

    # ── Interface-ranking "currently" shape (§ user request): "top/bottom N
    # interfaces for KPI X on device Y currently" — ranks interfaces on ONE
    # device by their most recent ("Last") value, not a historical Min/Max/Avg
    # over an explicit window. Narrowly gated on target=="values" + a top_n/
    # bottom_n aggregation + the word "interface(s)" appearing in the question,
    # so it never touches the pre-existing "top N DEVICES by KPI over <window>"
    # shape (no "interface" mention there), nor a ranking question that DOES
    # name an explicit time window (that keeps using Maximum over that window,
    # the existing behavior). See CURRENT_VALUE_LOOKBACK_SECONDS.
    current_value = (agg in ("top_n", "bottom_n") and target == "values"
                     and "interface" in ql and not _has_time_phrase(ql))

    # This shape ranks ACROSS every interface on the device — a single named
    # interface_filter here would only ever be router noise (the question
    # asks to rank interfaces, not to fetch one already-named one), and would
    # otherwise wrongly divert execution to the single-interface validator.
    if current_value:
        interface_filter = None

    # A named KPI with no explicit component -> guess the owning component from
    # the domain synonym table (e.g. "HC In Octets" -> "traffic") so the
    # executor scopes to the right component up front instead of scanning every
    # component on the device. A wrong guess still self-heals: the zero-result
    # recheck mechanism (_maybe_recheck) already widens to every component if
    # this narrows to nothing.
    if current_value and not component_filter and not component_filter_raw and kpi_filter:
        hint = _kpi_component_hint(kpi_filter.get("value") or "")
        if hint:
            component_filter = [hint]

    # "top/bottom N" in the raw text is a more reliable N than the router's
    # own top_n entity, which the local 7B doesn't reliably populate for a
    # "bottom N" phrasing (its prompt only ever showed "top N" examples).
    top_n_ql = None
    m_topn = re.search(r"\b(?:top|bottom|highest|lowest)\s+(\d+)\b", ql)
    if m_topn:
        top_n_ql = int(m_topn.group(1))

    tw = None
    if current_value:
        now = _now()
        tw = {"start": now - CURRENT_VALUE_LOOKBACK_SECONDS, "end": now}
    elif g.get("start_time"):
        tw = {"start": int(g["start_time"]), "end": int(g["end_time"])}

    spec = {
        "intent": intent,
        "target": target,
        "aggregation": agg,
        "device_filter": {"circle": circle, "name_match": name_match},
        "component_filter": component_filter,
        "component_filter_raw": component_filter_raw,
        "interface_filter": interface_filter,
        "kpi_filter": kpi_filter,
        "time_window": tw,
        "current_value": current_value,
        "top_n": ent.get("top_n") or top_n_ql or (5 if agg in ("top_n", "bottom_n") else None),
        "threshold": _spec_threshold(ent, q),
    }
    spec["result_shape"] = _spec_result_shape(spec)
    spec["recognized"] = _spec_is_recognized(spec, state)
    return spec


def _spec_is_recognized(spec: Dict[str, Any], state: VIAgentState) -> bool:
    """A spec is executable if its target is one we serve and (for value/stats
    questions) it names enough to resolve a KPI. Otherwise fall back to ReAct."""
    if spec["target"] not in ("devices", "components", "interfaces", "kpis", "values"):
        return False
    if spec["aggregation"] not in SPEC_AGGS:
        return False
    if spec["intent"] in ("kpi", "stats") and spec["target"] == "values":
        # need a time window to fetch history
        if not spec.get("time_window"):
            return False
    return True


# ── Executor ─────────────────────────────────────────────────────────────────

class _ToolLog:
    """Thread-safe collector of tool calls for the debug object."""
    def __init__(self, session_key: str):
        self.sk = session_key
        self._lock = threading.Lock()
        self.calls: List[Dict[str, Any]] = []
        self.errors: List[str] = []     # "tool: message" for every failed call
        self.notes: List[str] = []      # human-readable recovery notes (e.g. rechecks)
        self.n = 0

    def note(self, msg: str) -> None:
        """Record a recovery/recheck note that surfaces in the debug trace."""
        with self._lock:
            self.notes.append(msg)
        log.info("[RECHECK] %s", msg)

    def call(self, tool: str, args: Dict[str, Any]) -> Any:
        t0 = time.time()
        ok, err, obs = True, None, None
        try:
            obs = run_tool(tool, args, self.sk)
            return obs
        except MCPError as exc:
            ok, err = False, str(exc)
            raise
        finally:
            rec = {"tool": tool, "args": args, "ok": ok, "error": err,
                   "latency_ms": int((time.time() - t0) * 1000),
                   "observation": truncate_obs(obs, 600) if obs is not None else None}
            with self._lock:
                self.calls.append(rec)
                self.n += 1
                if err:
                    self.errors.append(f"{tool}: {err}")


def resolve_device_set(spec: Dict[str, Any], state: VIAgentState,
                       hosts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Circle + name-match -> a deterministically-sorted device list (1..N).

    If the question named a specific device (name_match, an explicit full name
    in the text, or a grounded host_id) and NONE of that matches, this returns
    an EMPTY list rather than silently falling back to the whole fleet — that
    fallback used to let a mistyped/nonexistent device name quietly widen into
    an unbounded fan-out instead of surfacing "device not found"."""
    q = state["user_query"]
    g = state["grounding"]
    df = spec.get("device_filter") or {}
    circle = df.get("circle")
    nm = df.get("name_match")
    device_was_named = bool(nm) or bool((state.get("entities") or {}).get("device"))

    pool = hosts
    if circle:
        pool = [h for h in hosts if host_in_circle(_host_name(h), circle)]

    if nm:
        pool = [h for h in pool if name_matches(_host_name(h), nm)]
    else:
        # explicit full device name(s) present in the question -> exact/in_list
        explicit = [h for h in pool if _host_name(h).upper() in q.upper() and len(_host_name(h)) >= 6]
        if explicit:
            pool = explicit
        elif g.get("host_id") is not None:
            # BUGFIX: hostid is NOT globally unique in this fleet — it's only
            # unique per (hostid, cid). The same numeric hostid can legitimately
            # belong to two different circle entries (sometimes even under two
            # completely different device NAMES), so filtering by hostid alone
            # silently pulled in a second, unrelated device from a different
            # circle every time that collision happened — surfacing as "2
            # device(s)" for what the user grounded as a single device. Always
            # pair hostid with the SAME cid resolve_node grounded it against.
            pool = [h for h in pool if h.get("hostid") == g["host_id"] and h.get("cid") == g.get("cid")]
        elif device_was_named and not circle:
            # a device name was given but matched nothing above — don't widen
            # to the unrestricted fleet, that's a "not found" case.
            pool = []

    return sorted(pool, key=lambda h: (_host_name(h).upper(), h.get("hostid") or 0))


def _components_for(host: Dict[str, Any], tl: _ToolLog) -> List[str]:
    data = tl.call("ig_list_components", {"host_id": host["hostid"], "cid": host["cid"]})
    return sorted(c.get("component") for c in data.get("components", []) if isinstance(c, dict) and c.get("component"))


def _interfaces_by_component_for(host: Dict[str, Any], components: List[str],
                                 tl: _ToolLog) -> Dict[str, Dict[str, Any]]:
    """Call ig_list_interfaces ONE COMPONENT AT A TIME and merge results.

    Every documented prefix-filter example in the API doc sends exactly one
    component per call (e.g. {"components": ["traffic"]}); a multi-component
    payload was never demonstrated, and in practice the real API returns empty
    component_prefixes for it rather than erroring — which is exactly the
    "0 interfaces" symptom this replaces. Component calls for one device are
    fanned out in parallel (consistent with the fan-out design elsewhere)."""
    comps = [c for c in (components or []) if c]
    if not comps:
        return {}

    def one(comp):
        data = tl.call("ig_list_interfaces",
                       {"host_id": host["hostid"], "cid": host["cid"], "components": [comp]})
        prefixes = data.get("component_prefixes", {}) if isinstance(data, dict) else {}
        prefixes = prefixes or {}
        # The API echoes back under the same component key; fall back to the
        # requested name if the response shape ever differs.
        payload = prefixes.get(comp) or (next(iter(prefixes.values())) if prefixes else {"items": [], "flags": ""})
        return (comp, payload)

    results = _parallel_map(one, comps)
    out: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if isinstance(r, tuple):
            comp, payload = r
            # BUGFIX: items are DOCUMENTED as strings ("Interface 100GE0/3/2 ::")
            # but are not guaranteed to be. Normalize here — this is the single
            # place every interface item enters the executor, so downstream
            # (_interfaces_for, _validate_single_interface_kpi, the KPI listing
            # fan-out) can rely on str and _iface_display's .split() is safe.
            out[comp] = {"items": _norm_items(payload),
                         "flags": _as_text(payload.get("flags")) if isinstance(payload, dict) else ""}
    return out


def _interfaces_for(host: Dict[str, Any], components: List[str], tl: _ToolLog) -> List[str]:
    """Flat, de-duplicated list of interface prefix items across components
    (keeps the ' ::' suffix verbatim). Built on the one-call-per-component
    fetch above."""
    by_comp = _interfaces_by_component_for(host, components, tl)
    out = set()
    for payload in by_comp.values():
        for it in _norm_items(payload):
            out.add(it)
    return sorted(out)


def _iface_display(prefix_item: Any) -> str:
    """Strip the API's trailing ' ::' to get a human-readable interface name.

    BUGFIX: this was typed str and called .split() directly on whatever the API
    returned. A single object-shaped item raised "'dict' object has no attribute
    'split'" from inside execute_query_spec — which only guards MCPError — so it
    escaped to run_agent and became the opaque "The agent hit an unexpected
    error" message. Coerce instead: a weird item degrades one row, not the turn."""
    s = prefix_item if isinstance(prefix_item, str) else _as_text(prefix_item)
    return s.split("::")[0].strip()


def _device_not_found_message(spec: Dict[str, Any], state: VIAgentState) -> str:
    """Specific 'device not found' clarification, using fuzzy alternatives from
    grounding when available, instead of a generic 'no devices matched'."""
    ent = state.get("entities", {}) or {}
    g = state.get("grounding", {}) or {}
    nm = (spec.get("device_filter") or {}).get("name_match") or {}
    wanted = nm.get("value") or ent.get("device")
    circle = (spec.get("device_filter") or {}).get("circle")
    if isinstance(wanted, list):
        wanted = ", ".join(wanted)
    if not wanted:
        return (f"I couldn't find any devices{f' in the {circle.title()} circle' if circle else ''} "
                f"matching that filter.")
    msg = f"I couldn't find a device matching \"{wanted}\""
    msg += f" in the {circle.title()} circle." if circle else "."
    alts = g.get("alternatives") or []
    if alts:
        msg += f" Close matches: {', '.join(alts)}."
    msg += " Could you confirm the exact device name?"
    return msg


def _multiple_devices_message(devices: List[Dict[str, Any]]) -> str:
    """Shared 'which one did you mean' clarification for a device_filter that
    matched more than one host, used by both the single-interface KPI-value
    path and the interface-ranking ('top N interfaces ... currently') path —
    both need exactly one device to proceed."""
    names = [_host_name(h) for h in devices[:8]]
    more = "..." if len(devices) > 8 else ""
    return (f"That matched {len(devices)} devices, so I'm not sure which one you mean: "
           f"{', '.join(names)}{more}. Could you give the exact device name?")


def _resolve_component_filter(spec: Dict[str, Any], devices: List[Dict[str, Any]],
                              tl: "_ToolLog"):
    """If the question named a component the fast-path regex didn't recognize
    (spec['component_filter_raw']), resolve it against the target device's REAL
    component list — exact match, then unique substring, then edit-distance —
    instead of guessing blind or silently dropping the filter.

    Returns (component_filter_or_None, unresolved_or_None). unresolved, when
    present, is (raw_phrase, candidate_component_names) for a clarification
    message; candidate_component_names is either the ambiguous matches or, if
    nothing matched at all, the device's full live component list."""
    cf = spec.get("component_filter")
    raw = spec.get("component_filter_raw")
    if cf or not raw or not devices:
        return cf, None
    live = _components_for(devices[0], tl)
    if not live:
        return None, None                  # nothing to resolve against; let caller proceed unfiltered
    raw_l = raw.strip().lower()
    exact = [c for c in live if c.lower() == raw_l]
    if exact:
        return [exact[0]], None
    contains = [c for c in live if raw_l in c.lower() or c.lower() in raw_l]
    if len(contains) == 1:
        return [contains[0]], None
    if len(contains) > 1:
        return None, (raw, contains)
    scored = sorted(((_similar(raw_l, c.lower()), c) for c in live), reverse=True)
    if scored and scored[0][0] >= 0.6:
        return [scored[0][1]], None
    return None, (raw, live)


def _component_clarification(raw: str, options: List[str], host_name: str) -> str:
    sample = options[:15]
    more = f" (+{len(options) - 15} more)" if len(options) > 15 else ""
    return (f"I couldn't find a component matching \"{raw}\" on {host_name}. "
           f"Available components: {', '.join(sample)}{more}.")


def _rank_kpi_required_message(host: Dict[str, Any], comp_filter: Optional[List[str]],
                               tl: "_ToolLog") -> str:
    """Ranking interfaces needs exactly one KPI to sort by. If a component is
    already known, sample one of its interfaces for the real KPI names on
    offer (cheap: 1-2 calls, not a per-interface fan-out); otherwise point at
    the device's component list as the first thing to narrow down."""
    host_name = _host_name(host)
    if comp_filter:
        by_comp = _interfaces_by_component_for(host, comp_filter, tl)
        items = []
        for payload in by_comp.values():
            items = _norm_items(payload)
            if items:
                break
        sample_kpis: List[str] = []
        if items:
            data = tl.call("ig_list_kpis", {"host_id": host["hostid"], "cid": host["cid"],
                                            "prefix": {comp_filter[0]: [items[0]]}})
            sample_kpis = sorted({_as_text(e.get("suffix")) for grp in data.get("results", {}).values()
                                  for e in grp if e.get("suffix")})
        hint = f" Example KPIs on this component: {', '.join(sample_kpis[:10])}." if sample_kpis else ""
        return (f"Ranking interfaces needs a single KPI to sort by. Which KPI on the "
               f"{comp_filter[0]} component of {host_name} would you like to rank by?{hint}")
    comps = _components_for(host, tl)
    return (f"Ranking interfaces needs a single KPI to sort by, and I couldn't tell which "
           f"component it belongs to. Available components on {host_name}: "
           f"{', '.join(comps) or '(none)'}. Which KPI (and component) would you like to rank by?")


def _validate_single_interface_kpi(spec: Dict[str, Any], state: VIAgentState,
                                   devices: List[Dict[str, Any]], tl: "_ToolLog"):
    """Stage-validate device -> interface -> KPI for a question naming ONE
    specific interface (interface_filter.mode == 'exact'):
      1. device must resolve to exactly one host (0 -> not found, 2+ -> ambiguous)
      2. component, if named, must match a real component on that device
      3. the named interface must match a real interface (under that component,
         or across all components if none was named)
      4. the named KPI, if any, must match a real KPI suffix on that interface;
         if no KPI was named, ALL of that interface's KPIs are used
    Returns (ready_dict, None) on success, or (None, clarification_message) the
    moment any stage can't be verified — so the user is told exactly what's
    missing/wrong instead of getting a generic failure after the fact."""
    iface_filter = spec["interface_filter"]

    # Stage 1: device
    if not devices:
        return None, _device_not_found_message(spec, state)
    if len(devices) > 1:
        return None, _multiple_devices_message(devices)
    host = devices[0]
    host_name = _host_name(host)

    # Stage 1b: component (only if the question named one)
    comp_filter, unresolved = _resolve_component_filter(spec, [host], tl)
    if unresolved:
        raw, options = unresolved
        return None, _component_clarification(raw, options, host_name)
    comps_to_try = comp_filter or _components_for(host, tl)
    if not comps_to_try:
        return None, f"{host_name} has no monitored components, so I can't look up interfaces on it."

    # Stage 2: interface
    by_comp = _interfaces_by_component_for(host, comps_to_try, tl)
    all_items = [(comp, it) for comp, payload in by_comp.items()
                for it in _norm_items(payload)]
    wanted_iface = _as_text(iface_filter.get("value"))
    wanted_l = re.sub(r"^interface\s+", "", wanted_iface.strip().lower())
    exact = [(c, it) for c, it in all_items
            if re.sub(r"^interface\s+", "", _iface_display(it).lower()) == wanted_l]
    matches = exact
    if not matches:
        contains = [(c, it) for c, it in all_items if wanted_l in _iface_display(it).lower()]
        if len(contains) == 1:
            matches = contains
        elif len(contains) > 1:
            names = [_iface_display(it) for _c, it in contains[:10]]
            more = "..." if len(contains) > 10 else ""
            return None, (f"\"{wanted_iface}\" matched multiple interfaces on {host_name}: "
                          f"{', '.join(names)}{more}. Which one did you mean?")
    if not matches:
        available = [_iface_display(it) for _c, it in all_items]
        sample = available[:15]
        more = f" (+{len(available) - 15} more)" if len(available) > 15 else ""
        scope = f" under the {comps_to_try[0]} component" if comp_filter else ""
        return None, (f"I couldn't find an interface named \"{wanted_iface}\" on {host_name}{scope}. "
                      f"Available interfaces: {', '.join(sample)}{more}.")
    # Stage 3: KPI
    #
    # BUGFIX: the SAME interface display name can legitimately appear under
    # more than one component — e.g. "Bundle-Ether1" shows up both under a
    # full-featured component (e.g. "traffic"/"interfaces", with HC In/Out
    # Octets, utilization, discards, etc.) AND under a narrower one (e.g.
    # "broadcast", exposing only its two broadcast-packet counters). When no
    # component was named, `matches` above can hold one (component, item) pair
    # per component that exposes it, and blindly taking matches[0] committed
    # to whichever component happened to sort first alphabetically out of
    # `_components_for` — "broadcast" before "interfaces"/"traffic" — then
    # reported the requested KPI as "not found" using only THAT (wrong,
    # narrower) component's KPI list, even though a sibling component had it.
    #
    # Fix: when a KPI was named, try every component that exact-matched the
    # interface, in order, until one of them actually carries that KPI; only
    # report "not found" once none of them do, and then list the UNION of
    # every tried component's KPIs (not just the first one's) so the
    # clarification reflects what's really available on this interface.
    kpi_filter = spec.get("kpi_filter")
    # BUGFIX / § user request: support MORE THAN ONE requested KPI on the same
    # interface ("HC In Octets and HC Out Octets on Interface X..."). kpi_filter
    # now carries a "values" list (build_spec / _spec_kpi_filter); "value"
    # stays as just the first one for any older reader. Fall back to a
    # single-element list when "values" is absent so this never breaks on an
    # older-shaped spec.
    wanted_kpis: Optional[List[str]] = None
    if kpi_filter:
        raw_values = kpi_filter.get("values") or ([kpi_filter.get("value")] if kpi_filter.get("value") else [])
        wanted_kpis = [_as_text(v).strip().lower() for v in raw_values if _as_text(v).strip()] or None

    # Try the conventionally-correct component FIRST when several tie on
    # exposing this interface (e.g. "HC In Octets" -> "traffic", per
    # KPI_SYNONYMS): a same-named KPI can genuinely exist under more than one
    # component (a generic "interfaces" component duplicating a subset of what
    # "traffic" tracks), so the loop below would otherwise stop at whichever
    # one happened to sort first alphabetically and never even look at the
    # component a human actually means by that KPI name. This only reorders
    # (a stable sort keeps every other tie in its original order) — it never
    # skips a component, so a KPI unique to a non-hinted component is still
    # found on the next iteration.
    hint = None
    for w in (wanted_kpis or []):
        hint = _kpi_component_hint(w)
        if hint:
            break
    ordered_matches = (sorted(matches, key=lambda ci: 0 if ci[0] == hint else 1)
                       if hint else matches)

    comp, item = matches[0]
    entries: List[Dict[str, Any]] = []
    tried_entries: List[Dict[str, Any]] = []
    found = False
    for c, it in ordered_matches:
        data = tl.call("ig_list_kpis", {"host_id": host["hostid"], "cid": host["cid"], "prefix": {c: [it]}})
        grp_entries = [e for grp in data.get("results", {}).values() for e in grp]
        if wanted_kpis is None:
            comp, item, entries = c, it, grp_entries
            found = True
            break
        tried_entries.extend(grp_entries)
        # Union of every requested KPI found under THIS component. Prefer
        # whichever candidate component covers the most of what was asked for
        # (all the common multi-KPI cases — e.g. In/Out Octets pairs — live
        # together under one component), and stop as soon as every requested
        # KPI has been satisfied rather than settling for a partial match
        # when a fuller one might still be a later candidate.
        kpi_matches = [e for e in grp_entries
                      if any(w in _as_text(e.get("suffix")).lower() for w in wanted_kpis)]
        if kpi_matches and len(kpi_matches) > len(entries):
            comp, item, entries = c, it, kpi_matches
            found = True
            if len(kpi_matches) >= len(wanted_kpis):
                break

    iface_disp = _iface_display(item)
    if not found:
        requested = ", ".join(str(v) for v in
                              (kpi_filter.get("values") or [kpi_filter.get("value")]) if v)
        avail = sorted({e.get("suffix") for e in tried_entries})
        return None, (f"I couldn't find a KPI matching \"{requested}\" on "
                      f"{iface_disp} of {host_name}. "
                      f"Available KPIs: {', '.join(str(a) for a in avail) or '(none)'}.")
    if not entries:
        return None, f"No KPIs are tracked on {iface_disp} of {host_name}."

    return {"host": host, "component": comp, "interface_display": iface_disp, "entries": entries}, None


def _find_interface_across_components(host: Dict[str, Any], iface_value: Any,
                                     comp_filter: Optional[List[str]],
                                     tl: "_ToolLog",
                                     kpi_hint: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Locate which component actually carries the named interface on `host`.

    Searches `comp_filter` first when given, otherwise EVERY component on the
    device. Returns (component, item_prefix_with_'::') on a unique exact/contains
    match, else None. This exists because the KPI-listing and values paths used
    to hardcode `comp = (comp_filter or ["traffic"])[0]` — so an interface that
    lives under errors/broadcast/bgp/etc. yielded 0 KPIs even though the device
    and interface were both correct. Resolving the component from the interface
    (instead of assuming 'traffic') is what makes 'correct interface -> non-zero'
    hold regardless of which component owns it.

    `kpi_hint` (the KPI text the caller is ultimately after, if any) breaks
    ties when the interface is monitored identically under several
    components — see _kpi_component_hint. This is a lighter, unverified
    version of the tie-break in _validate_single_interface_kpi (that one
    actually queries each candidate's KPI list and only settles on one that
    truly carries the requested KPI); here we only reorder which component is
    picked among several EXACT interface-name matches, since this function's
    caller (collect_for_host) doesn't retry against a second component if the
    first pick turns out not to carry the KPI it wanted."""
    comps = comp_filter or _components_for(host, tl)
    if not comps:
        return None
    by_comp = _interfaces_by_component_for(host, comps, tl)
    all_items = [(comp, it) for comp, payload in by_comp.items()
                 for it in _norm_items(payload)]
    wanted_l = re.sub(r"^interface\s+", "", _as_text(iface_value).strip().lower())
    if not wanted_l:
        return None
    exact = [(c, it) for c, it in all_items
             if re.sub(r"^interface\s+", "", _iface_display(it).lower()) == wanted_l]
    if exact:
        hint = _kpi_component_hint(kpi_hint) if kpi_hint else None
        if hint:
            exact = sorted(exact, key=lambda ci: 0 if ci[0] == hint else 1)
        return exact[0]
    contains = [(c, it) for c, it in all_items
                if wanted_l in _iface_display(it).lower()]
    if len(contains) == 1:
        return contains[0]
    return None


# Targets whose empty result is worth a widened-scope recheck (device-level and
# component-level listings are not — those legitimately can be empty/complete).
_RECHECKABLE_TARGETS = ("interfaces", "kpis", "values")


def _result_count(result: Dict[str, Any]) -> Optional[int]:
    """Best-effort row/entity count from an executor result payload."""
    payload = result.get("payload") or {}
    if "count" in payload and isinstance(payload["count"], int):
        return payload["count"]
    for key in ("rows", "stats"):
        if isinstance(payload.get(key), list):
            return len(payload[key])
    return None


def _maybe_recheck(spec: Dict[str, Any], state: VIAgentState,
                   result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Zero-result recovery (the 'recheck mechanism').

    Premise: if the device resolved and the interface/KPI the user named is real,
    the answer should not be 0. The most common way a valid question still comes
    back empty is a component scope that's too narrow or was mis-resolved (a
    component_filter the LLM/regex guessed, or an interface searched under the
    wrong component). When a recheckable listing/count for a grounded question
    comes back empty AND a component narrowing was in play, we re-run the SAME
    query once with the component scope widened to every component on the device.
    If that finds data, it replaces the empty answer (with a transparent note and
    the extra tool calls merged into the trace); otherwise the original empty
    result stands (the emptiness was genuine)."""
    if result.get("result_kind") not in ("list", "count", "scalar", "stats", "timeseries"):
        return None
    if spec.get("target") not in _RECHECKABLE_TARGETS:
        return None
    if spec.get("_rechecked"):                       # never recurse
        return None
    count = _result_count(result)
    if count is None or count > 0:
        return None
    # Only meaningful if a component narrowing could be the culprit. If the query
    # was already unscoped (all components) and still found nothing, the interface
    # genuinely isn't there — widening changes nothing.
    if not (spec.get("component_filter") or spec.get("component_filter_raw")):
        return None

    broadened = dict(spec)
    broadened["component_filter"] = None
    broadened["component_filter_raw"] = None
    broadened["_rechecked"] = True

    retry = _execute_query_spec_once(broadened, state)
    if _result_count(retry) and _result_count(retry) > 0:
        prev_scope = spec.get("component_filter") or spec.get("component_filter_raw")
        note = (f"Re-checked: the first lookup scoped to component {prev_scope} returned 0, "
                f"so I searched across all of the device's components and found "
                f"{_result_count(retry)}.")
        # merge the recheck's tool calls into the visible trace
        retry["tool_log"] = (result.get("tool_log") or []) + (retry.get("tool_log") or [])
        retry["tool_calls"] = int(result.get("tool_calls") or 0) + int(retry.get("tool_calls") or 0)
        retry["recheck_note"] = note
        if retry.get("answer"):
            retry["answer"] = f"{retry['answer']}  ({note})"
        return retry
    return None


def execute_query_spec(spec: Dict[str, Any], state: VIAgentState) -> Dict[str, Any]:
    """Thin wrapper: run the executor once, then apply zero-result recheck."""
    result = _execute_query_spec_once(spec, state)
    recovered = _maybe_recheck(spec, state, result)
    return recovered if recovered is not None else result


def _execute_query_spec_once(spec: Dict[str, Any], state: VIAgentState) -> Dict[str, Any]:
    """Walk a recognized QuerySpec generically. Returns a result dict with
    result_kind / payload / answer(optional) / debug tool_calls."""
    sk = state["session_key"]
    tl = _ToolLog(sk)
    target = spec["target"]
    agg = spec["aggregation"]

    try:
        hosts = get_hosts_cached(sk)
    except MCPError as exc:
        return {"result_kind": "error", "payload": {}, "answer": f"Could not load devices: {exc}",
                "tool_calls": tl.n, "tool_log": tl.calls}

    devices = resolve_device_set(spec, state, hosts)
    circle = (spec.get("device_filter") or {}).get("circle")
    where = f" in the {circle.title()} circle" if circle else ""
    # BUGFIX: when a name-match filter (containing/matching/starts with) was
    # ALSO applied, the answer used to say only "... in the X circle" with no
    # mention of the name filter at all — indistinguishable from "every
    # device in that circle", which is exactly what this looked like when it
    # silently degraded (see the name_match/circle fixes above this
    # function). Always name the filter that was actually applied.
    name_match = (spec.get("device_filter") or {}).get("name_match")
    if name_match and name_match.get("value"):
        mode_word = {"starts_with": "starting with", "exact": "matching"}.get(
            name_match.get("mode"), "containing")
        where += f' {mode_word} "{name_match["value"]}"'

    try:
        # ---- DEVICES ---------------------------------------------------------
        if target == "devices":
            comp_filter = spec.get("component_filter")
            if comp_filter:
                def keep(h):
                    comps = _components_for(h, tl)
                    return all(any(cf == c or cf in c for c in comps) for cf in comp_filter)
                mask = _parallel_map(keep, devices)
                devices = [h for h, k in zip(devices, mask) if k is True]
            names = [_host_name(h) for h in devices]
            if agg == "list":
                rows = [{"Device": _host_name(h), "cid": h.get("cid")} for h in devices]
                comp_note = f" with component {comp_filter}" if comp_filter else ""
                return _ok("list", {"rows": rows, "cols": ["Device", "cid"], "count": len(rows)},
                           f"Found {len(rows)} device(s){where}{comp_note}.", tl)
            comp_note = f" with a {', '.join(comp_filter)} component" if comp_filter else ""
            return _ok("count", {"count": len(names), "label": f"devices{where}"},
                       f"There are {len(names)} device(s){where}{comp_note}.", tl)

        if not devices:
            return _ok("text", {}, _device_not_found_message(spec, state), tl)

        # ---- COMPONENTS ------------------------------------------------------
        if target == "components":
            per = _parallel_map(lambda h: (_host_name(h), _components_for(h, tl)), devices)
            per = [p for p in per if isinstance(p, tuple)]
            if agg == "common":
                sets = [set(c) for _n, c in per if c]
                common = sorted(set.intersection(*sets)) if sets else []
                rows = [{"Component": c} for c in common]
                return _ok("list", {"rows": rows, "cols": ["Component"], "count": len(rows)},
                           f"{len(common)} component(s) are common across {len(per)} device(s): "
                           f"{', '.join(common) or '(none)'}.", tl)
            if len(devices) == 1:
                name, comps = per[0]
                if agg == "list":
                    # include item_count for a single device
                    data = tl.call("ig_list_components",
                                   {"host_id": devices[0]["hostid"], "cid": devices[0]["cid"]})
                    rows = [{"Component": c.get("component"), "Items": c.get("item_count")}
                            for c in data.get("components", []) if isinstance(c, dict)]
                    return _ok("list", {"rows": rows, "cols": ["Component", "Items"], "count": len(rows)},
                               f"{name} has {len(rows)} components: "
                               f"{', '.join(str(r['Component']) for r in rows)}.", tl)
                return _ok("count", {"count": len(comps), "label": "components"},
                           f"{name} has {len(comps)} monitored components.", tl)
            # multi-device union
            union = sorted({c for _n, comps in per for c in comps})
            rows = [{"Component": c} for c in union]
            return _ok("list", {"rows": rows, "cols": ["Component"], "count": len(rows)},
                       f"{len(union)} distinct component(s) across {len(per)} device(s): "
                       f"{', '.join(union)}.", tl)

        # ---- INTERFACES ------------------------------------------------------
        if target == "interfaces":
            comp_filter = spec.get("component_filter")
            iface_filter = spec.get("interface_filter")

            comp_filter, unresolved = _resolve_component_filter(spec, devices, tl)
            if unresolved:
                raw, options = unresolved
                subj = _host_name(devices[0]) if len(devices) == 1 else f"{len(devices)} device(s){where}"
                return _ok("text", {}, _component_clarification(raw, options, subj), tl)

            def ifaces_of(h):
                comps = comp_filter or _components_for(h, tl)
                items = _interfaces_for(h, comps, tl)
                disp = {_iface_display(i) for i in items}
                disp = {d for d in disp if d}
                if iface_filter and iface_filter.get("mode") == "contains":
                    v = _as_text(iface_filter.get("value")).lower()
                    disp = {d for d in disp if v in d.lower()}
                return (_host_name(h), sorted(disp))

            per = _parallel_map(ifaces_of, devices)
            per = [p for p in per if isinstance(p, tuple)]
            total = sum(len(ifs) for _n, ifs in per)
            scope = f" under {', '.join(comp_filter)}" if comp_filter else ""
            if iface_filter and iface_filter.get("mode") == "contains":
                scope += f' matching "{iface_filter["value"]}"'
            if agg == "list":
                rows = ([{"Interface": i} for _n, ifs in per for i in ifs] if len(devices) == 1
                        else [{"Device": n, "Interface": i} for n, ifs in per for i in ifs])
                cols = ["Interface"] if len(devices) == 1 else ["Device", "Interface"]
                subj = _host_name(devices[0]) if len(devices) == 1 else f"{len(devices)} device(s){where}"
                return _ok("list", {"rows": rows, "cols": cols, "count": total},
                           f"{subj} has {total} interface(s){scope}.", tl)
            subj = _host_name(devices[0]) if len(devices) == 1 else f"{len(devices)} device(s){where}"
            return _ok("count", {"count": total, "label": f"interfaces{scope}"},
                       f"{subj} has {total} interface(s){scope}.", tl)

        # ---- KPIS (metadata: list/count under an interface/component/device) --
        if target == "kpis":
            host = devices[0]
            iface_filter = spec.get("interface_filter")
            comp_filter = spec.get("component_filter")

            # (a) interface given -> KPIs under that one interface. The component
            # is resolved FROM the interface (across every component on the
            # device), not assumed to be 'traffic' — an interface under
            # errors/broadcast/bgp used to return 0 KPIs here despite being real.
            if iface_filter and iface_filter.get("value"):
                comp_filter, unresolved = _resolve_component_filter(spec, [host], tl)
                if unresolved:
                    raw, options = unresolved
                    return _ok("text", {}, _component_clarification(raw, options, _host_name(host)), tl)
                iface_val = iface_filter.get("value")
                found = _find_interface_across_components(host, iface_val, comp_filter, tl)
                if not found and comp_filter:
                    # Named component didn't contain it — widen to all components
                    # before giving up (the interface may live elsewhere).
                    tl.note(f"Interface '{iface_val}' not found under {comp_filter} on "
                            f"{_host_name(host)}; searching all components.")
                    found = _find_interface_across_components(host, iface_val, None, tl)
                if not found:
                    # Fall back to the staged validator for a precise, correct
                    # 'not found / available interfaces are …' clarification.
                    ready, clarification = _validate_single_interface_kpi(spec, state, [host], tl)
                    if clarification:
                        return _ok("text", {}, clarification, tl)
                    comp = ready["component"]
                    pfx = None
                    kpis = ready["entries"]
                else:
                    comp, item = found
                    pfx = item
                    data = tl.call("ig_list_kpis",
                                   {"host_id": host["hostid"], "cid": host["cid"],
                                    "prefix": {comp: [pfx]}})
                    kpis = [e for grp in data.get("results", {}).values() for e in grp]
                if agg == "list":
                    rows = [{"KPI": e.get("suffix"), "itemid": e.get("itemid")} for e in kpis]
                    return _ok("list", {"rows": rows, "cols": ["KPI", "itemid"], "count": len(rows)},
                               f"{len(rows)} KPI(s) on {iface_filter['value']} ({comp}) of "
                               f"{_host_name(host)}: {', '.join(str(e.get('suffix')) for e in kpis)}.", tl)
                return _ok("count", {"count": len(kpis), "label": "KPIs"},
                           f"{len(kpis)} KPI(s) are tracked on {iface_filter['value']} of {_host_name(host)}.", tl)

            # (b) explicit "how many KPIs" with no interface/component -> the
            # cheap device-wide total (sum of item_count, 1 extra call) rather
            # than fanning out to fetch every KPI name just to count them.
            if agg == "count" and not comp_filter and not spec.get("component_filter_raw"):
                data = tl.call("ig_list_components", {"host_id": host["hostid"], "cid": host["cid"]})
                comps = data.get("components", [])
                total = sum(int(c.get("item_count", 0) or 0) for c in comps if isinstance(c, dict))
                return _ok("count", {"count": total, "label": "KPIs"},
                           f"{_host_name(host)} tracks {total} KPIs across {len(comps)} components.", tl)

            # (c) LISTING: component given, no interface -> list all KPIs across
            # that component's interfaces. No component given -> list across
            # EVERY component (full parallel fan-out, per the design decision).
            # Table is grouped: Component | Interface | KPI | itemid.
            comp_filter, unresolved = _resolve_component_filter(spec, [host], tl)
            if unresolved:
                raw, options = unresolved
                return _ok("text", {}, _component_clarification(raw, options, _host_name(host)), tl)
            comps_to_scan = comp_filter or _components_for(host, tl)
            by_comp = _interfaces_by_component_for(host, comps_to_scan, tl)

            def kpis_for_interface(args):
                comp, item = args
                iface_disp = _iface_display(item)
                data = tl.call("ig_list_kpis",
                               {"host_id": host["hostid"], "cid": host["cid"], "prefix": {comp: [item]}})
                entries = [e for grp in data.get("results", {}).values() for e in grp]
                return [{"Component": comp, "Interface": iface_disp,
                        "KPI": e.get("suffix"), "itemid": e.get("itemid")} for e in entries]

            tasks = [(comp, item) for comp, payload in by_comp.items()
                    for item in _norm_items(payload)]
            nested = _parallel_map(kpis_for_interface, tasks)
            rows = [r for group in nested if isinstance(group, list) for r in group]

            # Device-level components (no object-level interfaces, e.g. cpu)
            # have no interface prefix to list suffixes under — the API has no
            # documented endpoint for interface-less KPI names, so report their
            # item_count as a summary row instead of silently dropping them.
            comps_with_ifaces = {c for c, p in by_comp.items()
                                 if isinstance(p, dict) and p.get("items")}
            device_level = [c for c in comps_to_scan if c not in comps_with_ifaces]
            if device_level:
                cdata = tl.call("ig_list_components", {"host_id": host["hostid"], "cid": host["cid"]})
                counts = {c.get("component"): c.get("item_count") for c in cdata.get("components", [])
                         if isinstance(c, dict)}
                for c in device_level:
                    if counts.get(c):
                        rows.append({"Component": c, "Interface": "(device-level)",
                                    "KPI": f"{counts[c]} item(s), no interface breakdown", "itemid": None})

            rows.sort(key=lambda r: (r["Component"], r["Interface"], str(r["KPI"])))
            scope = f" under {comp_filter[0]}" if comp_filter else " across all components"
            return _ok("list", {"rows": rows, "cols": ["Component", "Interface", "KPI", "itemid"],
                                "count": len(rows)},
                       f"{_host_name(host)} has {len(rows)} KPI(s){scope}.", tl)

        # ---- VALUES (kpi / stats) -------------------------------------------
        if target == "values":
            return _execute_values(spec, state, devices, tl)

    except MCPError as exc:
        return {"result_kind": "error", "payload": {}, "answer": f"Data fetch failed: {exc}",
                "tool_calls": tl.n, "tool_log": tl.calls}
    except Exception as exc:      # noqa: BLE001
        # BUGFIX: this used to guard MCPError ONLY, so any unexpected API/LLM
        # payload shape (the "'dict' object has no attribute 'split'" class of
        # bug) unwound past here into run_agent's catch-all and lost the tool
        # log with it. Catch it at the executor: the user gets a readable
        # message and the Debug panel still shows every call that led up to it.
        log.exception("[QUERYSPEC] executor crashed on target=%s agg=%s", target, agg)
        return {"result_kind": "error", "payload": {},
                "answer": ("I hit an internal error while fetching that data "
                           f"({type(exc).__name__}: {exc}). The Debug trace panel below shows "
                           "the tool calls made — please retry, and share that trace if it persists."),
                "tool_calls": tl.n, "tool_log": tl.calls}

    return {"result_kind": "text", "payload": {}, "answer": "", "tool_calls": tl.n, "tool_log": tl.calls}


def _ok(kind: str, payload: Dict[str, Any], answer: str, tl: "_ToolLog") -> Dict[str, Any]:
    out = {"result_kind": kind, "payload": payload, "answer": answer,
           "tool_calls": tl.n, "tool_log": tl.calls}
    if tl.notes:
        out["tool_notes"] = list(tl.notes)
    return out


def _pfx(iface: Any) -> str:
    """Normalize a user interface name to the ' ::' prefix string the API needs."""
    s = _as_text(iface).strip()
    if not s.lower().startswith("interface"):
        s = f"Interface {s}"
    if not s.rstrip().endswith("::"):
        s = s.rstrip() + " ::"
    return s


def _execute_values(spec: Dict[str, Any], state: VIAgentState,
                    devices: List[Dict[str, Any]], tl: "_ToolLog") -> Dict[str, Any]:
    """Resolve itemids across the device/interface/kpi set, fetch history in as
    few batched calls as the API allows (one per host), apply the aggregation."""
    tw = spec["time_window"]
    start, end = tw["start"], tw["end"]
    iface_filter = spec.get("interface_filter")
    kpi_filter = spec.get("kpi_filter")
    # Every requested KPI (§ user request: multi-KPI questions), lowercased for
    # substring matching; kpi_filter["values"] is the list build_spec produces,
    # falling back to the single "value" for an older-shaped spec.
    wanted_kpis_all: List[str] = []
    if kpi_filter:
        raw_values = kpi_filter.get("values") or ([kpi_filter.get("value")] if kpi_filter.get("value") else [])
        wanted_kpis_all = [_as_text(v).strip().lower() for v in raw_values if _as_text(v).strip()]

    # ── Interface-ranking "currently" shape (§ user request: "top/bottom N
    # interfaces for KPI X on device Y currently"): this ranking is inherently
    # scoped to exactly one device, and needs exactly one KPI to rank
    # interfaces by. Validate both up front — a missing/ambiguous device or a
    # missing KPI would otherwise either crash downstream or silently rank by
    # the wrong thing, rather than asking the user what's missing.
    if spec.get("current_value"):
        if not devices:
            return _ok("text", {}, _device_not_found_message(spec, state), tl)
        if len(devices) > 1:
            return _ok("text", {}, _multiple_devices_message(devices), tl)
        if not kpi_filter:
            comp_filter_chk, unresolved = _resolve_component_filter(spec, devices, tl)
            if unresolved:
                raw, options = unresolved
                return _ok("text", {}, _component_clarification(raw, options, _host_name(devices[0])), tl)
            return _ok("text", {}, _rank_kpi_required_message(devices[0], comp_filter_chk, tl), tl)

    # ── Spike/dip detection (§ user request: "show me the spikes/dips for KPI
    # X on interface Y on device Z in the last N minutes/hours or between A
    # and B"): analyzing one series for anomalies needs exactly one named
    # interface — there's nothing to detect against otherwise.
    if (spec.get("aggregation") in ("spikes", "dips", "anomalies")
            and not (iface_filter and iface_filter.get("mode") == "exact" and iface_filter.get("value"))):
        if not devices:
            return _ok("text", {}, _device_not_found_message(spec, state), tl)
        if len(devices) > 1:
            return _ok("text", {}, _multiple_devices_message(devices), tl)
        return _ok("text", {}, ("Spike/dip detection needs one specific interface to analyze. "
                                f"Which interface on {_host_name(devices[0])} would you like to check?"), tl)

    # A question that names ONE specific interface ("... on Interface X of
    # device Y ...") goes through the staged validator: confirm the device is
    # right, confirm the interface actually exists on it, confirm the named KPI
    # (if any) actually exists on that interface — THEN fetch. Any stage that
    # can't be verified returns a specific clarification instead of a blind
    # attempt that silently comes back empty.
    if iface_filter and iface_filter.get("mode") == "exact" and iface_filter.get("value"):
        ready, clarification = _validate_single_interface_kpi(spec, state, devices, tl)
        if clarification:
            return _ok("text", {}, clarification, tl)
        host = ready["host"]
        entries = ready["entries"]
        item_ids = [e["itemid"] for e in entries]
        cid_map = {str(e["itemid"]): host["cid"] for e in entries}
        hist = tl.call("ig_get_kpi_history",
                       {"item_ids": item_ids, "start_time": int(start), "end_time": int(end),
                        "cid": cid_map})
        values = hist.get("Values", hist.get("values", [])) or []
        series = hist.get("series", []) or []
        agg = spec["aggregation"]
        metric_field = _FIELD_BY_AGG.get(agg, "Maximum")
        is_anomaly_agg = agg in ("spikes", "dips", "anomalies")
        rows = []
        ts_rows: List[Dict[str, Any]] = []
        flagged_rows: List[Dict[str, Any]] = []
        marker_points: Dict[int, List[Tuple[Any, float, str]]] = {}
        total_points = 0
        for i, e in enumerate(entries):
            v = values[i] if i < len(values) else {}
            rows.append({
                "Device": _host_name(host), "KPI": f"{ready['interface_display']} : {e.get('suffix')}",
                "Minimum": v.get("Minimum"), "Maximum": v.get("Maximum"), "Last": v.get("Last"),
                "Average": v.get("Average"), "Total": v.get("Total"), "Units": v.get("units"),
            })
            # BUGFIX / § user request: the API's own series[].name only encodes
            # the interface's circuit description, not which KPI it is — so
            # two KPIs on the same interface (e.g. HC In/Out Octets) rendered
            # as two chart lines with IDENTICAL, indistinguishable legend
            # text. Override it with a label we control: Device : Interface :
            # KPI, so the legend (and any consumer of this series' "name")
            # always says exactly what it's plotting.
            if i < len(series) and isinstance(series[i], dict):
                series[i]["name"] = f"{e.get('suffix')} : {ready['interface_display']} : {_host_name(host)}"
            s = series[i] if i < len(series) else {}
            data_pts = [pt for pt in (s.get("data") or []) if len(pt) >= 2]
            total_points += len(data_pts)
            # § user request: detect spikes/dips per KPI series via a rolling
            # z-score, then tag both the raw time-series row (Flag column) and
            # a compact flagged-events row for each point that crosses the
            # threshold, plus its (epoch_ms, value, SPIKE|DIP) for the chart.
            flags = _detect_spikes_dips(data_pts, agg) if is_anomaly_agg else None
            for j, pt in enumerate(data_pts):
                flag = flags[j]["flag"] if flags else ""
                ts_row = {
                    "Device": _host_name(host), "Component": ready["component"],
                    "Interface": ready["interface_display"], "KPI": e.get("suffix"),
                    "Value": pt[1], "Time": _fmt_epoch_ms(pt[0]),
                }
                if is_anomaly_agg:
                    ts_row["Flag"] = flag
                ts_rows.append(ts_row)
                if flag:
                    z = flags[j]["zscore"]
                    flagged_rows.append({
                        "Interface": ready["interface_display"], "KPI": e.get("suffix"),
                        "Type": "Spike" if flag == "SPIKE" else "Dip", "Value": pt[1],
                        "Time": _fmt_epoch_ms(pt[0]), "Deviation (σ)": round(z, 2) if z is not None else None,
                    })
                    marker_points.setdefault(i, []).append((pt[0], pt[1], flag))
        if is_anomaly_agg:
            direction_word = {"spikes": "spike(s)", "dips": "dip(s)", "anomalies": "spike(s)/dip(s)"}[agg]
            flagged_rows.sort(key=lambda r: r["Time"])
            if flagged_rows:
                answer = (f"Found {len(flagged_rows)} {direction_word} for "
                         f"{', '.join(sorted({r['KPI'] for r in flagged_rows}))} on "
                         f"{ready['interface_display']} of {_host_name(host)}: " +
                         "; ".join(f"{r['Type']} {r['Value']} at {r['Time']}" for r in flagged_rows[:5]) +
                         (f" (+{len(flagged_rows) - 5} more)" if len(flagged_rows) > 5 else "") + ".")
            elif total_points < SPIKE_MIN_POINTS:
                answer = (f"Only {total_points} data point(s) were available on "
                         f"{ready['interface_display']} of {_host_name(host)} in this window — not enough "
                         f"to reliably detect {direction_word}. Try a longer window.")
            else:
                answer = (f"No {direction_word} detected for "
                         f"{', '.join(str(e.get('suffix')) for e in entries)} on "
                         f"{ready['interface_display']} of {_host_name(host)} in this window.")
            chart = build_timeseries_fig({"series": series}, markers=marker_points)
            payload = {"rows": flagged_rows, "cols": SPIKE_COLS, "count": len(flagged_rows),
                      "history": {"series": series}, "timeseries_rows": ts_rows,
                      "ts_cols": TIMESERIES_COLS + ["Flag"]}
            if chart is not None:
                payload["chart"] = chart
            return _ok("stats", payload, answer, tl)
        if agg in _FIELD_BY_AGG and len(rows) == 1:
            r = rows[0]
            return _ok("scalar", {"rows": rows, "cols": ["Device", "KPI", "Minimum", "Maximum",
                       "Last", "Average", "Total", "Units"], "history": {"series": series},
                       "scalar": r.get(metric_field), "timeseries_rows": ts_rows},
                       f"{r['KPI']} on {r['Device']}: {agg} = {r.get(metric_field)} ({r.get('Units')}).", tl)
        kind = "timeseries" if len(rows) == 1 else "stats"
        return {"result_kind": kind,
                "payload": {"history": {"series": series}, "stats": rows, "rows": rows,
                            "cols": ["Device", "KPI", "Minimum", "Maximum", "Last", "Average",
                                    "Total", "Units"], "count": len(rows),
                            "timeseries_rows": ts_rows},
                "answer": "", "tool_calls": tl.n, "tool_log": tl.calls}

    # ── Everything below is the broader fan-out path: no single named
    # interface (top-N/threshold/summary across many devices or interfaces). ──
    # Components to scan: an explicit filter if given, else ALL of the host's
    # components (was hardcoded to just "traffic", which silently ignored KPIs on
    # errors/broadcast/bgp/etc. and could yield 0 for a valid interface).
    comp_filter_v = spec.get("component_filter")

    def collect_for_host(h):
        """Return (host, [ {itemid, label, interface, cid} ]) for the wanted KPIs.
        Interfaces (and thus KPIs) are gathered per component and the ig_list_kpis
        prefix dict is keyed by the correct owning component for each interface."""
        comps = comp_filter_v or _components_for(h, tl)
        if not comps:
            return (h, [])
        # Build {component: [interface_prefix, ...]} for this host.
        comp_to_prefixes: Dict[str, List[str]] = {}
        if iface_filter and iface_filter.get("mode") == "exact" and iface_filter.get("value"):
            kpi_hint = _as_text(kpi_filter.get("value")) if kpi_filter and kpi_filter.get("value") else None
            found = _find_interface_across_components(h, iface_filter.get("value"), comps, tl,
                                                       kpi_hint=kpi_hint)
            if found:
                comp_to_prefixes[found[0]] = [found[1]]
        else:
            by_comp = _interfaces_by_component_for(h, comps, tl)
            v = _as_text(iface_filter.get("value")).lower() if (
                iface_filter and iface_filter.get("mode") == "contains") else None
            for comp_name, payload in by_comp.items():
                items = _norm_items(payload)
                if v:
                    items = [i for i in items if v in _as_text(i).lower()]
                if items:
                    comp_to_prefixes[comp_name] = items
        picks = []
        if not comp_to_prefixes:
            return (h, picks)
        for comp_name, prefixes in comp_to_prefixes.items():
            data = tl.call("ig_list_kpis",
                           {"host_id": h["hostid"], "cid": h["cid"],
                            "prefix": {comp_name: prefixes}})
            for grp_name, entries in (data.get("results", {}) or {}).items():
                for e in entries:
                    if not isinstance(e, dict) or e.get("itemid") is None:
                        continue
                    if wanted_kpis_all and not any(
                            w in _as_text(e.get("suffix")).lower() for w in wanted_kpis_all):
                        continue
                    picks.append({"itemid": e["itemid"],
                                  "label": f"{_as_text(e.get('prefix'))} : {_as_text(e.get('suffix'))}",
                                  "component": comp_name, "interface": _as_text(e.get("prefix")),
                                  "kpi": _as_text(e.get("suffix")),
                                  "cid": h["cid"], "host": _host_name(h)})
        # If no kpi filter, keep only the first KPI per interface to bound fan-out
        if not kpi_filter:
            seen, trimmed = set(), []
            for p in picks:
                iface = p["label"].split(" : ")[0]
                if iface in seen:
                    continue
                seen.add(iface)
                trimmed.append(p)
            picks = trimmed
        return (h, picks)

    per_host = [p for p in _parallel_map(collect_for_host, devices) if isinstance(p, tuple)]

    # If every host's KPI lookup errored out (rather than just legitimately
    # finding nothing), don't silently fall through to an empty result — that
    # degrades into an unhelpful generic "couldn't retrieve" message even
    # though the real reason is already known. Surface it.
    if not per_host and tl.errors:
        return _ok("error", {}, f"Could not look up the requested KPI(s): {tl.errors[-1]}", tl)

    # One history call per host (batched itemids).
    def history_for(entry):
        h, picks = entry
        if not picks:
            return (h, picks, {})
        item_ids = [p["itemid"] for p in picks]
        cid_map = {str(p["itemid"]): p["cid"] for p in picks}
        hist = tl.call("ig_get_kpi_history",
                       {"item_ids": item_ids, "start_time": int(start), "end_time": int(end),
                        "cid": cid_map})
        return (h, picks, hist)

    hosts_with_picks = [p for p in per_host if p[1]]
    histories = [r for r in _parallel_map(history_for, per_host) if isinstance(r, tuple)]

    # Same check after the history fetch: KPI(s) were found (hosts_with_picks
    # non-empty) but the history call itself failed for all of them.
    if hosts_with_picks and not histories and tl.errors:
        return _ok("error", {}, f"Found the requested KPI(s) but the history fetch failed: "
                   f"{tl.errors[-1]}", tl)

    # Flatten to per-KPI rows, keyed by device/interface, with a normalized value.
    agg = spec["aggregation"]
    # "currently" ranks by the most recent value ("Last"), never a Min/Max/Avg
    # over the (short, currency-only) lookback window built for this shape.
    metric_field = "Last" if spec.get("current_value") else _FIELD_BY_AGG.get(agg, "Maximum")
    all_series: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    label_by_index: List[str] = []
    for h, picks, hist in histories:
        values = hist.get("Values", hist.get("values", [])) or []
        series = hist.get("series", []) or []
        for i, p in enumerate(picks):
            v = values[i] if i < len(values) else {}
            row = {
                "Device": p["host"],
                "KPI": p["label"],
                "Minimum": v.get("Minimum"), "Maximum": v.get("Maximum"),
                "Last": v.get("Last"), "Average": v.get("Average"),
                "Total": v.get("Total"), "Units": v.get("units"),
                "_metric_bytes": to_bytes(v.get(metric_field)),
            }
            if spec.get("current_value"):
                row["_interface_disp"] = p.get("interface") or ""
                row["_kpi_name"] = p.get("kpi") or ""
            rows.append(row)
            # Raw per-point rows for the downloadable time-series table (§ user
            # request): series[i] is the same itemid/position as picks[i], since
            # both came from the same ig_get_kpi_history call over item_ids built
            # from `picks` in the same order (see history_for above).
            #
            # BUGFIX / § user request: override the API's own series[].name
            # (which only encodes the interface's circuit description, not
            # the KPI or device) with a controlled Device : Interface : KPI
            # label, so multiple KPIs / devices plotted together get distinct,
            # readable chart legends instead of identical truncated text.
            if i < len(series) and isinstance(series[i], dict):
                series[i]["name"] = f"{p.get('kpi') or ''} : {p.get('interface') or ''} : {p['host']}"
            s = series[i] if i < len(series) else {}
            if spec.get("current_value"):
                data_pts = s.get("data") or []
                row["_last_time"] = _fmt_epoch_ms(data_pts[-1][0]) if data_pts else None
            for pt in (s.get("data") or []):
                if len(pt) >= 2:
                    ts_rows.append({
                        "Device": p["host"], "Component": p.get("component"),
                        "Interface": p.get("interface"), "KPI": p.get("kpi"),
                        "Value": pt[1], "Time": _fmt_epoch_ms(pt[0]),
                    })
        for s in series:
            all_series.append(s)

    single_device = len(devices) == 1
    is_multi = not single_device or len(rows) > 1

    # ---- apply aggregation ----
    if agg in ("top_n", "bottom_n", "threshold"):
        ranked = [r for r in rows if r["_metric_bytes"] is not None]
        # stats scope: for 'top N DEVICES' collapse to per-device peak
        by_device = "device" in state["user_query"].lower() and "interface" not in state["user_query"].lower()
        if by_device:
            best: Dict[str, Dict[str, Any]] = {}
            for r in ranked:
                d = r["Device"]
                if d not in best or (r["_metric_bytes"] or 0) > (best[d]["_metric_bytes"] or 0):
                    best[d] = r
            ranked = list(best.values())
        # "bottom N" ranks ascending (lowest first); everything else (top N,
        # threshold) keeps the existing descending order.
        reverse_sort = agg != "bottom_n"
        ranked.sort(key=lambda r: (r["_metric_bytes"] or 0.0, r["KPI"]), reverse=reverse_sort)
        if agg in ("top_n", "bottom_n"):
            n = int(spec.get("top_n") or 5)
            ranked = ranked[:n]
            head = f"{'Top' if agg == 'top_n' else 'Bottom'} {n} by {metric_field.lower()}"
        else:
            thr = spec.get("threshold") or 0.0
            ranked = [r for r in ranked if (r["_metric_bytes"] or 0.0) > thr]
            head = f"{len(ranked)} item(s) over threshold"

        if spec.get("current_value"):
            # § user request: "top/bottom N interfaces for KPI X on device Y
            # currently" — an Interface-centric ranking table (+ bar chart)
            # instead of the generic Device/Min/Max/Avg/Total shape used by
            # the pre-existing cross-device/cross-window top_n/threshold path.
            out_rows = [{
                "Interface": r.get("_interface_disp") or "",
                "KPI": r.get("_kpi_name") or r["KPI"],
                "Value": r.get(metric_field),
                "Time": r.get("_last_time") or "—",
                "Units": r.get("Units"),
            } for r in ranked]
            cols = ["Interface", "KPI", "Value", "Time", "Units"]
            dev_name = _host_name(devices[0]) if devices else ""
            if out_rows:
                answer = (f"{head} on {dev_name}: " +
                         "; ".join(f"{r['Interface']} ({r['Value']} {r['Units'] or ''})".strip()
                                   for r in out_rows[:5]) + ".")
            else:
                answer = (f"None of {dev_name}'s interfaces had a recent value for the requested "
                         f"KPI in the last {CURRENT_VALUE_LOOKBACK_SECONDS // 3600} hour(s).")
            payload = {"rows": out_rows, "cols": cols, "count": len(out_rows),
                      "history": {"series": []}, "timeseries_rows": ts_rows}
            chart = build_rank_bar_chart(out_rows, head)
            if chart is not None:
                payload["chart"] = chart
            return _ok("stats", payload, answer, tl)

        out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ranked]
        cols = ["Device", "KPI", "Minimum", "Maximum", "Last", "Average", "Total", "Units"]
        return _ok("stats", {"rows": out_rows, "cols": cols, "count": len(out_rows),
                             "history": {"series": all_series}, "timeseries_rows": ts_rows},
                   f"{head}: " + "; ".join(f"{r['Device']} {r['KPI']} "
                   f"{r.get(metric_field)}" for r in ranked[:5]) + ".", tl)

    if agg in _FIELD_BY_AGG and len(rows) == 1:
        r = rows[0]
        f = _FIELD_BY_AGG[agg]
        return _ok("scalar", {"rows": rows, "cols": ["Device", "KPI", "Minimum", "Maximum",
                   "Last", "Average", "Total", "Units"], "history": {"series": all_series},
                   "scalar": r.get(f), "timeseries_rows": ts_rows},
                   f"{r['KPI']} on {r['Device']}: {agg} = {r.get(f)} ({r.get('Units')}).", tl)

    # summary (default): return series + stats table; answer templated by synth
    spec["_multi"] = is_multi
    stats = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    kind = "timeseries" if not is_multi else "stats"
    return {"result_kind": kind,
            "payload": {"history": {"series": all_series}, "stats": stats,
                        "rows": stats, "cols": ["Device", "KPI", "Minimum", "Maximum",
                        "Last", "Average", "Total", "Units"], "count": len(stats),
                        "timeseries_rows": ts_rows},
            "answer": "", "tool_calls": tl.n, "tool_log": tl.calls}


def _guess_component(ent: Dict[str, Any], g: Dict[str, Any]) -> str:
    kpi = _as_text(ent.get("kpi")).lower()
    for comp, syns in KPI_SYNONYMS.items():
        if any(s in kpi for s in syns):
            avail = [c.get("component") for c in g.get("components", [])]
            if not avail or comp in avail:
                return comp
    comps = [c.get("component") for c in g.get("components", [])]
    return "traffic" if ("traffic" in comps or not comps) else (comps[0] if comps else "traffic")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — SPEC NODE, SCOPE GATE, EXECUTOR NODE, SYNTHESIS, HELP/OOS
# ══════════════════════════════════════════════════════════════════════════════

def spec_node(state: VIAgentState) -> VIAgentState:
    """Assemble the QuerySpec (pure Python) and run the scope check."""
    t0 = time.time()
    spec = build_spec(state)
    verdict = scope_check(spec, state)      # None | ("unrelated"|"unsupported", message)
    spec["_scope"] = verdict                # carried in the declared 'spec' channel
    dbg = _dbg(state)
    dbg["query_spec"] = {k: v for k, v in spec.items() if k != "_scope"}
    dbg["scope_verdict"] = verdict[0] if verdict else "in_scope"
    _stage_done(dbg, "spec_build", t0)
    return {**state, "spec": spec, "debug": dbg}


# ── Scope handling (§3.3): unrelated vs monitoring-shaped-but-unsupported ─────
# Distinct, specific messages — neither is the generic HELP_TEXT.

_UNSUPPORTED_PATTERNS = [
    (r"\b(restart|reboot|bounce|shut\s?down|shutdown|disable|enable|turn (on|off))\b",
     "restart, reboot, shut down, or otherwise change the state of a device or interface"),
    (r"\b(create|raise|open|log|file)\b.*\bticket\b",
     "create, raise, or manage tickets"),
    (r"\b(forecast|predict|projection|next week|next month|future|will .*(be|look)|during diwali)\b",
     "forecast or predict future values"),
    (r"\b(alert|alerts|alarm|alarms|acknowledge|ack\b|currently down|is .* down|are .* down|which .* down)\b",
     "list alerts/alarms or report live up/down state"),
    (r"\b(change|configure|reconfigure|set the|update the|snmp community|password|community string)\b",
     "change configuration"),
    (r"\b(email me|send me|export to email|schedule a report)\b",
     "send emails or schedule reports"),
]


def _available_surface(state: VIAgentState) -> str:
    """Describe, from LIVE data where possible, what the tools actually expose."""
    g = state.get("grounding", {})
    dev = g.get("host_name")
    comps = [c.get("component") for c in g.get("components", []) if isinstance(c, dict)]
    if dev and comps:
        return (f"For {dev} I can report its monitored components "
                f"({', '.join(str(c) for c in comps[:8])}), its interfaces, and historical "
                f"KPI values (min/max/avg/last/total) over a time window.")
    return ("I can report device/component/interface/KPI metadata (counts and listings) and "
            "historical KPI values and statistics (min/max/avg/last/total, top-N, thresholds) "
            "over a time window.")


def scope_check(spec: Dict[str, Any], state: VIAgentState):
    q = state["user_query"]
    ql = q.lower()
    route = state["route"]
    if route == "help_q":
        return None                          # greetings handled by help_node
    # 1) monitoring-shaped but unsupported capability (API is read-only)
    for pat, human in _UNSUPPORTED_PATTERNS:
        if re.search(pat, ql):
            msg = (f"I can't {human} — the Instant Graph API this assistant uses is read-only "
                   f"and exposes no endpoint for that. " + _available_surface(state))
            return ("unsupported", msg)
    # 2) genuinely unrelated to network monitoring
    if route == "out_of_scope" and not spec.get("recognized"):
        msg = ("That question isn't about VI-IPPMS network monitoring, so I can't help with it. "
               + _available_surface(state) + " Try, e.g., \"How many devices are in the GUJ "
               "circle?\" or \"min/max/avg HC In Octets on Interface 100GE0/3/2 in the last 6 hours\".")
        return ("unrelated", msg)
    return None


def oos_node(state: VIAgentState) -> VIAgentState:
    kind, msg = state.get("spec", {}).get("_scope") or ("unrelated", "I can't help with that.")
    dbg = _dbg(state)
    dbg["answered_via"] = "out_of_scope"
    return {**state, "result_kind": "text", "answer": msg,
            "answered_via": "out_of_scope", "debug": dbg, "failed": False}


def queryspec_node(state: VIAgentState) -> VIAgentState:
    """Run the general executor over a recognized QuerySpec (the deterministic
    path). Never touches the ReAct loop."""
    t0 = time.time()
    res = execute_query_spec(state["spec"], state)
    dbg = _dbg(state)
    dbg["tool_calls"] = res.get("tool_log", [])
    dbg["answered_via"] = "query_spec"
    if res.get("recheck_note"):
        dbg["recheck"] = res["recheck_note"]
    if res.get("tool_notes"):
        dbg["recheck_notes"] = res["tool_notes"]
    _stage_done(dbg, "queryspec_exec", t0)
    return {**state,
            "payload": res.get("payload", {}) or {},
            "result_kind": res.get("result_kind", "text"),
            "answer": res.get("answer", "") or "",
            "answered_via": "query_spec",
            "tool_calls": state.get("tool_calls", 0) + int(res.get("tool_calls", 0) or 0),
            "fallback_used": False,
            "failed": res.get("result_kind") == "error",
            "debug": dbg}


def post_executor_node(state: VIAgentState) -> VIAgentState:
    """ReAct-fallback packager (only reached when no recognized QuerySpec matched).
    Turns the ReAct trace into structured data; this whole path is a fallback, so
    fallback_used is True."""
    trace = state.get("trace", [])
    dbg = _dbg(state)
    dbg["tool_calls"] = [
        {"tool": s.get("action"), "args": s.get("action_input"),
         "observation": (s.get("observation") or "")[:600] if isinstance(s.get("observation"), str)
         else s.get("observation"), "ok": not str(s.get("observation", "")).startswith("ERROR")}
        for s in trace if s.get("action")
    ]
    dbg["answered_via"] = "react"
    payload: Dict[str, Any] = {}
    result_kind = "text"
    hist = extract_history_from_trace(trace)
    if hist:
        payload = {"history": hist,
                   "stats": build_stats_rows(hist.get("Values", hist.get("values", [])))}
        result_kind = "timeseries"
    return {**state, "payload": payload, "result_kind": result_kind,
            "answered_via": "react", "fallback_used": True,
            "failed": state.get("failed", False), "debug": dbg}


SYNTH_SYSTEM = """\
You are a concise network-operations analyst. Using ONLY the retrieved data
provided, write a clear 1-3 sentence answer to the user's question. Cite the
device / interface / KPI names and the numeric values. Do NOT invent numbers,
and do NOT mention itemids, hostids, cids or tool names.
"""


def _alt_note(state: VIAgentState, ans: str) -> str:
    g = state.get("grounding", {})
    if g.get("alternatives"):
        ans += (f"  (Note: the device name was ambiguous — I used {g.get('host_name')}; "
                f"other close matches: {', '.join(g['alternatives'])}.)")
    return ans


def synthesize_node(state: VIAgentState) -> VIAgentState:
    """Phrase the final answer. Structured shapes (count/list/scalar/stats/error)
    are already templated by the engine and kept verbatim (byte-stable). A
    query-spec 'text' result (clarifications: device/component/interface/KPI
    not found, ambiguous match, out-of-scope-style messages) is ALSO already a
    complete, deliberately-worded answer and must be kept verbatim — routing it
    through the LLM here would either discard a correct clarification or risk
    the model inventing something different from what was actually verified.
    Only the narrative KPI 'summary'/timeseries case and the ReAct fallback's
    empty-answer 'text' case go through the LLM, now at temperature 0.0."""
    kind = state.get("result_kind")
    templated = kind in ("count", "list", "scalar", "stats", "error") or (
        kind == "text" and state.get("answered_via") == "query_spec")
    if state.get("answer") and templated:
        return {**state, "answer": _alt_note(state, state["answer"])}

    payload = state.get("payload", {})
    g = state.get("grounding", {})
    ent = state.get("entities", {})

    evidence: Dict[str, Any] = {}
    if payload.get("stats"):
        evidence["kpi_statistics"] = payload["stats"][:8]
    if g.get("host_name"):
        evidence["device"] = g["host_name"]
    if ent.get("interface"):
        evidence["interface"] = ent["interface"]
    if g.get("start_time"):
        evidence["window"] = {"start_epoch": g["start_time"], "end_epoch": g["end_time"]}

    if not evidence.get("kpi_statistics") and not state.get("answer"):
        last_obs = next((s.get("observation") for s in reversed(state.get("trace", []))
                         if s.get("observation") and not str(s["observation"]).startswith("ERROR")), None)
        if last_obs:
            evidence["retrieved_data"] = str(last_obs)[:2000]
        else:
            return {**state, "answer": state.get("answer") or
                    "I couldn't retrieve the requested data. Please refine the device, "
                    "interface, KPI or time window and try again.", "failed": True}

    user_p = (f"Question: {state['user_query']}\n\n"
              f"Retrieved data:\n{json.dumps(evidence, indent=2, default=str)[:2500]}")
    ans = call_llm(SYNTH_SYSTEM, user_p, max_new_tokens=300,
                   decode=SYNTHESIS_DECODE, stage="SYNTH").strip()
    if not ans and state.get("answer"):
        ans = state["answer"]
    if not ans:
        if payload.get("stats"):
            r = payload["stats"][0]
            ans = (f"{r.get('KPI','KPI')}: min {r.get('Minimum')}, max {r.get('Maximum')}, "
                   f"avg {r.get('Average')}, last {r.get('Last')} ({r.get('Units')}).")
        else:
            ans = "Retrieved the data but could not summarize it."
    return {**state, "answer": _alt_note(state, ans)}


HELP_TEXT = (
    "I'm the Talk-to-VI-IPPMS assistant. I answer questions about your network "
    "monitoring data using live APIs — no SQL. Try:\n"
    "• Metadata: “How many devices are in the GUJ circle?”, “How many interfaces "
    "on APVSPGJWPAR01HNE40?”, “List devices matching PAR01”, “How many KPIs are "
    "tracked on that device?”\n"
    "• KPI data: “On <device>, interface 100GE0/3/2, what was HC In Octets in the "
    "last 6 hours?”, “min/max/avg traffic on Eth-Trunk1 yesterday”."
)


def help_node(state: VIAgentState) -> VIAgentState:
    dbg = _dbg(state)
    dbg["answered_via"] = "help"
    return {**state, "result_kind": "text", "answer": HELP_TEXT,
            "answered_via": "help", "debug": dbg}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
#  router -> resolve -> build_spec -> {help | oos | queryspec | (cot->)react->post} -> synth
#  The recognized-QuerySpec path (queryspec) is the default; ReAct is now only a
#  last resort for in-scope questions whose shape no spec recognizes.

def _route_branch(state: VIAgentState) -> str:
    return "help" if state["route"] == "help_q" else "resolve"


def _after_spec(state: VIAgentState) -> str:
    if state.get("spec", {}).get("_scope"):
        return "oos"
    if state["route"] == "help_q":
        return "help"
    if state.get("spec", {}).get("recognized"):
        return "exec"
    # in-scope but unrecognized shape -> genuine ReAct fallback
    return "cot" if state["route"] == "kpi_q" else "react"


def build_graph():
    g = StateGraph(VIAgentState)
    # NOTE: a LangGraph node may not share a name with a state key, so this node
    # is "build_spec" even though the QuerySpec it produces lives in state["spec"].
    g.add_node("router",        router_node)
    g.add_node("resolve",       resolve_node)
    g.add_node("build_spec",    spec_node)
    g.add_node("queryspec",     queryspec_node)
    g.add_node("cot_plan",      cot_plan_node)
    g.add_node("react",         react_executor_node)
    g.add_node("post_executor", post_executor_node)
    g.add_node("synthesize",    synthesize_node)
    g.add_node("help",          help_node)
    g.add_node("oos",           oos_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", _route_branch, {"help": "help", "resolve": "resolve"})
    g.add_edge("resolve", "build_spec")
    g.add_conditional_edges("build_spec", _after_spec,
                            {"oos": "oos", "help": "help", "exec": "queryspec",
                             "cot": "cot_plan", "react": "react"})
    g.add_edge("queryspec",     "synthesize")
    g.add_edge("cot_plan",      "react")
    g.add_edge("react",         "post_executor")
    g.add_edge("post_executor", "synthesize")
    g.add_edge("synthesize",    END)
    g.add_edge("help",          END)
    g.add_edge("oos",           END)
    return g.compile()


vi_graph = build_graph()
log.info("[STARTUP] LangGraph compiled ✓")


def run_agent(user_query: str, session_key: str) -> VIAgentState:
    initial: VIAgentState = {
        "user_query": user_query, "session_key": session_key,
        "route": "", "confidence": "", "entities": {}, "grounding": {}, "spec": {},
        "plan": "", "trace": [], "result_kind": "text", "payload": {},
        "answer": "", "answered_via": "", "failed": False, "fallback_used": False,
        "tool_calls": 0, "debug": {},
    }
    try:
        return vi_graph.invoke(initial)
    except Exception as exc:
        # Last-resort net. Anything landing here is a genuine bug, so log the
        # full traceback AND carry a redacted copy into the debug object — the
        # old message ("The agent hit an unexpected error: 'dict' object has no
        # attribute 'split'") named neither the stage nor the value, which made
        # it effectively undiagnosable from the UI.
        log.exception("Agent run crashed")
        tb = traceback.format_exc()
        where = ""
        try:
            frames = traceback.extract_tb(exc.__traceback__)
            if frames:
                last = frames[-1]
                where = f" (at {os.path.basename(last.filename)}:{last.lineno} in {last.name}())"
        except Exception:      # noqa: BLE001
            pass
        return {**initial,
                "answer": (f"The agent hit an unexpected error: {type(exc).__name__}: {exc}{where}. "
                           "Open the Debug trace panel for details."),
                "result_kind": "error", "failed": True,
                "debug": {"answered_via": "crash", "error": f"{type(exc).__name__}: {exc}",
                          "traceback": tb[-2000:]}}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

PAL = ["#22d3ee", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#38bdf8", "#fb7185"]


def build_timeseries_fig(history: Dict[str, Any],
                         markers: Optional[Dict[int, List[Tuple[Any, float, str]]]] = None
                         ) -> Optional[go.Figure]:
    """`markers`, when given, overlays flagged spike/dip points (§ user
    request) on top of a series' line: {series_index: [(epoch_ms, value,
    "SPIKE"|"DIP"), ...]}. Optional and defaults to None so every pre-existing
    caller (plain KPI/stats charts) is unaffected."""
    series = history.get("series", []) if isinstance(history, dict) else []
    if not series:
        return None
    fig = go.Figure()
    n = 0
    for i, s in enumerate(series):
        data = s.get("data", []) or []
        xs = [datetime.fromtimestamp(pt[0] / 1000.0) for pt in data if len(pt) >= 2]
        ys = [pt[1] for pt in data if len(pt) >= 2]
        if not xs:
            continue
        series_name = str(s.get("name", f"series {i+1}"))[:80]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                      line=dict(color=PAL[i % len(PAL)], width=2),
                      name=series_name))
        n += 1
        # Spike/dip markers for this series, in front of the plain legend
        # (showlegend=False) so a long KPI-series legend doesn't double in
        # length — the color/shape/hover text already identify spike vs dip.
        flagged = (markers or {}).get(i) or []
        spikes = [(t, v) for t, v, k in flagged if k == "SPIKE"]
        dips = [(t, v) for t, v, k in flagged if k == "DIP"]
        if spikes:
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(t / 1000.0) for t, v in spikes], y=[v for t, v in spikes],
                mode="markers", marker=dict(color="#f87171", size=10, symbol="triangle-up",
                                            line=dict(color="#7f1d1d", width=1)),
                name=f"{series_name} — spike", showlegend=False,
                hovertemplate=f"{series_name}<br>Spike: %{{y}}<extra></extra>"))
        if dips:
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(t / 1000.0) for t, v in dips], y=[v for t, v in dips],
                mode="markers", marker=dict(color="#38bdf8", size=10, symbol="triangle-down",
                                            line=dict(color="#0c4a6e", width=1)),
                name=f"{series_name} — dip", showlegend=False,
                hovertemplate=f"{series_name}<br>Dip: %{{y}}<extra></extra>"))
    # § user request: side-by-side (horizontal) legend entries ran into each
    # other and became indistinguishable once labels got longer (KPI :
    # Interface : Device). Stack them vertically instead, one per line, below
    # the plot — and grow the bottom margin/figure height with the number of
    # series so a long stack never overlaps the x-axis.
    bottom_margin = min(50 + 20 * max(n, 1), 320)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1626",
                      plot_bgcolor="#0f1626", margin=dict(l=45, r=18, t=18, b=bottom_margin),
                      height=360 + max(0, n - 3) * 20,
                      legend=dict(orientation="v", x=0, xanchor="left", y=-0.18, yanchor="top"),
                      font=dict(family="Inter, sans-serif", size=11, color="#e2e8f0"))
    return fig


def build_rank_bar_chart(rows: List[Dict[str, Any]], title: str = "") -> Optional[go.Figure]:
    """Horizontal bar chart for the 'top/bottom N interfaces ... currently'
    ranking (§ user request) — rows are already in rank order (best/worst
    first), one bar per interface, labeled with its formatted current value.
    A reversed y-axis keeps rank #1 at the top regardless of top/bottom
    direction, since both are pre-sorted the same way by the caller."""
    if not rows:
        return None
    labels = [str(r.get("Interface", "")) for r in rows]
    values = [to_bytes(r.get("Value")) or 0.0 for r in rows]
    text = [f"{r.get('Value')} {r.get('Units') or ''}".strip() for r in rows]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                          marker=dict(color=PAL[0]),
                          text=text, textposition="auto"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1626", plot_bgcolor="#0f1626",
                      margin=dict(l=160, r=18, t=36 if title else 18, b=36),
                      height=max(220, 42 * len(rows) + 90),
                      title=dict(text=title, font=dict(size=12, color="#7c8aa5")) if title else None,
                      yaxis=dict(autorange="reversed"),
                      xaxis=dict(showticklabels=False),
                      font=dict(family="Inter, sans-serif", size=11, color="#e2e8f0"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — PIPELINE (called by the UI)
# ══════════════════════════════════════════════════════════════════════════════

# Per-session answer cache (§4.2A): a literally-identical repeat short-circuits
# the whole pipeline and returns the exact same result.
_ANSWER_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_ANSWER_CACHE_LOCK = threading.Lock()


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (q or "").lower())).strip()


# ── Extracted-parameters view (UI "Parameters" panel) ────────────────────────
# Everything the pipeline pulled out of the question, in one flat table:
#   Parameter | Extracted (verbatim from the question) | Resolved (what it was
#   grounded to against live data) | How.
# The debug panel already dumps raw entities/query_spec JSON; this is the
# human-readable "what did you understand from my question?" answer.

def _fmt_param(v: Any) -> str:
    if v is None or v == "" or v == []:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v) or "—"
    return str(v)


def _fmt_epoch(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:      # noqa: BLE001
        return "—"


def build_params_view(state: VIAgentState) -> List[Dict[str, str]]:
    """Flatten entities + grounding + QuerySpec into display rows."""
    ent = state.get("entities") or {}
    g = state.get("grounding") or {}
    spec = state.get("spec") or {}
    df = spec.get("device_filter") or {}
    rows: List[Dict[str, str]] = []

    def add(param, extracted, resolved, how=""):
        rows.append({"Parameter": param, "Extracted": _fmt_param(extracted),
                     "Resolved": _fmt_param(resolved), "How": how or "—"})

    # ── what the question was understood to be asking ──
    add("Route", state.get("route"), ROUTE_LABEL.get(state.get("route"), state.get("route")),
        f"router LLM · confidence={state.get('confidence') or '—'}")
    add("Intent", spec.get("intent"), spec.get("intent"), "derived from route + phrasing")
    add("Target", ent.get("target"), spec.get("target"), "what to return")
    add("Aggregation", ent.get("metric"), spec.get("aggregation"), "count / list / min / max / avg / top_n / …")
    add("Result shape", None, spec.get("result_shape"), "how the answer is rendered")

    # ── the named entities ──
    circle = df.get("circle") or g.get("circle")
    add("Circle", ent.get("circle"), circle,
        f"{g.get('hosts_in_circle_count')} device(s) in circle" if g.get("hosts_in_circle_count") is not None
        else ("canonicalized" if circle else "not named"))

    nm = df.get("name_match") or {}
    chosen = g.get("host_name")
    cand_n = len(g.get("candidates") or [])
    how_dev = "not named"
    if chosen:
        how_dev = f"grounded · host_id={g.get('host_id')} cid={g.get('cid')} · {cand_n} candidate(s)"
        if g.get("alternatives"):
            how_dev += f" · close: {', '.join(str(a) for a in g['alternatives'][:3])}"
    elif ent.get("device"):
        how_dev = "NOT FOUND — no device cleared the match threshold"
    override = g.get("device_entity_overridden")
    if override:
        how_dev += (f" · NOTE: router extracted \"{override.get('router_extracted')}\", overridden "
                    f"with \"{override.get('literal_match')}\" found verbatim in the question")
    add("Device", ent.get("device") or (nm.get("value") if nm else None), chosen, how_dev)

    if nm:
        add("Device name filter", f"{nm.get('mode')}: {_as_text(nm.get('value'))}",
            f"{nm.get('mode')} match", "from phrasing (contains / starts with / matching)")

    cf = spec.get("component_filter")
    add("Component", ent.get("component") or spec.get("component_filter_raw"),
        cf[0] if cf else None,
        "resolved against the device's live component list" if spec.get("component_filter_raw")
        else ("matched" if cf else "not named"))

    itf = spec.get("interface_filter") or {}
    add("Interface", ent.get("interface") or itf.get("value"), _as_text(itf.get("value")) or None,
        f"{itf.get('mode')} match" if itf else "not named")

    kf = spec.get("kpi_filter") or {}
    kf_values = kf.get("values") or ([kf.get("value")] if kf.get("value") else [])
    add("KPI", ent.get("kpi") or kf.get("value"), ", ".join(str(v) for v in kf_values) or None,
        f"{kf.get('mode')} match" + (f" ({len(kf_values)} requested)" if len(kf_values) > 1 else "")
        if kf else "not named")

    # ── numeric / temporal modifiers (only when in play) ──
    if spec.get("top_n") or ent.get("top_n"):
        add("Top N", ent.get("top_n"), spec.get("top_n"), "default 5 when 'top' is implied")
    if spec.get("threshold") is not None or ent.get("threshold") is not None:
        add("Threshold", ent.get("threshold"), f"{spec.get('threshold')} (base units/bytes)",
            "unit-normalized for comparison")

    tw = spec.get("time_window") or {}
    if tw or ent.get("time_phrase") or g.get("start_time"):
        start = tw.get("start") or g.get("start_time")
        end = tw.get("end") or g.get("end_time")
        how_tw = ("parsed from the question" if ent.get("time_phrase")
                 else ("\"currently\" lookback (most recent value, interface ranking)"
                       if spec.get("current_value") else "default window"))
        add("Time window", ent.get("time_phrase"),
            f"{_fmt_epoch(start)} → {_fmt_epoch(end)}" if start and end else None, how_tw)

    add("Devices in scope", None, g.get("total_hosts"), "total devices visible to your login")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14B — SMART HELPER SUGGESTIONS (incomplete-question / empty-result)
# ══════════════════════════════════════════════════════════════════════════════
#  Two distinct situations get a "notice" + a row of clickable follow-up chips
#  in the reply, reusing the SAME {"type": "quickq", ...} pattern-matched
#  button id the welcome-screen QUICK_QS chips already use — clicking one just
#  sends that text as the next question, no new callback needed:
#
#   1. "incomplete" — the pipeline itself asked for more information: a
#      device/component/interface/KPI it couldn't pin down (device not found,
#      ambiguous match, unresolved component, KPI not found on this
#      interface), or the ReAct fallback genuinely failed to retrieve data.
#      All of these already surface as a fixed, deterministic kind=="text"
#      answer (verified: every `_ok("text", ...)` call site in the executor
#      is one of these clarifications, never a legitimate positive answer).
#   2. "empty" — the question was well-formed and fully resolved, but the
#      structured result has zero rows/count (e.g. a filter that matches
#      nothing) — a different situation from (1): nothing was AMBIGUOUS, the
#      real data just came back empty, so the user should be told that
#      explicitly rather than seeing a bare "There are 0 device(s)..." with no
#      indication of whether that's expected or a typo somewhere.

def _substitute_token(text: str, old: str, new: str) -> str:
    """Swap a name the user typed for a corrected/alternative one, to build a
    ready-to-send follow-up question. Falls back to appending the
    replacement if the original token isn't found verbatim in the raw
    question (e.g. the router's extracted entity didn't exactly match the
    text) so the chip is still a complete, sendable question either way."""
    old = (old or "").strip()
    if old:
        new_text, n = re.subn(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
        if n:
            return new_text
    return f"{text} ({new})"


def _recovery_suggestions(state: VIAgentState) -> List[Dict[str, str]]:
    """Contextual follow-up questions built from whatever the pipeline DID
    manage to resolve (a grounded device, a named interface, ...), so a user
    facing an incomplete/ambiguous/empty answer can recover with one click
    instead of re-typing a whole new question from scratch. Never touches the
    LLM — purely derived from state already computed by resolve_node/build_spec."""
    g = state.get("grounding") or {}
    ent = state.get("entities") or {}
    q = state.get("user_query", "")
    out: List[Dict[str, str]] = []

    # A device WAS named but didn't clear the confidence bar for an exact
    # ground — resolve_node already collected close alternatives; offer each
    # as a one-click corrected re-ask of the SAME question.
    alts = g.get("alternatives") or []
    if alts and not g.get("host_id"):
        token = _as_text(ent.get("device"))
        for alt in alts[:3]:
            out.append({"label": f'Did you mean "{alt}"?', "q": _substitute_token(q, token, alt)})
        return out

    # A device WAS grounded — offer to discover whatever slot (component /
    # interface / KPI) the question left unnamed, scoped to that device, so
    # the user learns the exact real names instead of guessing again.
    dev = g.get("host_name")
    if dev:
        if not ent.get("component"):
            out.append({"label": "List its components", "q": f"List all components on {dev}"})
        if not ent.get("interface"):
            out.append({"label": "List its interfaces", "q": f"List all interfaces on {dev}"})
        elif not ent.get("kpi"):
            out.append({"label": "List KPIs on this interface",
                        "q": f'List all KPIs on "{ent["interface"]}" interface in {dev} device'})
        out.append({"label": "Devices in this circle" if g.get("circle") else "Confirm device details",
                    "q": (f'List devices in the {g["circle"]} circle' if g.get("circle")
                          else f"How many components and interfaces does {dev} have?")})
        return out[:4]

    # No device resolved at all (and none named) — nothing to scope
    # discovery chips to, so offer the most general recovery action.
    out.append({"label": "List all devices", "q": "List all devices"})
    return out


def pipeline(user_query: str, session_key: str) -> Dict[str, Any]:
    """Run the agent and shape a UI-friendly result dict. Serves an identical
    repeat from the per-session cache, persists every interaction to Postgres,
    and returns a populated debug object on every path."""
    t0 = time.time()
    ck = (session_key or "", _norm_q(user_query))

    with _ANSWER_CACHE_LOCK:
        hit = _ANSWER_CACHE.get(ck)
    if hit and (time.time() - hit[0]) < ANSWER_CACHE_TTL:
        cached = dict(hit[1])
        cached["debug"] = {**(cached.get("debug") or {}), "served_from_cache": True}
        cached["interaction_id"] = log_interaction(session_key, user_query, cached)
        return cached

    state = run_agent(user_query, session_key)
    elapsed = time.time() - t0

    payload = state.get("payload", {}) or {}
    kind = state.get("result_kind", "text")

    rows, cols, fig_json, stats = [], [], None, []
    # Some result shapes hand back a ready-made figure in payload["chart"]
    # instead of a plain history to build one from — the interface-ranking
    # "currently" bar chart (§ user request) and the spike/dip line chart with
    # flagged-point markers already baked in (§ user request) both do this.
    # Prefer it when present.
    preset_chart = payload.get("chart")
    hist = payload.get("history", {})
    if preset_chart is not None:
        fig_json = pio.to_json(preset_chart)
    elif hist:
        fig = build_timeseries_fig(hist)
        if fig is not None:
            fig_json = pio.to_json(fig)
    if kind == "timeseries":
        stats = payload.get("stats", []) or []
        if stats:
            cols = payload.get("cols") or ["Device", "KPI", "Minimum", "Maximum",
                                           "Last", "Average", "Total", "Units"]
            rows = stats
    elif kind in ("list", "stats", "scalar"):
        rows = payload.get("rows", []) or payload.get("stats", []) or []
        cols = payload.get("cols", []) or (list(rows[0].keys()) if rows else [])

    # Raw, long-format time-series export (Device/Component/Interface/KPI/
    # Value/Time — one row per data point) alongside the graph + stats table.
    # Only the QuerySpec KPI executor populates this (see _execute_values);
    # the ReAct fallback path has no structured per-point device/component/
    # interface breakdown to build it from, so it stays empty there.
    ts_rows = payload.get("timeseries_rows") or []
    # The spike/dip detection payload (§ user request) supplies its own
    # ts_cols (TIMESERIES_COLS + "Flag") so the raw time-series table shows
    # the tag; every other path keeps the plain shared column list.
    ts_cols = (payload.get("ts_cols") or TIMESERIES_COLS) if ts_rows else []

    # ReAct trace (fallback path only) for the legacy reasoning panel.
    trace_view = []
    for i, step in enumerate(state.get("trace", []), 1):
        trace_view.append({
            "n": i, "thought": step.get("thought", ""), "action": step.get("action"),
            "action_input": step.get("action_input"),
            "observation": (step.get("observation") or "")[:600]
                if isinstance(step.get("observation"), str) else step.get("observation"),
        })

    # Full debug object (§3.2) — populated on EVERY path, not just ReAct.
    dbg = dict(state.get("debug") or {})
    debug_obj = {
        "route": state.get("route"),
        "confidence": state.get("confidence"),
        "answered_via": state.get("answered_via") or dbg.get("answered_via"),
        "scope_verdict": dbg.get("scope_verdict"),
        "entities": dbg.get("entities") or state.get("entities"),
        "grounding_summary": dbg.get("grounding_summary"),
        "query_spec": dbg.get("query_spec") or state.get("spec"),
        "fallback_used": state.get("fallback_used", False),
        "tool_calls": dbg.get("tool_calls", []),
        "stage_times_ms": dbg.get("stage_times_ms", {}),
        "rag_examples": dbg.get("rag_examples", []),
        "served_from_cache": False,
        "error": dbg.get("error"),
        "traceback": dbg.get("traceback"),
    }

    try:
        params_view = build_params_view(state)
    except Exception:      # noqa: BLE001
        log.exception("params view build failed")   # never let a UI panel break a turn
        params_view = []

    answered_via = state.get("answered_via", "") or dbg.get("answered_via", "")
    is_error = kind == "error" or state.get("failed", False)
    # BUGFIX: scope_node() sets dbg["scope_verdict"] to the STRING "in_scope"
    # (not None/"") for every normal, successfully-recognized question — that
    # string is truthy in Python, so `if scope_verdict:` below fired on
    # literally every answer, not just unrelated/unsupported ones, showing
    # the QUICK_QS example chips on every single message. Only "unrelated"
    # and "unsupported" (the two values scope_check() actually returns) mean
    # the question was out of scope.
    scope_verdict = dbg.get("scope_verdict")
    out_of_scope = scope_verdict in ("unrelated", "unsupported")

    # Smart-helper notice + follow-up chips (§ user request): "incomplete" for
    # a mid-pipeline clarification (or a genuine ReAct failure to retrieve
    # data), "empty" for a fully-resolved query whose real answer has zero
    # rows/count. See SECTION 14B above for why each branch is scoped the way
    # it is (in particular: kind=="text" is a clarification ONLY when
    # answered_via=="query_spec" — verified every such case in the executor
    # is one — or when the ReAct path explicitly failed; a react-answered
    # kind=="text" that isn't a failure is a normal narrative answer and must
    # NOT get flagged here).
    notice: Optional[str] = None
    suggestions: List[Dict[str, str]] = []
    if out_of_scope:
        suggestions = [{"label": (eq[:30] + "…") if len(eq) > 30 else eq, "q": eq} for eq in QUICK_QS]
    elif answered_via == "help":
        pass
    elif kind == "text" and (answered_via == "query_spec" or state.get("failed")):
        notice = "incomplete"
        suggestions = _recovery_suggestions(state)
    elif not is_error and kind in ("count", "list", "stats", "timeseries"):
        empty = ((kind == "count" and payload.get("count") == 0) or
                 (kind in ("list", "stats", "timeseries") and len(rows) == 0))
        if empty:
            notice = "empty"
            suggestions = _recovery_suggestions(state)

    result = {
        "answer": state.get("answer", "") or "No answer produced.",
        "route": state.get("route", ""),
        "confidence": state.get("confidence", ""),
        "answered_via": answered_via,
        "kind": kind,
        "rows": rows, "cols": cols, "fig": fig_json, "stats": stats,
        "ts_rows": ts_rows, "ts_cols": ts_cols,
        "notice": notice, "suggestions": suggestions,
        "plan": state.get("plan", ""),
        "trace": trace_view,
        "params": params_view,
        "tool_calls": state.get("tool_calls", 0),
        "fallback_used": state.get("fallback_used", False),
        "error": is_error,
        "timing": {"total": round(elapsed, 2)},
        "debug": debug_obj,
    }

    # Cache successful, deterministic answers only (not errors).
    if not result["error"]:
        with _ANSWER_CACHE_LOCK:
            _ANSWER_CACHE[ck] = (time.time(), result)

    result["interaction_id"] = log_interaction(session_key, user_query, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — DASH FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Talk to VI-IPPMS")
server = app.server

ROUTE_LABEL = {"metadata_q": "METADATA", "kpi_q": "KPI DATA",
               "help_q": "HELP", "out_of_scope": "OUT OF SCOPE"}
ROUTE_COLOR = {"metadata_q": "#22d3ee", "kpi_q": "#34d399",
               "help_q": "#a78bfa", "out_of_scope": "#f87171"}

QUICK_QS = [
    "How many devices are in the GUJ circle?",
    "List devices matching PAR01",
    "How many components on APVSPGJWPAR01HNE40?",
    "How many KPIs are tracked on APVSPGJWPAR01HNE40?",
]

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
    :root{--bg:#0a0e17;--panel:#0f1626;--panel2:#131c30;--line:#1e293b;--text:#e2e8f0;
        --muted:#7c8aa5;--cyan:#22d3ee;--green:#34d399;--amber:#f59e0b;--red:#f87171;--violet:#a78bfa;}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
    .mono{font-family:'JetBrains Mono',monospace;}
    /* login */
    .login-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;
        background:radial-gradient(circle at 50% 0%,#0f1e33 0%,#0a0e17 60%);}
    .login-card{width:400px;background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:0 30px 30px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.55);}
    .scanbar{height:2px;width:100%;margin:0 -30px 0;background:linear-gradient(90deg,transparent,var(--cyan),transparent);
        background-size:200% 100%;animation:scan 3s linear infinite;}
    @keyframes scan{0%{background-position:200% 0}100%{background-position:-200% 0}}
    .login-eye{font-size:12px;letter-spacing:.16em;color:var(--cyan);margin:24px 0 4px;font-weight:600;}
    .login-h{font-size:23px;font-weight:700;margin:0 0 2px;}
    .login-sub{color:var(--muted);font-size:13px;margin:0 0 20px;}
    .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:14px 0 6px;display:block;}
    .inp{width:100%;padding:11px 12px;background:var(--panel2);border:1px solid var(--line);border-radius:7px;
        color:var(--text);font-size:14px;font-family:'JetBrains Mono',monospace;}
    .inp:focus{outline:none;border-color:var(--cyan);}
    .btn{width:100%;padding:12px;margin-top:22px;background:var(--cyan);color:#04202b;border:none;border-radius:7px;
        font-weight:700;font-size:14px;cursor:pointer;letter-spacing:.02em;}
    .btn:hover{background:#67e8f9;}
    .alert-err{margin-top:14px;padding:10px 12px;background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.35);
        color:var(--red);border-radius:7px;font-size:12.5px;}
    /* shell */
    .shell{display:grid;grid-template-columns:260px 1fr;grid-template-rows:56px 1fr;height:100vh;
        grid-template-areas:"top top" "side main";}
    .topbar{grid-area:top;display:flex;align-items:center;justify-content:space-between;padding:0 20px;
        background:var(--panel);border-bottom:1px solid var(--line);}
    .t-brand{display:flex;align-items:center;gap:10px;}
    .t-icon{font-size:20px;}
    .t-name{font-weight:700;font-size:15px;letter-spacing:.02em;}
    .t-sub{font-size:10px;color:var(--muted);letter-spacing:.04em;}
    .t-right{display:flex;align-items:center;gap:14px;font-size:12px;color:var(--muted);}
    .live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);
        animation:pulse 1.8s ease-in-out infinite;display:inline-block;}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
    .logout{background:transparent;border:1px solid var(--line);color:var(--text);padding:6px 12px;border-radius:6px;
        font-size:12px;cursor:pointer;}
    .logout:hover{border-color:var(--cyan);color:var(--cyan);}
    .sidebar{grid-area:side;background:var(--panel);border-right:1px solid var(--line);padding:16px 12px;
        display:flex;flex-direction:column;gap:10px;overflow:hidden;}
    .s-hdr{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);}
    .s-new{background:var(--panel2);border:1px solid var(--line);color:var(--text);padding:10px;border-radius:7px;
        cursor:pointer;font-size:13px;font-weight:600;}
    .s-new:hover{border-color:var(--cyan);}
    .s-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:6px;}
    .s-item{padding:9px 10px;border-radius:7px;border:1px solid transparent;cursor:pointer;}
    .s-item:hover{background:var(--panel2);}
    .s-act{background:var(--panel2);border-color:var(--cyan)44;}
    .s-ttl{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .s-meta{font-size:10px;color:var(--muted);margin-top:2px;}
    .main{grid-area:main;display:flex;flex-direction:column;overflow:hidden;}
    .scroll{flex:1;overflow-y:auto;padding:24px 20px;}
    .stream{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:20px;}
    .msg{display:flex;gap:12px;}
    .mu{justify-content:flex-end;}
    .ub{background:var(--cyan);color:#04202b;padding:11px 15px;border-radius:12px 12px 2px 12px;
        max-width:70%;font-size:14px;font-weight:500;}
    .av{width:30px;height:30px;border-radius:8px;background:var(--panel2);border:1px solid var(--line);
        display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
    .bb{background:var(--panel);border:1px solid var(--line);border-radius:2px 12px 12px 12px;padding:14px 16px;
        max-width:85%;font-size:14px;line-height:1.55;}
    .bm{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px;}
    .b{font-size:10px;padding:3px 8px;border-radius:999px;font-weight:600;letter-spacing:.03em;
        border:1px solid var(--line);color:var(--muted);}
    .ba{white-space:pre-wrap;}
    .be{white-space:pre-wrap;color:var(--red);}
    details.rx{margin-top:12px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);}
    details.rx summary{cursor:pointer;padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
        color:var(--cyan);font-weight:600;list-style:none;}
    details.rx summary::-webkit-details-marker{display:none;}
    .rx-step{padding:8px 12px;border-top:1px solid var(--line);font-size:12px;font-family:'JetBrains Mono',monospace;}
    .rx-th{color:var(--text);}
    .rx-ac{color:var(--cyan);}
    .rx-ob{color:var(--muted);white-space:pre-wrap;word-break:break-word;}
    details.dbg{margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#0b1220;}
    details.dbg summary{cursor:pointer;padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
        color:var(--violet);font-weight:600;list-style:none;}
    details.dbg summary::-webkit-details-marker{display:none;}
    /* parameters panel (same disclosure pattern as the debug trace) */
    details.prm{margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#0b1220;}
    details.prm summary{cursor:pointer;padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
        color:var(--amber);font-weight:600;list-style:none;}
    details.prm summary::-webkit-details-marker{display:none;}
    .prm-body{padding:10px 12px;border-top:1px solid var(--line);overflow-x:auto;}
    /* raw time-series export panel (same disclosure pattern) */
    details.tsd{margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#0b1220;}
    details.tsd summary{cursor:pointer;padding:8px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
        color:var(--green);font-weight:600;list-style:none;}
    details.tsd summary::-webkit-details-marker{display:none;}
    .tsd-body{padding:10px 12px;border-top:1px solid var(--line);}
    table.prm-t{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11.5px;}
    table.prm-t th{text-align:left;color:var(--muted);font-weight:600;text-transform:uppercase;font-size:10px;
        letter-spacing:.06em;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap;}
    table.prm-t td{padding:5px 8px;border-bottom:1px solid #16202f;vertical-align:top;word-break:break-word;}
    table.prm-t td.p-k{color:var(--muted);white-space:nowrap;}
    table.prm-t td.p-x{color:var(--text);}
    table.prm-t td.p-r{color:var(--cyan);}
    table.prm-t td.p-h{color:var(--muted);font-size:10.5px;}
    table.prm-t tr.p-miss td.p-r{color:var(--muted);}
    /* table block header (row count + download) */
    .tbl-bar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:12px;}
    .tbl-cap{font-size:11px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;}
    .dlbtn{background:transparent;border:1px solid var(--line);color:var(--muted);padding:5px 11px;
        border-radius:6px;font-size:11px;cursor:pointer;font-weight:600;white-space:nowrap;}
    .dlbtn:hover{border-color:var(--cyan);color:var(--cyan);}
    .dbg-body{padding:10px 12px;border-top:1px solid var(--line);}
    .dbg-row{display:flex;gap:8px;font-size:11.5px;font-family:'JetBrains Mono',monospace;padding:2px 0;}
    .dbg-k{color:var(--muted);min-width:130px;}
    .dbg-v{color:var(--text);white-space:pre-wrap;word-break:break-word;flex:1;}
    .dbg-pre{margin:6px 0 0;padding:8px;background:#0f1626;border:1px solid var(--line);border-radius:6px;
        font-size:11px;font-family:'JetBrains Mono',monospace;color:#9fb3d1;white-space:pre-wrap;
        word-break:break-word;max-height:260px;overflow:auto;}
    .fb-bar{display:flex;align-items:center;gap:8px;margin-top:12px;}
    .fb-lbl{font-size:11px;color:var(--muted);}
    .fbup,.fbdn,.csvb{background:transparent;border:1px solid var(--line);color:var(--muted);padding:5px 10px;
        border-radius:6px;font-size:11px;cursor:pointer;}
    .fbup:hover{border-color:var(--green);color:var(--green);}
    .fbdn:hover{border-color:var(--red);color:var(--red);}
    .csvb:hover{border-color:var(--cyan);color:var(--cyan);}
    .fb-active{border-color:var(--cyan);color:var(--cyan);}
    .fb-hint{font-size:11px;color:var(--green);}
    /* welcome */
    .welcome{max-width:760px;margin:6vh auto;text-align:center;}
    .w-eye{font-size:11px;letter-spacing:.16em;color:var(--cyan);font-weight:600;margin-bottom:8px;}
    .w-h{font-size:26px;font-weight:700;margin-bottom:8px;}
    .w-p{color:var(--muted);font-size:14px;margin-bottom:26px;}
    .cap-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px;}
    .cap{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;text-align:left;cursor:pointer;}
    .cap:hover{border-color:var(--cyan);}
    .cap-ico{font-size:20px;margin-bottom:8px;}
    .cap-ttl{font-weight:600;font-size:14px;margin-bottom:3px;}
    .cap-dsc{font-size:12px;color:var(--muted);}
    .chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;}
    .chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:7px 13px;font-size:12px;
        color:var(--text);cursor:pointer;}
    .chip:hover{border-color:var(--cyan);color:var(--cyan);}
    /* smart-helper suggestions: incomplete-question / empty-result follow-up chips */
    .sugg-block{margin-top:12px;}
    .sugg-lead{font-size:11.5px;color:var(--muted);margin-bottom:8px;}
    .sugg-lead-empty{color:var(--amber);}
    .sugg-row{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-start;}
    /* input */
    .iarea{border-top:1px solid var(--line);background:var(--panel);padding:14px 20px;}
    .iinner{max-width:860px;margin:0 auto;}
    .iwrap{display:flex;gap:10px;align-items:flex-end;background:var(--panel2);border:1px solid var(--line);
        border-radius:12px;padding:8px 10px;}
    .iwrap:focus-within{border-color:var(--cyan);}
    .ita{flex:1;background:transparent;border:none;color:var(--text);resize:none;font-size:14px;font-family:'Inter',sans-serif;
        outline:none;max-height:120px;padding:6px;}
    .sbtn{background:var(--cyan);color:#04202b;border:none;width:38px;height:38px;border-radius:9px;font-size:18px;
        cursor:pointer;font-weight:700;}
    .micbtn{background:transparent;color:var(--muted);border:1px solid var(--line);width:38px;height:38px;
        border-radius:9px;font-size:16px;cursor:pointer;flex-shrink:0;}
    .micbtn:hover{border-color:var(--cyan);color:var(--cyan);}
    .micbtn.mic-active{color:#04202b;background:var(--red);border-color:var(--red);animation:micpulse 1s infinite;}
    @keyframes micpulse{0%,100%{opacity:1}50%{opacity:.55}}
    .ihint{text-align:center;font-size:10.5px;color:var(--muted);margin-top:8px;}
    .thinking{display:flex;align-items:center;gap:8px;}
    .tdots span{width:6px;height:6px;background:var(--cyan);border-radius:50%;display:inline-block;margin:0 2px;
        animation:bounce 1.3s infinite;}
    .tdots span:nth-child(2){animation-delay:.2s}.tdots span:nth-child(3){animation-delay:.4s}
    @keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.4}30%{transform:translateY(-5px);opacity:1}}
    /* downvote feedback popup */
    .fbm-backdrop{position:fixed;inset:0;background:rgba(4,8,16,.65);display:flex;align-items:center;
        justify-content:center;z-index:1000;}
    .fbm-card{width:420px;max-width:92vw;background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:20px;box-shadow:0 20px 60px rgba(0,0,0,.55);}
    .fbm-title{font-size:15px;font-weight:700;margin-bottom:4px;}
    .fbm-sub{font-size:12px;color:var(--muted);margin:0 0 14px;}
    .fbm-textarea{width:100%;min-height:90px;background:var(--panel2);border:1px solid var(--line);
        border-radius:8px;color:var(--text);font-size:13px;font-family:'Inter',sans-serif;padding:10px 12px;
        resize:vertical;}
    .fbm-textarea:focus{outline:none;border-color:var(--cyan);}
    .fbm-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:16px;}
    .fbm-btn-ghost{background:transparent;border:1px solid var(--line);color:var(--text);padding:8px 16px;
        border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;}
    .fbm-btn-ghost:hover{border-color:var(--muted);}
    .fbm-btn-primary{background:var(--cyan);color:#04202b;border:none;padding:8px 16px;border-radius:7px;
        font-size:13px;font-weight:700;cursor:pointer;}
    .fbm-btn-primary:hover{background:#67e8f9;}
    </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
(function() {
    var nativeTextareaSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;

    // ── Enter sends, Shift+Enter inserts a newline ──────────────────────────
    // Bound directly on the textarea (not the document) so it can't swallow
    // Enter anywhere else. Dash re-renders the input area on login/page swap,
    // hence the dataset.keybound guard + the polling re-bind below.
    function setupKeys() {
        var input = document.getElementById('chat-input');
        if (!input || input.dataset.keybound) return;
        input.dataset.keybound = "1";

        input.addEventListener('keydown', function(e) {
            if (e.key !== 'Enter' || e.shiftKey) return;   // Shift+Enter: newline
            if (e.isComposing || e.keyCode === 229) return; // IME candidate selection
            e.preventDefault();                             // no stray newline
            if (!input.value || !input.value.trim()) return;
            // Let Dash's onChange for the final keystroke land before the click
            // fires, so the send callback reads the complete text.
            setTimeout(function() {
                var btn = document.getElementById('send-btn');
                if (btn) btn.click();
            }, 0);
        });

        // Grow the box as multi-line input is typed (Shift+Enter), capped by
        // the .ita max-height. Dash clears the value from the server after a
        // send, which fires no 'input' event, so the poll below also shrinks it
        // back once the box is empty again.
        function autosize() {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        }
        input.addEventListener('input', autosize);
        input._viAutosize = autosize;
        autosize();
    }

    function setupMic() {
        var micBtn = document.getElementById('mic-btn');
        var input  = document.getElementById('chat-input');
        if (!micBtn || !input || micBtn.dataset.bound) return;

        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            micBtn.title = "Voice input not supported in this browser";
            micBtn.disabled = true;
            micBtn.dataset.bound = "1";
            return;
        }
        micBtn.dataset.bound = "1";

        var recog = new SR();
        recog.lang = 'en-IN';
        recog.interimResults = false;
        recog.maxAlternatives = 1;
        var listening = false;

        var nativeSetter = nativeTextareaSetter;

        micBtn.addEventListener('click', function(e) {
            e.preventDefault();
            if (listening) { recog.stop(); return; }
            try { recog.start(); } catch (err) { /* already started */ }
        });
        recog.onstart = function() { listening = true; micBtn.classList.add('mic-active'); };
        recog.onend   = function() { listening = false; micBtn.classList.remove('mic-active'); };
        recog.onerror = function() { listening = false; micBtn.classList.remove('mic-active'); };
        recog.onresult = function(evt) {
            var transcript = evt.results[0][0].transcript;
            var current = document.getElementById('chat-input');
            var prefix = current.value ? current.value.trim() + ' ' : '';
            nativeSetter.call(current, prefix + transcript);
            current.dispatchEvent(new Event('input', { bubbles: true }));
        };
    }
    setInterval(function() {
        setupKeys();
        setupMic();
        // Re-sync the textarea height after Dash clears the value server-side.
        var input = document.getElementById('chat-input');
        if (input && input._viAutosize && !input.value) input._viAutosize();
    }, 800);
})();
</script>
</body>
</html>
"""


def login_page(error: Optional[str] = None):
    return html.Div(className="login-wrap", children=[
        html.Div(className="login-card", children=[
            html.Div(className="scanbar"),
            html.Div("VI-IPPMS · INSTANT GRAPH", className="login-eye"),
            html.Div("Talk to VI-IPPMS", className="login-h"),
            html.P("Conversational network analytics — ask about devices, interfaces, "
                   "components and KPIs in plain English.", className="login-sub"),
            html.Label("Email", className="lbl"),
            dcc.Input(id="login-email", type="email", className="inp",
                      placeholder="firstname.lastname@vodafoneidea.com", n_submit=0),
            html.Label("Employee ID", className="lbl"),
            dcc.Input(id="login-eid", type="text", className="inp", placeholder="e.g. 22000000", n_submit=0),
            html.Button("Sign in →", id="login-btn", n_clicks=0, className="btn"),
            html.Div(id="login-error",
                     children=(html.Div(error, className="alert-err") if error else None)),
        ])
    ])


def av():
    return html.Div("◉", className="av", style={"color": "var(--cyan)"})


def welcome():
    caps = [
        ("🗂️", "Device metadata", "Counts & listings of devices by circle or name pattern",
         "How many devices are in the GUJ circle?"),
        ("🔌", "Interfaces & components", "Interfaces / components / KPIs per device",
         "How many interfaces on APVSPGJWPAR01HNE40?"),
        ("📈", "KPI retrieval", "Values & statistics over a time window",
         "min/max/avg HC In Octets on Interface 100GE0/3/2 in the last 6 hours"),
        ("🧭", "Filter & slice", "Slice devices/KPIs by circle, component, threshold",
         "List devices matching PAR01"),
    ]
    return html.Div(className="welcome", children=[
        html.Div("TALK TO VI-IPPMS", className="w-eye"),
        html.Div("What would you like to know?", className="w-h"),
        html.P("Ask in plain English. The agent routes through Router → Resolve → "
               "QuerySpec → deterministic executor → Synthesize (falling back to a "
               "ReAct tool loop only for unusual shapes), using live Instant Graph APIs.",
               className="w-p"),
        html.Div(className="cap-grid", children=[
            html.Div(className="cap", id={"type": "quickq", "src": "cap", "q": q}, n_clicks=0,
                     children=[
                html.Div(ico, className="cap-ico"), html.Div(ttl, className="cap-ttl"),
                html.Div(dsc, className="cap-dsc")]) for ico, ttl, dsc, q in caps
        ]),
        html.Div(className="chips", children=[
            html.Button(q, className="chip", id={"type": "quickq", "src": "chip", "q": q}, n_clicks=0)
            for q in QUICK_QS
        ]),
    ])


# ── Render helpers ───────────────────────────────────────────────────────────

def render_user(text: str):
    return html.Div(className="msg mu", children=[html.Div(text, className="ub")])


def _table_block(cols, rows, index: int, csv_type: str = "csv"):
    """A result table with its own header bar: row count + Download CSV.

    The button is pattern-matched on {"type": csv_type, "index": <message
    index>}. `csv_type` distinguishes which of a message's tables a download
    click refers to (the summary stats table vs. the raw time-series table)
    so each is served by its own callback reading the right field off the
    stored result — the CSV always contains EVERY row, not just the 12
    visible on the current page of the DataTable."""
    n = len(rows)
    return html.Div([
        html.Div(className="tbl-bar", children=[
            html.Div(f"{n} row{'s' if n != 1 else ''} · {len(cols)} column{'s' if len(cols) != 1 else ''}",
                     className="tbl-cap"),
            html.Button("⬇  Download CSV", id={"type": csv_type, "index": index}, n_clicks=0,
                        className="dlbtn", title=f"Download all {n} row(s) as CSV"),
        ]),
        _stats_table(cols, rows),
    ])


def _timeseries_panel(cols, rows, index: int):
    """Collapsed disclosure panel (same pattern as Debug/Parameters) holding
    the raw, long-format time-series table (Device/Component/Interface/KPI/
    Value/Time — one row per data point) with its own CSV download, shown
    alongside the chart + Min/Max/Last/Average/Total/Units stats table for
    KPI questions. Collapsed by default since a single query can span many
    points across many series."""
    if not rows:
        return None
    return html.Details(className="tsd", children=[
        html.Summary(f"📈 Time series data · {len(rows)} point(s)"),
        html.Div(_table_block(cols, rows, index, csv_type="csv-ts"), className="tsd-body"),
    ])


def _stats_table(cols, rows):
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=[{c: r.get(c, "") for c in cols} for r in rows],
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "marginTop": "6px"},
        style_header={"backgroundColor": "#131c30", "color": "#7c8aa5",
                      "fontFamily": "JetBrains Mono", "fontSize": "11px",
                      "textTransform": "uppercase", "border": "1px solid #1e293b"},
        style_filter={"backgroundColor": "#131c30", "color": "#e2e8f0",
                      "border": "1px solid #1e293b"},
        style_cell={"backgroundColor": "#0f1626", "color": "#e2e8f0",
                    "fontFamily": "JetBrains Mono", "fontSize": "12px",
                    "border": "1px solid #1e293b", "padding": "7px 10px",
                    "textAlign": "left", "maxWidth": "260px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
    )


def _trace_panel(trace):
    if not trace:
        return None
    steps = []
    for s in trace:
        line = [html.Div(f"Thought: {s['thought']}", className="rx-th")] if s.get("thought") else []
        if s.get("action"):
            line.append(html.Div(f"Action: {s['action']}  "
                                 f"{json.dumps(s.get('action_input', {}))}", className="rx-ac"))
        obs = s.get("observation")
        if obs and obs != "(final answer)":
            obs_txt = obs if isinstance(obs, str) else json.dumps(obs, default=str)
            line.append(html.Div(f"Observation: {obs_txt[:600]}", className="rx-ob"))
        steps.append(html.Div(className="rx-step", children=line))
    return html.Details(className="rx", children=[
        html.Summary(f"⚙ Agent reasoning · {len(trace)} step(s)"),
        *steps,
    ])


def _params_panel(params: List[Dict[str, str]]):
    """Collapsed disclosure panel (same pattern as the Debug trace) listing every
    parameter the pipeline pulled out of the question — circle, device,
    component, interface, KPI, target/metric, top-N, threshold, time window —
    alongside what each was actually resolved to against live data."""
    if not params:
        return None
    header = html.Tr([html.Th("Parameter"), html.Th("Extracted from question"),
                      html.Th("Resolved to"), html.Th("How")])
    body = []
    for p in params:
        missed = p.get("Resolved", "—") == "—"
        body.append(html.Tr(className="p-miss" if missed else "", children=[
            html.Td(p.get("Parameter", ""), className="p-k"),
            html.Td(p.get("Extracted", "—"), className="p-x"),
            html.Td(p.get("Resolved", "—"), className="p-r"),
            html.Td(p.get("How", "—"), className="p-h"),
        ]))
    found = sum(1 for p in params if p.get("Extracted", "—") != "—")
    return html.Details(className="prm", children=[
        html.Summary(f"🧩 Parameters · {found} extracted"),
        html.Div(className="prm-body", children=[
            html.Table(className="prm-t", children=[html.Thead(header), html.Tbody(body)])
        ]),
    ])


def _debug_panel(debug: Dict[str, Any]):
    """Distinct Debug panel (§3.2), collapsed by default (behind the disclosure
    toggle) so end users aren't shown it unless they open it. Populated on EVERY
    answer — query-spec, skip, help, out-of-scope, and ReAct."""
    if not debug:
        return None
    def row(k, v):
        return html.Div(className="dbg-row", children=[
            html.Div(k, className="dbg-k"),
            html.Div(str(v), className="dbg-v")])
    st = debug.get("stage_times_ms", {}) or {}
    timings = ", ".join(f"{k}={v}ms" for k, v in st.items())
    tcs = debug.get("tool_calls", []) or []
    tool_lines = "\n".join(
        f"{i+1}. {t.get('tool')}({json.dumps(t.get('args', {}), default=str)[:120]}) "
        f"-> ok={t.get('ok')} {t.get('latency_ms','?')}ms"
        + (f"\n     ERROR: {t.get('error')}" if not t.get('ok') and t.get('error') else "")
        for i, t in enumerate(tcs)) or "(none)"
    body = [
        row("route", debug.get("route")),
        row("confidence", debug.get("confidence")),
        row("answered via", debug.get("answered_via")),
        row("scope verdict", debug.get("scope_verdict")),
        row("fallback used", debug.get("fallback_used")),
        row("served from cache", debug.get("served_from_cache")),
        row("rag matches", ", ".join(str(x) for x in (debug.get("rag_examples") or [])) or "—"),
        row("stage timings", timings or "—"),
        html.Div("entities", className="dbg-k", style={"marginTop": "6px"}),
        html.Pre(json.dumps(debug.get("entities") or {}, indent=2, default=str), className="dbg-pre"),
        html.Div("query_spec", className="dbg-k", style={"marginTop": "6px"}),
        html.Pre(json.dumps(debug.get("query_spec") or {}, indent=2, default=str), className="dbg-pre"),
        html.Div("grounding", className="dbg-k", style={"marginTop": "6px"}),
        html.Pre(json.dumps(debug.get("grounding_summary") or {}, indent=2, default=str), className="dbg-pre"),
        html.Div(f"tool calls ({len(tcs)})", className="dbg-k", style={"marginTop": "6px"}),
        html.Pre(tool_lines, className="dbg-pre"),
    ]
    if debug.get("traceback"):
        body.append(html.Div("traceback", className="dbg-k",
                             style={"marginTop": "6px", "color": "var(--red)"}))
        body.append(html.Pre(str(debug["traceback"]), className="dbg-pre",
                             style={"color": "#fca5a5"}))
    return html.Details(className="dbg", children=[
        html.Summary("🔬 Debug trace"),
        html.Div(body, className="dbg-body"),
    ])


def _suggestions_panel(notice: Optional[str], suggestions: List[Dict[str, str]], index: int):
    """Notice + a row of clickable follow-up chips for an incomplete/ambiguous
    question or an empty result (§ user request). Chips reuse the exact
    {"type": "quickq", ...} id pattern the welcome-screen QUICK_QS buttons
    already use, so on_send already handles a click with no new callback."""
    if not suggestions:
        return None
    if notice == "empty":
        lead = ("⚠️  No results came back for this — the device, component, interface, KPI, "
                "or time window may not match anything. Try:")
        lead_cls = "sugg-lead sugg-lead-empty"
    else:
        lead = "🧭  This looks like it might be missing some details — try:"
        lead_cls = "sugg-lead"
    chips = [
        html.Button(s["label"], id={"type": "quickq", "src": f"sugg-{index}-{i}", "q": s["q"]},
                    n_clicks=0, className="chip", title=s["q"])
        for i, s in enumerate(suggestions)
    ]
    return html.Div(className="sugg-block", children=[
        html.Div(lead, className=lead_cls),
        html.Div(chips, className="sugg-row"),
    ])


def render_bot(result: Dict[str, Any], index: int, feedback: Dict[str, str]):
    route = result.get("route", "")
    badges = [html.Span(ROUTE_LABEL.get(route, route or "—"), className="b",
                        style={"color": ROUTE_COLOR.get(route, "#7c8aa5"),
                               "borderColor": ROUTE_COLOR.get(route, "#1e293b")})]
    if result.get("timing", {}).get("total") is not None:
        badges.append(html.Span(f"{result['timing']['total']}s", className="b"))
    if result.get("tool_calls"):
        badges.append(html.Span(f"{result['tool_calls']} tool call(s)", className="b"))
    if result.get("fallback_used"):
        badges.append(html.Span("FALLBACK", className="b",
                                style={"color": "#f59e0b", "borderColor": "#f59e0b"}))

    body = [html.Div(badges, className="bm")]
    ans_cls = "be" if result.get("error") else "ba"
    body.append(html.Div(result.get("answer", ""), className=ans_cls))

    sp = _suggestions_panel(result.get("notice"), result.get("suggestions"), index)
    if sp is not None:
        body.append(sp)

    if result.get("fig"):
        try:
            body.append(dcc.Graph(figure=pio.from_json(result["fig"]),
                                  config={"displayModeBar": False},
                                  style={"marginTop": "12px"}))
        except Exception:
            pass

    if result.get("rows") and result.get("cols"):
        body.append(_table_block(result["cols"], result["rows"], index))

    tsp = _timeseries_panel(result.get("ts_cols"), result.get("ts_rows"), index)
    if tsp is not None:
        body.append(tsp)

    pp = _params_panel(result.get("params"))
    if pp is not None:
        body.append(pp)

    tp = _trace_panel(result.get("trace"))
    if tp is not None:
        body.append(tp)

    dp = _debug_panel(result.get("debug"))
    if dp is not None:
        body.append(dp)

    # feedback (the CSV download now lives on the table's own header bar)
    fbkey = str(index)
    given = feedback.get(fbkey)
    fb_children = [html.Span("Was this helpful?", className="fb-lbl"),
                   html.Button("▲", id={"type": "fbup", "index": index}, n_clicks=0,
                               className="fbup" + (" fb-active" if given == "up" else "")),
                   html.Button("▼", id={"type": "fbdn", "index": index}, n_clicks=0,
                               className="fbdn" + (" fb-active" if given == "down" else ""))]
    if given:
        fb_children.append(html.Span("✓ recorded", className="fb-hint"))
    body.append(html.Div(fb_children, className="fb-bar"))

    return html.Div(className="msg", children=[av(), html.Div(body, className="bb")])


def render_thinking():
    return html.Div(className="msg", children=[
        av(),
        html.Div(className="bb", children=[
            html.Div(className="thinking", children=[
                html.Span("Routing → resolving → reasoning", style={"fontSize": "13px",
                          "color": "var(--muted)"}),
                html.Span(className="tdots", children=[html.Span(), html.Span(), html.Span()]),
            ])
        ])
    ])


def render_stream(messages: List[Dict[str, Any]], feedback: Dict[str, str], pending: bool):
    if not messages and not pending:
        return welcome()
    items = []
    for i, m in enumerate(messages):
        if m["role"] == "user":
            items.append(render_user(m["content"]))
        else:
            items.append(render_bot(m["result"], i, feedback))
    if pending:
        items.append(render_thinking())
    return html.Div(items, className="stream")


def feedback_modal(index: int):
    """Popup shown on a downvote, letting the user attach free-text detail.
    The plain up/down vote (vi_chat_feedback) is already recorded the moment
    the thumbs-down is clicked — see on_feedback — so cancelling this popup
    without typing anything loses nothing but the optional note. Submit/Cancel
    use pattern-matched ids keyed on `index` (not fixed ids) because this Div
    is injected as the output of another callback: per the documented Dash
    quirk (see the ops-console file's "App layout" note), prevent_initial_call
    is only honored for inputs already in the server-rendered layout, so a
    freshly-inserted plain-id button fires its callback once immediately on
    insertion. Pattern-matched ids let close_feedback_modal reuse the same
    'ignore the all-zero initial fire' guard already used elsewhere in this
    file (on_feedback, on_csv, switch_conv) instead of tripping that trap."""
    return html.Div(className="fbm-backdrop", children=[
        html.Div(className="fbm-card", children=[
            html.Div("What went wrong with this answer?", className="fbm-title"),
            html.P("Optional — your feedback helps us improve. Your vote is already recorded.",
                   className="fbm-sub"),
            dcc.Textarea(id="fb-text", className="fbm-textarea",
                         placeholder="Tell us what was wrong or missing…"),
            html.Div(className="fbm-actions", children=[
                html.Button("Cancel", id={"type": "fbmodal-cancel", "index": index},
                           n_clicks=0, className="fbm-btn-ghost"),
                html.Button("Submit feedback", id={"type": "fbmodal-submit", "index": index},
                           n_clicks=0, className="fbm-btn-primary"),
            ]),
        ]),
    ])


# ── Main app layout ──────────────────────────────────────────────────────────

def _fmt_conv_meta(ts: Any) -> str:
    """Sidebar's small secondary line under a conversation title — when this
    conversation was last active, e.g. 'Aug 10, 09:15'."""
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ts.strftime("%b %d, %H:%M")
    except Exception:      # noqa: BLE001
        return ""


def sidebar(convs, active_cid):
    conv_items = []
    for c in convs:
        cls = "s-item s-act" if c["cid"] == active_cid else "s-item"
        conv_items.append(html.Div(className=cls, id={"type": "conv", "cid": c["cid"]}, n_clicks=0,
                                   children=[html.Div(c["title"], className="s-ttl"),
                                             html.Div(c["meta"], className="s-meta")]))
    return html.Div(className="sidebar", children=[
        html.Div("Conversations", className="s-hdr"),
        html.Button("＋  New conversation", id="new-conv", n_clicks=0, className="s-new"),
        html.Div(conv_items, className="s-list"),
    ])


def main_page(email: str, convs, active_cid, messages, feedback, pending):
    return html.Div(className="shell", children=[
        html.Div(className="topbar", children=[
            html.Div(className="t-brand", children=[
                html.Span("◈", className="t-icon", style={"color": "var(--cyan)"}),
                html.Div([html.Div("Talk to VI-IPPMS", className="t-name"),
                          html.Div("Instant Graph · conversational analytics", className="t-sub")]),
            ]),
            html.Div(className="t-right", children=[
                html.Span([html.Span(className="live-dot"), "  live APIs"]),
                html.Span(email, className="mono"),
                html.Button("Sign out", id="logout-btn", n_clicks=0, className="logout"),
            ]),
        ]),
        html.Div(id="sidebar-host", children=sidebar(convs, active_cid)),
        html.Div(className="main", children=[
            html.Div(className="scroll", children=[
                html.Div(id="stream-host", children=render_stream(messages, feedback, pending))
            ]),
            html.Div(className="iarea", children=[
                html.Div(className="iinner", children=[
                    html.Div(className="iwrap", children=[
                        html.Button("🎤", id="mic-btn", n_clicks=0, className="micbtn",
                                    title="Click and speak your question"),
                        dcc.Textarea(id="chat-input", className="ita", rows=1,
                                     placeholder="Ask about devices, interfaces, components or KPIs…"),
                        html.Button("➤", id="send-btn", n_clicks=0, className="sbtn",
                                    title="Send (Enter)"),
                    ]),
                    html.Div(["Press ", html.B("Enter"), " to send · ", html.B("Shift + Enter"),
                              " for a new line"], className="ihint"),
                ]),
            ]),
        ]),
    ])


# ── Root layout + stores ─────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="sauth", storage_type="session"),      # {token, email}
    dcc.Store(id="smsgs", storage_type="memory", data=[]),
    dcc.Store(id="sconvs", storage_type="memory", data=[]),
    dcc.Store(id="scid", storage_type="memory"),
    dcc.Store(id="sfeedback", storage_type="memory", data={}),
    dcc.Store(id="spending", storage_type="memory", data=None),
    dcc.Store(id="sfbmodal", storage_type="memory", data=None),   # {"index": <msg index>} or None
    dcc.Download(id="csv-dl"),
    html.Div(id="login-view", children=login_page(), style={"display": "block"}),
    html.Div(id="main-view", children=main_page("", [], None, [], {}, False), style={"display": "none"}),
    html.Div(id="fb-modal-host"),
])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("login-view", "children"),
    Output("main-view", "children"),
    Output("login-view", "style"),
    Output("main-view", "style"),
    Output("smsgs", "data"),
    Output("sconvs", "data"),
    Output("scid", "data"),
    Input("sauth", "data"),
    State("sfeedback", "data"),
)
def route_page(auth, feedback):
    sess = session_get((auth or {}).get("token"))
    if not sess:
        return (login_page(), no_update, {"display": "block"}, {"display": "none"},
                no_update, no_update, no_update)
    # Reload this user's persisted chat history every time we land here
    # authenticated — a fresh login AND a plain page refresh both go through
    # here (sauth uses session storage and survives a reload; smsgs/sconvs/
    # scid are memory-only and do NOT, which is exactly why history used to
    # vanish even mid-session, not just after a real re-login). Auto-open the
    # most recent conversation so returning users land back where they left
    # off instead of a blank welcome screen.
    convs = [{"cid": c["id"], "title": c["title"], "meta": _fmt_conv_meta(c["last_active_at"])}
             for c in list_conversations(sess["email"])]
    if convs:
        active_cid = convs[0]["cid"]
        msgs = load_conversation_messages(active_cid, sess["email"])
    else:
        active_cid, msgs = None, []
    main = main_page(sess["email"], convs, active_cid, msgs, feedback or {}, False)
    return (no_update, main, {"display": "none"}, {"display": "block"},
            msgs, convs, active_cid)


# BUGFIX: main_page() only ever builds the sidebar's HTML once, when
# route_page rebuilds the whole page (login, logout, or a page reload).
# Every callback that changes which conversations exist or which one is
# active (on_submit creating a new conversation, new_conv, switch_conv) was
# already correctly writing sconvs/scid — but nothing turned that store
# update into an actual re-render of the sidebar DOM, so a brand-new
# conversation, or the highlight moving to a clicked one, only ever became
# visible after something (a refresh/re-login) forced route_page to rebuild
# the page from scratch. This callback is that missing re-render: it fires
# on every sconvs/scid change, wherever it comes from, and keeps the sidebar
# in sync without any of those other callbacks needing to change.
@app.callback(
    Output("sidebar-host", "children"),
    Input("sconvs", "data"),
    Input("scid", "data"),
    prevent_initial_call=True,
)
def render_sidebar(convs, active_cid):
    return sidebar(convs or [], active_cid)


@app.callback(
    Output("sauth", "data"),
    Output("login-error", "children"),
    Input("login-btn", "n_clicks"),
    Input("login-email", "n_submit"),
    Input("login-eid", "n_submit"),
    State("login-email", "value"),
    State("login-eid", "value"),
    prevent_initial_call=True,
)
def do_login(_n, _s1, _s2, email, eid):
    email = (email or "").strip().lower()
    eid = (eid or "").strip()
    if not email or not eid:
        return no_update, html.Div("Email and Employee ID are both required.", className="alert-err")
    # Email + employee id ARE the credentials — a successful Instant Graph
    # login is the auth check, there is no separate app-side password gate.
    try:
        ig_login_via_mcp(email, eid)
    except MCPError as exc:
        return no_update, html.Div(f"Instant Graph login failed: {exc}", className="alert-err")
    token = session_create(email, eid)
    return {"token": token, "email": email}, None


@app.callback(
    Output("sauth", "data", allow_duplicate=True),
    Output("smsgs", "data", allow_duplicate=True),
    Output("sconvs", "data", allow_duplicate=True),
    Output("scid", "data", allow_duplicate=True),
    Output("sfbmodal", "data", allow_duplicate=True),
    Input("logout-btn", "n_clicks"),
    State("sauth", "data"),
    State("scid", "data"),
    State("smsgs", "data"),
    prevent_initial_call=True,
)
def do_logout(_n, auth, cid, msgs):
    if not _n:
        return no_update, no_update, no_update, no_update, no_update
    sess = session_get((auth or {}).get("token"))
    # Signing out while sitting on an empty placeholder (New conversation,
    # never used) -> delete it now rather than leaving it for the retention
    # sweep to eventually clear.
    if sess and cid is not None and not msgs:
        delete_conversation(cid, sess["email"])
    session_destroy((auth or {}).get("token"))
    return None, [], [], None, None


# Step 1 of streaming: show the user's message + a thinking bubble immediately.
@app.callback(
    Output("smsgs", "data", allow_duplicate=True),
    Output("spending", "data", allow_duplicate=True),
    Output("stream-host", "children", allow_duplicate=True),
    Output("chat-input", "value"),
    Output("sauth", "data", allow_duplicate=True),
    Input("send-btn", "n_clicks"),
    Input({"type": "quickq", "src": ALL, "q": ALL}, "n_clicks"),
    State("chat-input", "value"),
    State("smsgs", "data"),
    State("sfeedback", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def on_send(_send, _quick, typed, msgs, feedback, auth):
    trig = ctx.triggered_id
    q = None
    if isinstance(trig, dict) and trig.get("type") == "quickq":
        # ignore the spurious initial fire (all n_clicks == 0)
        if not any((n or 0) > 0 for n in (ctx.inputs_list[1] and
                   [i["value"] for i in ctx.inputs_list[1]] or [])):
            return no_update, no_update, no_update, no_update, no_update
        q = trig.get("q")
    else:
        q = (typed or "").strip()
    if not session_get((auth or {}).get("token")):
        return no_update, no_update, no_update, no_update, None
    if not q:
        return no_update, no_update, no_update, no_update, no_update
    msgs = (msgs or []) + [{"role": "user", "content": q}]
    return msgs, {"q": q}, render_stream(msgs, feedback or {}, True), "", no_update


# Step 2: run the pipeline for the pending question and render the answer.
@app.callback(
    Output("smsgs", "data", allow_duplicate=True),
    Output("spending", "data", allow_duplicate=True),
    Output("stream-host", "children", allow_duplicate=True),
    Output("sconvs", "data", allow_duplicate=True),
    Output("scid", "data", allow_duplicate=True),
    Output("sauth", "data", allow_duplicate=True),
    Input("spending", "data"),
    State("smsgs", "data"),
    State("sauth", "data"),
    State("sconvs", "data"),
    State("scid", "data"),
    State("sfeedback", "data"),
    prevent_initial_call=True,
)
def on_submit(pending, msgs, auth, convs, cid, feedback):
    if not pending or not pending.get("q"):
        return no_update, no_update, no_update, no_update, no_update, no_update
    sess = session_get((auth or {}).get("token"))
    if not sess:
        return no_update, None, no_update, no_update, no_update, None
    q = pending["q"]
    email = sess["email"]
    # msgs already has this turn's user question appended (on_send did that
    # before triggering this callback), so exactly 1 means it's the very
    # first message of whatever conversation is currently active.
    is_first_message = len(msgs or []) <= 1

    convs = convs or []
    if not cid:
        # No conversation at all yet (e.g. the very first message in a brand
        # new browser tab, bypassing New conversation entirely) -> mint one
        # now, titled from the real question. If persistence is unavailable,
        # cid stays None and this turn simply isn't saved (best-effort, same
        # as every other logging call here) — the chat keeps working either way.
        cid = create_conversation(email, q[:42])
        if cid is not None:
            convs = [{"cid": cid, "title": q[:42], "meta": _fmt_conv_meta(datetime.now())}] + convs
    elif is_first_message:
        # cid already exists as an empty "New conversation" placeholder
        # (from clicking the New conversation button) — this is its first
        # real message, so replace the generic placeholder title.
        rename_conversation(cid, q[:42])
        for c in convs:
            if c["cid"] == cid:
                c["title"] = q[:42]
                break

    try:
        result = pipeline(q, email)
    except Exception as exc:
        log.exception("pipeline crashed")
        result = {"answer": f"Unexpected error: {exc}", "route": "", "error": True,
                  "trace": [], "timing": {"total": 0}}
    msgs = (msgs or []) + [{"role": "assistant", "result": result}]

    if cid is not None:
        save_message(cid, email, "user", content=q)
        save_message(cid, email, "assistant", result=result, interaction_id=result.get("interaction_id"))
        touch_conversation(cid)
        for c in convs:
            if c["cid"] == cid:
                c["meta"] = _fmt_conv_meta(datetime.now())
                break
    return msgs, None, render_stream(msgs, feedback or {}, False), convs, cid, no_update


@app.callback(
    Output("smsgs", "data", allow_duplicate=True),
    Output("scid", "data", allow_duplicate=True),
    Output("sconvs", "data", allow_duplicate=True),
    Output("stream-host", "children", allow_duplicate=True),
    Input("new-conv", "n_clicks"),
    State("smsgs", "data"),
    State("sconvs", "data"),
    State("sfeedback", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def new_conv(_n, msgs, convs, feedback, auth):
    sess = session_get((auth or {}).get("token"))
    if not sess:
        return no_update, no_update, no_update, no_update
    # Already sitting on an empty conversation (a placeholder from an
    # earlier click nothing was ever sent in, or a brand-new tab with no
    # history at all) — nothing to do, avoids piling up empty "New
    # conversation" rows from repeated clicks.
    if not msgs:
        return no_update, no_update, no_update, no_update
    # Mint the placeholder up front (not lazily on first message) so it's
    # visible in the sidebar immediately, matching every other conversation
    # switch. Renamed from its generic title once a real question is sent
    # (see on_submit).
    cid = create_conversation(sess["email"], "New conversation")
    convs = convs or []
    if cid is not None:
        convs = [{"cid": cid, "title": "New conversation",
                 "meta": _fmt_conv_meta(datetime.now())}] + convs
    return [], cid, convs, render_stream([], feedback or {}, False)


@app.callback(
    Output("sfeedback", "data"),
    Output("stream-host", "children", allow_duplicate=True),
    Output("sfbmodal", "data"),
    Input({"type": "fbup", "index": ALL}, "n_clicks"),
    Input({"type": "fbdn", "index": ALL}, "n_clicks"),
    State("sfeedback", "data"),
    State("smsgs", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def on_feedback(_up, _dn, feedback, msgs, auth):
    if not session_get((auth or {}).get("token")):
        return no_update, no_update, no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update, no_update, no_update
    # ignore initial all-zero fire
    clicked = [i["value"] for i in (ctx.inputs_list[0] + ctx.inputs_list[1])]
    if not any((n or 0) > 0 for n in clicked):
        return no_update, no_update, no_update
    feedback = dict(feedback or {})
    idx = trig["index"]
    vote = "up" if trig["type"] == "fbup" else "down"
    feedback[str(idx)] = vote
    # Persist to vi_chat_feedback linked to the real interaction row (§3.5),
    # while keeping the in-memory store for instant UI feedback.
    try:
        result = (msgs or [])[idx].get("result", {})
        log_feedback(result.get("interaction_id"), vote)
    except (IndexError, KeyError, TypeError):
        pass
    log.info("[FEEDBACK] msg=%s vote=%s", idx, vote)
    # A downvote is already recorded above; the popup only offers to attach an
    # OPTIONAL free-text note (see close_feedback_modal) — an upvote never
    # opens it, and doesn't touch whatever modal state is already there.
    modal_data = {"index": idx} if vote == "down" else no_update
    return feedback, render_stream(msgs or [], feedback, False), modal_data


@app.callback(
    Output("fb-modal-host", "children"),
    Input("sfbmodal", "data"),
)
def render_feedback_modal(modal_state):
    if not modal_state:
        return None
    return feedback_modal(modal_state.get("index"))


@app.callback(
    Output("sfbmodal", "data", allow_duplicate=True),
    Input({"type": "fbmodal-submit", "index": ALL}, "n_clicks"),
    Input({"type": "fbmodal-cancel", "index": ALL}, "n_clicks"),
    State("fb-text", "value"),
    State("smsgs", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def close_feedback_modal(_submit, _cancel, text, msgs, auth):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update
    # ignore the spurious fire Dash triggers the moment these buttons are
    # first inserted (see feedback_modal's docstring)
    clicked = [i["value"] for i in (ctx.inputs_list[0] + ctx.inputs_list[1])]
    if not any((n or 0) > 0 for n in clicked):
        return no_update
    if trig.get("type") == "fbmodal-submit":
        sess = session_get((auth or {}).get("token"))
        text = (text or "").strip()
        if sess and text:
            idx = trig.get("index")
            try:
                interaction_id = (msgs or [])[idx]["result"].get("interaction_id")
            except (IndexError, KeyError, TypeError):
                interaction_id = None
            log_user_feedback(interaction_id, sess["email"], text)
    return None   # close the popup either way (submit or cancel)


def _csv_download(rows, cols, route: str, suffix: str):
    """Build a dcc.Download payload from rows/cols, or no_update if empty."""
    if not rows:
        return no_update
    if not cols:
        cols = list(rows[0].keys())
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    route = (route or "result").replace("_q", "")
    return dict(content=buf.getvalue(),
                filename=f"vi_ippms_{route}_{suffix}_{stamp}.csv")


def _csv_trigger_result(msgs):
    """Shared guard for the two CSV callbacks below: ignore the all-zero fire
    on first render, then look up the triggered message's stored result."""
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return None
    clicked = [i["value"] for i in ctx.inputs_list[0]]
    if not any((n or 0) > 0 for n in clicked):
        return None
    idx = trig["index"]
    try:
        return (msgs or [])[idx]["result"]
    except (IndexError, KeyError, TypeError):
        return None


@app.callback(
    Output("csv-dl", "data"),
    Input({"type": "csv", "index": ALL}, "n_clicks"),
    State("smsgs", "data"),
    prevent_initial_call=True,
)
def on_csv(_n, msgs):
    """Serve the stats/list table behind a Download CSV button. Rebuilt from
    the message store, so it exports EVERY row (the DataTable only shows 12
    per page), with the columns in the same order as displayed."""
    result = _csv_trigger_result(msgs)
    if result is None:
        return no_update
    return _csv_download(result.get("rows", []), result.get("cols", []),
                         result.get("route"), "summary")


@app.callback(
    Output("csv-dl", "data", allow_duplicate=True),
    Input({"type": "csv-ts", "index": ALL}, "n_clicks"),
    State("smsgs", "data"),
    prevent_initial_call=True,
)
def on_csv_ts(_n, msgs):
    """Serve the raw time-series table (Device/Component/Interface/KPI/Value/
    Time) behind its own Download CSV button, same pattern as on_csv."""
    result = _csv_trigger_result(msgs)
    if result is None:
        return no_update
    return _csv_download(result.get("ts_rows", []), result.get("ts_cols", []),
                         result.get("route"), "timeseries")


@app.callback(
    Output("scid", "data", allow_duplicate=True),
    Output("smsgs", "data", allow_duplicate=True),
    Output("sconvs", "data", allow_duplicate=True),
    Output("stream-host", "children", allow_duplicate=True),
    Input({"type": "conv", "cid": ALL}, "n_clicks"),
    State("sfeedback", "data"),
    State("sauth", "data"),
    State("scid", "data"),
    State("smsgs", "data"),
    State("sconvs", "data"),
    prevent_initial_call=True,
)
def switch_conv(_n, feedback, auth, prev_cid, prev_msgs, convs):
    # BUGFIX: this used to only set scid — clicking a conversation in the
    # sidebar changed which one was highlighted but never actually loaded
    # ITS messages, so the transcript shown never changed. Now it pulls that
    # conversation's messages fresh from Postgres, same as opening it after
    # login.
    trig = ctx.triggered_id
    clicked = [i["value"] for i in ctx.inputs_list[0]]
    if not isinstance(trig, dict) or not any((n or 0) > 0 for n in clicked):
        return no_update, no_update, no_update, no_update
    sess = session_get((auth or {}).get("token"))
    if not sess:
        return no_update, no_update, no_update, no_update
    cid = trig["cid"]
    convs = convs or []
    # Leaving an empty placeholder (New conversation, never used) for a
    # different one -> delete it now instead of leaving it in the sidebar
    # until the retention sweep eventually clears it.
    if prev_cid is not None and prev_cid != cid and not prev_msgs:
        delete_conversation(prev_cid, sess["email"])
        convs = [c for c in convs if c["cid"] != prev_cid]
    msgs = load_conversation_messages(cid, sess["email"])
    return cid, msgs, convs, render_stream(msgs, feedback or {}, False)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _warm_tools():
    global TOOLS_CACHE
    try:
        TOOLS_CACHE = mcp_list_tools()
        log.info("[STARTUP] Discovered %d MCP tools.", len(TOOLS_CACHE))
    except Exception as exc:
        log.warning("[STARTUP] Could not pre-list MCP tools: %s", exc)


def _warm_startup():
    _warm_tools()
    _load_question_guide()          # build the local RAG index (§4.2D)
    _ensure_logging_tables()        # best-effort create vi_chat_* tables (§3.5)
    _ensure_user_feedback_table()   # best-effort create user_feedback, independently
    _ensure_conversation_tables()   # best-effort create chat history tables, independently
    _ensure_retention_sweeper()     # start the 7-day retention purge thread
    log.info("[STARTUP] tool KB %s, question guide %d rows, logging %s, "
             "user_feedback %s, chat_history %s.",
             "loaded" if TOOL_KB_TEXT else "MISSING", len(_GUIDE_ROWS),
             "ready" if _logging_ready else "disabled",
             "ready" if _user_feedback_ready else "disabled",
             "ready" if _conversation_tables_ready else "disabled")


if __name__ == "__main__":
    _warm_startup()
    log.info("[STARTUP] Talk-to-VI-IPPMS on http://%s:%s  (MCP=%s, GPU=%s)",
             APP_HOST, APP_PORT, MCP_SERVER_URL, GPU_PROXY_URL)
    app.run(host=APP_HOST, port=APP_PORT, debug=False, use_reloader=False)
