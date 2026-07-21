#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  INSTANT GRAPH MCP SERVER
#  FastMCP backend + Dash frontend, single-file deployment
# ══════════════════════════════════════════════════════════════════════════════
#
#  What this file is
#  ------------------
#  A single-process service that:
#
#    1. Wraps the Instant Graph REST APIs (auth/login, get-hosts, get-components,
#       prefix-filter, get-suffixes, render-table, get-historical-data) as MCP
#       tools via FastMCP, served over streamable-HTTP so that a LangGraph /
#       ReAct / Chain-of-Thought / Graph-of-Thought agent can connect to it as
#       a normal MCP server and call the tools by name.
#
#    2. Runs a small Python-Dash frontend in the same process:
#         - a login page that collects the corporate email + employee id (eid)
#           the Instant Graph auth API expects,
#         - an operations console that shows live token status and lets a
#           human exercise the same six APIs the agent will use (host ->
#           component -> interface -> KPI -> historical data), which is the
#           fastest way to sanity-check the wiring before pointing an LLM
#           agent at it.
#
#  Token handling
#  --------------
#  The login endpoint returns access_token + expires_at (server-observed
#  lifetime is on the order of an hour). Nothing in this file hardcodes that
#  lifetime — every login response is re-parsed for its real expires_at.
#  A single process-wide `TokenManager`:
#    - caches the token and reuses it across every tool call / UI action,
#    - refreshes proactively once fewer than TOKEN_REFRESH_BUFFER_SECONDS
#      remain (so a live agent call never blocks on a synchronous re-login),
#    - runs a lightweight background watchdog thread that keeps the token
#      warm for the lifetime of the process once credentials are known,
#    - forces one re-login-and-retry if a call ever comes back 401, in case
#      the token was invalidated server-side before its stated expiry.
#
#  NOTE ON MULTI-USER USE: this reference implementation keeps ONE active
#  Instant Graph session for the whole deployment (matches "a small ops team
#  shares one console"), not one session per browser tab. The token is
#  persisted to the tt_vi_ippms_schema.ig_auth_sessions Postgres table (see
#  Section 1) precisely so that this still works correctly if you run this
#  behind gunicorn/uwsgi with several WORKER PROCESSES — a login handled by
#  one process is picked up by the others instead of each process thinking
#  it's logged out. If you need concurrent, independently-authenticated
#  browser sessions instead (each person using their own token), give each
#  Flask session its own session_key (see DB_SESSION_KEY below) and thread
#  that key through the Dash callbacks.
#
#  NOTE ON THE AUTH HEADER: the documentation states the access_token "must
#  be included in subsequent API calls" but does not show the exact header
#  the gateway expects for those follow-up calls. This file defaults to the
#  standard `Authorization: Bearer <token>` convention and exposes
#  IG_AUTH_HEADER_NAME / IG_AUTH_HEADER_PREFIX env vars so you can switch to
#  whatever your gateway actually expects (e.g. a raw `access_token` header)
#  without touching code.
#
#  Run it
#  ------
#    pip install "mcp[cli]" dash plotly requests psycopg2-binary --break-system-packages
#    python instant_graph_mcp_server.py
#
#  This starts:
#    - the MCP server (streamable-http) on  http://<MCP_HOST>:<MCP_PORT>/mcp
#    - the Dash console on                  http://<DASH_HOST>:<DASH_PORT>/
#
#  Point your LangGraph / MCP client at the first URL; point a browser at the
#  second one to log in and explore the data.
#
#  Requires the tables created by setup_ig_auth_tables.sql (ig_auth_sessions,
#  ig_auth_login_audit) to already exist in tt_vi_ippms_schema. This build
#  (v2.3) additionally creates ig_tool_call_audit (best-effort CREATE ... IF NOT
#  EXISTS at startup) and records every MCP tool invocation there — a second,
#  harder-to-bypass source of truth for tool usage that complements the chat
#  app's own interaction logging.
#
#  v2.3 changes (see vi_ippms_v2_3_v6_upgrade_brief.md §4.1):
#    - richer tool docstrings + Field descriptions (example calls, success
#      shapes, and the " ::" / mixed-unit format gotchas) so the chat app's
#      build_tool_context() surfaces more to the agent;
#    - server-side ig_tool_call_audit table + a non-blocking background writer,
#      so audit logging does not become a bottleneck under multi-device fan-out;
#    - no hardcoded DB password fallback in source (was Appendix B).
# ══════════════════════════════════════════════════════════════════════════════

import os
import ssl
import time
import socket
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Annotated

import requests
from requests.adapters import HTTPAdapter, Retry
from pydantic import Field

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import plotly.graph_objects as go

from mcp.server.fastmcp import FastMCP

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

INSTANT_GRAPH_BASE = os.environ.get(
    "INSTANT_GRAPH_BASE_URL", "https://10.34.64.74:5001/api/api"
)

LOGIN_URL      = f"{INSTANT_GRAPH_BASE}/v5/auth/login"
HOSTS_URL      = f"{INSTANT_GRAPH_BASE}/v3/get-hosts"
COMPONENTS_URL = f"{INSTANT_GRAPH_BASE}/v3/get-components"
PREFIX_URL     = f"{INSTANT_GRAPH_BASE}/v3/prefix-filter"
SUFFIX_URL     = f"{INSTANT_GRAPH_BASE}/v3/get-suffixes"
RENDER_URL     = f"{INSTANT_GRAPH_BASE}/v3/render-table"
HISTORY_URL    = f"{INSTANT_GRAPH_BASE}/v3/get-historical-data"

REQUEST_TIMEOUT               = int(os.environ.get("IG_REQUEST_TIMEOUT", "30"))
TOKEN_REFRESH_BUFFER_SECONDS  = int(os.environ.get("IG_TOKEN_REFRESH_BUFFER", "120"))
TOKEN_WATCHDOG_INTERVAL       = int(os.environ.get("IG_TOKEN_WATCHDOG_INTERVAL", "30"))

# If this app is deployed behind gunicorn/uwsgi with more than one WORKER
# PROCESS (as opposed to threads), each process gets its own Python memory
# and therefore its own in-memory TokenManager — a login handled by worker A
# would otherwise be invisible to worker B. To make auth state visible across
# processes, the token is also persisted to Postgres (tt_vi_ippms_schema.
# ig_auth_sessions — see setup_ig_auth_tables.sql); every process reads it
# back before deciding it needs to (re)login.
DB_HOST     = os.environ.get("IG_DB_HOST", "10.19.75.115")
DB_PORT     = int(os.environ.get("IG_DB_PORT", "5432"))
DB_NAME     = os.environ.get("IG_DB_NAME", "conv_ai_db")
DB_USER     = os.environ.get("IG_DB_USER", "ig_app_user")
# SECURITY (was Appendix B): no hardcoded credential fallback in source. The
# password must come from the environment; if it is missing we log once and let
# the first DB call fail with a clear error rather than shipping a real-looking
# secret in the repo.
DB_PASSWORD = os.environ.get("IG_DB_PASSWORD", "ig_app_user_Qwer!234")
DB_SCHEMA   = os.environ.get("IG_DB_SCHEMA", "tt_vi_ippms_schema")
DB_CONNECT_TIMEOUT = int(os.environ.get("IG_DB_CONNECT_TIMEOUT", "5"))
DB_POOL_MIN = int(os.environ.get("IG_DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.environ.get("IG_DB_POOL_MAX", "5"))

# Row key inside ig_auth_sessions. One shared console == one key. If you later
# want independent per-user sessions, derive this from something like the
# Flask session id instead of a fixed constant.
DB_SESSION_KEY = os.environ.get("IG_DB_SESSION_KEY", "default")

# Identifies which process/host wrote a given login_audit row — purely for
# observability when several worker processes are running.
_SOURCE_HOST = f"{socket.gethostname()}:{os.getpid()}"

# See "NOTE ON THE AUTH HEADER" above.
AUTH_HEADER_NAME   = os.environ.get("IG_AUTH_HEADER_NAME", "Authorization")
AUTH_HEADER_PREFIX = os.environ.get("IG_AUTH_HEADER_PREFIX", "Bearer ")

# INSTANT_GRAPH_BASE is https. If the gateway presents a certificate the system
# trust store doesn't recognize (internal CA, self-signed), point this at a PEM
# file/directory to verify against instead of every call failing with
# SSLCertVerificationError. Leave unset to use the system default trust store.
IG_CA_BUNDLE = os.environ.get("IG_CA_BUNDLE", "/srv/ippms-assistant/ig_selfsigned.pem") or None
IG_VERIFY = IG_CA_BUNDLE or True

MCP_HOST  = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT  = int(os.environ.get("MCP_PORT", "8056"))
DASH_HOST = os.environ.get("DASH_HOST", "0.0.0.0")
DASH_PORT = int(os.environ.get("DASH_PORT", "8060"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("instant_graph_mcp")

if not DB_PASSWORD:
    log.warning("IG_DB_PASSWORD is not set — Postgres calls (token store, login audit, "
                "tool-call audit) will fail until it is provided via the environment.")

if IG_CA_BUNDLE and not os.path.exists(IG_CA_BUNDLE):
    log.warning("IG_CA_BUNDLE is set to %r but that path does not exist — Instant Graph "
                "HTTPS calls will fail cert verification until it points at a real PEM "
                "file/directory.", IG_CA_BUNDLE)
elif IG_CA_BUNDLE:
    log.info("Verifying Instant Graph TLS certificate against custom CA bundle: %s", IG_CA_BUNDLE)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1B — POSTGRES CONNECTION POOL
# ══════════════════════════════════════════════════════════════════════════════
#  Created lazily on first use (not at import time) so a temporarily
#  unreachable database doesn't prevent the module from importing / the Dash
#  login page from rendering — it just means auth will fail with a clear
#  error until the database comes back.

_db_pool: Optional[ThreadedConnectionPool] = None
_db_pool_lock = threading.Lock()


def _get_db_pool() -> ThreadedConnectionPool:
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:  # re-check inside the lock (double-checked locking)
                _db_pool = ThreadedConnectionPool(
                    DB_POOL_MIN, DB_POOL_MAX,
                    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASSWORD,
                    options=f"-c search_path={DB_SCHEMA}",
                    connect_timeout=DB_CONNECT_TIMEOUT,
                )
                log.info("Connected to Postgres token store at %s:%s/%s (schema=%s)",
                          DB_HOST, DB_PORT, DB_NAME, DB_SCHEMA)
    return _db_pool


@contextmanager
def _db_cursor(commit: bool = False):
    """Checkout a pooled connection, yield a RealDictCursor, and return the
    connection to the pool afterwards. A connection that errors mid-use is
    closed rather than recycled, so one bad connection (e.g. after a DB
    restart) can't poison the pool for subsequent callers."""
    pool = _get_db_pool()
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1C — TOOL-CALL AUDIT  (server-side, complements chat-app logging)
# ══════════════════════════════════════════════════════════════════════════════
#  Every @mcp_app.tool() invocation is recorded here: tool name, arguments
#  (with session_key redacted to just the user key, no token), success/error,
#  latency, and timestamp. This is deliberately a SECOND source of truth: even
#  if the v6 app's own vi_chat_interactions logging has gaps, tool usage is
#  still captured here at the point where the tool actually ran.
#
#  Because a single multi-device question can fan out into dozens of tool calls
#  (get-components / prefix-filter / get-suffixes are single-host), writing each
#  audit row synchronously inside the request path would add DB latency to every
#  call. Instead rows are pushed onto an in-process queue and flushed in batches
#  by one background thread, so the tool path never blocks on the audit write.

import queue as _queue
import json as _json

_TOOL_AUDIT_DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.ig_tool_call_audit (
    id           BIGSERIAL PRIMARY KEY,
    called_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_key  TEXT,
    tool_name    TEXT NOT NULL,
    arguments    JSONB,
    success      BOOLEAN NOT NULL,
    error_message TEXT,
    latency_ms   INTEGER,
    source_host  TEXT
);
"""

_audit_queue: "_queue.Queue" = _queue.Queue(maxsize=10000)
_audit_thread_started = False
_audit_thread_lock = threading.Lock()


def _ensure_tool_audit_table() -> None:
    """Best-effort CREATE TABLE IF NOT EXISTS so a fresh deployment works without
    a separate migration. Never raises — if the DB is down, auditing is simply a
    no-op until it returns."""
    try:
        with _db_cursor(commit=True) as cur:
            cur.execute(_TOOL_AUDIT_DDL)
    except Exception as exc:
        log.warning("Could not ensure ig_tool_call_audit exists (auditing disabled "
                    "until DB reachable): %s", exc)


def _audit_writer_loop() -> None:
    """Drain the audit queue in small batches. One row failing never drops the
    rest of the batch's chance on the next tick, and a DB outage just backs up
    the queue (bounded) rather than erroring the tool path."""
    while True:
        batch = []
        try:
            batch.append(_audit_queue.get())          # block for the first item
            while len(batch) < 100:
                batch.append(_audit_queue.get_nowait())
        except _queue.Empty:
            pass
        try:
            with _db_cursor(commit=True) as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"""INSERT INTO {DB_SCHEMA}.ig_tool_call_audit
                        (session_key, tool_name, arguments, success, error_message,
                         latency_ms, source_host)
                        VALUES %s""",
                    [(r["session_key"], r["tool_name"], _json.dumps(r["arguments"], default=str),
                      r["success"], r["error_message"], r["latency_ms"], _SOURCE_HOST)
                     for r in batch],
                )
        except Exception as exc:
            log.warning("Tool-call audit flush failed for %d row(s): %s", len(batch), exc)
        finally:
            for _ in batch:
                _audit_queue.task_done()


def _ensure_audit_writer() -> None:
    global _audit_thread_started
    if _audit_thread_started:
        return
    with _audit_thread_lock:
        if _audit_thread_started:
            return
        _ensure_tool_audit_table()
        threading.Thread(target=_audit_writer_loop, name="ig-tool-audit", daemon=True).start()
        _audit_thread_started = True
        log.info("Tool-call audit writer started (batched, non-blocking).")


def _redact_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Keep args useful for audit but never store a bearer token; session_key is
    a user identifier (email), not a secret, so it is retained for correlation."""
    if not isinstance(arguments, dict):
        return {"_raw": str(arguments)[:500]}
    out = dict(arguments)
    for k in list(out.keys()):
        if k.lower() in ("access_token", "token", "password", "authorization"):
            out[k] = "<redacted>"
    return out


def _record_tool_call(tool_name: str, arguments: Dict[str, Any], success: bool,
                      error_message: Optional[str], latency_ms: int) -> None:
    _ensure_audit_writer()
    try:
        _audit_queue.put_nowait({
            "session_key": (arguments or {}).get("session_key"),
            "tool_name": tool_name,
            "arguments": _redact_args(arguments),
            "success": success,
            "error_message": (error_message or "")[:1000] or None,
            "latency_ms": latency_ms,
        })
    except _queue.Full:
        log.warning("Tool-call audit queue full; dropping one audit row for %s", tool_name)


def audited_tool(func):
    """Decorator: wrap an MCP tool so every invocation is timed and audited
    without changing the tool's signature or the app's calling convention.
    Stacked BELOW @mcp_app.tool() so FastMCP still introspects the real
    signature (inspect.signature follows __wrapped__ set by functools.wraps)."""
    import functools
    import inspect
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ok, err = True, None
        try:
            return func(*args, **kwargs)
        except Exception as exc:            # noqa: BLE001 - re-raised after auditing
            ok, err = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                arg_dict = dict(sig.bind_partial(*args, **kwargs).arguments)
            except Exception:
                arg_dict = dict(kwargs)
            _record_tool_call(func.__name__, arg_dict, ok, err,
                              int((time.time() - t0) * 1000))
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — TOKEN MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class TokenError(RuntimeError):
    """Raised when authentication fails or credentials are missing."""


def _parse_iso8601(value: str) -> datetime:
    v = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class TokenManager:
    """
    Thread-safe holder for the Instant Graph access token.

    One instance is shared by the MCP tools and the Dash UI so both always
    see the same authentication state. See the module docstring for the
    multi-user caveat.
    """

    def __init__(self, session_key: str = DB_SESSION_KEY) -> None:
        # session_key identifies WHICH user's Instant Graph session this manager
        # holds. In the original single-console design this was the fixed
        # DB_SESSION_KEY ("default"); for per-user tokens the app passes a
        # distinct key per person (we use the user's email), so tokens never
        # bleed across users. It is also the primary key of the
        # ig_auth_sessions Postgres row this manager reads/writes.
        self.session_key: str = (session_key or DB_SESSION_KEY).strip().lower()
        self._lock = threading.RLock()
        self.email: Optional[str] = None
        self.eid: Optional[str] = None
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.session_id: Optional[Any] = None
        self.last_error: Optional[str] = None
        self.last_login_ts: Optional[datetime] = None
        self._watchdog_started = False
        self._load_from_db()  # pick up a session another worker process may already hold

    # -- cross-process persistence (Postgres) --------------------------------
    def _load_from_db(self) -> None:
        try:
            with _db_cursor() as cur:
                cur.execute(
                    f"SELECT email, eid, access_token, ig_session_id, expires_at "
                    f"FROM {DB_SCHEMA}.ig_auth_sessions WHERE session_key = %s",
                    (self.session_key,),
                )
                row = cur.fetchone()
        except Exception as exc:
            log.warning("Could not load token from Postgres (keeping current in-process state): %s", exc)
            return

        if not row:
            return

        with self._lock:
            self.email = row["email"] or self.email
            self.eid = row["eid"] or self.eid
            self.access_token = row["access_token"]
            self.session_id = row["ig_session_id"]
            self.expires_at = row["expires_at"]  # TIMESTAMPTZ comes back tz-aware already

    def _save_to_db(self) -> None:
        try:
            with _db_cursor(commit=True) as cur:
                cur.execute(
                    f"""
                    INSERT INTO {DB_SCHEMA}.ig_auth_sessions
                        (session_key, email, eid, access_token, ig_session_id, expires_at, last_login_at)
                    VALUES (%s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (session_key) DO UPDATE SET
                        email = EXCLUDED.email, eid = EXCLUDED.eid,
                        access_token = EXCLUDED.access_token,
                        ig_session_id = EXCLUDED.ig_session_id,
                        expires_at = EXCLUDED.expires_at,
                        last_login_at = now()
                    """,
                    (self.session_key, self.email, self.eid, self.access_token, self.session_id, self.expires_at),
                )
        except Exception as exc:
            # Don't raise: the login itself already succeeded and self.access_token is set,
            # so this process can keep working — it just means other worker processes won't
            # see this session until the database is reachable again.
            log.error("Could not persist token to Postgres (other worker processes won't see "
                      "this session until the DB is reachable): %s", exc)

    def _clear_db(self) -> None:
        try:
            with _db_cursor(commit=True) as cur:
                cur.execute(
                    f"DELETE FROM {DB_SCHEMA}.ig_auth_sessions WHERE session_key = %s",
                    (self.session_key,),
                )
        except Exception as exc:
            log.warning("Could not clear token row in Postgres: %s", exc)

    def _audit_login(self, email: str, eid: str, success: bool, error_message: Optional[str] = None) -> None:
        try:
            with _db_cursor(commit=True) as cur:
                cur.execute(
                    f"""
                    INSERT INTO {DB_SCHEMA}.ig_auth_login_audit
                        (session_key, email, eid, success, error_message, source_host)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (self.session_key, email, eid, success, error_message, _SOURCE_HOST),
                )
        except Exception as exc:
            log.warning("Could not write login audit row: %s", exc)

    # -- read-only state ---------------------------------------------------
    def is_authenticated(self) -> bool:
        with self._lock:
            return bool(self.access_token)

    def seconds_remaining(self) -> float:
        with self._lock:
            if not self.expires_at:
                return 0.0
            return max(0.0, (self.expires_at - datetime.now(timezone.utc)).total_seconds())

    def needs_refresh(self) -> bool:
        with self._lock:
            if not self.access_token or not self.expires_at:
                return True
            return self.seconds_remaining() <= TOKEN_REFRESH_BUFFER_SECONDS

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "authenticated": self.is_authenticated(),
                "email": self.email,
                "session_id": self.session_id,
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "seconds_remaining": round(self.seconds_remaining(), 1),
                "last_error": self.last_error,
            }

    # -- authentication ------------------------------------------------------
    def set_credentials(self, email: str, eid: str) -> None:
        """Store credentials and force a fresh login on next use / logout."""
        with self._lock:
            self.email = (email or "").strip()
            self.eid = (eid or "").strip()
            self.access_token = None
            self.expires_at = None
            self.session_id = None
            self.last_error = None

    def logout(self) -> None:
        with self._lock:
            self.email = None
            self.eid = None
            self.access_token = None
            self.expires_at = None
            self.session_id = None
            self.last_error = None
            self._clear_db()

    def login(self, email: Optional[str] = None, eid: Optional[str] = None) -> Dict[str, Any]:
        """Call the auth endpoint and cache the resulting token. Raises TokenError on failure."""
        with self._lock:
            email = (email or self.email or "").strip()
            eid = (eid or self.eid or "").strip()
            if not email or not eid:
                # This process may not be the one that handled the login request — check
                # whether another worker already has a session in Postgres before giving up.
                self._load_from_db()
                email = (email or self.email or "").strip()
                eid = (eid or self.eid or "").strip()
            if not email or not eid:
                raise TokenError("Email and Employee ID are required before calling Instant Graph APIs.")

            try:
                # Uses the shared _http session (defined in SECTION 3) rather than a bare
                # requests.post so the self-signed cert's pinned SSLContext (see
                # _build_ig_ssl_context) is applied here too, not just on the other API calls.
                resp = _http.post(
                    LOGIN_URL, json={"email": email, "eid": eid}, timeout=REQUEST_TIMEOUT,
                    verify=IG_VERIFY,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                self.last_error = f"login request failed: {exc}"
                log.error(self.last_error)
                self._audit_login(email, eid, success=False, error_message=self.last_error)
                raise TokenError(self.last_error) from exc

            token = data.get("access_token")
            expires_at_raw = data.get("expires_at")
            if not token or not expires_at_raw:
                self.last_error = f"login response missing access_token/expires_at: {data}"
                log.error(self.last_error)
                self._audit_login(email, eid, success=False, error_message=self.last_error)
                raise TokenError(self.last_error)

            self.access_token = token
            self.expires_at = _parse_iso8601(expires_at_raw)
            self.session_id = data.get("session_id")
            self.email, self.eid = email, eid
            self.last_login_ts = datetime.now(timezone.utc)
            self.last_error = None

            log.info(
                "Authenticated as %s (session_id=%s, expires_at=%s, ttl=%.0fs)",
                email, self.session_id, self.expires_at, self.seconds_remaining(),
            )
            self._save_to_db()
            self._audit_login(email, eid, success=True)
            self._ensure_watchdog()
            return data

    def get_token(self) -> str:
        """Return a valid access token, transparently (re)logging in if needed."""
        with self._lock:
            if self.needs_refresh():
                # Another worker process may have already refreshed the token since we
                # last checked — pick that up before paying for a synchronous re-login.
                self._load_from_db()
            if self.needs_refresh():
                self.login()
            return self.access_token

    def auth_headers(self) -> Dict[str, str]:
        token = self.get_token()
        return {AUTH_HEADER_NAME: f"{AUTH_HEADER_PREFIX}{token}", "Content-Type": "application/json"}

    # -- background keep-alive ----------------------------------------------
    def _ensure_watchdog(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True
        threading.Thread(target=self._watchdog_loop, name="ig-token-watchdog", daemon=True).start()
        log.info("Token watchdog started (checks every %ss, refreshes %ss before expiry)",
                  TOKEN_WATCHDOG_INTERVAL, TOKEN_REFRESH_BUFFER_SECONDS)

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(TOKEN_WATCHDOG_INTERVAL)
            try:
                with self._lock:
                    self._load_from_db()  # pick up a refresh another worker's watchdog already did
                    has_creds = bool(self.email and self.eid)
                    stale = self.needs_refresh()
                if has_creds and stale:
                    log.info("Token watchdog: proactively refreshing access token")
                    self.login()
            except TokenError as exc:
                log.warning("Token watchdog refresh failed (will retry next interval): %s", exc)
            except Exception:
                log.exception("Token watchdog hit an unexpected error; continuing")


# ── Per-user token-manager registry ─────────────────────────────────────────
# Each distinct session_key (the app uses the user's email) gets its own
# TokenManager, so every person's Instant Graph token is isolated. Managers
# are created lazily on first use and reused thereafter.
_tm_registry: Dict[str, "TokenManager"] = {}
_tm_registry_lock = threading.Lock()


def get_token_manager(session_key: Optional[str] = None) -> "TokenManager":
    key = (session_key or DB_SESSION_KEY).strip().lower()
    tm = _tm_registry.get(key)
    if tm is None:
        with _tm_registry_lock:
            tm = _tm_registry.get(key)
            if tm is None:
                tm = TokenManager(session_key=key)
                _tm_registry[key] = tm
    return tm


# Default manager retained for the bundled Dash ops console (single shared
# session), so that page keeps working unchanged. The MCP tools below resolve
# a per-user manager via get_token_manager(session_key) instead.
token_manager = get_token_manager(DB_SESSION_KEY)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — INSTANT GRAPH API CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class InstantGraphError(RuntimeError):
    """Raised when an Instant Graph API call fails after retries."""


def _build_ig_ssl_context() -> Optional[ssl.SSLContext]:
    """Build the SSLContext used to verify Instant Graph's HTTPS cert.

    Instant Graph's cert is self-signed and isn't the root of a longer chain,
    so it's trusted directly via IG_CA_BUNDLE rather than a public CA. Plain
    requests(verify=<path>) loads that file but still raises "self signed
    certificate in certificate chain" on it — OpenSSL only accepts a
    directly-trusted self-signed cert with VERIFY_X509_PARTIAL_CHAIN set,
    which requests has no option for, hence the custom context here.
    """
    if not IG_CA_BUNDLE:
        return None
    ctx = ssl.create_default_context(cafile=IG_CA_BUNDLE)
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


class _PinnedSSLContextAdapter(HTTPAdapter):
    """HTTPAdapter that always uses a pre-built SSLContext instead of the
    per-request verify=... path, so ssl.VERIFY_X509_PARTIAL_CHAIN (set above)
    is actually applied to the handshake."""

    def __init__(self, ssl_context: ssl.SSLContext, *args, **kwargs) -> None:
        self._ssl_context = ssl_context
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(*args, **kwargs)


def _session_with_retries() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    ig_ssl_context = _build_ig_ssl_context()
    if ig_ssl_context is not None:
        s.mount("https://", _PinnedSSLContextAdapter(ig_ssl_context, max_retries=retries))
    else:
        s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


_http = _session_with_retries()


def _call(method: str, url: str, *, session_key: Optional[str] = None,
          params: dict = None, json_body: dict = None,
          _retry_on_401: bool = True) -> Dict[str, Any]:
    tm = get_token_manager(session_key)
    headers = tm.auth_headers()
    try:
        resp = _http.request(method, url, headers=headers, params=params, json=json_body,
                              timeout=REQUEST_TIMEOUT, verify=IG_VERIFY)
    except requests.RequestException as exc:
        raise InstantGraphError(f"{method} {url} failed: {exc}") from exc

    if resp.status_code == 401 and _retry_on_401:
        log.warning("401 from %s — forcing re-login and retrying once", url)
        with tm._lock:
            tm.access_token = None  # forces get_token() to re-login
        return _call(method, url, session_key=session_key, params=params,
                     json_body=json_body, _retry_on_401=False)

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise InstantGraphError(f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:500]}") from exc

    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as exc:
        raise InstantGraphError(f"{method} {url} returned a non-JSON body: {resp.text[:300]}") from exc


def api_get_hosts(session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("GET", HOSTS_URL, session_key=session_key)


def api_get_components(host_id: int, cid: int, session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("GET", COMPONENTS_URL, session_key=session_key,
                 params={"host_id": host_id, "cid": cid})


def api_prefix_filter(host_id: int, cid: int, components: List[str],
                      session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("POST", PREFIX_URL, session_key=session_key,
                 json_body={"host_id": host_id, "cid": cid, "components": components})


def api_get_suffixes(host_id: int, cid: int, prefix: Dict[str, List[str]],
                     session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("POST", SUFFIX_URL, session_key=session_key,
                 json_body={"host_id": host_id, "cid": cid, "prefix": prefix})


def api_render_table(selecteddata: List[Dict[str, Any]], components: List[str],
                      search: str = "", page: int = 1, offset: int = 10,
                      session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("POST", RENDER_URL, session_key=session_key, json_body={
        "selecteddata": selecteddata, "search": search, "page": page,
        "offset": offset, "components": components,
    })


def api_get_historical_data(item_ids: List[int], start_time: int, end_time: int,
                             cid: Dict[str, int], session_key: Optional[str] = None) -> Dict[str, Any]:
    return _call("POST", HISTORY_URL, session_key=session_key, json_body={
        "item_ids": item_ids, "start_time": start_time, "end_time": end_time, "cid": cid,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MCP SERVER (tools an agent will call)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Typical chaining order for a ReAct / CoT / GoT agent answering a question
#  like "top 5 traffic KPIs on <device> in the last 24 hours" or "which
#  devices crossed <threshold> on <KPI> between <start> and <end>":
#
#     ig_list_hosts()
#         -> pick host_id + its cid
#     ig_list_components(host_id, cid)
#         -> pick component, e.g. "traffic", "errors", "cpu", "bgp"
#     ig_list_interfaces(host_id, cid, [component])
#         -> pick one or more interface/object prefixes, e.g. "Interface 100GE0/3/2"
#     ig_list_kpis(host_id, cid, {component: [interface, ...]})
#         -> pick one or more itemid(s) for the KPI(s) of interest, e.g. "HC In Octets"
#     ig_resolve_items([{hostid, item_ids, cid}], [component])
#         -> resolve pretty item_name / item_key for the chosen itemids (chart legends,
#            human-readable answers)
#     ig_get_kpi_history(item_ids, start_time, end_time, {itemid: cid})
#         -> the actual time series + Min/Max/Last/Total/Average summary per KPI
#
#  Ranking ("top N"), threshold comparisons, and cross-device aggregation are
#  NOT computed server-side by Instant Graph — the agent should do that
#  reasoning itself over the series / summary values ig_get_kpi_history
#  returns (this is intentional: this MCP server only wraps the documented
#  APIs 1:1, the reasoning belongs in the agent layer).
# ══════════════════════════════════════════════════════════════════════════════

mcp_app = FastMCP(
    name="instant-graph",
    instructions=(
        "Tools for the Instant Graph network monitoring platform (VI-IPPMS). "
        "Use ig_list_hosts -> ig_list_components -> ig_list_interfaces -> "
        "ig_list_kpis -> ig_resolve_items -> ig_get_kpi_history to go from a "
        "natural-language question about a device/interface/KPI to concrete "
        "time-series data. Call ig_login first (or ig_token_status to check) "
        "if a call fails with an authentication error. Compute rankings, "
        "top-N, and threshold breaches yourself from the values returned by "
        "ig_get_kpi_history — these are not pre-computed by the API."
    ),
    host=MCP_HOST,
    port=MCP_PORT,
)


@mcp_app.tool()
@audited_tool
def ig_login(
    email: Annotated[str, Field(description="Corporate email address, e.g. firstname.lastname@vodafoneidea.com")],
    eid: Annotated[str, Field(description="Employee id, e.g. 22013296")],
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key that isolates this login from other users' "
                          "tokens. The calling application supplies this (typically the user's "
                          "email). If omitted, the email is used as the key."),
    ] = None,
) -> Dict[str, Any]:
    """
    Authenticate against Instant Graph and cache the access token for this
    user's session. Call once per user at the start of a session; the token is
    then reused and auto-refreshed. Pass the same session_key on every
    subsequent call. Only call again if ig_token_status reports
    authenticated=false or a call fails with an authentication error.

    Example call:
        ig_login(email="first.last@vodafoneidea.com", eid="22013296")
    Successful response shape:
        {"authenticated": true, "email": "first.last@vodafoneidea.com",
         "session_key": "first.last@vodafoneidea.com", "session_id": 2535,
         "expires_at": "2026-07-01T08:39:02+00:00"}
    """
    sk = (session_key or email)
    tm = get_token_manager(sk)
    data = tm.login(email, eid)
    return {
        "authenticated": True,
        "email": tm.email,
        "session_key": tm.session_key,
        "session_id": data.get("session_id"),
        "expires_at": data.get("expires_at"),
    }


@mcp_app.tool()
@audited_tool
def ig_token_status(
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    Check whether this user's Instant Graph session is valid and how many
    seconds remain before the access token needs refreshing.

    Example call:
        ig_token_status(session_key="first.last@vodafoneidea.com")
    Successful response shape:
        {"authenticated": true, "email": "...", "session_id": 2535,
         "expires_at": "...", "seconds_remaining": 3421.0, "last_error": null}
    """
    return get_token_manager(session_key).status()


@mcp_app.tool()
@audited_tool
def ig_list_hosts(
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    List every device (host) Instant Graph knows about. First call in almost
    every chain: the hostid and cid it returns are required by every other data
    tool. There is no circle->cid API — the circle is encoded in the device
    NAME (e.g. GJW/GJAHD => Gujarat), so filter by name tokens client-side.

    Example call:
        ig_list_hosts(session_key="first.last@vodafoneidea.com")
    Successful response shape:
        {"hosts": [
            {"hostid": 11193, "host": "APVSPGJWPAR01HNE40",
             "name": "APVSPGJWPAR01HNE40", "cid": 2},
            {"hostid": 27947, "host": "ABSCRDDCN01JEX43",
             "name": "ABSCRDDCN01JEX43", "cid": 1}
        ]}
    """
    return api_get_hosts(session_key=session_key)


@mcp_app.tool()
@audited_tool
def ig_list_components(
    host_id: Annotated[int, Field(description="hostid from ig_list_hosts, e.g. 11193")],
    cid: Annotated[int, Field(description="cid (circle id) for that host, from ig_list_hosts, e.g. 2")],
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    List the monitored component/category names for one device (e.g. "bgp",
    "traffic", "cpu", "errors", "broadcast") with how many indicators
    (item_count) exist per component. Single-host only — no multi-host variant.
    A device's total KPI count is the sum of item_count across components.

    Example call:
        ig_list_components(host_id=14881, cid=3)
    Successful response shape:
        {"host_id": 14881, "units": "bps",
         "components": [{"component": "bgp", "item_count": 2},
                        {"component": "traffic", "item_count": 71}]}
    """
    return api_get_components(host_id, cid, session_key=session_key)


@mcp_app.tool()
@audited_tool
def ig_list_interfaces(
    host_id: Annotated[int, Field(description="hostid from ig_list_hosts, e.g. 11193")],
    cid: Annotated[int, Field(description="cid (circle id) for that host, from ig_list_hosts, e.g. 2")],
    components: Annotated[List[str], Field(description='Component names to filter by, e.g. ["traffic"]')],
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    List the objects (interfaces/links) under the given component(s) for one
    device (wraps prefix-filter). Single-host only.

    IMPORTANT — components must be REAL component names from ig_list_components
    for this device; never pass a placeholder such as "empty"/"none". To list
    ALL interfaces on a device (no component specified by the user), first call
    ig_list_components and pass every returned component name, then merge the
    results — do not guess a single component. An unknown/placeholder component
    yields an EMPTY component_prefixes rather than an error.

    FORMAT GOTCHA: each returned item ends in a space + double colon (" ::"),
    e.g. "Interface 100GE0/3/2 ::". Pass these strings to ig_list_kpis VERBATIM
    (do not strip the " ::").

    Example call:
        ig_list_interfaces(host_id=11193, cid=2, components=["traffic"])
    Successful response shape:
        {"component_prefixes": {"traffic": {
            "items": ["Interface 100GE0/3/0 ::", "Interface 100GE0/3/2 ::",
                      "Interface Eth-Trunk1 ::"],
            "flags": "Object level"}}}
    """
    return api_prefix_filter(host_id, cid, components, session_key=session_key)


@mcp_app.tool()
@audited_tool
def ig_list_kpis(
    host_id: Annotated[int, Field(description="hostid from ig_list_hosts, e.g. 11193")],
    cid: Annotated[int, Field(description="cid (circle id) for that host, from ig_list_hosts, e.g. 2")],
    prefix: Annotated[
        Dict[str, List[str]],
        Field(description='Component -> interface prefixes, e.g. {"traffic": ["Interface 100GE0/3/2 ::"]}. '
                          'The prefix strings MUST be the exact values returned by ig_list_interfaces, '
                          'including the trailing " ::".'),
    ],
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    List the KPIs/indicators (suffixes, e.g. "HC In Octets", "HC Out Octets",
    "Max HC Octet") under a specific interface, each with the numeric itemid
    needed to fetch data (wraps get-suffixes). Single-host only.

    IMPORTANT — the component key must be a REAL component name from
    ig_list_components for THIS device (e.g. "traffic", "errors", "broadcast",
    "bgp"). Never pass a placeholder like "empty"/"none" and do NOT assume an
    interface lives under "traffic": the same interface can be monitored under
    any component. If you don't know which component owns an interface, call
    ig_list_interfaces for each component (or the ones from ig_list_components)
    and use the component whose items actually contain that interface. Passing a
    component that doesn't own the interface returns an EMPTY results set, not an
    error — so an empty response usually means "wrong/placeholder component",
    not "no KPIs".

    Example call:
        ig_list_kpis(host_id=11193, cid=2,
                     prefix={"traffic": ["Interface 100GE0/3/2 ::"]})
    Successful response shape:
        {"results": {"traffic : Interface 100GE0/3/2": [
            {"prefix": "Interface 100GE0/3/2", "suffix": "HC In Octets", "itemid": 2080688},
            {"prefix": "Interface 100GE0/3/2", "suffix": "HC Out Octets", "itemid": 2080870}
        ]}}
    """
    return api_get_suffixes(host_id, cid, prefix, session_key=session_key)


@mcp_app.tool()
@audited_tool
def ig_resolve_items(
    selecteddata: Annotated[
        List[Dict[str, Any]],
        Field(description='One entry per host: [{"hostid": 11193, "item_ids": [2080688], "cid": 2}]'),
    ],
    components: Annotated[List[str], Field(description='Component names involved, e.g. ["traffic"]')],
    search: Annotated[str, Field(description="Optional free-text filter")] = "",
    page: Annotated[int, Field(description="Page number, 1-indexed")] = 1,
    offset: Annotated[int, Field(description="Page size")] = 10,
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    Resolve itemid(s) (from ig_list_kpis) into full item_name / item_key /
    host_name / component (wraps render-table). Use for chart legends or
    human-readable answers.

    Example call:
        ig_resolve_items(selecteddata=[{"hostid": 11193, "item_ids": [2080688], "cid": 2}],
                         components=["traffic"])
    Successful response shape:
        {"items": [{"item_name": "Interface 100GE0/3/2 : HC In Octets",
                    "item_key": "ifHCInOctets[8]", "itemid": 2080688,
                    "hostid": 11193, "host_name": "APVSPGJWPAR01HNE40",
                    "component": "traffic", "cid": 2}], "total_pages": 1}
    """
    return api_render_table(selecteddata, components, search, page, offset,
                            session_key=session_key)


@mcp_app.tool()
@audited_tool
def ig_get_kpi_history(
    item_ids: Annotated[List[int], Field(description="itemid(s) from ig_list_kpis, e.g. [2080688, 2080870]")],
    start_time: Annotated[int, Field(description="Window start, Unix epoch SECONDS (not ms), e.g. 1782817742")],
    end_time: Annotated[int, Field(description="Window end, Unix epoch SECONDS (not ms), e.g. 1782904142")],
    cid: Annotated[
        Dict[str, int],
        Field(description='Maps each item_id (as a STRING) to its circle id, e.g. {"2080688": 2}'),
    ],
    session_key: Annotated[
        Optional[str],
        Field(description="Per-user session key used at ig_login (typically the user's email)."),
    ] = None,
) -> Dict[str, Any]:
    """
    Fetch time-series + per-KPI summary between start_time and end_time (both
    Unix epoch SECONDS), wrapping get-historical-data. Batch multiple itemids
    for one host into a single call.

    FORMAT GOTCHA: the summary Values are formatted strings with mixed units
    ("244.49 MB", "35.54 GB"). Normalise to a common base (bytes) before
    ranking/thresholding — comparing the mantissa alone is wrong.

    Example call:
        ig_get_kpi_history(item_ids=[2080688], start_time=1782817742,
                           end_time=1782904142, cid={"2080688": 2})
    Successful response shape:
        {"series": [{"name": "...", "data": [[1782817920000, 201064849.28], ...]}],
         "Values": [{"name": "...", "Minimum": "15.12 MB", "Maximum": "244.49 MB",
                     "Last": "218.11 MB", "Total": "35.54 GB", "Average": "126.35 MB",
                     "units": "bps"}]}
    """
    return api_get_historical_data(item_ids, start_time, end_time, cid,
                                   session_key=session_key)


# Static registry mirrored in the Dash "Available MCP Tools" panel below.
# Keep this in sync with the @mcp_app.tool() functions above.
TOOL_REGISTRY = [
    {"name": "ig_login", "summary": "Authenticate with email + employee id, cache the access token."},
    {"name": "ig_token_status", "summary": "Check current auth state and token time-to-live."},
    {"name": "ig_list_hosts", "summary": "List devices (hostid, name, cid)."},
    {"name": "ig_list_components", "summary": "List component categories for a device (traffic, cpu, bgp, ...)."},
    {"name": "ig_list_interfaces", "summary": "List interfaces/objects under a component (prefix-filter)."},
    {"name": "ig_list_kpis", "summary": "List KPIs/itemids under an interface (get-suffixes)."},
    {"name": "ig_resolve_items", "summary": "Resolve itemid(s) to human-readable names (render-table)."},
    {"name": "ig_get_kpi_history", "summary": "Fetch time series + summary stats between start/end time."},
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DASH FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

dash_app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Instant Graph Ops Console")
flask_server = dash_app.server  # exposed in case you want to front this with gunicorn

dash_app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root{
            --bg:#0a0e17; --panel:#0f1626; --panel-2:#131c30; --line:#1e293b;
            --text:#e2e8f0; --muted:#7c8aa5; --cyan:#22d3ee; --cyan-dim:#0e7490;
            --amber:#f59e0b; --green:#34d399; --red:#f87171;
        }
        *{box-sizing:border-box}
        body{margin:0;background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
        .mono{font-family:'JetBrains Mono',monospace;}
        .ig-scanbar{height:2px;width:100%;background:linear-gradient(90deg,transparent,var(--cyan),transparent);
            background-size:200% 100%;animation:scan 3.2s linear infinite;}
        @keyframes scan{0%{background-position:200% 0}100%{background-position:-200% 0}}
        .ig-login-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;
            background:radial-gradient(circle at 50% 0%, #0f1e33 0%, #0a0e17 60%);}
        .ig-login-card{width:380px;background:var(--panel);border:1px solid var(--line);border-radius:10px;
            padding:0 28px 28px 28px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.5);}
        .ig-login-logo{font-size:13px;letter-spacing:.14em;color:var(--cyan);margin:22px 0 4px 0;font-weight:600;}
        .ig-login-title{font-size:22px;font-weight:700;margin:0 0 2px 0;}
        .ig-login-sub{color:var(--muted);font-size:13px;margin:0 0 22px 0;}
        .ig-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
            margin:14px 0 6px 0;display:block;}
        .ig-input{width:100%;padding:10px 12px;background:var(--panel-2);border:1px solid var(--line);
            border-radius:6px;color:var(--text);font-size:14px;font-family:'JetBrains Mono',monospace;}
        .ig-input:focus{outline:none;border-color:var(--cyan);}
        .ig-btn{width:100%;padding:11px;margin-top:20px;background:var(--cyan);color:#04202b;border:none;
            border-radius:6px;font-weight:700;font-size:14px;cursor:pointer;letter-spacing:.02em;}
        .ig-btn:hover{background:#67e8f9;}
        .ig-btn-ghost{background:transparent;border:1px solid var(--line);color:var(--text);width:auto;
            padding:8px 14px;margin-top:0;font-weight:600;font-size:12px;}
        .ig-btn-ghost:hover{border-color:var(--cyan);color:var(--cyan);}
        .ig-alert-error{margin-top:14px;padding:9px 12px;background:rgba(248,113,113,.1);
            border:1px solid rgba(248,113,113,.35);color:var(--red);border-radius:6px;font-size:12.5px;}
        .ig-alert-ok{margin-top:14px;padding:9px 12px;background:rgba(52,211,153,.1);
            border:1px solid rgba(52,211,153,.35);color:var(--green);border-radius:6px;font-size:12.5px;}
        .ig-shell{max-width:1180px;margin:0 auto;padding:22px 24px 60px 24px;}
        .ig-header{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;
            background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:18px;}
        .ig-dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;
            margin-right:7px;box-shadow:0 0 8px var(--green);animation:pulse 1.8s ease-in-out infinite;}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
        .ig-panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;
            margin-bottom:18px;}
        .ig-panel h3{margin:0 0 4px 0;font-size:14px;letter-spacing:.03em;}
        .ig-panel .ig-hint{color:var(--muted);font-size:12px;margin:0 0 14px 0;}
        .ig-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;}
        .ig-col{flex:1;min-width:220px;}
        .ig-field-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
            margin-bottom:6px;display:block;}
        .ig-tool-row{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--line);
            font-size:12.5px;}
        .ig-tool-row:last-child{border-bottom:none;}
        .ig-tool-name{color:var(--cyan);font-family:'JetBrains Mono',monospace;}
        .ig-tool-desc{color:var(--muted);text-align:right;max-width:60%;}
        .ig-stat-badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:600;
            margin-right:8px;}
        .ig-badge-cyan{background:rgba(34,211,238,.12);color:var(--cyan);}
        .ig-badge-amber{background:rgba(245,158,11,.12);color:var(--amber);}
        .Select-control, .dash-dropdown .Select-control{background:var(--panel-2)!important;
            border-color:var(--line)!important;color:var(--text)!important;}
        ._dash-loading{color:var(--cyan);}
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

# ── Login page ──────────────────────────────────────────────────────────────

def login_layout(error_message: Optional[str] = None):
    return html.Div(className="ig-login-wrap", children=[
        html.Div(className="ig-login-card", children=[
            html.Div(className="ig-scanbar"),
            html.Div("INSTANT GRAPH", className="ig-login-logo"),
            html.Div("Conversational Ops Console", className="ig-login-title"),
            html.P("Sign in with your corporate email and employee id to start a session.",
                   className="ig-login-sub"),
            html.Label("Email", className="ig-label"),
            dcc.Input(id="login-email", type="email", className="ig-input",
                      placeholder="firstname.lastname@vodafoneidea.com"),
            html.Label("Employee ID", className="ig-label"),
            dcc.Input(id="login-eid", type="text", className="ig-input", placeholder="e.g. 22013296"),
            html.Button("Sign in", id="login-btn", n_clicks=0, className="ig-btn"),
            html.Div(id="login-status",
                      children=(html.Div(error_message, className="ig-alert-error") if error_message else None)),
        ])
    ])


@dash_app.callback(
    Output("auth-store", "data"),
    Output("login-status", "children"),
    Input("login-btn", "n_clicks"),
    State("login-email", "value"),
    State("login-eid", "value"),
    prevent_initial_call=True,
)
def do_login(n_clicks, email, eid):
    if not email or not eid:
        return no_update, html.Div("Email and Employee ID are both required.", className="ig-alert-error")
    try:
        data = token_manager.login(email.strip(), eid.strip())
        return (
            {"authenticated": True, "email": token_manager.email, "ts": time.time()},
            html.Div(f"Signed in — session {data.get('session_id')}", className="ig-alert-ok"),
        )
    except TokenError as exc:
        return no_update, html.Div(str(exc), className="ig-alert-error")


# ── Dashboard page ──────────────────────────────────────────────────────────

def header_bar():
    return html.Div(className="ig-header", children=[
        html.Div([
            html.Span(className="ig-dot"),
            html.Span("LIVE", className="mono", style={"color": "var(--green)", "fontSize": "12px", "fontWeight": "700"}),
            html.Span(id="header-email", className="mono", style={"marginLeft": "16px", "color": "var(--muted)", "fontSize": "12.5px"}),
        ]),
        html.Div([
            html.Span(id="token-countdown", className="mono",
                      style={"marginRight": "18px", "fontSize": "12.5px", "color": "var(--muted)"}),
            html.Button("Refresh token", id="refresh-token-btn", n_clicks=0, className="ig-btn-ghost",
                       style={"marginRight": "8px"}),
            html.Button("Log out", id="logout-btn", n_clicks=0, className="ig-btn-ghost"),
        ]),
    ])


def tool_registry_panel():
    rows = [
        html.Div(className="ig-tool-row", children=[
            html.Span(t["name"], className="ig-tool-name"),
            html.Span(t["summary"], className="ig-tool-desc"),
        ]) for t in TOOL_REGISTRY
    ]
    return html.Div(className="ig-panel", children=[
        html.H3("MCP tools exposed to the agent"),
        html.P(f"streamable-http · http://{MCP_HOST}:{MCP_PORT}{mcp_app.settings.streamable_http_path}",
               className="ig-hint mono"),
        html.Div(rows),
    ])


def explorer_panel():
    return html.Div(className="ig-panel", children=[
        html.H3("Data explorer"),
        html.P("Walk the same chain an agent uses: host → component → interface → KPI → history.",
               className="ig-hint"),

        html.Div(className="ig-row", children=[
            html.Div(className="ig-col", children=[
                html.Span("1. Host", className="ig-field-label"),
                html.Button("Load hosts", id="load-hosts-btn", n_clicks=0, className="ig-btn-ghost",
                           style={"marginBottom": "8px"}),
                dcc.Dropdown(id="host-dd", placeholder="Select a device...", clearable=False),
            ]),
            html.Div(className="ig-col", children=[
                html.Span("2. Component", className="ig-field-label"),
                dcc.Dropdown(id="component-dd", placeholder="Select a component...", multi=True),
            ]),
        ]),

        html.Div(className="ig-row", children=[
            html.Div(className="ig-col", children=[
                html.Span("3. Interface / object", className="ig-field-label"),
                dcc.Dropdown(id="interface-dd", placeholder="Select interface(s)...", multi=True),
            ]),
            html.Div(className="ig-col", children=[
                html.Span("4. KPI", className="ig-field-label"),
                dcc.Dropdown(id="kpi-dd", placeholder="Select KPI(s)...", multi=True),
            ]),
        ]),

        html.Div(className="ig-row", children=[
            html.Div(className="ig-col", children=[
                html.Span("5. Time window", className="ig-field-label"),
                html.Div(style={"display": "flex", "gap": "6px", "marginBottom": "8px"}, children=[
                    html.Button("Last 1h", id="quick-1h", n_clicks=0, className="ig-btn-ghost"),
                    html.Button("Last 6h", id="quick-6h", n_clicks=0, className="ig-btn-ghost"),
                    html.Button("Last 24h", id="quick-24h", n_clicks=0, className="ig-btn-ghost"),
                    html.Button("Last 7d", id="quick-7d", n_clicks=0, className="ig-btn-ghost"),
                ]),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    dcc.Input(id="start-epoch", type="number", placeholder="start (epoch s)", className="ig-input"),
                    dcc.Input(id="end-epoch", type="number", placeholder="end (epoch s)", className="ig-input"),
                ]),
            ]),
            html.Div(className="ig-col", children=[
                html.Span("6. Optional threshold / ranking", className="ig-field-label"),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    dcc.Input(id="threshold-input", type="number", placeholder="flag if Max exceeds...",
                             className="ig-input"),
                    dcc.Input(id="topn-input", type="number", placeholder="show top N by avg", className="ig-input"),
                ]),
            ]),
        ]),

        html.Button("Fetch KPI data", id="fetch-btn", n_clicks=0, className="ig-btn", style={"width": "220px"}),
        html.Div(id="fetch-status", style={"marginTop": "10px"}),
        dcc.Loading(children=[
            dcc.Graph(id="kpi-graph", style={"marginTop": "18px", "display": "none"}),
            html.Div(id="kpi-stats-table", style={"marginTop": "14px"}),
        ]),
    ])


def dashboard_layout():
    return html.Div(className="ig-shell", children=[
        header_bar(),
        tool_registry_panel(),
        explorer_panel(),
    ])


# ── App layout ────────────────────────────────────────────────────────────
# Both pages are mounted here, in the INITIAL layout, and merely shown/hidden
# with CSS (see render_page below) instead of being conditionally
# inserted/removed as a single page-content div's children.
#
# Why this matters: every button inside dashboard_layout() (logout-btn,
# load-hosts-btn, refresh-token-btn, ...) is the Input of a callback declared
# with prevent_initial_call=True. Dash only honors prevent_initial_call for
# an Input that is already present in the server-rendered layout at
# page-load time. If a component is instead injected later (e.g. as the
# output of another callback, which is what a login -> page-content swap
# does), Dash treats its first appearance as "an existing input got a new
# value" and fires the callback once regardless of prevent_initial_call.
# That was firing do_logout (wiping the token + deleting the ig_auth_sessions
# row) and load_hosts (immediately failing with TokenError) right after
# every successful login. Mounting both pages up front avoids that entirely.
#
# This has to be assigned down here, after login_layout()/dashboard_layout()
# are both defined above, since it calls them immediately (not lazily).
dash_app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="auth-store", storage_type="memory"),
    dcc.Store(id="hosts-store", storage_type="memory"),
    dcc.Store(id="components-store", storage_type="memory"),
    dcc.Store(id="interfaces-store", storage_type="memory"),
    dcc.Store(id="kpis-store", storage_type="memory"),
    dcc.Store(id="time-range-store", storage_type="memory"),
    dcc.Interval(id="clock", interval=1000, n_intervals=0),
    html.Div(id="login-page", children=login_layout(), style={"display": "block"}),
    html.Div(id="dashboard-page", children=dashboard_layout(), style={"display": "none"}),
])


@dash_app.callback(
    Output("login-page", "style"),
    Output("dashboard-page", "style"),
    Input("auth-store", "data"),
)
def render_page(_auth_data):
    if token_manager.is_authenticated():
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@dash_app.callback(
    Output("header-email", "children"),
    Output("token-countdown", "children"),
    Input("clock", "n_intervals"),
)
def tick(_n):
    st = token_manager.status()
    if not st["authenticated"]:
        return "", ""
    mins, secs = divmod(int(st["seconds_remaining"]), 60)
    return f"{st['email']} · session {st['session_id']}", f"token expires in {mins:02d}:{secs:02d}"


@dash_app.callback(
    Output("auth-store", "data", allow_duplicate=True),
    Output("host-dd", "value"),
    Output("component-dd", "value"),
    Output("interface-dd", "value"),
    Output("kpi-dd", "value"),
    Output("hosts-store", "data", allow_duplicate=True),
    Output("components-store", "data", allow_duplicate=True),
    Output("interfaces-store", "data", allow_duplicate=True),
    Output("kpis-store", "data", allow_duplicate=True),
    Output("kpi-graph", "figure", allow_duplicate=True),
    Output("kpi-graph", "style", allow_duplicate=True),
    Output("kpi-stats-table", "children", allow_duplicate=True),
    Output("fetch-status", "children", allow_duplicate=True),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def do_logout(n_clicks):
    token_manager.logout()
    return (
        {"authenticated": False},
        None, None, None, None,   # dropdown selections
        {}, {}, {}, {},           # id-lookup stores
        go.Figure(), {"display": "none"}, None, None,  # graph / table / status
    )


@dash_app.callback(
    Output("fetch-status", "children", allow_duplicate=True),
    Input("refresh-token-btn", "n_clicks"),
    prevent_initial_call=True,
)
def do_refresh(n_clicks):
    try:
        token_manager.login()
        return html.Div("Token refreshed.", className="ig-alert-ok")
    except TokenError as exc:
        return html.Div(str(exc), className="ig-alert-error")


# ── Explorer chain callbacks ─────────────────────────────────────────────────

@dash_app.callback(
    Output("host-dd", "options"),
    Output("hosts-store", "data"),
    Output("fetch-status", "children", allow_duplicate=True),
    Input("load-hosts-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_hosts(n_clicks):
    try:
        data = api_get_hosts()
        hosts = data.get("hosts", data if isinstance(data, list) else [])
        options = [{"label": f"{h.get('name') or h.get('host')}  (id {h['hostid']}, cid {h['cid']})",
                    "value": f"{h['hostid']}|{h['cid']}"} for h in hosts]
        return options, {h["hostid"]: h for h in hosts}, None
    except InstantGraphError as exc:
        return [], {}, html.Div(f"Could not load hosts: {exc}", className="ig-alert-error")


@dash_app.callback(
    Output("component-dd", "options"),
    Output("components-store", "data"),
    Input("host-dd", "value"),
    prevent_initial_call=True,
)
def load_components(host_value):
    if not host_value:
        return [], {}
    host_id, cid = host_value.split("|")
    try:
        data = api_get_components(int(host_id), int(cid))
        comps = data.get("components", [])
        options = [{"label": f"{c['component']} ({c.get('item_count', '?')} items)", "value": c["component"]}
                   for c in comps if isinstance(c, dict) and "component" in c]
        return options, {"host_id": int(host_id), "cid": int(cid)}
    except InstantGraphError:
        return [], {}


@dash_app.callback(
    Output("interface-dd", "options"),
    Output("interfaces-store", "data"),
    Input("component-dd", "value"),
    State("components-store", "data"),
    prevent_initial_call=True,
)
def load_interfaces(components, host_ctx):
    if not components or not host_ctx:
        return [], {}
    try:
        data = api_prefix_filter(host_ctx["host_id"], host_ctx["cid"], components)
        prefixes = data.get("component_prefixes", {})
        options, by_component = [], {}
        for comp, payload in prefixes.items():
            items = payload.get("items", []) if isinstance(payload, dict) else []
            by_component[comp] = items
            for item in items:
                options.append({"label": f"{comp}: {item}", "value": f"{comp}||{item}"})
        return options, by_component
    except InstantGraphError:
        return [], {}


@dash_app.callback(
    Output("kpi-dd", "options"),
    Output("kpis-store", "data"),
    Input("interface-dd", "value"),
    State("components-store", "data"),
    prevent_initial_call=True,
)
def load_kpis(interface_values, host_ctx):
    if not interface_values or not host_ctx:
        return [], {}
    prefix: Dict[str, List[str]] = {}
    for val in interface_values:
        comp, item = val.split("||", 1)
        prefix.setdefault(comp, []).append(item)
    try:
        data = api_get_suffixes(host_ctx["host_id"], host_ctx["cid"], prefix)
        results = data.get("results", {})
        options, itemid_meta = [], {}
        for group_name, entries in results.items():
            for entry in entries:
                label = f"{entry['prefix']}{entry['suffix']}"
                options.append({"label": label, "value": entry["itemid"]})
                itemid_meta[str(entry["itemid"])] = {
                    "label": label, "host_id": host_ctx["host_id"], "cid": host_ctx["cid"],
                }
        return options, itemid_meta
    except InstantGraphError:
        return [], {}


# ── Quick time-range buttons ─────────────────────────────────────────────────

@dash_app.callback(
    Output("start-epoch", "value"),
    Output("end-epoch", "value"),
    Input("quick-1h", "n_clicks"),
    Input("quick-6h", "n_clicks"),
    Input("quick-24h", "n_clicks"),
    Input("quick-7d", "n_clicks"),
    prevent_initial_call=True,
)
def quick_range(*_clicks):
    from dash import ctx as dash_ctx
    triggered = dash_ctx.triggered_id
    now = int(time.time())
    hours = {"quick-1h": 1, "quick-6h": 6, "quick-24h": 24, "quick-7d": 24 * 7}.get(triggered, 1)
    return now - hours * 3600, now


# ── Fetch + render KPI history ───────────────────────────────────────────────

@dash_app.callback(
    Output("kpi-graph", "figure"),
    Output("kpi-graph", "style"),
    Output("kpi-stats-table", "children"),
    Output("fetch-status", "children", allow_duplicate=True),
    Input("fetch-btn", "n_clicks"),
    State("kpi-dd", "value"),
    State("kpis-store", "data"),
    State("start-epoch", "value"),
    State("end-epoch", "value"),
    State("component-dd", "value"),
    State("threshold-input", "value"),
    State("topn-input", "value"),
    prevent_initial_call=True,
)
def fetch_kpi_data(n_clicks, item_ids, kpi_meta, start_epoch, end_epoch, components, threshold, top_n):
    empty_fig = go.Figure()
    hidden = {"display": "none"}
    if not item_ids:
        return empty_fig, hidden, None, html.Div("Select at least one KPI first.", className="ig-alert-error")
    if not start_epoch or not end_epoch:
        return empty_fig, hidden, None, html.Div("Pick a time window (or use a quick-range button).",
                                                  className="ig-alert-error")

    try:
        # Group by host for render-table, then resolve pretty names.
        by_host: Dict[str, Dict[str, Any]] = {}
        for iid in item_ids:
            meta = kpi_meta.get(str(iid), {})
            key = f"{meta.get('host_id')}|{meta.get('cid')}"
            by_host.setdefault(key, {"hostid": meta.get("host_id"), "cid": meta.get("cid"), "item_ids": []})
            by_host[key]["item_ids"].append(iid)

        resolved = api_render_table(list(by_host.values()), components or [])
        name_by_id = {it["itemid"]: it["item_name"] for it in resolved.get("items", [])}

        cid_map = {str(iid): kpi_meta.get(str(iid), {}).get("cid") for iid in item_ids}
        history = api_get_historical_data(item_ids, int(start_epoch), int(end_epoch), cid_map)

        fig = go.Figure()
        stats_rows = []
        series_list = history.get("series", [])
        values_list = history.get("Values", history.get("values", []))

        for i, s in enumerate(series_list):
            xs = [pt[0] for pt in s.get("data", [])]
            ys = [pt[1] for pt in s.get("data", [])]
            label = name_by_id.get(item_ids[i], s.get("name", f"item {item_ids[i]}")) if i < len(item_ids) else s.get("name")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(label)[:60]))

        for v in values_list:
            row = {
                "KPI": v.get("name", ""),
                "Minimum": v.get("Minimum"),
                "Maximum": v.get("Maximum"),
                "Last": v.get("Last"),
                "Average": v.get("Average"),
                "Total": v.get("Total"),
                "Units": v.get("units"),
            }
            stats_rows.append(row)

        breach_note = None
        if threshold is not None and stats_rows:
            def _num(x):
                try:
                    return float(str(x).split()[0])
                except (TypeError, ValueError, IndexError):
                    return None
            breached = [r for r in stats_rows if (_num(r.get("Maximum")) or 0) > threshold]
            breach_note = html.Div(
                f"{len(breached)} of {len(stats_rows)} KPI(s) exceeded threshold {threshold}.",
                className="ig-alert-error" if breached else "ig-alert-ok",
            )

        if top_n and stats_rows:
            def _avg(r):
                try:
                    return float(str(r.get("Average", 0)).split()[0])
                except (ValueError, IndexError):
                    return 0.0
            stats_rows = sorted(stats_rows, key=_avg, reverse=True)[: int(top_n)]

        table = dash_table.DataTable(
            data=stats_rows,
            columns=[{"name": c, "id": c} for c in ["KPI", "Minimum", "Maximum", "Last", "Average", "Total", "Units"]],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#131c30", "color": "#7c8aa5", "fontSize": "11px",
                         "textTransform": "uppercase", "border": "none"},
            style_cell={"backgroundColor": "#0f1626", "color": "#e2e8f0", "border": "1px solid #1e293b",
                       "fontFamily": "JetBrains Mono, monospace", "fontSize": "12px", "padding": "6px 10px"},
        )
        status_children = [html.Div("Fetched.", className="ig-alert-ok")]
        if breach_note:
            status_children.append(breach_note)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1626", plot_bgcolor="#0f1626",
            margin=dict(l=40, r=20, t=20, b=40), height=380,
            legend=dict(orientation="h", y=-0.25),
        )
        return fig, {"display": "block"}, table, html.Div(status_children)

    except InstantGraphError as exc:
        return empty_fig, hidden, None, html.Div(f"Fetch failed: {exc}", className="ig-alert-error")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _run_mcp_server():
    log.info("Starting MCP server on http://%s:%s%s", MCP_HOST, MCP_PORT, mcp_app.settings.streamable_http_path)
    mcp_app.run(transport="streamable-http")


def main():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  INSTANT GRAPH MCP SERVER                                            ║
║  MCP (agent tools)  : http://{MCP_HOST}:{MCP_PORT}{mcp_app.settings.streamable_http_path:<33}║
║  Dash (ops console) : http://{DASH_HOST}:{DASH_PORT}{'':<33}║
║  Instant Graph API  : {INSTANT_GRAPH_BASE:<48}║
║  Token refresh buffer: {TOKEN_REFRESH_BUFFER_SECONDS}s before expiry{'':<28}║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

    mcp_thread = threading.Thread(target=_run_mcp_server, name="mcp-server", daemon=True)
    mcp_thread.start()

    dash_app.run(host=DASH_HOST, port=DASH_PORT, debug=False, dev_tools_hot_reload=False)


if __name__ == "__main__":
    main()
