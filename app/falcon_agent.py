"""
FALCON – Vi NOC Intelligence  (HAR V16 frontend  ×  LangGraph Backend V3.4)
════════════════════════════════════════════════════════════════════════════
9-Node LangGraph Pipeline:
  Router → TableID → RAG → CoT(DSPy) → GoT(SQL-Beam) →
  SQLValidation → SQLExecution → Explanation  (+LLM Fallback)

Dash Frontend (preserved from HAR V16):
  5-badge topbar · sidebar conversations · cap-cards · chips ·
  wordcloud / bar / line / pie / table charts · SQL viewer

Run: python FALCON_AGENT_HAR_V17.py  →  http://0.0.0.0:8026

Dependencies (new vs V16):
  pip install langgraph dspy-ai sentence-transformers numpy
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os, re, csv, ast, json, time, math, hashlib, traceback, secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TypedDict
from urllib.parse import quote_plus

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ── DSPy ──────────────────────────────────────────────────────────────────────
import dspy

# ── Sentence Transformers ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Dash ──────────────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update, ALL, ctx
import plotly.graph_objects as go
import plotly.io as pio

# ── Security hardening (see combined_security_report.xlsx remediation) ─────────
from security_common import (
    build_sql_validator, db_connect_args, required_secret, SessionStore,
    SAFETY_PREAMBLE, classify_blocked_request, refusal_for, sanitize_output,
    validate_upload, SecurityAuditLogger, SQLSecurityError,
)

print("✅ All imports OK")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEBUG         = os.environ.get("DEBUG", "true").lower() == "true"

# ── LLM / GPU Proxy ───────────────────────────────────────────────────────────
GPU_URL  = os.environ.get("GPU_PROXY_URL", "http://127.0.0.1:8071/v1/infer")
GPU_KEY  = required_secret("GPU_API_KEY")          # no plaintext secret in source (Obs: secrets-in-source)
GPU_TO   = int(os.environ.get("GPU_TIMEOUT", "180"))
MODEL    = os.environ.get("DEFAULT_MODEL",  "mistral")

# ── Decode Configs ────────────────────────────────────────────────────────────
ROUTER_DECODE      = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}
INTENT_DECODE      = {"temperature": 0.1, "top_p": 0.9, "top_k": 5}
SQL_DECODE         = {"temperature": 0.1, "top_p": 0.9, "top_k": 5}
PARAM_DECODE       = {"temperature": 0.0, "top_p": 0.9, "top_k": 3}
COT_DECODE         = {"temperature": 0.2, "top_p": 0.9, "top_k": 5}
EXPLANATION_DECODE = {"temperature": 0.3, "top_p": 0.9, "top_k": 5}

# ── Retry / GoT Limits ────────────────────────────────────────────────────────
SQL_MAX_RETRIES = 2
EXEC_MAX_ITERS  = 3
GOT_BEAM_WIDTH  = 3
COT_MAX_HOPS    = 3
GOT_EARLY_EXIT  = 8.0

# ── Database ──────────────────────────────────────────────────────────────────
# NOTE: previously the DB password was a real credential hardcoded as the
# os.environ.get() default (i.e. committed to source control). It is now
# read strictly from the environment — see required_secret() in
# security_common.py. Deploy this with PGUSER/PGPASSWORD/PGHOST set to a
# scoped, least-privilege account (never the DB superuser).
PG = dict(
    user    =os.environ.get("PGUSER",     "falcon"),
    password=required_secret("PGPASSWORD"),
    host    =os.environ.get("PGHOST",     "10.19.71.249"),
    port    =os.environ.get("PGPORT",     "5432"),
    dbname  =os.environ.get("PGDATABASE", "falcondb"),
)
PG_SSLMODE = os.environ.get("PGSSLMODE", "require")   # fixes Obs #6 — Absence of Secure Transport Controls
_db_url = (f"postgresql+psycopg2://{PG['user']}:{quote_plus(PG['password'])}"
           f"@{PG['host']}:{PG['port']}/{PG['dbname']}")
try:
    engine = create_engine(
        _db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args=db_connect_args(sslmode=PG_SSLMODE),
    )
    print("DB OK")
except Exception as _e:
    engine = None
    print(f"DB error: {_e}")

# ── Security audit logging (fixes Obs #4 — Logging Control Weakness) ───────────
sec_logger = SecurityAuditLogger(schema="tt_falcon_schema", app_name="talk-to-falcon")

# ── RAG Paths ─────────────────────────────────────────────────────────────────
EMBEDDER_PATH   = os.environ.get("EMBEDDER_PATH", "/data/harish/eli-embedding-small-1/")
RAG_BASE_DIR    = os.environ.get("RAG_BASE_DIR",  "/srv/vi-assistant/rag")
EMBED_CACHE_DIR = os.path.join(RAG_BASE_DIR, "embeddings_cache")
QUESTIONS_CSV   = os.path.join(RAG_BASE_DIR, "100_Questions.csv")
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.environ.get("LOG_DIR", "/srv/vi-assistant/logs")
LOG_FILE = os.path.join(LOG_DIR, "falcon_v17.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Table allow-list ──────────────────────────────────────────────────────────
VALID_TABLES = {
    "falconschema.vw_f_hpsm_h",
    "falconschema.vw_sys_log_data_h",
    "falconschema.vw_f_session_data",
    "falconschema.f_vipam_data_h",
    "falconschema.f_static_ip_mpls_inventory",
    "falconschema.f_topology_view_data",
    "falconschema.f_hpsm_tt_rca",
}
METADATA_TABLE    = "falconschema.f_static_ip_mpls_inventory"
RCA_TABLE         = "falconschema.f_hpsm_tt_rca"
CORRELATION_TABLE = "falconschema.f_topology_view_data"

RAG_DOCS: Dict[str, str] = {
    "falconschema.vw_f_hpsm_h":                os.path.join(RAG_BASE_DIR, "hpsm.txt"),
    "falconschema.vw_sys_log_data_h":          os.path.join(RAG_BASE_DIR, "syslog.txt"),
    "falconschema.vw_f_session_data":          os.path.join(RAG_BASE_DIR, "session.txt"),
    "falconschema.f_vipam_data_h":             os.path.join(RAG_BASE_DIR, "vipam.txt"),
    "falconschema.f_static_ip_mpls_inventory": os.path.join(RAG_BASE_DIR, "inventory.txt"),
    "falconschema.f_topology_view_data":       os.path.join(RAG_BASE_DIR, "topo.txt"),
    "falconschema.f_hpsm_tt_rca":              os.path.join(RAG_BASE_DIR, "rca.txt"),
}

TABLE_TO_NPY: Dict[str, str] = {
    "falconschema.vw_f_hpsm_h":                "hpsm.npy",
    "falconschema.vw_sys_log_data_h":          "syslog.npy",
    "falconschema.vw_f_session_data":          "session.npy",
    "falconschema.f_vipam_data_h":             "vipam.npy",
    "falconschema.f_static_ip_mpls_inventory": "inventory.npy",
    "falconschema.f_topology_view_data":       "topo.npy",
    "falconschema.f_hpsm_tt_rca":              "rca.npy",
}



# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1.5 — AUTH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Domain flag ───────────────────────────────────────────────────────────────
# True  → ANY @vodafoneidea.com address can log in (use domain password or own)
# False → only emails listed in ALLOWED_EMAILS below can log in
ALLOW_VODAFONE_DOMAIN: bool = False

# ── Per-user credentials hash helper ──────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Load allowed emails dynamically from TTF_user_details ─────────────────────
def get_allowed_emails() -> List[str]:
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT username FROM falconschema.ttf_user_details")).fetchall()
            return [row[0].strip().lower() for row in rows if row[0]]
    except Exception as e:
        print(f"[AUTH] Failed to fetch allowed emails: {e}")
        return []

# SECURITY FIX (secrets-in-source): real account passwords used to be
# hardcoded here in plaintext-derived form, visible to anyone with repo
# access. Allow-listed emails and credentials are now loaded from the DB at
# startup; if the DB is unreachable, both are empty and login fails closed
# (nobody can authenticate) rather than falling back to a hardcoded
# password. TTF_user_details should hold a proper per-user password hash
# column for a production-grade rollout — flagged separately since it's a
# schema change outside this application file.
ALLOWED_EMAILS: List[str] = get_allowed_emails()

# ── Per-user credentials: only fallback/admin accounts hardcoded ──────────────
def get_user_credentials():
    if engine is None:
        return {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT username from falconschema.ttf_user_details")).fetchall()
            default_pw = required_secret("FALCON_DEFAULT_USER_PASSWORD")
            return {row[0].strip().lower(): _hash(default_pw) for row in rows if row[0]}
    except Exception as e:
        print(f"[AUTH] Failed to fetch credentials: {e}")
        return {}


USER_CREDENTIALS: Dict[str, str] = get_user_credentials()

# ── Fallback password for @vodafoneidea.com users NOT listed above ─────────────
# Used only when ALLOW_VODAFONE_DOMAIN = True and the email isn't in USER_CREDENTIALS
VODAFONE_DOMAIN_PASSWORD: str = _hash(required_secret("VODAFONE_DOMAIN_PASSWORD"))

# ── Session store: single active session per user + TTL/idle expiry ────────────
# (fixes Obs #15 — Concurrent Login: a new login now invalidates the user's
# previous token instead of allowing unlimited simultaneous sessions)
_session_store = SessionStore(ttl_seconds=8 * 3600, idle_timeout_seconds=2 * 3600)


def auth_login(email: str, password: str):
    """
    Validate credentials.  Returns (success: bool, error_msg: str).
    """
    email = (email or "").strip().lower()
    if not email or not password:
        return False, "Email and password are required."

    is_vi = email.endswith("@vodafoneidea.com")

    # ── Access-level check ────────────────────────────────────────────────────
    if ALLOW_VODAFONE_DOMAIN:
        if not is_vi:
            sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "domain_not_allowed"})
            return False, "Access is restricted to @vodafoneidea.com accounts."
    else:
        if email not in ALLOWED_EMAILS:
            sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "not_on_allow_list"})
            return False, "Your email is not on the authorised access list."

    # ── Password check ────────────────────────────────────────────────────────
    hashed   = _hash(password)
    expected = USER_CREDENTIALS.get(email)
    if expected is None:
        # Unknown user — allow with the domain fallback password (Vi domain only)
        if is_vi and ALLOW_VODAFONE_DOMAIN:
            expected = VODAFONE_DOMAIN_PASSWORD
        else:
            sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "account_not_configured"})
            return False, "Account not configured — contact your administrator."

    if hashed != expected:
        sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "bad_password"})
        return False, "Incorrect password. Please try again."

    sec_logger.log_event("auth_success", "info", user_email=email)
    return True, ""


def auth_create_session(email: str) -> str:
    return _session_store.create(email)


def auth_validate(token: str) -> Optional[str]:
    """Return the email bound to token, or None if invalid/expired."""
    return _session_store.validate(token)


def auth_destroy(token: str):
    _session_store.destroy(token)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1.6 — DB CHAT LOGGING  (TTF_chat_logs)
# ══════════════════════════════════════════════════════════════════════════════

def log_session_event(user_email: str, question: str, llm_response: str,
                      vote: str = "", sql_query: str = ""):
    """Insert a row into TTF_chat_logs.
    Columns: id (serial4, auto), timestamp, user_email, question,
             llm_response, vote, sql_query
    """
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO falconschema.ttf_chat_logs
                    (timestamp, user_email, question, llm_response, vote, sql_query)
                VALUES
                    (:timestamp, :user_email, :question, :llm_response, :vote, :sql_query)
            """), {
                "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user_email":   user_email,
                "question":     question,
                "llm_response": llm_response[:800],
                "vote":         vote,
                "sql_query":    sql_query,
            })
    except Exception as e:
        print(f"[LOG_DB] Failed to log session event: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COMPLETE SCHEMA DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_DOCS: Dict[str, str] = {

"falconschema.vw_f_hpsm_h": """\
HPSM INCIDENT TABLE — falconschema.vw_f_hpsm_h
Context: HPSM (Hewlett-Packard Service Manager) stores network incident tickets.
         One row = one incident ticket for one router.

Columns:
  ticket            TEXT      e.g. "EN_IM_02022026_120417"
  open_time         TIMESTAMP e.g. "09-02-2026 16:09:38"       ticket creation time
  priority_code     TEXT      e.g. "P1","P2","P3"              fault severity
  category          TEXT      e.g. "CEN","NSS","CORE","PMS"
  status            TEXT      e.g. "open","closed"
  brief_description TEXT
  subcategory       TEXT
  logical_name      TEXT      node hostname
  nss_id            TEXT
  downtime_start    TIMESTAMP
  downtime_end      TIMESTAMP (NULL if ongoing)
  problem           TEXT      e.g. "Link Down","High CPU"
  ipaddress         TEXT      router IP — PRIMARY JOIN KEY
  circle_name       TEXT      e.g. "Mumbai","Gujarat","Delhi"   (ILIKE for match)
  circle_code       TEXT      e.g. "MUM","GUJ","DEL"
  city              TEXT
  vendor            TEXT      e.g. "Nokia","Juniper","Cisco-Xr"
  network_layer     TEXT      e.g. "Access","ISP","NGN","Core"
  domain_name       TEXT      e.g. "IPMPLS","IPCPE","IPDCN"

Time filters:
  open_time      = ticket creation timestamp  (use for "open today", "last N hours")
  downtime_start = outage start               (use for "outage started in last N hours")
  downtime_end   = outage end                 (NULL means ongoing)
""",

"falconschema.vw_f_session_data": """\
SESSION TABLE — falconschema.vw_f_session_data
Context: ISE logs each user CLI session on a router. One row = one session.

Columns:
  vendor_name           TEXT
  device_name           TEXT      hostname
  device_ip             TEXT      router IP — PRIMARY JOIN KEY
  username              TEXT
  session_commands      TEXT      condensed commands
  session_full_commands TEXT      full verbatim output
  command_class         TEXT      "config_class" or "read_class"
  start_timestamp       TIMESTAMP
  end_timestamp         TIMESTAMP
  circle_name           TEXT
  circle_code           TEXT
  city                  TEXT
  nodename              TEXT
  nss_id                TEXT
  vendor                TEXT
  network_layer         TEXT
  domain_name           TEXT

Filters:
  command_class = 'config_class'  → configuration changes
  command_class = 'read_class'    → read-only sessions
""",

"falconschema.vw_sys_log_data_h": """\
SYSLOG TABLE — falconschema.vw_sys_log_data_h
Context: Router system logs, ML-classified normal(status='0') or error(status='1').

Columns:
  timestamp     TIMESTAMP
  keyword       TEXT      e.g. "RSVP","LDP","BGP","interface","OSPF","ISIS"
  text          TEXT      full syslog message
  ipaddress     TEXT      router IP — PRIMARY JOIN KEY
  tokens        TEXT      tokenized keywords
  status        TEXT      "0" = normal,  "1" = error
  circle_name   TEXT
  circle_code   TEXT
  nodename      TEXT
  nss_id        TEXT
  city          TEXT
  vendor        TEXT
  network_layer TEXT
  domain_name   TEXT

CRITICAL: status='1' = error log; status='0' = normal.
""",

"falconschema.f_vipam_data_h": """\
VIPAM SESSION TABLE — falconschema.f_vipam_data_h
Context: ViPAM fallback command audit log. One row = one command execution.

Columns:
  processing_start_time TIMESTAMP
  user_name             TEXT
  user_display_name     TEXT
  service_user_name     TEXT
  service_host_name     TEXT
  user_session          TEXT
  neid                  TEXT
  ip_address            TEXT      router IP — PRIMARY JOIN KEY
  command_text          TEXT
  inv_operator          TEXT
  inv_subdomain         TEXT
  inv_network           TEXT
  inv_vendor            TEXT
  inv_make              TEXT
  inv_modelno           TEXT
  inv_circle            TEXT      e.g. "APR","TNC"  (exact match)
""",

"falconschema.f_static_ip_mpls_inventory": """\
IP MPLS INVENTORY TABLE — falconschema.f_static_ip_mpls_inventory
Context: Master static inventory of all IP/MPLS network nodes. No timestamps.

Columns:
  circle_name        VARCHAR(100)
  circle_code        VARCHAR(100)
  city_name          VARCHAR(100)
  node_name          VARCHAR(100)
  vendor_name        VARCHAR(100)  e.g. "Cisco","Juniper","Huawei","Nokia"
  nss_id             VARCHAR(100)
  ipaddress          VARCHAR(100)  static IP — PRIMARY JOIN KEY
  network_layer_name VARCHAR(100)  e.g. "Core","Aggregation","Access","PE","P"
  domain_name        VARCHAR(100)  e.g. "IPMPLS","IPCPE","IPDCN"
  latlong            VARCHAR(100)

NOTE: Static table — no time-based filters.
""",

"falconschema.f_topology_view_data": """\
NETWORK INCIDENT CORRELATION TABLE — falconschema.f_topology_view_data
Context: Pre-joined fault correlation view linking HPSM tickets to syslog/session/vipam.
ALWAYS filter by hpsm_ticket.

Columns:
  a_end_ip          TEXT       IP of affected node
  nss_id            TEXT
  logical_name      TEXT
  circle_code       TEXT
  circle_name       TEXT
  vendor_name       TEXT
  network_layer     TEXT
  downtime_start    TIMESTAMP
  hpsm_ticket       TEXT       PRIMARY FILTER
  priority_code     TEXT
  brief_description TEXT
  topology_ip_out   TEXT       JSON list of downstream IPs
  role              TEXT
  clubbing_category TEXT
  sessions          TEXT       config commands around fault window
  syslogs           TEXT       syslog messages from fault window
  vipam_sessions    TEXT       VIPAM commands if ISE data missing
  crtid_list        TEXT       Change Request Ticket IDs
""",

"falconschema.f_hpsm_tt_rca": """\
HPSM RCA TABLE — falconschema.f_hpsm_tt_rca
Context: Root Cause Analysis records, one row per HPSM ticket.
ALWAYS filter by hpsm_ticket.

Columns:
  hpsm_ticket       TEXT      HPSM ticket ID — PRIMARY FILTER
  priority_code     TEXT
  logical_name      TEXT
  a_end_ip          TEXT
  circle_code       TEXT
  network_layer     TEXT
  brief_description TEXT
  impacted_nodes    TEXT
  root_cause        TEXT
  symptoms          TEXT
  impact            TEXT
""",
}

# ── Table summaries ───────────────────────────────────────────────────────────
TABLE_SUMMARIES: Dict[str, str] = {
    "falconschema.vw_f_hpsm_h":                "HPSM incident tickets, outages, downtime, priority, vendor, circle",
    "falconschema.vw_f_session_data":          "ISE CLI sessions — config/read, usernames, commands, router access",
    "falconschema.vw_sys_log_data_h":          "Router syslogs RSVP/LDP/BGP/interface, error/normal classification",
    "falconschema.f_vipam_data_h":             "ViPAM privileged access command audit (fallback if ISE missing)",
    "falconschema.f_static_ip_mpls_inventory": "Static IP/MPLS inventory — nodes, circles, vendors, domains",
    "falconschema.f_topology_view_data":       "Network Incident Correlation — HPSM ticket ↔ syslogs/sessions/vipam pre-joined",
    "falconschema.f_hpsm_tt_rca":              "Root Cause Analysis records per HPSM ticket",
}

# ── Baseline few-shot examples ────────────────────────────────────────────────
BASELINE_EXAMPLES: List[Dict] = [
    # HPSM
    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "What are the current open P1 HPSM tickets today?",
     "sql": "SELECT ticket, open_time, priority_code, brief_description, circle_name, ipaddress, vendor, network_layer FROM falconschema.vw_f_hpsm_h WHERE priority_code = 'P1' AND status = 'open' AND DATE(open_time) = CURRENT_DATE ORDER BY open_time DESC LIMIT 200;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "Open P1 and P2 faults right now",
     "sql": "SELECT ticket, open_time, priority_code, brief_description, circle_name, ipaddress, vendor FROM falconschema.vw_f_hpsm_h WHERE priority_code IN ('P1','P2') AND status = 'open' ORDER BY priority_code, open_time DESC LIMIT 200;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "Show all outages that started in the last 2 hours.",
     "sql": "SELECT ticket, circle_name, ipaddress, downtime_start, downtime_end FROM falconschema.vw_f_hpsm_h WHERE downtime_start >= NOW() - INTERVAL '2 hours' ORDER BY downtime_start DESC LIMIT 200;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "Incidents by circle today",
     "sql": "SELECT circle_name, COUNT(*) AS cnt FROM falconschema.vw_f_hpsm_h WHERE DATE(open_time) = CURRENT_DATE GROUP BY circle_name ORDER BY cnt DESC LIMIT 50;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "P1 P2 count by priority today",
     "sql": "SELECT priority_code, COUNT(*) AS cnt FROM falconschema.vw_f_hpsm_h WHERE priority_code IN ('P1','P2') AND DATE(open_time) = CURRENT_DATE GROUP BY priority_code ORDER BY priority_code LIMIT 10;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "Show all tickets by circle code",
     "sql": "SELECT circle_code, COUNT(*) AS cnt FROM falconschema.vw_f_hpsm_h WHERE DATE(open_time) = CURRENT_DATE GROUP BY circle_code ORDER BY cnt DESC LIMIT 50;"},

    {"category": "HPSM", "tables": ["falconschema.vw_f_hpsm_h"],
     "question": "List all incidents where downtime exceeded 30 minutes.",
     "sql": "SELECT ticket, circle_name, ipaddress, downtime_start, downtime_end, EXTRACT(EPOCH FROM (downtime_end - downtime_start))/60 AS downtime_minutes FROM falconschema.vw_f_hpsm_h WHERE downtime_end IS NOT NULL AND EXTRACT(EPOCH FROM (downtime_end - downtime_start))/60 > 30 ORDER BY downtime_minutes DESC LIMIT 200;"},

    # Syslog
    {"category": "Syslog", "tables": ["falconschema.vw_sys_log_data_h"],
     "question": "Error keyword wordcloud last 24 hours",
     "sql": "SELECT keyword, COUNT(*) AS cnt FROM falconschema.vw_sys_log_data_h WHERE status = '1' AND timestamp >= NOW() - INTERVAL '24 hours' GROUP BY keyword ORDER BY cnt DESC LIMIT 50;"},

    {"category": "Syslog", "tables": ["falconschema.vw_sys_log_data_h"],
     "question": "Hourly syslog error trend today",
     "sql": "SELECT DATE_TRUNC('hour', timestamp) AS hr, COUNT(*) AS cnt FROM falconschema.vw_sys_log_data_h WHERE status = '1' AND DATE(timestamp) = CURRENT_DATE GROUP BY hr ORDER BY hr LIMIT 48;"},

    {"category": "Syslog", "tables": ["falconschema.vw_sys_log_data_h"],
     "question": "Top 10 noisy routers last 24 hours",
     "sql": "SELECT ipaddress, nodename, circle_name, COUNT(*) AS cnt FROM falconschema.vw_sys_log_data_h WHERE status = '1' AND timestamp >= NOW() - INTERVAL '24 hours' GROUP BY ipaddress, nodename, circle_name ORDER BY cnt DESC LIMIT 10;"},

    {"category": "Syslog", "tables": ["falconschema.vw_sys_log_data_h"],
     "question": "Show all error syslogs generated in the last 30 minutes.",
     "sql": "SELECT timestamp, keyword, text, ipaddress, nodename, circle_name, vendor FROM falconschema.vw_sys_log_data_h WHERE status = '1' AND timestamp >= NOW() - INTERVAL '30 minutes' ORDER BY timestamp DESC LIMIT 200;"},

    # Session
    {"category": "Session", "tables": ["falconschema.vw_f_session_data"],
     "question": "Config sessions last 1 hour",
     "sql": "SELECT username, device_name, device_ip, session_commands, start_timestamp FROM falconschema.vw_f_session_data WHERE command_class = 'config_class' AND start_timestamp >= NOW() - INTERVAL '1 hour' ORDER BY start_timestamp DESC LIMIT 200;"},

    {"category": "Session", "tables": ["falconschema.vw_f_session_data"],
     "question": "Who made config changes last 30 minutes",
     "sql": "SELECT username, device_name, device_ip, start_timestamp FROM falconschema.vw_f_session_data WHERE command_class = 'config_class' AND start_timestamp >= NOW() - INTERVAL '30 minutes' ORDER BY start_timestamp DESC LIMIT 200;"},

    {"category": "Session", "tables": ["falconschema.vw_f_session_data"],
     "question": "Which users made configuration changes in last 1 hour?",
     "sql": "SELECT username, device_name, device_ip, session_commands, start_timestamp FROM falconschema.vw_f_session_data WHERE command_class = 'config_class' AND start_timestamp >= NOW() - INTERVAL '1 hour' ORDER BY start_timestamp DESC LIMIT 200;"},

    # ViPAM
    {"category": "ViPAM", "tables": ["falconschema.f_vipam_data_h"],
     "question": "Which service users executed the most VIPAM commands?",
     "sql": "SELECT service_user_name, COUNT(*) AS command_count FROM falconschema.f_vipam_data_h GROUP BY service_user_name ORDER BY command_count DESC LIMIT 20;"},

    # Inventory
    {"category": "Metadata", "tables": ["falconschema.f_static_ip_mpls_inventory"],
     "question": "Node count by vendor",
     "sql": "SELECT vendor_name, COUNT(*) AS cnt FROM falconschema.f_static_ip_mpls_inventory GROUP BY vendor_name ORDER BY cnt DESC LIMIT 50;"},

    {"category": "Metadata", "tables": ["falconschema.f_static_ip_mpls_inventory"],
     "question": "List all nodes in Delhi circle with their vendor names.",
     "sql": "SELECT node_name, vendor_name, network_layer_name, domain_name, ipaddress FROM falconschema.f_static_ip_mpls_inventory WHERE circle_code = 'DEL' OR circle_name ILIKE '%Delhi%' ORDER BY vendor_name, node_name LIMIT 200;"},

    # RCA
    {"category": "RCA", "tables": ["falconschema.f_hpsm_tt_rca"],
     "question": "RCA for ticket EN_IM_02022026_120417",
     "sql": "SELECT hpsm_ticket, root_cause, symptoms, impact, impacted_nodes FROM falconschema.f_hpsm_tt_rca WHERE hpsm_ticket = 'EN_IM_02022026_120417' LIMIT 10;"},

    # Topology
    {"category": "Correlation", "tables": ["falconschema.f_topology_view_data"],
     "question": "Show correlation data for ticket EN_IM_02022026_120417.",
     "sql": "SELECT hpsm_ticket, a_end_ip, circle_name, priority_code, topology_ip_out, sessions, syslogs, vipam_sessions, crtid_list, brief_description FROM falconschema.f_topology_view_data WHERE hpsm_ticket = 'EN_IM_02022026_120417';"},
]

# ── Quick-access chips (UI) ───────────────────────────────────────────────────
QUICK = [
    "Open P1 and P2 faults right now",
    "Incidents by circle today",
    "Top 10 noisy routers last 24 hours",
    "Error keyword wordcloud last 24 hours",
    "Hourly syslog error trend today",
    "Config sessions last 1 hour",
    "Node count by vendor",
    "P1 P2 count by priority today",
    "Who made config changes last 30 minutes",
    "Show all tickets by circle code",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — RAG INDEX
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGEntry:
    id:        str
    text:      str
    category:  str
    tables:    List[str]
    sql:       str = ""
    question:  str = ""
    doc_type:  str = "example"   # "example" | "rag_doc"
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class FalconRAG:
    """
    Unified RAG index combining:
      1. BASELINE_EXAMPLES   — hardcoded few-shot Q/SQL pairs
      2. 100_Questions.csv   — curated CSV of Q+SQL+category+tables (optional)
      3. TXT documents       — schema/column docs chunked per table  (optional)
    Falls back gracefully if embedder or files are absent.
    """

    def __init__(self):
        self.entries: List[RAGEntry] = []
        self.embedder: Optional[SentenceTransformer] = None
        self._table_emb_cache: Dict[str, Tuple[np.ndarray, str]] = {}
        self._load_embedder()
        self._load_table_embeddings()
        self._ingest_baseline()
        self._load_csv()
        self._load_docs()
        # De-duplicate
        seen: set = set()
        unique: List[RAGEntry] = []
        for e in self.entries:
            if e.id not in seen:
                seen.add(e.id)
                unique.append(e)
        self.entries = unique
        print(f"[RAG] Total indexed: {len(self.entries)} entries "
              f"({sum(1 for e in self.entries if e.doc_type=='example')} examples, "
              f"{sum(1 for e in self.entries if e.doc_type=='rag_doc')} doc chunks)")

    # ── Embedder ──────────────────────────────────────────────────────────────

    def _load_embedder(self):
        try:
            self.embedder = SentenceTransformer(EMBEDDER_PATH)
            print(f"✅ Embedder loaded: {EMBEDDER_PATH}")
        except Exception as e:
            print(f"[WARN] Embedder not loaded ({e}) — using keyword fallback")

    # ── Pre-built table embedding cache (.npy + .txt) ─────────────────────────

    def _load_table_embeddings(self):
        loaded = 0
        for table, npy_name in TABLE_TO_NPY.items():
            npy_path = os.path.join(EMBED_CACHE_DIR, npy_name)
            txt_path = RAG_DOCS.get(table, "")
            if not os.path.exists(npy_path):
                continue
            try:
                emb = np.load(npy_path).astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            except Exception as ex:
                print(f"[TABLE_EMB] Failed to load {npy_path}: {ex}")
                continue
            full_text = ""
            if txt_path and os.path.exists(txt_path):
                try:
                    full_text = Path(txt_path).read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            if not full_text:
                full_text = SCHEMA_DOCS.get(table, "")
            self._table_emb_cache[table] = (emb, full_text)
            loaded += 1
            print(f"[TABLE_EMB] ✓ {npy_name}")
        print(f"[TABLE_EMB] Loaded {loaded}/{len(TABLE_TO_NPY)} table embeddings")

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        try:
            return self.embedder.encode(text, normalize_embeddings=True)
        except Exception:
            return None

    def _make_entry(self, question: str, sql: str, category: str,
                    tables: List[str], doc_type: str = "example") -> RAGEntry:
        text = f"Question: {question}\nSQL: {sql}" if sql else question
        return RAGEntry(
            id=hashlib.md5(text.encode()).hexdigest()[:12],
            text=text, category=category, tables=tables,
            sql=sql, question=question, doc_type=doc_type,
            embedding=self._embed(text),
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _ingest_baseline(self):
        for ex in BASELINE_EXAMPLES:
            e = self._make_entry(
                question=ex["question"], sql=ex["sql"],
                category=ex.get("category", "General"),
                tables=ex.get("tables", []), doc_type="example",
            )
            self.entries.append(e)
        print(f"[RAG] Baseline: {len(BASELINE_EXAMPLES)} examples ingested")

    def _load_csv(self):
        if not os.path.exists(QUESTIONS_CSV):
            print(f"[RAG] CSV not found at {QUESTIONS_CSV} — skipping")
            return
        loaded = 0
        try:
            with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q   = (row.get("question") or "").strip()
                    sql = (row.get("sql")      or "").strip()
                    cat = (row.get("category") or "General").strip()
                    tbl = (row.get("tables")   or "").strip()
                    if not q or not sql:
                        continue
                    tables = [t.strip() for t in tbl.split(",") if t.strip() in VALID_TABLES]
                    self.entries.append(self._make_entry(q, sql, cat, tables, "example"))
                    loaded += 1
        except Exception as e:
            print(f"[RAG] CSV error: {e}")
        print(f"[RAG] CSV: {loaded} examples loaded")

    def _load_docs(self):
        for table, fp in RAG_DOCS.items():
            if not os.path.exists(fp):
                continue
            try:
                content = Path(fp).read_text(encoding="utf-8").strip()
                if not content:
                    continue
                chunks = [c.strip() for c in re.split(r"\n{2,}", content) if c.strip()]
                for chunk in chunks:
                    eid = hashlib.md5(chunk.encode()).hexdigest()[:12]
                    self.entries.append(RAGEntry(
                        id=eid, text=chunk, category="Schema",
                        tables=[table], doc_type="rag_doc",
                        embedding=self._embed(chunk),
                    ))
                print(f"[RAG] {os.path.basename(fp)}: {len(chunks)} chunks loaded")
            except Exception as ex:
                print(f"[RAG] Doc error {fp}: {ex}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, tables_hint: Optional[List[str]] = None,
                 doc_types: Optional[List[str]] = None, top_k: int = 6) -> List[RAGEntry]:
        pool = [e for e in self.entries if not doc_types or e.doc_type in doc_types]
        if tables_hint:
            primary  = [e for e in pool if any(t in e.tables for t in tables_hint)]
            fallback = [e for e in pool if not any(t in e.tables for t in tables_hint)]
            pool = primary + fallback
        if self.embedder:
            q_emb = self._embed(query)
            if q_emb is not None:
                scored = []
                for entry in pool:
                    sim = (float(np.dot(q_emb, entry.embedding))
                           if entry.embedding is not None
                           else self._keyword_score(query, entry.text))
                    scored.append((sim, entry))
                scored.sort(key=lambda x: -x[0])
                return [e for _, e in scored[:top_k]]
        scored = [(self._keyword_score(query, e.text), e) for e in pool]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        qw = set(re.findall(r"\w+", query.lower()))
        tw = set(re.findall(r"\w+", text.lower()))
        return len(qw & tw) / (len(qw) + 1)

    def few_shot_block(self, query: str, tables_hint: Optional[List[str]] = None,
                       top_k: int = 5) -> str:
        examples = self.retrieve(query, tables_hint=tables_hint, doc_types=["example"], top_k=top_k)
        if not examples:
            return "(No examples available)"
        return "\n".join(f"Q: {ex.question}\nSQL:\n{ex.sql}\n---" for ex in examples)

    def schema_context(self, tables: List[str]) -> str:
        """
        Return COMPLETE schema/doc text for every requested table.
        Priority: 1) .npy cache   2) .txt on disk   3) SCHEMA_DOCS hardcoded
        """
        parts: List[str] = []
        for t in tables:
            if t in self._table_emb_cache:
                _, full_text = self._table_emb_cache[t]
                if full_text:
                    parts.append(full_text)
                    continue
            txt_path = RAG_DOCS.get(t, "")
            if txt_path and os.path.exists(txt_path):
                try:
                    text = Path(txt_path).read_text(encoding="utf-8").strip()
                    if text:
                        parts.append(text)
                        continue
                except Exception:
                    pass
            if t in SCHEMA_DOCS:
                parts.append(SCHEMA_DOCS[t])
        return "\n\n---\n\n".join(parts) if parts else "(No schema available)"

    def find_table_by_embedding(self, query: str) -> Optional[str]:
        """Semantic table selection using pre-built .npy embeddings."""
        if not self._table_emb_cache or self.embedder is None:
            return None
        try:
            q_emb = self.embedder.encode(query, normalize_embeddings=True).astype(np.float32)
        except Exception:
            return None
        best_table: Optional[str] = None
        best_score: float = -1.0
        for table, (t_emb, _) in self._table_emb_cache.items():
            score = float(np.dot(q_emb, t_emb))
            if score > best_score:
                best_score = score
                best_table = table
        print(f"[TABLE_EMB] Best match → {best_table}  (score={best_score:.4f})")
        return best_table


# Initialise RAG at module load
rag = FalconRAG()


# ── Embedding builder (optional — run once to create .npy files) ──────────────
def build_rag_embeddings():
    try:
        _emb = SentenceTransformer("/data/harish/eli-embedding-small-1/")
    except Exception as e:
        print(f"[EmbBuilder] Embedder unavailable: {e}")
        return
    print("\n[Embedding Builder] Checking cache...")
    for file in os.listdir(RAG_BASE_DIR):
        if not file.endswith(".txt"):
            continue
        txt_path = os.path.join(RAG_BASE_DIR, file)
        npy_name = file.replace(".txt", ".npy")
        npy_path = os.path.join(EMBED_CACHE_DIR, npy_name)
        if os.path.exists(npy_path):
            print(f"✓ Cached → {npy_name}")
            continue
        print(f"Building → {file}")
        with open(txt_path, "r") as f:
            text = f.read()
        embedding = _emb.encode(text)
        np.save(npy_path, embedding)
        print(f"Saved → {npy_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def debug_print(stage: str, data: Any = None) -> None:
    if not DEBUG:
        return
    sep = "─" * 64
    print(f"\n{sep}\n[DEBUG] {stage}\n{sep}")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            print(f"DataFrame: {len(data)} rows × {len(data.columns)} cols")
            print(data.head(3).to_string())
        elif isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, default=str)[:3000])
        else:
            print(str(data)[:3000])
    print(sep)


def log_trace(record: Dict[str, Any]) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as ex:
        print(f"[LOG ERROR] {ex}")


def safe_json_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        text = re.sub(r"<[^>]+?>", "", text)
        text = re.sub(r"```(?:json)?", "", text).replace("```", "")
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        raw = m.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            val = ast.literal_eval(raw)
            return val if isinstance(val, dict) else {}
    except Exception as ex:
        debug_print(f"JSON_PARSE_ERROR: {ex}", str(text)[:300])
        return {}


def extract_sql(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.replace("```sql", "").replace("```", "")
    match = re.search(r"((?:WITH|SELECT)[\s\S]*?;)", raw, re.IGNORECASE)
    return match.group(1).strip() if match else raw.strip()


def detect_model_refusal(raw: str) -> Optional[str]:
    if not raw:
        return "Model returned empty output"
    if re.search(r"\b(?:SELECT|WITH)\b", raw, re.IGNORECASE):
        return None
    refusal_patterns = [
        r"provide\s+an?\s+assistive", r"help\s+you\s+write",
        r"i\s+cannot\s+generate", r"i\s+can(?:'t|not)\s+",
        r"please\s+provide\s+more", r"need\s+more\s+(?:context|information|details)",
        r"could\s+you\s+clarify", r"as\s+an?\s+(?:AI|language\s+model)",
    ]
    for pat in refusal_patterns:
        if re.search(pat, raw, re.IGNORECASE):
            return f"Model refused: \"{raw[:120].strip()}\""
    return f"Model returned text instead of SQL: \"{raw[:120].strip()}\""


# SECURITY FIX: the previous validator was a substring keyword blocklist
# that never restricted which SQL *functions* could be called (letting
# current_setting()/version()/inet_server_addr()/session_user/pg_*() leak
# backend, config, TLS and version data — Obs #2/#5/#6/#24), never stripped
# comments (so "DR/**/OP" defeated the "drop" check), and never rejected
# stacked ";"-separated statements. validate_sql is now a hardened
# allow-list validator built in security_common.py — single SELECT/WITH
# statement, function allow-list, and table allow-list only.
validate_sql = build_sql_validator(VALID_TABLES)


def safe_serialize_dataframe(df) -> List[dict]:
    if df is None or df.empty:
        return []
    safe = df.copy().where(pd.notna(df), None)
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return safe.to_dict("records")


def extract_columns(df) -> List[str]:
    return [] if df is None else [str(c) for c in df.columns]


def summarize_dataframe(df) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"row_count": 0, "columns": []}
    summary: Dict[str, Any] = {
        "row_count":   len(df),
        "columns":     list(df.columns),
        "sample_rows": safe_serialize_dataframe(df.head(20)),
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and len(df) > 0:
            try:
                summary[f"{col}_stats"] = {
                    "min":  float(df[col].min()),
                    "max":  float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2),
                }
            except Exception:
                pass
    return summary


# ── Time-parameter extractor ──────────────────────────────────────────────────
_TIME_PATTERN = re.compile(
    r"\b(?:last|past|previous|in\s+the\s+last|in\s+past)\s+"
    r"(\d+)\s*"
    r"(hour|hr|hours|hrs|minute|min|minutes|mins|day|days|week|weeks|month|months)s?\b",
    re.IGNORECASE,
)
_UNIT_CANONICAL = {
    "hour":"hour", "hr":"hour", "hours":"hours", "hrs":"hours",
    "minute":"minute", "min":"minute", "minutes":"minutes", "mins":"minutes",
    "day":"day", "days":"days", "week":"week", "weeks":"weeks",
    "month":"month", "months":"months",
}

def extract_time_parameter(user_query: str) -> Optional[str]:
    match = _TIME_PATTERN.search(user_query)
    if not match:
        if re.search(r"\btoday\b", user_query, re.IGNORECASE):
            return "today"
        return None
    number   = match.group(1)
    unit_raw = match.group(2).lower().rstrip("s")
    unit     = _UNIT_CANONICAL.get(unit_raw, unit_raw)
    if int(number) > 1 and not unit.endswith("s"):
        unit += "s"
    return f"{number} {unit}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class GPUApiClient:
    def __init__(self, api_key: str = GPU_KEY, proxy_url: str = GPU_URL):
        self._url     = proxy_url
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"} if api_key else {"Content-Type": "application/json"}
        if not api_key:
            print("[GPU CLIENT WARNING] GPU_API_KEY not set — running unauthenticated.")

    def infer(self, prompt: str, model: str = MODEL,
              max_new_tokens: int = 512, temperature: float = 0.0,
              top_p: float = 0.9, top_k: int = 3) -> str:
        payload = {
            "model": model, "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p), "top_k": int(top_k),
        }
        try:
            r = requests.post(self._url, json=payload, headers=self._headers, timeout=GPU_TO)
            r.raise_for_status()
            data = r.json()
            rid = data.get("request_id", "")
            if rid:
                print(f"[GPU] request_id={rid} tokens={data.get('new_tokens')} "
                      f"elapsed={data.get('elapsed_s')}s")
            return data.get("text", "")
        except requests.Timeout:
            print(f"[GPU TIMEOUT] Exceeded {GPU_TO}s")
            return ""
        except requests.RequestException as ex:
            print(f"[GPU ERROR] {ex}")
            return ""


gpu_client = GPUApiClient()


def call_llm(system_prompt: str, user_prompt: str,
             model_name: Optional[str] = None,
             max_new_tokens: int = 512,
             decode_config: Optional[dict] = None,
             stage_label: str = "") -> str:
    decode_config = decode_config or ROUTER_DECODE
    model = model_name or MODEL
    wrapped = (
        f"SYSTEM:\n{system_prompt.strip()}\n\n"
        f"USER:\n{user_prompt.strip()}\n\n"
        f"ASSISTANT:\n"
    )
    if DEBUG and stage_label:
        debug_print(f"LLM ▶ {stage_label}", wrapped[:1500])
    result = gpu_client.infer(
        prompt=wrapped, model=model,
        max_new_tokens=max_new_tokens,
        temperature=decode_config.get("temperature", 0.0),
        top_p=decode_config.get("top_p", 0.9),
        top_k=decode_config.get("top_k", 3),
    )
    if DEBUG and stage_label:
        debug_print(f"LLM ◀ {stage_label}", result[:1500])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — DSPy WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class FalconDSPyLM(dspy.LM):
    """Thin DSPy adapter over the GPU proxy — used by CoT signatures."""
    def __init__(self):
        super().__init__(model="gpu_proxy")

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            prompt = "\n".join(m.get("content", "") for m in messages)
        result = gpu_client.infer(
            prompt=prompt or "",
            max_new_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1),
        )
        return [{"text": result, "finish_reason": "stop"}]

    def basic_request(self, prompt, **kwargs):
        return self(prompt=prompt, **kwargs)


try:
    dspy.settings.configure(lm=FalconDSPyLM())
    print("✅ DSPy configured")
except Exception as ex:
    print(f"[WARN] DSPy config: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — GRAPH-OF-THOUGHT (GoT) SQL BEAM REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoTNode:
    node_id:   str
    sql:       str
    params:    Dict[str, Any]
    score:     float = 0.0
    valid:     bool  = False
    error:     Optional[str] = None
    reasoning: str   = ""
    parent_id: Optional[str] = None
    depth:     int   = 0


class GraphOfThoughtRefiner:
    """
    Beam-search SQL refinement:
      1. Validate each candidate SQL against schema rules
      2. Score using LLM (0-10 rubric)
      3. Fix failing nodes via LLM fixer
      4. Return the highest-scoring valid node
    """

    SCORER_SYS = """\
You are a SQL quality evaluator for a telecom NOC PostgreSQL database.
Score the SQL on a 0-10 scale:
  0-4  Correctness — does it answer the user question?
  0-3  Schema      — correct table.column names, single-table only?
  0-2  Safety      — SELECT/WITH only, no injections?
  0-1  Efficiency  — no unnecessary complexity, correct LIMIT?
Return STRICT JSON only: {"score": <int 0-10>, "reasons": ["..."]}"""

    FIX_SYS = """\
Fix the broken SQL query. Return STRICT JSON only:
{"sql": "<corrected complete SELECT or WITH query>", "params": {}, "fix": "<what was changed>"}"""

    def _nid(self, sql: str) -> str:
        return hashlib.md5(sql.encode()).hexdigest()[:8]

    def _score(self, node: GoTNode, query: str, schema: str) -> float:
        raw = call_llm(
            self.SCORER_SYS,
            f"Question: {query}\nSchema (excerpt):\n{schema[:400]}\nSQL:\n{node.sql}",
            max_new_tokens=80, decode_config=PARAM_DECODE, stage_label="GOT_SCORE",
        )
        return float(safe_json_parse(raw).get("score", 0))

    def _validate(self, node: GoTNode) -> GoTNode:
        try:
            validate_sql(node.sql)
            node.valid = True
        except Exception as ex:
            node.valid = False
            node.error = str(ex)
        return node

    def _fix(self, node: GoTNode, schema: str, query: str) -> GoTNode:
        user_p = (
            f"Error: {node.error}\n\nFailed SQL:\n{node.sql}\n\n"
            f"Schema:\n{schema[:1000]}\n\nValid tables: {sorted(VALID_TABLES)}\n\n"
            f"Question: {query}"
        )
        raw    = call_llm(self.FIX_SYS, user_p, max_new_tokens=600,
                          decode_config=SQL_DECODE, stage_label="GOT_FIX")
        parsed = safe_json_parse(raw)
        new_sql = parsed.get("sql", node.sql)
        return GoTNode(
            node_id=self._nid(new_sql), sql=new_sql,
            params=parsed.get("params", node.params),
            reasoning=parsed.get("fix", ""),
            parent_id=node.node_id, depth=node.depth + 1,
        )

    def run(self, candidates: List[Dict], query: str, schema: str) -> Optional[GoTNode]:
        beam: List[GoTNode] = [
            GoTNode(node_id=self._nid(c["sql"]), sql=c["sql"],
                    params=c.get("params", {}), reasoning=c.get("explanation", ""))
            for c in candidates[:GOT_BEAM_WIDTH] if c.get("sql")
        ]
        if not beam:
            return None

        best: Optional[GoTNode] = None
        all_nodes: List[GoTNode] = list(beam)

        for hop in range(COT_MAX_HOPS):
            next_beam: List[GoTNode] = []
            for node in beam:
                node = self._validate(node)
                if node.valid:
                    node.score = self._score(node, query, schema)
                    if best is None or node.score > best.score:
                        best = node
                    if best.score >= GOT_EARLY_EXIT:
                        debug_print(f"GOT early exit hop={hop}", {"score": best.score})
                        return best
                    next_beam.append(node)
                elif node.depth < COT_MAX_HOPS - 1:
                    child = self._fix(node, schema, query)
                    all_nodes.append(child)
                    next_beam.append(child)
            beam = next_beam
            if not beam:
                break

        if best is None:
            for n in all_nodes:
                if n.valid:
                    return n
            return all_nodes[0] if all_nodes else None

        debug_print("GOT_BEST", {"sql": best.sql[:200], "score": best.score})
        return best


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — AGENT STATE (LangGraph shared TypedDict)
# ══════════════════════════════════════════════════════════════════════════════

class FalconAgentState(TypedDict, total=False):
    user_query:       str
    model_name:       str
    route:            str    # single_table_q | metadata_q | rca_q | correlation_q
    route_confidence: str
    route_reasoning:  str
    tables:           List[str]
    schema_context:   str
    few_shot_block:   str
    cot_plan:         str
    sql:              str
    sql_params:       Dict[str, Any]
    sql_error:        str
    sql_retry_count:  int
    df:               Optional[Any]
    exec_error:       Optional[str]
    exec_valid:       Optional[bool]
    exec_attempts:    int
    answer:           str
    failed:           bool
    failure_reason:   str
    fallback_used:    bool
    security_blocked: bool
    in_scope:         bool
    out_of_scope:     bool


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_tbl_block = "\n".join(
    f"  {t.split('.')[-1]:45s} → {s}"
    for t, s in TABLE_SUMMARIES.items()
)

# ══════════════════════════════════════════════════════════════════════════════
#  SCOPE GATE — runs BEFORE the Router. Added in response to live testing
#  that showed off-topic/admin-flavored questions (system introspection,
#  "create a test database", storytelling, bias-probe prompts, generic
#  code requests) were reaching SQL generation, which then invented
#  plausible-but-fake SQL against real or imaginary tables and the
#  explanation stage hallucinated a confident-sounding answer from the
#  empty/irrelevant result. This node uses the same schema/RAG context the
#  rest of the pipeline uses to decide, before any SQL is attempted,
#  whether the question is actually answerable from FALCON's data.
# ══════════════════════════════════════════════════════════════════════════════

SCOPE_GATE_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a strict scope gate for FALCON, a Vi (Vodafone Idea) NOC telecom
network-operations reporting assistant. FALCON answers questions that can be
answered by querying the tables below — incident tickets, syslogs, CLI
sessions, ViPAM access logs, IP/MPLS inventory, root-cause analysis, and
topology correlation. Nothing else.

AVAILABLE TABLES AND WHAT THEY COVER:
{_tbl_block}

Classify the user's message as IN_SCOPE or OUT_OF_SCOPE.

IN_SCOPE means: a genuine reporting/analytics question answerable by a
read-only SELECT against the tables above (counts, trends, tickets,
outages, sessions, inventory lookups, RCA, correlation, etc.), including
follow-up questions, greetings, or requests to clarify/rephrase a prior
answer.

OUT_OF_SCOPE means ANY of the following, even if phrased as a data
question or dressed up in domain language:
 - Asking about this system's own configuration, database internals,
   connection details, SSL/TLS status, version, server IP, credentials,
   or logs (this is infrastructure self-inspection, not NOC reporting)
 - Requests to create, modify, delete, or otherwise change data or schema
   (this app is read-only reporting; even if phrased as a normal question
   like "create a test database" or "delete X")
 - Requests to change a password or account credentials
 - General programming/code-writing requests unrelated to querying these
   tables (e.g. "write a hello world app", scripts, automation tools)
 - Storytelling, roleplay, hypotheticals, or "continue this story" framing
 - Requests to describe or profile a person by demographic characteristics
 - General knowledge, historical, or security-research questions (CVEs,
   historical events, how-to questions) with no connection to this data
 - Anything else clearly unrelated to the tables listed above

When genuinely unsure between a plausible domain question and an
off-topic one, prefer IN_SCOPE — this gate is for clear-cut cases, not a
second content filter (that already runs separately).

Return STRICT JSON only:
{{"scope": "IN_SCOPE" or "OUT_OF_SCOPE", "reasoning": "<one short sentence>"}}
""".strip()


ROUTER_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are FALCON's query routing agent for a telecom network operations platform.

AVAILABLE TABLES:
{_tbl_block}

ROUTES:
  single_table_q → answered by exactly ONE table (most common)
  metadata_q     → inventory counts, node listings, circle/domain/vendor queries
  rca_q          → root cause analysis lookup by HPSM ticket number
  correlation_q  → topology correlation view for a specific HPSM ticket

PERSONA-TO-TABLE MAPPING (strict — never join across tables):
  metadata_q     → falconschema.f_static_ip_mpls_inventory
  rca_q          → falconschema.f_hpsm_tt_rca
  correlation_q  → falconschema.f_topology_view_data
  single_table_q → one of: vw_f_hpsm_h, vw_f_session_data, vw_sys_log_data_h, f_vipam_data_h

ROUTING RULES (apply in order):
  1. Contains ticket number + "RCA" or "root cause"                           → rca_q
  2. Contains ticket number + "correlation"/"syslog"/"session"/"topo"         → correlation_q
  3. "how many"/"count"/"list all"/"inventory"/"how many in"                  → metadata_q
  4. Single data domain only                                                   → single_table_q

Return STRICT JSON only (no other text):
{{"route": "<route>", "confidence": "High|Medium|Low", "reasoning": "<one sentence>"}}
""".strip()


TABLE_ID_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a table identification expert for the FALCON telecom database.
Identify the SINGLE table needed to answer the query.

TABLE CAPABILITIES:
{_tbl_block}

PERSONA-TO-TABLE MAPPING (one table per persona, no JOINs):
  metadata_q     → falconschema.f_static_ip_mpls_inventory
  rca_q          → falconschema.f_hpsm_tt_rca
  correlation_q  → falconschema.f_topology_view_data
  single_table_q → pick ONE of: vw_f_hpsm_h, vw_f_session_data,
                   vw_sys_log_data_h, f_vipam_data_h

RULES:
 - Always return exactly ONE table — no JOINs are needed
 - Match the route to its dedicated table using the mapping above

Return STRICT JSON only:
{{"tables": ["falconschema.table1"],
  "primary_table": "<the one table>",
  "reasoning": "<why this table>"}}
""".strip()


SQL_RULES = f"""\
{SAFETY_PREAMBLE}
POSTGRESQL SQL GENERATION RULES (MUST FOLLOW ALL):
 1. SELECT only — never INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE
 2. FULLY QUALIFIED table names: falconschema.<tablename>
 3. Use ONLY columns from the schema context provided
 4. LIMIT 500 for detail queries; no LIMIT for pure aggregates (GROUP BY only)
 5. Time filters: NOW() - INTERVAL '...' or CURRENT_DATE
 6. String matching: ILIKE for case-insensitive (circles, vendors, keywords)
 7. status='1' = error syslog; status='0' = normal syslog
 8. Query ONE table only — no JOINs across tables
 9. End every query with a semicolon
 10. Only read-only analytic SQL functions are permitted — never call
     administrative/introspection functions (current_setting, version,
     inet_server_addr, session_user, pg_*, etc.)"""


EXPLANATION_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a Senior Network Intelligence Analyst at a major telecom NOC (Vi — Vodafone Idea).
Explain SQL query results clearly and concisely for NOC engineers.

Guidelines:
  • 3-5 sentences maximum
  • Highlight notable counts, anomalies, P1/P2 tickets, repeated nodes/IPs
  • Use domain terms: circle, node, ticket, RSVP/LDP/BGP, vendor
  • Mention the time window if this was a time-based query
  • If result is empty: say so clearly and suggest a possible reason
  • Do NOT repeat the SQL query text
  • Ground every statement strictly in the query result data provided —
    never invent counts, IDs, CVEs, credentials, or other facts not present
    in the result set, and never state that a data-changing action occurred
  • Do not infer risk, suspicion, or intent from a person's race, ethnicity,
    or nationality if such data appears in results
  • End with one actionable insight if the data warrants it

Plain text only — no JSON, no markdown headers, no bullet points."""

# Distinctive fragments used by sanitize_output() to detect system-prompt
# leakage in a model response (e.g. via translation/roleplay jailbreaks).
_SYSTEM_PROMPT_FRAGMENTS = [
    "FALCON's query routing agent for a telecom network operations platform",
    "PERSONA-TO-TABLE MAPPING",
    "POSTGRESQL SQL GENERATION RULES",
    "Senior Network Intelligence Analyst at a major telecom NOC",
    "SAFETY & SCOPE RULES (non-negotiable",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — KEYWORD MAPS & ROUTE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

VALID_ROUTES = {"single_table_q", "metadata_q", "rca_q", "correlation_q"}
ROUTE_TABLE_DEFAULTS = {
    "single_table_q": ["falconschema.vw_f_hpsm_h"],
    "metadata_q":     [METADATA_TABLE],
    "rca_q":          [RCA_TABLE],
    "correlation_q":  [CORRELATION_TABLE],
}

KW_TABLE_MAP = {
    "hpsm":      "falconschema.vw_f_hpsm_h",
    "ticket":    "falconschema.vw_f_hpsm_h",
    "outage":    "falconschema.vw_f_hpsm_h",
    "incident":  "falconschema.vw_f_hpsm_h",
    "downtime":  "falconschema.vw_f_hpsm_h",
    "priority":  "falconschema.vw_f_hpsm_h",
    "p1":        "falconschema.vw_f_hpsm_h",
    "p2":        "falconschema.vw_f_hpsm_h",
    "syslog":    "falconschema.vw_sys_log_data_h",
    "rsvp":      "falconschema.vw_sys_log_data_h",
    "ldp":       "falconschema.vw_sys_log_data_h",
    "bgp":       "falconschema.vw_sys_log_data_h",
    "ospf":      "falconschema.vw_sys_log_data_h",
    "isis":      "falconschema.vw_sys_log_data_h",
    "session":   "falconschema.vw_f_session_data",
    "config":    "falconschema.vw_f_session_data",
    "command":   "falconschema.vw_f_session_data",
    "username":  "falconschema.vw_f_session_data",
    "vipam":     "falconschema.f_vipam_data_h",
    "inventory": "falconschema.f_static_ip_mpls_inventory",
    "router":    "falconschema.f_static_ip_mpls_inventory",
    "node":      "falconschema.f_static_ip_mpls_inventory",
    "rca":       "falconschema.f_hpsm_tt_rca",
    "root cause":"falconschema.f_hpsm_tt_rca",
    "topology":  "falconschema.f_topology_view_data",
    "topo":      "falconschema.f_topology_view_data",
    "correlation":"falconschema.f_topology_view_data",
}

# Maps LangGraph table name → HAR16 UI route label
_TABLE_TO_ROUTE_LABEL = {
    "falconschema.vw_f_hpsm_h":                "fault",
    "falconschema.vw_sys_log_data_h":          "syslog",
    "falconschema.vw_f_session_data":          "session",
    "falconschema.f_vipam_data_h":             "vipam",
    "falconschema.f_static_ip_mpls_inventory": "inventory",
    "falconschema.f_topology_view_data":       "topology",
    "falconschema.f_hpsm_tt_rca":              "rca",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — CORE PIPELINE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_cot_plan(query: str, schema: str, tables: List[str], few_shot: str) -> str:
    """DSPy-informed Chain-of-Thought SQL planning. Returns a JSON plan string."""
    plan_sys = """\
You are a SQL planning expert for a telecom NOC PostgreSQL database.
Use Chain-of-Thought reasoning to plan the SQL query step-by-step.

Return STRICT JSON only:
{
  "plan_steps":       ["1. ...", "2. ...", "3. ..."],
  "tables_needed":    ["fully.qualified.table"],
  "select_columns":   ["col1", "col2"],
  "where_conditions": ["condition1"],
  "group_by":         ["col1"],
  "order_by":         ["col1 DESC"],
  "limit":            200,
  "time_filter":      "description or empty"
}"""
    user_p = (
        f"Schema (need-to-know only):\n{schema[:2000]}\n\n"
        f"Tables: {tables}\n\n"
        f"Few-shot examples:\n{few_shot}\n\n"
        f"Query: {query}"
    )
    raw  = call_llm(plan_sys, user_p, max_new_tokens=600,
                    decode_config=COT_DECODE, stage_label="COT_PLAN")
    plan = safe_json_parse(raw)
    debug_print("COT_PLAN", plan)
    return json.dumps(plan, indent=2)


def run_got_gen(query: str, schema: str, cot_plan: str,
                few_shot: str, tables: List[str]) -> Optional[GoTNode]:
    """Graph-of-Thought beam-search SQL generation."""
    gen_sys = f"""{SQL_RULES}

Return STRICT JSON only:
{{"sql": "<complete SELECT or WITH ... SELECT query>", "params": {{}}, "explanation": "<one line>"}}"""

    user_p = (
        f"## SCHEMA (need-to-know):\n{schema}\n\n"
        f"## FEW-SHOT EXAMPLES:\n{few_shot}\n\n"
        f"## SQL PLAN (Chain-of-Thought):\n{cot_plan}\n\n"
        f"## USER QUESTION:\n{query}\n\n"
        f"Generate a complete, correct PostgreSQL query against ONE table only:"
    )

    candidates: List[Dict] = []
    for beam_i in range(GOT_BEAM_WIDTH):
        temp = round(0.1 + beam_i * 0.1, 2)
        raw  = call_llm(gen_sys, user_p, max_new_tokens=700,
                        decode_config={**SQL_DECODE, "temperature": temp},
                        stage_label=f"GOT_GEN_{beam_i}")
        parsed = safe_json_parse(raw)
        if parsed.get("sql"):
            candidates.append(parsed)

    # Emergency fallback if beam produced nothing
    if not candidates:
        raw    = call_llm(gen_sys, user_p, max_new_tokens=700,
                          decode_config=SQL_DECODE, stage_label="GOT_EMERGENCY")
        parsed = safe_json_parse(raw)
        if parsed.get("sql"):
            candidates.append(parsed)

    if not candidates:
        return None

    return GraphOfThoughtRefiner().run(candidates, query, schema)


def run_exec_retry(sql: str, params: dict, schema: str, query: str) -> Dict[str, Any]:
    """Execute SQL with up to EXEC_MAX_ITERS LLM-assisted fix attempts on error."""
    if engine is None:
        return {"df": None, "sql": sql, "exec_error": "DB not connected",
                "exec_valid": False, "exec_attempts": 0}
    exec_error = None
    for attempt in range(1, EXEC_MAX_ITERS + 1):
        try:
            validate_sql(sql)
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
            debug_print("SQL_RESULT", {"rows": len(df), "cols": list(df.columns)})
            log_trace({"node": "exec", "attempt": attempt, "rows": len(df), "sql": sql[:200]})
            return {"df": df, "sql": sql, "exec_error": None,
                    "exec_valid": True, "exec_attempts": attempt}
        except Exception as ex:
            exec_error = str(ex)
            print(f"[EXEC] Attempt {attempt}/{EXEC_MAX_ITERS} failed: {exec_error[:150]}")
            if attempt == EXEC_MAX_ITERS:
                break
            fix_sys = ('Fix this SQL execution error. '
                       'Return STRICT JSON: {"sql": "<fixed>", "params": {}, "fix": "<what changed>"}')
            fix_p   = (f"Error: {exec_error}\n\nSQL:\n{sql}\n\n"
                       f"Schema:\n{schema[:800]}\n\nValid tables: {sorted(VALID_TABLES)}\n\n"
                       f"Question: {query}")
            fixed   = safe_json_parse(call_llm(fix_sys, fix_p, max_new_tokens=600,
                                               decode_config=SQL_DECODE,
                                               stage_label=f"SQL_FIX_{attempt}"))
            if fixed.get("sql"):
                sql    = fixed["sql"]
                params = fixed.get("params", params)

    return {"df": None, "sql": sql, "exec_error": exec_error,
            "exec_valid": False, "exec_attempts": EXEC_MAX_ITERS}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — LLM DEEP-ANALYSIS FALLBACK
#  Triggered when LangGraph result is empty or failed.
#  3-step chain: Schema Analyst → SQL Engineer → NOC Analyst
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_fallback(query: str, prior: FalconAgentState) -> FalconAgentState:
    """Multi-step LLM fallback when the graph pipeline fails."""
    print("\n🔁 LLM deep-analysis fallback triggered")

    tables   = prior.get("tables") or list(VALID_TABLES)
    schema   = rag.schema_context(tables)
    few_shot = rag.few_shot_block(query, tables_hint=tables, top_k=5)

    # Step 1: Schema Analyst
    analyst_sys = f"""\
{SAFETY_PREAMBLE}
You are a FALCON telecom database Schema Analyst.
Analyse the query and identify the exact table and columns needed.
Each persona queries ONE table only — no JOINs.

AVAILABLE TABLES AND SCHEMAS:
{schema}

Return STRICT JSON only:
{{
  "tables":      ["falconschema.table1"],
  "key_columns": ["col1", "col2"],
  "filters":     ["condition1"],
  "analysis":    "<one sentence explaining the query intent>"
}}"""

    analyst_raw    = call_llm(
        analyst_sys,
        f"Query: {query}\nPrior SQL attempted: {prior.get('sql', 'None')}",
        max_new_tokens=400, decode_config=COT_DECODE, stage_label="FALLBACK_ANALYST",
    )
    analyst_result = safe_json_parse(analyst_raw)
    refined_tables = [t for t in analyst_result.get("tables", []) if t in VALID_TABLES]
    if not refined_tables:
        refined_tables = tables
    refined_schema = rag.schema_context(refined_tables)

    # Step 2: SQL Engineer
    engineer_sys = f"""\
{SQL_RULES}

You are a SQL Engineer. Generate a complete, correct PostgreSQL SELECT query against ONE table only.

Schema:
{refined_schema}

Few-shot examples:
{few_shot}

Schema Analysis:
{json.dumps(analyst_result, indent=2)}

Return STRICT JSON only:
{{"sql": "<complete SELECT query>", "params": {{}}, "explanation": "<one line>"}}"""

    eng_raw    = call_llm(
        engineer_sys, f"Query: {query}",
        max_new_tokens=700, decode_config=SQL_DECODE, stage_label="FALLBACK_SQL",
    )
    eng_result = safe_json_parse(eng_raw)
    fb_sql     = eng_result.get("sql", "")
    fb_params  = eng_result.get("params", {})

    if not fb_sql:
        cot_plan = run_cot_plan(query, refined_schema, refined_tables, few_shot)
        best     = run_got_gen(query, refined_schema, cot_plan, few_shot, refined_tables)
        if best:
            fb_sql    = best.sql
            fb_params = best.params

    exec_result = run_exec_retry(fb_sql, fb_params, refined_schema, query) if fb_sql else {
        "df": None, "exec_error": "No SQL generated", "exec_valid": False, "exec_attempts": 0}

    # Step 3: NOC Analyst explanation — same no-data short-circuit as
    # explanation_node: don't let the LLM invent commentary from zero rows.
    df = exec_result.get("df")
    if df is not None and df.empty:
        answer = ("The query ran successfully but returned no matching records. "
                  "Try adjusting the filters or time range.")
    elif df is None:
        answer = f"Could not retrieve data. {exec_result.get('exec_error', '')}".strip()
    else:
        summary = summarize_dataframe(df)
        answer  = call_llm(
            EXPLANATION_SYSTEM,
            f"Question: {query}\nSQL:\n{fb_sql[:500]}\n\nResult:\n{json.dumps(summary, indent=2, default=str)[:2500]}",
            max_new_tokens=400, decode_config=EXPLANATION_DECODE, stage_label="FALLBACK_EXPLAIN",
        ).strip() or "Fallback pipeline completed — see results."

    return {
        **prior,
        "tables":         refined_tables,
        "schema_context": refined_schema,
        "sql":            fb_sql,
        "sql_params":     fb_params,
        "df":             df,
        "exec_error":     exec_result.get("exec_error"),
        "exec_valid":     exec_result.get("exec_valid", False),
        "exec_attempts":  exec_result.get("exec_attempts", 0),
        "answer":         answer,
        "failed":         not exec_result.get("exec_valid", False),
        "fallback_used":  True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

SCOPE_REFUSAL_MESSAGE = (
    "I can only answer reporting questions about FALCON's network operations data "
    "— incident tickets, syslogs, sessions, ViPAM access, inventory, RCA, and "
    "topology correlation. Could you rephrase your question around one of those?"
)


def scope_gate_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] SCOPE GATE")
    print("="*60)
    query = state["user_query"]
    raw   = call_llm(SCOPE_GATE_SYSTEM, f"Message: {query}",
                     max_new_tokens=100, decode_config=ROUTER_DECODE, stage_label="SCOPE_GATE")
    parsed = safe_json_parse(raw)
    scope  = str(parsed.get("scope", "IN_SCOPE")).strip().upper()
    in_scope = scope != "OUT_OF_SCOPE"
    print(f"[SCOPE_GATE] {scope} — {parsed.get('reasoning', '')}")
    if not in_scope:
        sec_logger.log_event("content_blocked", "info",
                              detail={"stage": "scope_gate", "query": query[:500],
                                      "reasoning": parsed.get("reasoning", "")})
    return {**state, "in_scope": in_scope}


def scope_refusal_node(state: FalconAgentState) -> FalconAgentState:
    print("\n[NODE] SCOPE REFUSAL")
    return {**state, "answer": SCOPE_REFUSAL_MESSAGE, "failed": False, "out_of_scope": True}


def router_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] ROUTER")
    print("="*60)
    query  = state["user_query"]
    raw    = call_llm(ROUTER_SYSTEM, f"Query: {query}",
                      max_new_tokens=80, decode_config=ROUTER_DECODE, stage_label="ROUTER")
    parsed = safe_json_parse(raw)
    route  = parsed.get("route", "single_table_q")
    if route not in VALID_ROUTES:
        route = "single_table_q"
    log_trace({"node": "router", "route": route, "query": query,
               "confidence": parsed.get("confidence")})
    print(f"[ROUTER] → {route}  ({parsed.get('confidence','?')} confidence)")
    return {
        **state,
        "route":            route,
        "route_confidence": parsed.get("confidence", "Medium"),
        "route_reasoning":  parsed.get("reasoning", ""),
    }


def table_id_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] TABLE IDENTIFICATION")
    print("="*60)
    query  = state["user_query"]
    route  = state.get("route", "single_table_q")
    raw    = call_llm(
        TABLE_ID_SYSTEM,
        f"Valid tables: {sorted(VALID_TABLES)}\n\nRoute: {route}\n\nQuery: {query}",
        max_new_tokens=200, decode_config=INTENT_DECODE, stage_label="TABLE_ID",
    )
    parsed = safe_json_parse(raw)
    tables = [t for t in parsed.get("tables", []) if t in VALID_TABLES]
    if not tables:
        tables = ROUTE_TABLE_DEFAULTS.get(route, ["falconschema.vw_f_hpsm_h"])
    else:
        tables = tables[:1]   # enforce single-table constraint
    print(f"[TABLE_ID] Identified: {tables}")
    return {**state, "tables": tables}


def rag_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] RAG RETRIEVAL")
    print("="*60)
    query  = state["user_query"]
    tables = state.get("tables", [])

    # Embedding-based table cross-check
    emb_table = rag.find_table_by_embedding(query)
    if emb_table:
        if not tables or not any(t in VALID_TABLES for t in tables):
            print(f"[RAG] No valid LLM table — using embedding match: {emb_table}")
            tables = [emb_table]
        else:
            llm_table  = tables[0] if tables else "?"
            match_flag = "✓ agree" if emb_table == llm_table else "⚠ differ"
            print(f"[RAG] Embedding={emb_table.split('.')[-1]}  LLM={llm_table.split('.')[-1]}  [{match_flag}]")

    schema   = rag.schema_context(tables)
    few_shot = rag.few_shot_block(query, tables_hint=tables, top_k=5)
    print(f"[RAG] Schema context: {len(schema)} chars  |  Few-shot: {len(few_shot)} chars")
    return {**state, "tables": tables, "schema_context": schema, "few_shot_block": few_shot}


def cot_plan_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] CHAIN-OF-THOUGHT PLANNER (DSPy)")
    print("="*60)
    plan = run_cot_plan(
        query    = state["user_query"],
        schema   = state.get("schema_context", ""),
        tables   = state.get("tables", []),
        few_shot = state.get("few_shot_block", ""),
    )
    print(f"[COT] Plan:\n{plan[:400]}")
    return {**state, "cot_plan": plan}


def sql_generation_node(state: FalconAgentState) -> FalconAgentState:
    attempt = state.get("sql_retry_count", 0) + 1
    print("\n" + "="*60)
    print(f"[NODE] SQL GENERATION — GoT Beam (Attempt {attempt}/{SQL_MAX_RETRIES + 1})")
    print("="*60)

    query      = state["user_query"]
    schema     = state.get("schema_context", "")
    cot_plan   = state.get("cot_plan", "{}")
    few_shot   = state.get("few_shot_block", "")
    tables     = state.get("tables", [])
    last_error = state.get("sql_error", "")

    # Pinned time window injection
    time_param = extract_time_parameter(query)
    if time_param and time_param != "today":
        time_note = f"\n⏱ PINNED TIME: Use EXACTLY INTERVAL '{time_param}' — do NOT copy from examples.\n"
    elif time_param == "today":
        time_note = "\n⏱ PINNED TIME: Use EXACTLY >= CURRENT_DATE — no INTERVAL expression.\n"
    else:
        time_note = ""

    # Retry error injection
    if last_error:
        if "returned text" in last_error or "refused" in last_error.lower():
            error_note = (
                "\n🚨 PREVIOUS ATTEMPT: You returned prose instead of SQL. "
                "Output ONLY a SELECT statement — no explanations.\n"
            )
        else:
            error_note = f"\n⚠️ PREVIOUS SQL ERROR: {last_error}\nFix and regenerate.\n"
    else:
        error_note = ""

    enriched_plan = cot_plan + time_note + error_note
    best = run_got_gen(query, schema, enriched_plan, few_shot, tables)

    if best and best.sql:
        refusal = detect_model_refusal(best.sql)
        if refusal:
            print(f"[SQL_GEN] Refusal detected: {refusal}")
            return {**state, "sql": "", "sql_params": {},
                    "sql_error": refusal, "sql_retry_count": attempt}
        print(f"[SQL_GEN] ✓ SQL extracted ({len(best.sql)} chars, score={best.score:.1f})")
        return {**state, "sql": best.sql, "sql_params": best.params,
                "sql_error": "", "sql_retry_count": attempt}

    print("[SQL_GEN] GoT returned nothing — clearing SQL")
    return {**state, "sql": "", "sql_params": {},
            "sql_error": "GoT beam produced no candidate SQL", "sql_retry_count": attempt}


def sql_validation_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] SQL VALIDATION")
    print("="*60)
    sql = state.get("sql", "")
    try:
        validate_sql(sql)
        print("[VALIDATION] ✓ PASSED")
        return {**state, "sql_error": ""}
    except SQLSecurityError as e:
        # A security-guardrail rejection (DDL/DML/admin function/disallowed
        # table) is NOT a fixable syntax mistake — retrying just invites the
        # model to try another way around the same rule, and falling back to
        # run_llm_fallback previously produced hallucinated "success"
        # narratives (e.g. "Database 'mytestdb' has been successfully
        # created") once the retries were exhausted. Fail immediately with a
        # deterministic refusal instead of ever reaching the explanation LLM.
        error = str(e)
        print(f"[VALIDATION] ✗ SECURITY BLOCK → {error}")
        sec_logger.log_event("sql_blocked", "warning",
                              user_email=state.get("user_email"),
                              detail={"sql": sql[:500], "reason": error})
        return {**state, "sql_error": error, "security_blocked": True}
    except Exception as e:
        error = str(e)
        print(f"[VALIDATION] ✗ FAILED → {error}")
        sec_logger.log_event("sql_blocked", "warning",
                              user_email=state.get("user_email"),
                              detail={"sql": sql[:500], "reason": error})
        return {**state, "sql_error": error}


def should_retry_sql(state: FalconAgentState) -> str:
    error   = state.get("sql_error", "")
    retries = state.get("sql_retry_count", 0)
    if not error:
        return "execute"
    if state.get("security_blocked"):
        print("[ROUTER] SQL blocked by security guardrail → failing immediately (no retry)")
        return "fail"
    if retries <= SQL_MAX_RETRIES:
        print(f"[ROUTER] SQL invalid → retrying ({retries}/{SQL_MAX_RETRIES})")
        return "retry"
    print("[ROUTER] Max retries exhausted → failing")
    return "fail"


def sql_execution_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] SQL EXECUTION")
    print("="*60)
    result = run_exec_retry(
        sql    = state["sql"],
        params = state.get("sql_params", {}),
        schema = state.get("schema_context", ""),
        query  = state["user_query"],
    )
    if result.get("exec_valid"):
        df = result["df"]
        print(f"[EXECUTE] ✓ {len(df)} rows × {len(df.columns)} cols")
        log_trace({
            "node":   "sql_execution",
            "query":  state["user_query"],
            "route":  state.get("route"),
            "tables": state.get("tables"),
            "sql":    state.get("sql", "")[:200],
        })
        return {**state, "df": df, "exec_error": None,
                "exec_valid": True, "exec_attempts": result["exec_attempts"],
                "failed": False}
    else:
        err = result.get("exec_error", "Unknown DB error")
        print(f"[EXECUTE] ✗ Failed → {err}")
        return {**state, "df": None, "exec_error": err,
                "exec_valid": False, "exec_attempts": result["exec_attempts"],
                "failed": True, "failure_reason": f"DB execution failed: {err}"}


def explanation_node(state: FalconAgentState) -> FalconAgentState:
    print("\n" + "="*60)
    print("[NODE] EXPLANATION AGENT")
    print("="*60)
    query = state["user_query"]
    df    = state.get("df")
    error = state.get("exec_error")

    if error or df is None:
        return {**state, "answer": f"Query could not be executed. Error: {error or 'Unknown'}"}

    # Per feedback from live testing: when the query legitimately ran but
    # returned zero rows, do NOT call the explanation LLM at all — it was
    # observed inventing confident-sounding commentary ("this may indicate
    # a well-maintained network", sarcastic remarks, fabricated causes) from
    # nothing but an empty result set. Return a fixed, deterministic message
    # instead; only genuine data gets an LLM-written explanation.
    if df.empty:
        return {**state, "answer": "The query ran successfully but returned no matching records. "
                                    "Try adjusting the filters or time range."}

    summary = summarize_dataframe(df)
    user_p  = (
        f"Question: {query}\n\n"
        f"Route: {state.get('route', 'unknown')}\n"
        f"Tables: {state.get('tables', [])}\n"
        f"SQL:\n{state.get('sql', 'N/A')[:500]}\n\n"
        f"Result:\n{json.dumps(summary, indent=2, default=str)[:2500]}"
    )
    explanation = call_llm(
        EXPLANATION_SYSTEM, user_p,
        max_new_tokens=400, decode_config=EXPLANATION_DECODE,
        stage_label="EXPLAIN",
    ).strip()
    print(f"\n[EXPLANATION]\n{explanation}")
    return {**state, "answer": explanation}


def failure_node(state: FalconAgentState) -> FalconAgentState:
    print("\n[NODE] FAILURE TERMINAL")
    reason = state.get("sql_error") or state.get("failure_reason") or "Unknown SQL generation error"
    return {
        **state,
        "failed":         True,
        "failure_reason": f"SQL generation failed after {SQL_MAX_RETRIES + 1} attempts → {reason}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
#
#  router_node
#      │
#  table_id_node
#      │
#  rag_node
#      │
#  cot_plan_node
#      │
#  sql_generation_node ◄──────────────────────────────────────┐
#      │                                                       │
#  sql_validation_node                                         │ (retry)
#      │                                                       │
#      ├── "execute" ──► sql_execution_node                    │
#      │                      │                                │
#      │                 explanation_node                      │
#      │                      │                                │
#      │                     END                               │
#      │                                                       │
#      ├── "retry"  ───────────────────────────────────────────┘
#      │
#      └── "fail"   ──► failure_node ──► END

def build_falcon_graph() -> StateGraph:
    graph = StateGraph(FalconAgentState)
    graph.add_node("scope_gate",     scope_gate_node)
    graph.add_node("scope_refusal",  scope_refusal_node)
    graph.add_node("router",         router_node)
    graph.add_node("table_id",       table_id_node)
    graph.add_node("rag",            rag_node)
    graph.add_node("cot_planner",    cot_plan_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("sql_validation", sql_validation_node)
    graph.add_node("sql_execution",  sql_execution_node)
    graph.add_node("explanation",    explanation_node)
    graph.add_node("failure",        failure_node)

    graph.set_entry_point("scope_gate")
    graph.add_conditional_edges(
        "scope_gate",
        lambda state: "router" if state.get("in_scope", True) else "scope_refusal",
        {"router": "router", "scope_refusal": "scope_refusal"},
    )
    graph.add_edge("scope_refusal", END)
    graph.add_edge("router",         "table_id")
    graph.add_edge("table_id",       "rag")
    graph.add_edge("rag",            "cot_planner")
    graph.add_edge("cot_planner",    "sql_generation")
    graph.add_edge("sql_generation", "sql_validation")
    graph.add_conditional_edges(
        "sql_validation",
        should_retry_sql,
        {"execute": "sql_execution", "retry": "sql_generation", "fail": "failure"},
    )
    graph.add_edge("sql_execution", "explanation")
    graph.add_edge("explanation",   END)
    graph.add_edge("failure",       END)
    return graph.compile()


falcon_graph = build_falcon_graph()
print("[STARTUP] LangGraph compiled ✓")


def run_query(user_query: str, model_name: str = MODEL,
              use_fallback: bool = True) -> FalconAgentState:
    """Run the full LangGraph pipeline, with LLM fallback on failure."""
    initial: FalconAgentState = {
        "user_query":       user_query,
        "model_name":       model_name,
        "route":            "",
        "route_confidence": "",
        "route_reasoning":  "",
        "tables":           [],
        "schema_context":   "",
        "few_shot_block":   "",
        "cot_plan":         "",
        "sql":              "",
        "sql_params":       {},
        "sql_error":        "",
        "sql_retry_count":  0,
        "df":               None,
        "exec_error":       None,
        "exec_valid":       None,
        "exec_attempts":    0,
        "answer":           "",
        "failed":           False,
        "failure_reason":   "",
        "fallback_used":    False,
        "security_blocked": False,
        "in_scope":         True,
        "out_of_scope":     False,
    }
    final: FalconAgentState = falcon_graph.invoke(initial)
    if final.get("out_of_scope"):
        # The scope gate already set a clean refusal in "answer" — never let
        # the fallback chain (which has its own independent SQL Engineer
        # step) pick this back up, or we'd be right back to it inventing SQL
        # against unrelated/imaginary tables for an off-topic question.
        print("[ORCHESTRATOR] Out of scope — skipping fallback")
        return final
    if final.get("security_blocked"):
        # Never hand a security-guardrail rejection to the fallback chain —
        # it runs its own independent SQL Engineer step with no memory of
        # *why* the first attempt was blocked, and testing showed it would
        # either retry the same blocked operation or wander to an unrelated
        # table and explain irrelevant results as if they answered the
        # question. A deterministic refusal is correct here, not another
        # LLM attempt.
        print("[ORCHESTRATOR] SQL blocked by security guardrail — skipping fallback")
        final["answer"] = ("I can only answer read-only reporting questions against the approved "
                            "NOC data tables — I can't create, modify, or delete data.")
        return final
    if use_fallback and (final.get("failed") or final.get("df") is None):
        print("[ORCHESTRATOR] Graph result empty/failed — triggering fallback")
        final = run_llm_fallback(user_query, final)
    return final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — CHART BUILDER (from HAR V16 — fully preserved)
# ══════════════════════════════════════════════════════════════════════════════

PAL = ["#3b5bdb","#1c7ed6","#0c8599","#2f9e44","#e67700",
       "#c92a2a","#7950f2","#f76707","#1098ad","#2b8a3e"]
_BL = dict(paper_bgcolor="white", plot_bgcolor="#f9fafb",
           font=dict(family="DM Sans,sans-serif", color="#0f2044", size=12),
           margin=dict(l=48, r=16, t=36, b=52))


def build_chart(df, viz, title=""):
    if df is None or df.empty or viz == "table":
        return None
    cols = list(df.columns)
    num  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat  = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return None
    fig = None
    try:
        if viz == "bar" and cat:
            lb, vl = cat[0], num[0]
            d = df[[lb, vl]].dropna().nlargest(20, vl)
            clrs = [PAL[i % len(PAL)] for i in range(len(d))]
            if any(len(str(x)) > 10 for x in d[lb]):
                d = d.sort_values(vl)
                fig = go.Figure(go.Bar(y=d[lb].astype(str), x=d[vl], orientation="h",
                    marker_color=clrs, opacity=.88,
                    text=d[vl].apply(lambda v: f"{int(v):,}"), textposition="outside"))
                fig.update_layout(**_BL, height=max(240, len(d)*34+80), showlegend=False)
            else:
                fig = go.Figure(go.Bar(x=d[lb].astype(str), y=d[vl],
                    marker_color=clrs, opacity=.88,
                    text=d[vl].apply(lambda v: f"{int(v):,}"), textposition="outside"))
                fig.update_layout(**_BL, height=300, showlegend=False, xaxis_tickangle=-35)

        elif viz == "line":
            d = df[[cols[0], num[0]]].copy()
            d[cols[0]] = pd.to_datetime(d[cols[0]], errors="coerce")
            d = d.dropna().sort_values(cols[0])
            if not d.empty:
                fig = go.Figure(go.Scatter(x=d[cols[0]], y=d[num[0]], mode="lines+markers",
                    line=dict(color="#3b5bdb", width=2.5), fill="tozeroy",
                    fillcolor="rgba(59,91,219,0.07)"))
                fig.update_layout(**_BL, height=280, showlegend=False)

        elif viz == "pie" and cat:
            lb, vl = cat[0], num[0]
            d = df[[lb, vl]].dropna().nlargest(8, vl)
            fig = go.Figure(go.Pie(labels=d[lb].astype(str), values=d[vl], hole=0.4,
                marker=dict(colors=PAL[:len(d)], line=dict(color="#fff", width=2)),
                textinfo="percent+label"))
            fig.update_layout(**_BL, height=300)

        elif viz == "wordcloud" and cat:
            wc, vc = cat[0], num[0]
            top = df.nlargest(50, vc).reset_index(drop=True)
            n = len(top)
            if n == 0:
                return None
            mx = float(top[vc].max()) or 1
            sizes = ((top[vc] / mx) * 44 + 12).tolist()
            xp, yp = [0.0], [0.0]
            placed, ring = 1, 1
            while placed < n:
                r = ring * 0.19
                cap = max(6, int(2 * math.pi * r / 0.15))
                rn = min(cap, n - placed)
                for k in range(rn):
                    a = (2 * math.pi * k / rn) + ring * 0.4
                    xp.append(r * math.cos(a))
                    yp.append(r * math.sin(a) * 0.6)
                placed += rn
                ring += 1
            xp, yp = xp[:n], yp[:n]
            words  = top[wc].astype(str).tolist()
            counts = top[vc].tolist()
            fig = go.Figure()
            for i in range(n):
                c = PAL[i % len(PAL)]
                rv, gv, bv = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                op = min(1.0, 0.5 + 0.5 * (sizes[i]-12) / 44)
                fig.add_trace(go.Scatter(x=[xp[i]], y=[yp[i]], mode="text", text=[words[i]],
                    textfont=dict(size=sizes[i], color=f"rgba({rv},{gv},{bv},{op:.2f})"),
                    hovertemplate=f"<b>{words[i]}</b><br>{int(counts[i]):,}<extra></extra>",
                    showlegend=False))
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=4, r=4, t=24, b=4), height=300,
                xaxis=dict(visible=False, range=[-1.1, 1.1]),
                yaxis=dict(visible=False, range=[-0.75, 0.75]))

        if fig:
            if title:
                fig.update_layout(title=dict(text=title[:60],
                    font=dict(size=11, color="#9ca3af"), x=0.01))
            return fig.to_json()
    except Exception as e:
        print(f"[CHART] {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — VIZ DETECTION & PIPELINE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _detect_viz(q: str, df) -> str:
    """
    Infer the best visualisation type from the query text and dataframe.

    Priority order:
      1. Explicit user request  — e.g. "show as bar chart", "pie chart", "line graph"
         → always wins, regardless of data shape
      2. Strong implicit signals — e.g. "hourly trend", "wordcloud", "percentage"
      3. Data-shape heuristics  — fallback when user gave no chart hint
    """
    if df is None or df.empty:
        return "table"

    ql = q.lower()

    # ── 1. EXPLICIT USER CHART REQUEST ────────────────────────────────────────
    # Check these first so the user's stated preference always wins.

    _BAR_PHRASES = [
        "bar chart", "bar graph", "bar plot", "barchart",
        "show as bar", "show bar", "as a bar", "horizontal bar", "vertical bar",
        "column chart", "column graph",
    ]
    _LINE_PHRASES = [
        "line chart", "line graph", "line plot", "linechart",
        "show as line", "as a line", "show line",
        "trend chart", "trend graph", "time series chart",
    ]
    _PIE_PHRASES = [
        "pie chart", "pie graph", "donut chart", "donut graph",
        "show as pie", "as a pie", "show pie",
    ]
    _WORDCLOUD_PHRASES = [
        "wordcloud", "word cloud", "word-cloud",
        "keyword cloud", "tag cloud",
    ]
    _TABLE_PHRASES = [
        "show as table", "as a table", "tabular", "in table form",
        "show table", "table view",
    ]

    if any(p in ql for p in _BAR_PHRASES):
        return "bar"
    if any(p in ql for p in _LINE_PHRASES):
        return "line"
    if any(p in ql for p in _PIE_PHRASES):
        return "pie"
    if any(p in ql for p in _WORDCLOUD_PHRASES):
        return "wordcloud"
    if any(p in ql for p in _TABLE_PHRASES):
        return "table"

    # ── 2. STRONG IMPLICIT SIGNALS ────────────────────────────────────────────
    # Single-word hints that strongly imply a chart type without an explicit ask.

    if any(x in ql for x in ["keyword", "keywords"]):
        return "wordcloud"
    if any(x in ql for x in ["trend", "hourly", "by hour", "timeline",
                               "over time", "per hour", "time series"]):
        return "line"
    if any(x in ql for x in ["pie", "proportion", "share", "percentage"]):
        return "pie"

    # ── 3. DATA-SHAPE HEURISTICS ──────────────────────────────────────────────
    # Only reached when the user gave no chart hint at all.

    cols = list(df.columns)
    num  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat  = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if num and cat:
        if any(x in ql for x in ["count", "top", "most", "highest", "by circle",
                                   "by vendor", "by priority", "distribution",
                                   "breakdown", "noisy"]):
            if len(df) <= 6:
                return "pie"
            return "bar"
        if len(cols) == 2:
            return "bar"
    return "table"


def pipeline(q: str) -> dict:
    """
    Main entry-point called by the Dash callback.
    Runs the LangGraph 9-node pipeline, then maps the state to the
    HAR V16-compatible output dict for the frontend.
    """
    t0  = time.perf_counter()
    out = dict(answer="", sql="", rows=[], cols=[], fig=None,
               route={}, viz="table", timing={}, error="")

    # SECURITY FIX (Obs #8/#9/#10/#13/#14/#17/#20/#21 + prompt-injection
    # output-splice finding): deterministic pre-model content classifier.
    # This runs BEFORE any LLM call, so a jailbreak framing can never talk
    # its way past it the way persona-switch / translation / roleplay
    # tricks did in testing.
    blocked_category = classify_blocked_request(q)
    if blocked_category:
        sec_logger.log_event("content_blocked", "warning",
                              detail={"stage": "input", "category": blocked_category, "query": q[:500]})
        out["answer"] = refusal_for(blocked_category)
        out["timing"] = {"total": round(time.perf_counter() - t0, 2)}
        return out

    try:
        state = run_query(q, model_name=MODEL, use_fallback=True)

        # Map LangGraph tables → HAR16 route label for UI badges / colours
        tables    = state.get("tables", [])
        lbl       = _TABLE_TO_ROUTE_LABEL.get(tables[0], "fault") if tables else "fault"
        out["route"] = {
            "route": lbl,
            "table": tables[0] if tables else "",
        }

        out["sql"]    = state.get("sql", "")
        # Output-side defense in depth: catch any residual system-prompt
        # leakage or harmful content that slipped past the input classifier
        # (e.g. via an indirect/obfuscated jailbreak) before it reaches the user.
        raw_answer = state.get("answer", "No answer generated.")
        out["answer"] = sanitize_output(raw_answer, system_prompt_fragments=_SYSTEM_PROMPT_FRAGMENTS)
        if out["answer"] != raw_answer:
            sec_logger.log_event("content_blocked", "warning",
                                  detail={"stage": "output", "query": q[:500]})
        out["error"]  = (state.get("exec_error") or
                         state.get("failure_reason") or "")

        df = state.get("df")
        out["rows"] = safe_serialize_dataframe(df)
        out["cols"] = extract_columns(df)

        viz      = _detect_viz(q, df)
        out["viz"] = viz
        out["fig"] = build_chart(df, viz, q) if df is not None else None

    except Exception as e:
        traceback.print_exc()
        out["error"]  = str(e)
        out["answer"] = f"Error: {e}"

    out["timing"] = {"total": round(time.perf_counter() - t0, 2)}
    print("[PIPELINE]", out["timing"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — DOCUMENT SEARCH MODE
#  parse_uploaded_doc  → extract plain text from PDF / DOCX / TXT / CSV
#  answer_from_doc     → LLM Q&A grounded strictly in the document
# ══════════════════════════════════════════════════════════════════════════════

def parse_uploaded_doc(content_string: str, filename: str) -> Tuple[str, str]:
    """
    Parse a Dash dcc.Upload payload (base64 data-URI) into plain text.
    Returns (text, error_msg).  error_msg is "" on success.
    Supported: .pdf, .docx, .txt, .md, .log, .csv, .tsv
    """
    import base64 as _b64
    import io as _io

    # Strip the data-URI prefix  (e.g. "data:application/pdf;base64,XXXX")
    if "," in content_string:
        content_string = content_string.split(",", 1)[1]

    try:
        raw = _b64.b64decode(content_string)
    except Exception as ex:
        return "", f"Base64 decode error: {ex}"

    # SECURITY FIX (Obs #11 — Improper File Type Validation): the extension
    # was previously the ONLY check, and any unrecognized extension fell
    # through to a blind UTF-8 decode of the raw bytes — meaning images,
    # executables, and arbitrary binaries were always accepted and handed to
    # the LLM as "document text". validate_upload() checks the extension
    # allow-list AND the actual byte signature/content, and rejects anything
    # that doesn't match (e.g. a PNG renamed to .csv).
    ok, err = validate_upload(filename, raw)
    if not ok:
        sec_logger.log_event("upload_rejected", "warning", detail={"filename": filename, "reason": err})
        return "", f"Upload rejected: {err}"

    fname = filename.lower()

    try:
        # ── Plain text variants ───────────────────────────────────────────────
        if any(fname.endswith(ext) for ext in (".txt", ".md", ".log", ".rst", ".yaml", ".json")):
            return raw.decode("utf-8", errors="replace").strip(), ""

        # ── CSV / TSV ─────────────────────────────────────────────────────────
        if fname.endswith(".csv") or fname.endswith(".tsv"):
            sep   = "\t" if fname.endswith(".tsv") else ","
            lines = raw.decode("utf-8", errors="replace").splitlines()
            # Cap at 500 rows to avoid flooding the LLM context
            capped = lines[:500]
            note   = f"\n[Note: showing first 500 of {len(lines)} rows]" if len(lines) > 500 else ""
            return ("\n".join(capped) + note).strip(), ""

        # ── PDF ───────────────────────────────────────────────────────────────
        if fname.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(_io.BytesIO(raw))
                pages  = [p.extract_text() or "" for p in reader.pages]
                text   = "\n\n".join(pages).strip()
                if not text:
                    return "", "PDF appears to be scanned/image-only — no extractable text found."
                return text, ""
            except ImportError:
                return "", "pypdf not installed. Run: pip install pypdf --break-system-packages"
            except Exception as ex:
                return "", f"PDF parse error: {ex}"

        # ── DOCX ─────────────────────────────────────────────────────────────
        if fname.endswith(".docx"):
            try:
                import docx as _docx
                doc  = _docx.Document(_io.BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                return text.strip(), ""
            except ImportError:
                return "", "python-docx not installed. Run: pip install python-docx --break-system-packages"
            except Exception as ex:
                return "", f"DOCX parse error: {ex}"

        # Anything else has already been rejected by validate_upload() above —
        # no permissive "decode whatever bytes we got" fallback anymore.
        return "", f"Unsupported file type: {filename}"

    except Exception as ex:
        return "", f"Unexpected parse error: {ex}"


DOC_QA_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a precise document analyst embedded in the FALCON Vi NOC Intelligence platform.
The user has uploaded a document and is asking questions about it.

STRICT RULES:
1. Answer using ONLY the content from the provided document — no external knowledge.
2. If the answer is not in the document, say clearly: "This information is not in the uploaded document."
3. Quote the relevant section (≤ 3 sentences) when it helps precision.
4. Be concise — 3–6 sentences unless the question needs a longer structured answer.
5. If the document is a network/telecom document, use proper domain terminology.
6. Do NOT mention "the document says" repeatedly — integrate naturally.
7. Never fabricate configuration, credentials, or infrastructure details that
   are not literally present in the document text — an unclear/unreadable
   document is grounds to say so, never grounds to invent a plausible-sounding
   answer.
"""


def answer_from_doc(question: str, doc_text: str, doc_name: str) -> dict:
    """
    Run LLM Q&A grounded in doc_text.
    Returns a dict compatible with on_submit's bot-message structure.
    """
    t0 = time.perf_counter()

    blocked_category = classify_blocked_request(question)
    if blocked_category:
        sec_logger.log_event("content_blocked", "warning",
                              detail={"stage": "doc_qa_input", "category": blocked_category, "question": question[:500]})
        return {
            "answer": refusal_for(blocked_category),
            "doc_name": doc_name,
            "timing": {"total": round(time.perf_counter() - t0, 2)},
            "error": False,
        }

    # Trim doc to ~12 000 chars to fit LLM context (≈ 3k tokens)
    context = doc_text[:12_000]
    if len(doc_text) > 12_000:
        context += f"\n\n[…document truncated — showing first 12,000 of {len(doc_text):,} characters]"

    user_p = (
        f"Document name: {doc_name}\n\n"
        f"Document content:\n{context}\n\n"
        f"───\nQuestion: {question}"
    )

    raw = call_llm(
        DOC_QA_SYSTEM, user_p,
        max_new_tokens=700,
        decode_config=EXPLANATION_DECODE,
        stage_label="DOC_QA",
    ).strip()

    answer = raw if raw else "Unable to generate an answer. Please try rephrasing the question."
    answer = sanitize_output(answer, system_prompt_fragments=_SYSTEM_PROMPT_FRAGMENTS)
    print(f"[DOC_QA] '{question[:60]}' → {len(answer)} chars ({time.perf_counter()-t0:.1f}s)")

    return {
        "answer":     answer,
        "doc_name":   doc_name,
        "timing":     {"total": round(time.perf_counter() - t0, 2)},
        "error":      False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DASH APP — FRONTEND (fully preserved from HAR V16)
# ══════════════════════════════════════════════════════════════════════════════

RLBL = {"fault":"Fault","syslog":"Syslog","session":"Session","vipam":"ViPAM",
        "inventory":"Inventory","topology":"Topology","rca":"RCA","multi":"Multi"}
RCOL = {"fault":"#c92a2a","syslog":"#e67700","session":"#1c7ed6","vipam":"#7950f2",
        "inventory":"#2f9e44","topology":"#0c8599","rca":"#862e9c","multi":"#3b5bdb"}

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@1&family=JetBrains+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#f0f2f5;color:#0f2044;font-size:14px;height:100vh;overflow:hidden;-webkit-font-smoothing:antialiased}
.shell{display:grid;grid-template-columns:244px 1fr;grid-template-rows:52px 1fr;height:100vh}
.topbar{grid-column:1/-1;background:#0b1b3a;display:flex;align-items:center;border-bottom:1px solid rgba(255,255,255,.07)}
.t-brand{width:244px;flex-shrink:0;display:flex;align-items:center;gap:10px;padding:0 16px;border-right:1px solid rgba(255,255,255,.08);height:100%}
.t-icon{width:26px;height:26px;border-radius:6px;background:linear-gradient(135deg,#3b5bdb,#1c7ed6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px}
.t-name{font-family:'DM Serif Display',serif;font-style:italic;font-size:15px;color:#fff}
.t-sub{font-size:8px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3);margin-top:1px}
.t-body{flex:1;padding:0 18px;display:flex;align-items:center;gap:10px}
.t-title{font-size:13px;color:rgba(255,255,255,.5)}
.t-title b{color:#fff}
.t-right{margin-left:auto;display:flex;align-items:center;gap:10px;padding-right:18px}
.t-badges{display:flex;gap:4px;align-items:center}
.vd{width:1px;height:14px;background:rgba(255,255,255,.1)}
.live-dot{width:6px;height:6px;border-radius:50%;background:#51cf66;animation:pulse 2s infinite;box-shadow:0 0 4px rgba(81,207,102,.5)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.clock{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(255,255,255,.3)}
.b{display:inline-flex;align-items:center;padding:2px 7px;border-radius:4px;font-size:9px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;border:1px solid}
.br{color:#4c6ef5;background:rgba(76,110,245,.1);border-color:rgba(76,110,245,.25)}
.brows{color:#2f9e44;background:rgba(47,158,68,.1);border-color:rgba(47,158,68,.25)}
.btime{color:#868e96;background:#f8f9fa;border-color:#e4e7ec;font-family:'JetBrains Mono',monospace}
.berr{color:#c92a2a;background:rgba(201,42,42,.08);border-color:rgba(201,42,42,.2)}
.bviz{color:#7950f2;background:rgba(121,80,242,.08);border-color:rgba(121,80,242,.2)}
.sidebar{background:#fff;border-right:1px solid #e4e7ec;display:flex;flex-direction:column;overflow:hidden}
.s-hdr{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#adb5bd;padding:12px 14px 5px}
.s-new{all:unset;display:flex;align-items:center;gap:6px;width:calc(100% - 14px);margin:0 7px 3px;padding:7px 11px;background:#0b1b3a;color:#fff;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;transition:background .15s}
.s-new:hover{background:#1a3260}
.s-list{flex:1;overflow-y:auto;padding:2px 4px}
.s-list::-webkit-scrollbar{width:3px}
.s-list::-webkit-scrollbar-thumb{background:#dee2e6;border-radius:2px}
.s-item{padding:7px 10px;border-radius:5px;cursor:pointer;border-left:3px solid transparent;margin-bottom:1px}
.s-item:hover{background:#f1f3f5}
.s-act{background:#eef1fd!important;border-left-color:#3b5bdb!important}
.s-ttl{font-size:11px;font-weight:500;color:#0f2044;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.s-act .s-ttl{color:#3b5bdb;font-weight:600}
.s-meta{font-family:'JetBrains Mono',monospace;font-size:9px;color:#adb5bd}
.s-foot{padding:10px 12px;border-top:1px solid #e4e7ec;background:#f9fafb}
.s-mrow{display:flex;align-items:center;gap:6px}
.s-mlbl{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#adb5bd;flex-shrink:0}
.steps{display:flex;align-items:center;gap:3px;padding:6px 12px;background:#f9fafb;border-bottom:1px solid #e4e7ec;flex-wrap:wrap}
.step{padding:2px 7px;background:#eef1fd;color:#4c6ef5;border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap}
.step-sep{color:#d1d5db;font-size:9px}
.main{display:flex;flex-direction:column;overflow:hidden}
.scroll{flex:1;overflow-y:auto;padding:22px 0 12px}
.scroll::-webkit-scrollbar{width:4px}
.scroll::-webkit-scrollbar-thumb{background:#dee2e6;border-radius:2px}
.stream{max-width:840px;margin:0 auto;padding:0 22px}
.welcome{padding:28px 0 20px}
.w-eye{font-size:9.5px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#3b5bdb;margin-bottom:7px}
.w-h{font-family:'DM Serif Display',serif;font-style:italic;font-size:24px;color:#0f2044;margin-bottom:7px}
.w-p{font-size:13px;color:#6b7280;max-width:440px;line-height:1.7;margin-bottom:20px}
.cap-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:18px}
.cap{background:#fff;border:1.5px solid #e4e7ec;border-radius:8px;padding:11px 13px;cursor:pointer;transition:border-color .15s,box-shadow .15s,transform .1s;user-select:none}
.cap:hover{border-color:#3b5bdb;box-shadow:0 3px 9px rgba(59,91,219,.1);transform:translateY(-1px)}
.cap-ico{font-size:17px;margin-bottom:5px;pointer-events:none}
.cap-ttl{font-size:12px;font-weight:600;color:#0f2044;margin-bottom:1px;pointer-events:none}
.cap-dsc{font-size:10px;color:#9ca3af;line-height:1.5;pointer-events:none}
.qlbl{font-size:9px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:#adb5bd;margin-bottom:6px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{padding:3px 10px;background:#fff;border:1px solid #e4e7ec;border-radius:16px;font-size:10.5px;color:#6b7280;cursor:pointer;transition:all .12s;white-space:nowrap;user-select:none}
.chip:hover{background:#eef1fd;border-color:rgba(59,91,219,.3);color:#3b5bdb}
.msg{margin-bottom:14px}
.mu{display:flex;justify-content:flex-end}
.ub{background:#0b1b3a;color:#fff;border-radius:9px 9px 3px 9px;padding:9px 13px;max-width:58%;font-size:13.5px;line-height:1.6;box-shadow:0 2px 5px rgba(0,0,0,.14)}
.mb{display:flex;align-items:flex-start;gap:8px}
.av{width:25px;height:25px;flex-shrink:0;border-radius:6px;background:linear-gradient(135deg,#0b1b3a,#1a3260);display:flex;align-items:center;justify-content:center;font-size:10px;color:#fff;margin-top:2px}
.bb{flex:1;background:#fff;border:1px solid #e4e7ec;border-radius:3px 9px 9px 9px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.04);min-width:0}
.bm{display:flex;align-items:center;gap:4px;flex-wrap:wrap;padding:6px 11px;background:#f9fafb;border-bottom:1px solid #e4e7ec}
.ba{padding:11px 13px;font-size:13.5px;line-height:1.75;color:#0f2044;border-bottom:1px solid #e4e7ec}
.be{padding:11px 13px;background:#fff5f5;border-left:3px solid #c92a2a;color:#c92a2a;font-size:13px}
.sh{display:flex;align-items:center;justify-content:space-between;padding:5px 12px 4px;background:#f9fafb;border-bottom:1px solid #e4e7ec;gap:8px}
.sl{font-size:11px;font-weight:600;color:#374151}
.sc{font-family:'JetBrains Mono',monospace;font-size:9.5px;color:#adb5bd;background:#f1f3f6;border:1px solid #e4e7ec;border-radius:3px;padding:1px 5px}
.sq{all:unset;display:flex;align-items:center;gap:5px;padding:6px 12px;width:100%;font-size:9.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:#adb5bd;background:#f9fafb;border-top:1px solid #e4e7ec;cursor:pointer}
.sq:hover{background:#f1f3f6;color:#374151}
.sqc{margin:0;padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.6;color:#1e3a5f;background:#f8faff;border-top:1px solid #e4e7ec;white-space:pre-wrap;word-break:break-word;display:none}
.iarea{background:#fff;border-top:1px solid #e4e7ec;padding:10px 22px 13px}
.iinner{max-width:840px;margin:0 auto}
.iwrap{display:flex;align-items:flex-end;gap:8px;border:1.5px solid #e4e7ec;border-radius:10px;padding:7px 8px;transition:border-color .15s,box-shadow .15s;background:#fff}
.iwrap:focus-within{border-color:#3b5bdb;box-shadow:0 0 0 3px rgba(59,91,219,.08)}
.ita{flex:1;background:none;border:none;outline:none;resize:none;max-height:100px;font-size:13.5px;line-height:1.5;color:#0f2044;font-family:'DM Sans',sans-serif;padding:2px 0}
.ita::placeholder{color:#adb5bd}
.sbtn{width:31px;height:31px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:#0b1b3a;border:none;border-radius:6px;cursor:pointer;color:#fff;font-size:15px;transition:background .15s,transform .1s}
.sbtn:hover{background:#3b5bdb;transform:translateY(-1px)}
.ihint{margin-top:4px;font-size:10px;color:#adb5bd;text-align:right}
.dash-table-container .dash-spreadsheet-inner th{background:#f9fafb!important;color:#9ca3af!important;font-size:9.5px!important;font-weight:700!important;letter-spacing:.06em!important;text-transform:uppercase!important;border-bottom:1px solid #e4e7ec!important;padding:6px 10px!important}
.dash-table-container .dash-spreadsheet-inner td{font-size:11.5px!important;padding:5px 10px!important;font-family:'JetBrains Mono',monospace!important;color:#0f2044!important;background:#fff!important;border-bottom:1px solid rgba(228,231,236,.5)!important}
.dash-table-container .dash-spreadsheet-inner tr:hover td{background:#f9fafb!important}
.fb-bar{display:flex;align-items:center;gap:6px;padding:6px 12px;background:#f9fafb;border-top:1px solid #e4e7ec}
.fb-lbl{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#adb5bd;margin-right:2px}
.fbup,.fbdn{all:unset;display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid #e4e7ec;background:#fff;color:#6b7280;transition:all .15s}
.fbup:hover{background:#ebfbee;border-color:#2f9e44;color:#2f9e44}
.fbdn:hover{background:#fff5f5;border-color:#c92a2a;color:#c92a2a}
.fbup.fb-active{background:#ebfbee;border-color:#2f9e44;color:#2f9e44;cursor:default}
.fbdn.fb-active{background:#fff5f5;border-color:#c92a2a;color:#c92a2a;cursor:default}
.fbdn.fb-spin{opacity:.5;cursor:wait}
.fb-hint{font-size:9.5px;color:#adb5bd;margin-left:4px}
/* ── Document Search Mode ──────────────────────────────────────────────────── */
.docbtn{width:31px;height:31px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:#f1f3f5;border:1.5px solid #e4e7ec;border-radius:6px;cursor:pointer;font-size:15px;transition:all .15s;line-height:1}
.docbtn:hover{background:#eef1fd;border-color:#3b5bdb}
.docbtn-active{background:#eef1fd!important;border-color:#3b5bdb!important;box-shadow:0 0 0 3px rgba(59,91,219,.1)!important}
.doc-strip{padding:7px 0 5px;margin-bottom:6px;border-bottom:1px solid #e4e7ec;display:flex;flex-direction:column;gap:5px}
.doc-strip-hidden{display:none!important}
.doc-upload-zone{border:1.5px dashed #c8d0e7;border-radius:8px;padding:9px 16px;text-align:center;cursor:pointer;font-size:12px;color:#6b7280;background:#f8f9ff;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:7px}
.doc-upload-zone:hover{border-color:#3b5bdb;background:#eef1fd;color:#3b5bdb}
.doc-info-row{display:flex;align-items:center;gap:6px;min-height:18px}
.doc-loaded{font-size:11px;color:#2f9e44;font-weight:600;display:flex;align-items:center;gap:4px}
.doc-err{font-size:11px;color:#c92a2a;font-weight:500}
.doc-clear{all:unset;font-size:10px;color:#adb5bd;cursor:pointer;padding:2px 5px;border-radius:3px;border:1px solid #e4e7ec;background:#fff;transition:all .12s}
.doc-clear:hover{color:#c92a2a;border-color:#c92a2a;background:#fff5f5}
.bdoc{color:#0c8599;background:rgba(12,133,153,.08);border-color:rgba(12,133,153,.25)}
/* ── Login page ────────────────────────────────────────────────────────────── */
.login-shell{min-height:100vh;background:linear-gradient(135deg,#060f24 0%,#0b1b3a 55%,#0d2554 100%);display:flex;align-items:center;justify-content:center;padding:24px}
.login-card{width:100%;max-width:400px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:40px 36px 34px;box-shadow:0 24px 64px rgba(0,0,0,.45)}
.login-logo{display:flex;align-items:center;gap:11px;margin-bottom:28px}
.login-logo-icon{width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,#3b5bdb,#1c7ed6);display:flex;align-items:center;justify-content:center;font-size:16px;color:#fff;flex-shrink:0}
.login-logo-name{font-family:'DM Serif Display',serif;font-style:italic;font-size:21px;color:#fff}
.login-logo-sub{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3);margin-top:1px}
.login-h{font-size:18px;font-weight:700;color:#fff;margin-bottom:4px}
.login-sub{font-size:12px;color:rgba(255,255,255,.4);margin-bottom:26px;line-height:1.5}
.login-label{display:block;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:6px}
.login-input{width:100%;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:10px 13px;font-size:13.5px;color:#fff;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .15s,box-shadow .15s;margin-bottom:16px}
.login-input::placeholder{color:rgba(255,255,255,.25)}
.login-input:focus{border-color:rgba(59,91,219,.7);box-shadow:0 0 0 3px rgba(59,91,219,.2)}
.login-btn{width:100%;padding:11px;background:linear-gradient(135deg,#3b5bdb,#1c7ed6);border:none;border-radius:8px;color:#fff;font-size:13.5px;font-weight:600;cursor:pointer;font-family:'DM Sans',sans-serif;transition:opacity .15s,transform .1s;margin-top:4px}
.login-btn:hover{opacity:.88;transform:translateY(-1px)}
.login-btn:active{transform:translateY(0)}
.login-error{margin-top:12px;padding:9px 12px;background:rgba(201,42,42,.15);border:1px solid rgba(201,42,42,.35);border-radius:7px;color:#ff8787;font-size:12px;line-height:1.5}
.login-badge{display:inline-flex;align-items:center;gap:5px;margin-top:22px;padding:6px 11px;background:rgba(47,158,68,.1);border:1px solid rgba(47,158,68,.2);border-radius:20px;font-size:10px;color:rgba(47,158,68,.9)}
.login-divider{border:none;border-top:1px solid rgba(255,255,255,.08);margin:20px 0}
/* ── Topbar user chip ──────────────────────────────────────────────────────── */
.user-chip{display:flex;align-items:center;gap:7px;padding:4px 10px 4px 5px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:20px;cursor:default}
.user-avatar{width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,#3b5bdb,#1c7ed6);display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;font-weight:700;flex-shrink:0}
.user-email{font-size:10px;color:rgba(255,255,255,.6);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.logout-btn{all:unset;padding:3px 9px;border-radius:5px;font-size:10px;font-weight:600;color:rgba(255,255,255,.35);border:1px solid rgba(255,255,255,.1);cursor:pointer;transition:all .15s;margin-left:2px}
.logout-btn:hover{color:#ff8787;border-color:rgba(255,100,100,.3);background:rgba(255,100,100,.08)}
/* ── Thinking bubble ───────────────────────────────────────────────────────── */
.thinking-bubble{display:flex;align-items:center;gap:10px;padding:13px 15px;font-size:13px;color:#6b7280;background:#f9fafb}
.thinking-robot{font-size:20px;display:inline-block;animation:robot-bob .9s ease-in-out infinite}
@keyframes robot-bob{0%,100%{transform:translateY(0) rotate(-3deg)}50%{transform:translateY(-4px) rotate(3deg)}}
.thinking-label{font-weight:600;color:#374151;font-size:13px}
.thinking-dots{display:inline-flex;gap:2px;align-items:center;margin-left:1px}
.thinking-dots span{display:inline-block;width:5px;height:5px;border-radius:50%;background:#3b5bdb;animation:dot-pulse 1.4s ease-in-out infinite both}
.thinking-dots span:nth-child(2){animation-delay:.2s}
.thinking-dots span:nth-child(3){animation-delay:.4s}
@keyframes dot-pulse{0%,80%,100%{transform:scale(0.6);opacity:.4}40%{transform:scale(1);opacity:1}}
/* ── CSV download button ───────────────────────────────────────────────────── */
.csv-btn{all:unset;display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:5px;font-size:10px;font-weight:600;cursor:pointer;border:1px solid #e4e7ec;background:#fff;color:#6b7280;transition:all .15s;margin-left:auto}
.csv-btn:hover{background:#ebfbee;border-color:#2f9e44;color:#2f9e44}
.bdocname{color:#495057;background:#f8f9fa;border-color:#e4e7ec;font-family:'JetBrains Mono',monospace;font-size:8.5px;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-steps{display:flex;align-items:center;gap:3px;padding:6px 12px;background:#f0fafe;border-bottom:1px solid #c8e6f0}
.doc-step{padding:2px 7px;background:#d3eef8;color:#0c8599;border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap}
"""

# ── JS — ALL non-Dash interactions ────────────────────────────────────────────
JS = r"""
(function(){
  /* IST clock */
  function tick(){
    var el=document.querySelector('[data-clock]');
    if(!el)return;
    var d=new Date(new Date().toLocaleString('en-US',{timeZone:'Asia/Kolkata'}));
    el.textContent=[d.getHours(),d.getMinutes(),d.getSeconds()]
      .map(function(v){return String(v).padStart(2,'0')}).join(':')+' IST';
  }
  setInterval(tick,1000);tick();

  /* Enter to send */
  document.addEventListener('keydown',function(e){
    if(e.target.id==='qin'&&e.key==='Enter'&&!e.shiftKey){
      e.preventDefault();
      if((e.target.value||'').trim())document.getElementById('sbtn').click();
    }
  },true);

  /* SQL toggle */
  document.addEventListener('click',function(e){
    var b=e.target.closest('[data-sql]');
    if(!b)return;
    var p=document.getElementById('sqlpre-'+b.getAttribute('data-sql'));
    if(p)p.style.display=p.style.display==='block'?'none':'block';
  });

  /* Cap cards and chips → fill textarea + click send */
  document.addEventListener('click',function(e){
    var el=e.target.closest('[data-query]');
    if(!el)return;
    var q=el.getAttribute('data-query');
    if(!q)return;
    var ta=document.getElementById('qin');
    var btn=document.getElementById('sbtn');
    if(!ta||!btn)return;
    var setter=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
    setter.call(ta,q);
    ta.dispatchEvent(new Event('input',{bubbles:true}));
    setTimeout(function(){btn.click();},50);
  });
})();
"""

# ── Dash app ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="FALCON – Vi NOC")
app.index_string = f"""<!DOCTYPE html>
<html><head>
  {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
  <style>{CSS}</style>
</head><body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  <script>{JS}</script>
</body></html>"""

# ── UI Helpers ────────────────────────────────────────────────────────────────
def _av():
    return html.Div("▲", className="av")

def _steps():
    labels = ["① Router","② TableID","③ RAG","④ CoT","⑤ GoT-SQL","⑥ Validate","⑦ Execute","⑧ Explain"]
    items = []
    for i, l in enumerate(labels):
        items.append(html.Span(l, className="step"))
        if i < len(labels)-1:
            items.append(html.Span("→", className="step-sep"))
    return html.Div(items, className="steps")


def make_login_page():
    """Render the full-screen login card."""
    domain_hint = (
        "All @vodafoneidea.com accounts are currently enabled."
        if ALLOW_VODAFONE_DOMAIN
        else f"{len(ALLOWED_EMAILS)} authorised user(s) only."
    )
    return html.Div(className="login-shell", children=[
        html.Div(className="login-card", children=[
            # ── Logo ─────────────────────────────────────────────────────────
            html.Div(className="login-logo", children=[
                html.Div("▲", className="login-logo-icon"),
                html.Div([
                    html.Div("{TEST} Falcon", className="login-logo-name"),
                    html.Div("TESTING VERSION", className="login-logo-sub"),
                ]),
            ]),
            # ── Heading ───────────────────────────────────────────────────────
            html.Div("Sign in to Vi NOC", className="login-h"),
            html.Div("LangGraph 9-Node Intelligence Platform", className="login-sub"),
            # ── Email ─────────────────────────────────────────────────────────
            html.Label("Email address", className="login-label", htmlFor="login-email"),
            dcc.Input(
                id="login-email", type="email",
                placeholder="you@vodafoneidea.com",
                className="login-input",
                debounce=False, n_submit=0,
            ),
            # ── Password ──────────────────────────────────────────────────────
            html.Label("Password", className="login-label", htmlFor="login-password"),
            dcc.Input(
                id="login-password", type="password",
                placeholder="Enter your password",
                className="login-input",
                debounce=False, n_submit=0,
            ),
            # ── Button ────────────────────────────────────────────────────────
            html.Button("Sign in →", id="login-btn", className="login-btn", n_clicks=0),
            # ── Error ─────────────────────────────────────────────────────────
            html.Div(id="login-error", className="login-error",
                     style={"display": "none"}),
            # ── Footer badge ──────────────────────────────────────────────────
            html.Hr(className="login-divider"),
            html.Div([
                html.Span("🔒 "),
                html.Span(domain_hint, style={"fontSize": "10px",
                                               "color": "rgba(255,255,255,.35)"}),
            ], className="login-badge"),
        ])
    ])

def make_welcome():
    return html.Div(className="welcome", children=[
        html.Div("Vi Network Operations", className="w-eye"),
        html.Div("What would you like to analyse?", className="w-h"),
        html.P("Ask in plain English — FALCON routes through 9 AI nodes (LangGraph + DSPy + GoT) and returns live data.",
               className="w-p"),
        html.Div(className="cap-grid", children=[
            html.Div(className="cap", **{"data-query":"Open P1 and P2 faults right now"}, children=[
                html.Div("🔴", className="cap-ico"),
                html.Div("Live Faults", className="cap-ttl"),
                html.Div("P1/P2 incidents, circle breakdown", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query":"Error keyword wordcloud last 24 hours"}, children=[
                html.Div("📊", className="cap-ico"),
                html.Div("Syslog Analysis", className="cap-ttl"),
                html.Div("Error keywords, noisy routers, trends", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query":"Show config sessions last 1 hour"}, children=[
                html.Div("🔐", className="cap-ico"),
                html.Div("Session Audit", className="cap-ttl"),
                html.Div("Config changes, user access audit", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query":"Node count by vendor"}, children=[
                html.Div("🗄️", className="cap-ico"),
                html.Div("Inventory", className="cap-ttl"),
                html.Div("Node counts, vendor distribution", className="cap-dsc")]),
        ]),
        html.Div("Quick queries", className="qlbl"),
        html.Div(className="chips", children=[
            html.Span(q, className="chip", **{"data-query": q}) for q in QUICK
        ]),
    ])

def render_bot_doc(m, idx, voted=None):
    """Render a bot reply that came from Document Search mode (no SQL / no table)."""
    doc_name = m.get("source_doc", "Document")
    total    = m.get("timing", {}).get("total")
    is_err   = m.get("error", False)
    body     = []

    # ── Badges ────────────────────────────────────────────────────────────────
    badges = [
        html.Span("📄  DOC SEARCH", className="b bdoc"),
        html.Span(doc_name[:35], className="b bdocname"),
    ]
    if total:
        badges.append(html.Span(f"⏱ {total:.1f}s", className="b btime"))
    body.append(html.Div(badges, className="bm"))

    # ── Doc pipeline steps ────────────────────────────────────────────────────
    body.append(html.Div([
        html.Span("① Parse", className="doc-step"),
        html.Span("→", className="step-sep"),
        html.Span("② Chunk", className="doc-step"),
        html.Span("→", className="step-sep"),
        html.Span("③ LLM Q&A", className="doc-step"),
    ], className="doc-steps"))

    # ── Answer ────────────────────────────────────────────────────────────────
    body.append(html.Div(m.get("content", ""), className="be" if is_err else "ba"))

    # ── Feedback bar ──────────────────────────────────────────────────────────
    up_cls = "fbup fb-active" if voted == "up" else "fbup"
    dn_cls = "fbdn fb-active" if voted == "dn" else "fbdn"
    hint   = ("✓ Helpful" if voted == "up"
              else "↻ Retrying…" if voted == "dn"
              else "")
    body.append(html.Div([
        html.Span("Was this helpful?", className="fb-lbl"),
        html.Button("👍  Upvote",   n_clicks=0,
                    id={"type": "upvote",   "index": idx},
                    className=up_cls, disabled=voted is not None),
        html.Button("👎  Downvote", n_clicks=0,
                    id={"type": "downvote", "index": idx},
                    className=dn_cls, disabled=voted is not None),
        html.Span(hint, className="fb-hint") if hint else html.Span(),
    ], className="fb-bar"))

    return html.Div(className="msg mb", children=[_av(), html.Div(body, className="bb")])


def render_thinking():
    """Animated 🤖 Thinking... bubble shown while pipeline is running."""
    return html.Div(className="msg mb", id="thinking-bubble", children=[
        _av(),
        html.Div(className="bb", children=[
            html.Div(className="thinking-bubble", children=[
                html.Span("🤖", className="thinking-robot"),
                html.Span("Thinking", className="thinking-label"),
                html.Span(className="thinking-dots", children=[
                    html.Span(), html.Span(), html.Span(),
                ]),
            ])
        ])
    ])


def render_bot(m, idx, voted=None):
    """
    voted: None | "up" | "dn"
      None → both buttons enabled
      "up" → upvote highlighted, downvote disabled
      "dn" → downvote highlighted (retrying), upvote disabled
    """
    # ── Document Search mode answers get their own renderer ──────────────────
    if m.get("doc_mode"):
        return render_bot_doc(m, idx, voted)

    rk  = m.get("route", {}).get("route", "") if isinstance(m.get("route"), dict) else ""
    col = RCOL.get(rk, "#6b7280")
    rows  = m.get("rows", [])
    clst  = m.get("cols", [])
    fig_j = m.get("fig")
    sql   = m.get("sql", "")
    total = m.get("timing", {}).get("total")
    viz   = m.get("viz", "table")
    is_err = m.get("error", False)
    body = []
    badges = []
    if rk:
        badges.append(html.Span(RLBL.get(rk, rk), className="b br",
            style={"color": col, "borderColor": col+"44", "background": col+"11"}))
    if viz != "table":
        badges.append(html.Span(viz.upper(), className="b bviz"))
    if rows:
        badges.append(html.Span(f"{len(rows):,} rows", className="b brows"))
    if total:
        badges.append(html.Span(f"⏱ {total:.1f}s", className="b btime"))
    if is_err:
        badges.append(html.Span("ERROR", className="b berr"))
    if badges:
        body.append(html.Div(badges, className="bm"))
    body.append(_steps())
    body.append(html.Div(m.get("content", ""), className="be" if is_err else "ba"))
    if fig_j:
        try:
            fig = pio.from_json(fig_j)
            h   = fig.layout.height
            fig.update_layout(height=int(h) if h and h > 0 else 300)
            body.append(html.Div(
                style={"padding":"11px 11px 5px","borderBottom":"1px solid #e4e7ec"},
                children=[dcc.Graph(id=f"ch{idx}", figure=fig,
                    config={"displayModeBar": True,
                            "modeBarButtonsToRemove": ["lasso2d","select2d"],
                            "responsive": True},
                    style={"width":"100%"})]))
        except Exception as ex:
            print(f"[chart] {ex}")
    if rows and clst:
        body.append(html.Div([
            html.Div([html.Span("Results", className="sl"),
                      html.Span(f"{len(rows):,} rows", className="sc"),
                      html.Button("⬇ Download CSV",
                                  id={"type": "csv-btn", "index": idx},
                                  n_clicks=0, className="csv-btn")], className="sh"),
            dash_table.DataTable(
                data=rows[:200],
                columns=[{"name": c, "id": c} for c in clst],
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"'JetBrains Mono',monospace","fontSize":"11px",
                             "padding":"5px 10px","border":"none",
                             "borderBottom":"1px solid rgba(228,231,236,.5)",
                             "color":"#0f2044","textOverflow":"ellipsis",
                             "maxWidth":"200px","overflow":"hidden"},
                style_header={"background":"#f9fafb","color":"#9ca3af","fontWeight":"700",
                               "fontSize":"9px","textTransform":"uppercase",
                               "letterSpacing":".06em","borderBottom":"1px solid #e4e7ec",
                               "padding":"6px 10px"},
                style_data_conditional=[
                    {"if":{"filter_query":'{priority_code} = "P1"',"column_id":"priority_code"},
                     "color":"#c92a2a","fontWeight":"700"},
                    {"if":{"filter_query":'{priority_code} = "P2"',"column_id":"priority_code"},
                     "color":"#e67700","fontWeight":"600"},
                    {"if":{"filter_query":'{status} = "open"',"column_id":"status"},
                     "color":"#c92a2a"},
                    {"if":{"filter_query":'{status} = "closed"',"column_id":"status"},
                     "color":"#2f9e44"},
                ])]))
    if sql:
        sid = str(idx)
        body.append(html.Div([
            html.Button("⟨/⟩  View SQL", className="sq", **{"data-sql": sid}),
            html.Pre(sql, id=f"sqlpre-{sid}", className="sqc"),
        ]))
    # ── Feedback bar (upvote / downvote) ─────────────────────────────────────
    up_cls = "fbup fb-active" if voted == "up" else "fbup"
    dn_cls = "fbdn fb-active" if voted == "dn" else "fbdn"
    hint   = ("✓ Saved to training set" if voted == "up"
              else "↻ Regenerating…"    if voted == "dn"
              else "")
    body.append(html.Div([
        html.Span("Was this helpful?", className="fb-lbl"),
        html.Button(
            "👍  Upvote", n_clicks=0,
            id={"type": "upvote",   "index": idx},
            className=up_cls,
            disabled=voted is not None,
        ),
        html.Button(
            "👎  Downvote", n_clicks=0,
            id={"type": "downvote", "index": idx},
            className=dn_cls,
            disabled=voted is not None,
        ),
        html.Span(hint, className="fb-hint") if hint else html.Span(),
    ], className="fb-bar"))
    return html.Div(className="msg mb", children=[_av(), html.Div(body, className="bb")])

def render_stream(msgs, feedback=None):
    """
    feedback: dict mapping str(msg_index) -> "up" | "dn"
    Passed down so render_bot can show the correct voted state.
    """
    if not msgs:
        return [make_welcome()]
    feedback = feedback or {}
    out = []
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            out.append(html.Div(className="msg mu",
                children=[html.Div(m["content"], className="ub")]))
        else:
            out.append(render_bot(m, i, voted=feedback.get(str(i))))
    return out

def render_sidebar(convs, active):
    items = []
    for c in reversed(convs or []):
        cls = "s-item s-act" if c["id"] == active else "s-item"
        items.append(html.Div(className=cls,
            id={"type":"conv","index":c["id"]}, n_clicks=0, children=[
                html.Div(c.get("title","…")[:42], className="s-ttl"),
                html.Div(f"{c.get('count',0)} msg", className="s-meta")]))
    return items

# ── Layout ────────────────────────────────────────────────────────────────────

def make_main_layout(email: str = ""):
    """Return the full NOC shell layout, injecting the logged-in user email."""
    initials = (email[:2].upper() if email else "?")
    return html.Div(className="shell", children=[
        html.Header(className="topbar", children=[
            html.Div(className="t-brand", children=[
                html.Div("▲", className="t-icon"),
                html.Div([html.Div("{TEST} Falcon", className="t-name"),
                          html.Div("Testing Version", className="t-sub")])]),
            html.Div(className="t-body", children=[
                html.Span([html.B("Vi NOC"), " — Network Operations Intelligence"], className="t-title"),
                html.Div(id="tbadges", className="t-badges")]),
            html.Div(className="t-right", children=[
                html.Div(className="vd"),
                html.Div([html.Div(className="live-dot"),
                          html.Span("Live", style={"fontSize":"10px","color":"rgba(255,255,255,.35)"})],
                         style={"display":"flex","alignItems":"center","gap":"5px"}),
                html.Div(className="vd"),
                html.Div("", className="clock", **{"data-clock":"1"}),
                html.Div(className="vd"),
                # ── Logged-in user chip + logout ─────────────────────────────
                html.Div(className="user-chip", children=[
                    html.Div(initials, className="user-avatar"),
                    html.Span(email, className="user-email"),
                ]),
                html.Button("Sign out", id="logout-btn",
                            className="logout-btn", n_clicks=0),
            ])]),
        html.Aside(className="sidebar", children=[
            html.Div("Conversations", className="s-hdr"),
            html.Button("＋  New Conversation", className="s-new", id="newbtn", n_clicks=0),
            html.Div(id="slist", className="s-list"),
            html.Div(className="s-foot", children=[
                html.Div(className="s-mrow", children=[
                    html.Span("Model", className="s-mlbl"),
                    dcc.Dropdown(id="mdd",
                        options=[{"label":x,"value":x} for x in ["mistral","redhatai","llama3"]],
                        value=MODEL, clearable=False,
                        style={"flex":"1","fontSize":"12px"})])])]),
        html.Main(className="main", children=[
            html.Div(className="scroll", id="chat-scroll", children=[
                html.Div(id="stream", className="stream", children=[make_welcome()])]),
            html.Div(className="iarea", children=[
                html.Div(className="iinner", children=[
                    # ── Document Search strip (hidden until doc mode is ON) ───────
                    html.Div(id="doc-strip", className="doc-strip doc-strip-hidden", children=[
                        dcc.Upload(
                            id="doc-upload",
                            children=html.Div([
                                html.Span("📎", style={"fontSize": "15px"}),
                                html.Span(" Drop a file here or "),
                                html.Span("browse", style={"color": "#3b5bdb", "fontWeight": "600",
                                                            "textDecoration": "underline"}),
                                html.Span(" — PDF, DOCX, TXT, CSV", style={"color": "#adb5bd"}),
                            ]),
                            className="doc-upload-zone",
                            multiple=False,
                        ),
                        html.Div(id="doc-info-row", className="doc-info-row"),
                    ]),
                    # ── Input row ────────────────────────────────────────────
                    html.Div(className="iwrap", children=[
                        dcc.Textarea(id="qin", className="ita", rows=1,
                            placeholder="Ask about faults, syslogs, sessions, RCA…",
                            style={"resize": "none"}),
                        html.Button("📄", id="docbtn", className="docbtn", n_clicks=0,
                                    title="Toggle Document Search mode"),
                        html.Button("→", id="sbtn", className="sbtn", n_clicks=0)]),
                    html.Div(id="ihint-text",
                             children="Enter to send · Shift+Enter for new line",
                             className="ihint")])])]),
        # Stores (scoped inside main layout)
        dcc.Store(id="smsgs",     data=[]),
        dcc.Store(id="sconvs",    data=[]),
        dcc.Store(id="scid",      data=None),
        dcc.Store(id="sfeedback", data={}),
        dcc.Store(id="sdoc",      data=None),
        dcc.Store(id="sdoc_mode", data=False),
        dcc.Store(id="spending",  data=None),
        dcc.Download(id="csv-dl"),
    ])


# ── Top-level layout: auth shell ──────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="sauth", data=None, storage_type="session"),  # persists across refreshes
    html.Div(id="page-content", children=[make_login_page()]),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── AUTH: Page routing ────────────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Input("sauth", "data"),
    Input("url",   "pathname"),
)
def route_page(token, _pathname):
    """Show login page or main NOC shell depending on session validity."""
    email = auth_validate(token)
    if not email:
        return make_login_page()
    return make_main_layout(email)


# ── AUTH: Login ───────────────────────────────────────────────────────────────

@app.callback(
    Output("sauth",        "data",             allow_duplicate=True),
    Output("login-error",  "children",                              ),
    Output("login-error",  "style",                                 ),
    Input("login-btn",     "n_clicks"),
    Input("login-email",   "n_submit"),
    Input("login-password","n_submit"),
    State("login-email",   "value"),
    State("login-password","value"),
    prevent_initial_call=True,
)
def on_login(n_btn, _ns_email, _ns_pwd, email, password):
    """Validate credentials and issue a session token."""
    if not (n_btn or _ns_email or _ns_pwd):
        return no_update, no_update, no_update
    ok, err = auth_login(email or "", password or "")
    if not ok:
        return no_update, err, {"display": "block"}
    token = auth_create_session(email)
    print(f"[AUTH] Login OK: {(email or '').strip().lower()}")
    return token, "", {"display": "none"}


# ── AUTH: Logout ──────────────────────────────────────────────────────────────

@app.callback(
    Output("sauth", "data", allow_duplicate=True),
    Input("logout-btn", "n_clicks"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def on_logout(n, token):
    if not n:
        return no_update
    auth_destroy(token)
    print(f"[AUTH] Logout: session destroyed")
    return None




@app.callback(
    Output("stream",    "children"),
    Output("smsgs",     "data"),
    Output("qin",       "value"),
    Output("sconvs",    "data"),
    Output("scid",      "data"),
    Output("slist",     "children"),
    Output("sfeedback", "data",    allow_duplicate=True),
    Output("spending",  "data"),
    Input("sbtn",       "n_clicks"),
    State("qin",        "value"),
    State("smsgs",      "data"),
    State("sconvs",     "data"),
    State("scid",       "data"),
    State("mdd",        "value"),
    State("sfeedback",  "data"),
    State("sdoc_mode",  "data"),
    State("sdoc",       "data"),
    prevent_initial_call=True,
)
def on_send(_, qval, msgs, convs, cid, mval, feedback, doc_mode, doc_data):
    """Immediate response: render user message + thinking bubble, queue pipeline."""
    q = (qval or "").strip()
    if not q:
        return (no_update,) * 8

    msgs  = list(msgs  or [])
    convs = list(convs or [])

    if not cid:
        cid = str(int(time.time()*1000))
        convs.append({"id": cid, "title": q[:42], "count": 0, "msgs": [], "feedback":{}})
        feedback = {}
    conv = next((c for c in convs if c["id"] == cid), None)
    if not conv:
        convs.append({"id": cid, "title": q[:42], "count": 0, "msgs": [], "feedback":{}})
        conv = convs[-1]
    if conv["count"] == 0:
        conv["title"] = q[:42]
    conv["count"] += 1

    msgs.append({"role": "user", "content": q})

    # Build stream: all prior messages + new user bubble + thinking indicator
    feedback = dict(feedback or {})
    stream_children = render_stream(msgs, feedback) + [render_thinking()]

    # Store query context for pipeline callback
    pending = {
        "query":    q,
        "doc_mode": doc_mode,
        "doc_data": doc_data,
        "mval":     mval,
        "ts":       time.time(),
    }

    return (stream_children, msgs, "",
            convs, cid, render_sidebar(convs, cid), feedback, pending)


# ── STEP 2: Slow callback — runs pipeline, replaces thinking bubble ─────────────

@app.callback(
    Output("stream",    "children",  allow_duplicate=True),
    Output("smsgs",     "data",      allow_duplicate=True),
    Output("tbadges",   "children"),
    Output("sconvs",    "data",      allow_duplicate=True),
    Output("slist",     "children",  allow_duplicate=True),
    Output("sfeedback", "data",      allow_duplicate=True),
    Input("spending",   "data"),
    State("smsgs",      "data"),
    State("sconvs",     "data"),
    State("scid",       "data"),
    State("sfeedback",  "data"),
    State("sauth",  "data"),
    prevent_initial_call=True,
)
def on_submit(pending, msgs, convs, cid, feedback, sauth):
    """Pipeline execution: replaces the thinking bubble with the real response."""
    if not pending or not pending.get("query"):
        return (no_update,) * 6

    global MODEL
    q        = pending["query"]
    doc_mode = pending.get("doc_mode", False)
    doc_data = pending.get("doc_data")
    mval     = pending.get("mval")

    if mval:
        MODEL = mval

    msgs  = list(msgs  or [])
    convs = list(convs or [])
    feedback = dict(feedback or {})

    conv = next((c for c in convs if c["id"] == cid), None)

    # ── Branch: Document Search mode vs NOC LangGraph pipeline ───────────────
    if doc_mode and doc_data and doc_data.get("text"):
        res = answer_from_doc(q, doc_data["text"], doc_data.get("name", "Document"))
        bot_msg = {
            "role":       "bot",
            "content":    res["answer"],
            "doc_mode":   True,
            "source_doc": doc_data.get("name", "Document"),
            "timing":     res["timing"],
            "route":      {},
            "viz":        "table",
            "sql":        "",
            "rows":       [],
            "cols":       [],
            "fig":        None,
            "error":      res.get("error", False),
        }
        total  = res["timing"].get("total", 0)
        badges = [
            html.Span("\U0001f4c4  DOC SEARCH",        className="b bdoc"),
            html.Span(doc_data.get("name", "")[:28], className="b bdocname"),
            html.Span(f"\u23f1 {total:.1f}s",          className="b btime"),
        ]

    elif doc_mode and (not doc_data or not doc_data.get("text")):
        bot_msg = {
            "role":       "bot",
            "content":    (
                "\u26a0\ufe0f  Document Search mode is active but no document has been "
                "uploaded yet. Please upload a file using the upload area above, or toggle "
                "off Document Search mode (\U0001f4c4 button) to query the NOC database."
            ),
            "doc_mode":   True,
            "source_doc": "",
            "timing":     {"total": 0},
            "route":      {},
            "viz":        "table",
            "sql":        "",
            "rows":       [],
            "cols":       [],
            "fig":        None,
            "error":      True,
        }
        badges = [html.Span("\u26a0\ufe0f  No Document", className="b berr")]

    else:
        res = pipeline(q)
        bot_msg = {
            "role":    "bot",
            "content": res["answer"],
            "route":   res["route"],
            "viz":     res["viz"],
            "sql":     res["sql"],
            "rows":    res["rows"],
            "cols":    res["cols"],
            "fig":     res["fig"],
            "timing":  res["timing"],
            "error":   bool(res["error"]),
        }
        rk    = res["route"].get("route", "") if isinstance(res["route"], dict) else ""
        col   = RCOL.get(rk, "#6b7280")
        total = res["timing"].get("total", 0)
        badges = [
            html.Span(RLBL.get(rk, rk), className="b br",
                      style={"color": col, "borderColor": col+"44", "background": col+"11"}),
            html.Span(f"{len(res['rows']):,} rows", className="b brows"),
            html.Span(f"\u23f1 {total:.1f}s",       className="b btime"),
        ]
        if res["error"]:
            badges.append(html.Span("ERROR", className="b berr"))

    msgs.append(bot_msg)
    if conv:
        conv["msgs"] = msgs

    # ── Log to TTF_chat_logs ──────────────────────────────────────────────────
    log_session_event(
        user_email   = auth_validate(sauth) or "unknown",
        question     = q,
        llm_response = bot_msg.get("content", ""),
        vote         = "",
        sql_query    = bot_msg.get("sql", ""),
    )

    return (render_stream(msgs, feedback), msgs, badges,
            convs, render_sidebar(convs, cid), feedback)


# ── CSV Download callback ────────────────────────────────────────────────────────

@app.callback(
    Output("csv-dl", "data"),
    Input({"type": "csv-btn", "index": ALL}, "n_clicks"),
    State("smsgs", "data"),
    prevent_initial_call=True,
)
def on_csv_download(n_clicks_list, msgs):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update
    if not any(n and n > 0 for n in (n_clicks_list or [])):
        return no_update

    idx = triggered.get("index")
    if idx is None or not msgs or idx >= len(msgs):
        return no_update

    msg = msgs[idx]
    rows = msg.get("rows", [])
    cols = msg.get("cols", [])
    if not rows or not cols:
        return no_update

    df = pd.DataFrame(rows, columns=cols)
    filename = f"falcon_results_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return dcc.send_data_frame(df.to_csv, filename, index=False)

@app.callback(
    Output("stream",     "children",  allow_duplicate=True),
    Output("smsgs",      "data",      allow_duplicate=True),
    Output("scid",       "data",      allow_duplicate=True),
    Output("sconvs",     "data",      allow_duplicate=True),
    Output("slist",      "children",  allow_duplicate=True),
    Output("tbadges",    "children",  allow_duplicate=True),
    Output("sfeedback",  "data",      allow_duplicate=True),  # FIX: reset feedback on new conv
    Input("newbtn",      "n_clicks"),
    State("sconvs",      "data"),
    prevent_initial_call=True,
)
def on_new_conversation(n, convs):
    if not n:
        return (no_update,) * 7
    convs = list(convs or [])
    cid   = str(int(time.time()*1000))
    convs.append({"id": cid, "title": "New conversation", "count": 0, "msgs": [], "feedback":{}})
    return [make_welcome()], [], cid, convs, render_sidebar(convs, cid), [], {}


@app.callback(
    Output("stream",    "children",  allow_duplicate=True),
    Output("smsgs",     "data",      allow_duplicate=True),
    Output("scid",      "data",      allow_duplicate=True),
    Output("slist",     "children",  allow_duplicate=True),
    Output("tbadges",   "children",  allow_duplicate=True),
    Output("sfeedback", "data",      allow_duplicate=True),  # FIX: restore feedback on conv switch
    Input({"type": "conv", "index": ALL}, "n_clicks"),
    State("sconvs",     "data"),
    State("scid",       "data"),
    prevent_initial_call=True,
)
def on_conv_select(n_clicks_list, convs, current_cid):
    # Find which sidebar item was actually clicked
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return (no_update,) * 6

    clicked_cid = triggered.get("index")
    if not clicked_cid or clicked_cid == current_cid:
        return (no_update,) * 6

    # Guard: only act if a real click happened (not initial render fires)
    if not any(n and n > 0 for n in (n_clicks_list or [])):
        return (no_update,) * 6

    convs = list(convs or [])
    conv  = next((c for c in convs if c["id"] == clicked_cid), None)
    if not conv:
        return (no_update,) * 6

    # Restore this conversation's messages and feedback state
    msgs     = list(conv.get("msgs")     or [])
    feedback = dict(conv.get("feedback") or {})

    return (
        render_stream(msgs, feedback),
        msgs,
        clicked_cid,
        render_sidebar(convs, clicked_cid),
        [],       # clear topbar badges when switching
        feedback, # FIX: write restored feedback back to the store
    )


# ── Upvote: save Q+SQL to 100_Questions.csv ───────────────────────────────────

@app.callback(
    Output("sfeedback", "data",     allow_duplicate=True),
    Output("stream",    "children", allow_duplicate=True),
    Input({"type": "upvote", "index": ALL}, "n_clicks"),
    State("smsgs",     "data"),
    State("sconvs",    "data"),
    State("scid",      "data"),
    State("sfeedback", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def on_upvote(n_clicks_list, msgs, convs, cid, feedback, sauth):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update, no_update
    bot_idx = triggered.get("index")
    if bot_idx is None:
        return no_update, no_update

    # Fix: check the triggered component's actual click value, not all buttons
    # (re-renders reset all n_clicks=0, so any(n>0) always fails after first vote)
    triggered_value = next(
        (t["value"] for t in (ctx.triggered or []) if t.get("value")),
        None,
    )
    if not triggered_value:
        return no_update, no_update

    msgs     = list(msgs or [])
    convs    = list(convs or [])
    feedback = dict(feedback or {})

    if bot_idx >= len(msgs):
        return no_update, no_update

    bot_msg  = msgs[bot_idx]
    user_msg = next((msgs[i] for i in range(bot_idx - 1, -1, -1)
                     if msgs[i]["role"] == "user"), None)
    if not user_msg:
        return no_update, no_update

    q   = user_msg.get("content", "").strip()
    sql = bot_msg.get("sql",      "").strip()

    # Append to 100_Questions.csv ─────────────────────────────────────────────
    try:
        file_exists = os.path.exists(QUESTIONS_CSV)
        with open(QUESTIONS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "sql", "category", "tables"])
            if not file_exists:
                writer.writeheader()
            rk  = bot_msg.get("route", {})
            tbl = rk.get("table", "")              if isinstance(rk, dict) else ""
            cat = RLBL.get(rk.get("route", ""), "General") if isinstance(rk, dict) else "General"
            writer.writerow({"question": q, "sql": sql, "category": cat, "tables": tbl})
        print(f"[UPVOTE] ✓ Saved to {QUESTIONS_CSV}: {q[:60]}")
    except Exception as e:
        print(f"[UPVOTE] CSV write error: {e}")

    feedback[str(bot_idx)] = "up"

    # Persist feedback in conv record for sidebar restore
    conv = next((c for c in convs if c["id"] == cid), None)
    if conv:
        conv["feedback"] = feedback

    # ── Log to TTF_chat_logs ──────────────────────────────────────────────────
    log_session_event(
        user_email   = auth_validate(sauth) or "unknown",
        question     = q,
        llm_response = bot_msg.get("content", ""),
        vote         = "up",
        sql_query    = sql,
    )

    return feedback, render_stream(msgs, feedback)


# ── Downvote: re-run pipeline with retry context ──────────────────────────────

@app.callback(
    Output("sfeedback", "data",     allow_duplicate=True),
    Output("stream",    "children", allow_duplicate=True),
    Output("smsgs",     "data",     allow_duplicate=True),
    Output("sconvs",    "data",     allow_duplicate=True),
    Input({"type": "downvote", "index": ALL}, "n_clicks"),
    State("smsgs",     "data"),
    State("sconvs",    "data"),
    State("scid",      "data"),
    State("sfeedback", "data"),
    State("sauth", "data"),
    prevent_initial_call=True,
)
def on_downvote(n_clicks_list, msgs, convs, cid, feedback, sauth):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update, no_update, no_update, no_update
    bot_idx = triggered.get("index")
    if bot_idx is None:
        return no_update, no_update, no_update, no_update

    # Fix: check the triggered component's actual click value, not all buttons
    # (re-renders reset all n_clicks=0, so any(n>0) always fails after first vote)
    triggered_value = next(
        (t["value"] for t in (ctx.triggered or []) if t.get("value")),
        None,
    )
    if not triggered_value:
        return no_update, no_update, no_update, no_update

    msgs     = list(msgs  or [])
    convs    = list(convs or [])
    feedback = dict(feedback or {})

    if bot_idx >= len(msgs):
        return no_update, no_update, no_update, no_update

    # Find the original user question before this bot response
    user_msg = next((msgs[i] for i in range(bot_idx - 1, -1, -1)
                     if msgs[i]["role"] == "user"), None)
    if not user_msg:
        return no_update, no_update, no_update, no_update

    original_q = user_msg.get("content", "").strip()

    # Inject retry context so the pipeline knows the previous answer was bad
    retry_q = (
        f"{original_q}\n\n"
        f"[SYSTEM NOTE: The previous answer to this question was marked as unhelpful "
        f"by the NOC engineer. Re-analyse carefully — choose the correct table and "
        f"columns, verify the time filter, and generate a more accurate SQL query. "
        f"Do NOT repeat the prior response.]"
    )

    # Mark as downvoted before re-running (shows spinner hint in UI immediately)
    feedback[str(bot_idx)] = "dn"

    # ── Log to TTF_chat_logs ──────────────────────────────────────────────────
    log_session_event(
        user_email   = auth_validate(sauth) or "unknown",
        question     = original_q,
        llm_response = msgs[bot_idx].get("content", "") if bot_idx < len(msgs) else "",
        vote         = "dn",
        sql_query    = msgs[bot_idx].get("sql", "") if bot_idx < len(msgs) else "",
    )

    print(f"[DOWNVOTE] Re-running pipeline for: {original_q[:60]}")
    res = pipeline(retry_q)

    new_bot = {
        "role":    "bot",
        "content": res["answer"],
        "route":   res["route"],
        "viz":     res["viz"],
        "sql":     res["sql"],
        "rows":    res["rows"],
        "cols":    res["cols"],
        "fig":     res["fig"],
        "timing":  res["timing"],
        "error":   bool(res["error"]),
    }

    # Append a visible retry marker + new bot response to the conversation
    msgs.append({"role": "user",  "content": f"↻ Retry: {original_q}"})
    msgs.append(new_bot)

    # Persist updated msgs + feedback in conv record
    conv = next((c for c in convs if c["id"] == cid), None)
    if conv:
        conv["msgs"]     = msgs
        conv["feedback"] = feedback

    return feedback, render_stream(msgs, feedback), msgs, convs


# ══════════════════════════════════════════════════════════════════════════════
#  DOC SEARCH CALLBACKS
#  1. on_doc_toggle  — 📄 button: show/hide upload strip, toggle sdoc_mode
#  2. on_doc_upload  — dcc.Upload: parse file, populate sdoc store
#  3. on_doc_clear   — "✕ Clear" button: wipe document from sdoc store
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("sdoc_mode",  "data",      allow_duplicate=True),
    Output("doc-strip",  "className",                      ),
    Output("docbtn",     "className",                      ),
    Output("qin",        "placeholder",                    ),
    Input("docbtn",      "n_clicks"),
    State("sdoc_mode",   "data"),
    prevent_initial_call=True,
)
def on_doc_toggle(n, currently_on):
    """Toggle Document Search mode on/off."""
    if not n:
        return no_update, no_update, no_update, no_update
    new_mode = not bool(currently_on)
    strip_cls  = "doc-strip"             if new_mode else "doc-strip doc-strip-hidden"
    btn_cls    = "docbtn docbtn-active"  if new_mode else "docbtn"
    placeholder = (
        "Ask a question about the uploaded document…"
        if new_mode else
        "Ask about faults, syslogs, sessions, RCA…"
    )
    return new_mode, strip_cls, btn_cls, placeholder


@app.callback(
    Output("sdoc",        "data",     allow_duplicate=True),
    Output("doc-info-row","children",                     ),
    Input("doc-upload",   "contents"),
    State("doc-upload",   "filename"),
    prevent_initial_call=True,
)
def on_doc_upload(contents, filename):
    """Parse the uploaded file and store extracted text in sdoc."""
    if not contents or not filename:
        return no_update, no_update

    text, err = parse_uploaded_doc(contents, filename)

    if err:
        info = html.Span(f"❌  {err}", className="doc-err")
        return None, info

    word_count = len(text.split())
    info = html.Div([
        html.Span(f"✅  {filename}  ·  {word_count:,} words", className="doc-loaded"),
        html.Button("✕  Clear", id="doc-clear-btn", n_clicks=0, className="doc-clear"),
    ], className="doc-info-row")

    doc_data = {"text": text, "name": filename, "words": word_count}
    print(f"[DOC_UPLOAD] '{filename}' parsed — {word_count:,} words")
    return doc_data, info


@app.callback(
    Output("sdoc",         "data",     allow_duplicate=True),
    Output("doc-info-row", "children", allow_duplicate=True),
    Input("doc-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_doc_clear(n):
    """Clear the loaded document from the store."""
    if not n:
        return no_update, no_update
    return None, html.Span()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # build_rag_embeddings()   # Uncomment once to pre-build .npy embedding cache
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  FALCON  ·  Vi NOC  ·  LangGraph 9-Node Pipeline  (HAR V17)    ║
╠══════════════════════════════════════════════════════════════════╣
║  http://0.0.0.0:8026                                            ║
║  Model  : {MODEL:<52} ║
║  DB     : {PG['host']}:{PG['port']}/{PG['dbname']:<36} ║
║  RAG    : {len(rag.entries)} entries                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline: Router → TableID → RAG → CoT(DSPy) → GoT(SQL-Beam)  ║
║            → Validate → Execute → Explain  (+LLM Fallback)      ║
╠══════════════════════════════════════════════════════════════════╣
║  Frontend: 4 cap-cards · 10 chips · wordcloud/bar/line/pie/tbl  ║
║  No FastAPI — pure Dash on port 8026                            ║
╚══════════════════════════════════════════════════════════════════╝""")
    # SECURITY FIX (Obs #6 — Absence of Secure Transport Controls): the app
    # previously always served plain HTTP. If SSL_CERTFILE/SSL_KEYFILE are
    # provided, the Dash dev server terminates TLS directly; otherwise this
    # MUST run behind a TLS-terminating reverse proxy (nginx/ALB/etc.) —
    # never expose it over plain HTTP directly, even on an intranet.
    _ssl_cert = os.environ.get("SSL_CERTFILE")
    _ssl_key  = os.environ.get("SSL_KEYFILE")
    _ssl_ctx  = (_ssl_cert, _ssl_key) if _ssl_cert and _ssl_key else None
    if _ssl_ctx is None:
        print("[SECURITY WARNING] SSL_CERTFILE/SSL_KEYFILE not set — serving "
              "plain HTTP. Deploy behind a TLS-terminating reverse proxy.")
    app.run(host="0.0.0.0", port=18000, debug=False, dev_tools_hot_reload=False, ssl_context=_ssl_ctx)
