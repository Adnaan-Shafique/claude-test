"""
GRE – SQM Global Roamers Excellence  (LangGraph 9-Node Pipeline + Dash UI)
════════════════════════════════════════════════════════════════════════════
Version: V2.4

Pipeline:
  Router → TableID → RAG → CoT(DSPy) → GoT(SQL-Beam) →
  SQLValidation → SQLExecution → Explanation  (+LLM Fallback)

Dash Frontend:
  5-badge topbar · sidebar conversations · cap-cards · chips ·
  wordcloud / bar / line / pie / table charts · SQL viewer · Login

Active Tables:
  public.view_inbound_failure_rate        — inbound failure rates by country/module
  public.view_outbound_failure_rate       — outbound failure rates by operator/module
  public.analytics_uniq_roamers_inbound  — unique in-roamer counts by country/circle
  public.analytics_uniq_roamers          — unique out-roamer counts by country/circle
  public.analytics_inbound_response_data_hr  — hourly inbound error/response codes
  public.analytics_outbound_response_data_hr — hourly outbound error/response codes
  public.ir_wsms_logs        — Welcome SMS dispatch log
  public.ir_steering_master  — Traffic steering master config
  public.ir_ntr_logs         — Network Transaction Routing events

Routes (GRE-specific):
  footprint_q   → unique roamer counts by country/circle
  performance_q → inbound/outbound failure rates & threshold analysis
  error_q       → granular error/response code distribution
  circle_q      → circle-level comparison & partner analytics
  trend_q       → time-series / hourly / dip analysis
  sms_q         → Welcome SMS dispatch, delivery status, MSISDN/IMSI history
  steering_q    → roaming partners, preferred/forbidden, LBTR/SRDC, MCC/MNC
  ntr_q         → NTR attach events, SS7/Diameter ULR, visited network, IMSI trace

Run: python GRE_AGENT_V2_4_clean.py  →  http://0.0.0.0:18003
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
    validate_upload, SecurityAuditLogger, DDL_DML_KEYWORDS,
)

print("✅ All imports OK")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEBUG = os.environ.get("DEBUG", "true").lower() == "true"

# ── LLM / GPU Proxy ───────────────────────────────────────────────────────────
GPU_URL = os.environ.get("GPU_PROXY_URL", "http://127.0.0.1:8071/v1/infer")
GPU_KEY = required_secret("GPU_API_KEY")          # no plaintext secret in source (Obs: secrets-in-source)
GPU_TO  = int(os.environ.get("GPU_TIMEOUT", "180"))
MODEL   = os.environ.get("DEFAULT_MODEL",  "mistral")

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
# SECURITY FIX (secrets-in-source + Obs #2/#5/#24 "session_user = postgres"
# finding): the app previously connected as the Postgres SUPERUSER with a
# hardcoded plaintext password. Both are removed — PGUSER must be set to a
# scoped, least-privilege, read-only role and PGPASSWORD is read strictly
# from the environment (see required_secret() in security_common.py).
PG = dict(
    user    =os.environ.get("PGUSER",     "gre_readonly"),
    password=required_secret("PGPASSWORD"),
    host    =os.environ.get("PGHOST",     "10.19.71.234"),
    port    =os.environ.get("PGPORT",     "5432"),
    dbname  =os.environ.get("PGDATABASE", "ir"),
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
sec_logger = SecurityAuditLogger(schema="tt_gre_schema", app_name="talk-to-GRE")

# ── RAG Paths ─────────────────────────────────────────────────────────────────
EMBEDDER_PATH   = os.environ.get("EMBEDDER_PATH", "/data/harish/eli-embedding-small-1/")
RAG_BASE_DIR    = os.environ.get("RAG_BASE_DIR",  "/srv/gre-assistant/rag")
EMBED_CACHE_DIR = os.path.join(RAG_BASE_DIR, "embeddings_cache")
QUESTIONS_CSV   = os.path.join(RAG_BASE_DIR, "gre_questions.csv")
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.environ.get("LOG_DIR", "/srv/gre-assistant/logs")
LOG_FILE = os.path.join(LOG_DIR, "gre_agent_v2_4.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)
        
# ── DB Logging ───────────────────────────────────────────────────────────────────
def log_session_event(user_email: str, question: str, llm_response: str, vote: str = ""):
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO chatbot_details (timestamp, user_email, question, llm_response, vote)
                VALUES (:timestamp, :user_email, :question, :llm_response, :vote)
            """), {
                "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user_email":   user_email,
                "question":     question,
                "llm_response": llm_response[:800],
                "vote":         vote,
            })
    except Exception as e:
        print(f"[LOG_DB] Failed to log session event: {e}")

# ── Table allow-list ──────────────────────────────────────────────────────────
VALID_TABLES = {
    "public.analytics_uniq_roamers_inbound",
    "public.analytics_uniq_roamers",
    "public.analytics_inbound_response_data_hr",
    "public.analytics_outbound_response_data_hr",
    "public.view_inbound_failure_rate",
    "public.view_outbound_failure_rate",
    "public.ir_wsms_logs",
    "public.ir_steering_master",
    "public.ir_ntr_logs",
}

# Semantic shortcuts
INBOUND_TABLE          = "public.view_inbound_failure_rate"
OUTBOUND_TABLE         = "public.view_outbound_failure_rate"
ROAMERS_INBOUND_TABLE  = "public.analytics_uniq_roamers_inbound"
ROAMERS_OUTBOUND_TABLE = "public.analytics_uniq_roamers"
RESP_INBOUND_TABLE     = "public.analytics_inbound_response_data_hr"
RESP_OUTBOUND_TABLE    = "public.analytics_outbound_response_data_hr"
WSMS_TABLE             = "public.ir_wsms_logs"
STEERING_TABLE         = "public.ir_steering_master"
NTR_TABLE              = "public.ir_ntr_logs"

RAG_DOCS: Dict[str, str] = {
    "public.analytics_uniq_roamers_inbound":  os.path.join(RAG_BASE_DIR, "uniq_roamers_inbound.txt"),
    "public.analytics_uniq_roamers":          os.path.join(RAG_BASE_DIR, "uniq_roamers_outbound.txt"),
    "public.analytics_inbound_response_data_hr": os.path.join(RAG_BASE_DIR, "inbound_response_hr.txt"),
    "public.analytics_outbound_response_data_hr":os.path.join(RAG_BASE_DIR, "outbound_response_hr.txt"),
    "public.view_inbound_failure_rate":         os.path.join(RAG_BASE_DIR, "view_inbound_failure_rate.txt"),
    "public.view_outbound_failure_rate":        os.path.join(RAG_BASE_DIR, "view_outbound_failure_rate.txt"),
    "public.ir_wsms_logs":                      os.path.join(RAG_BASE_DIR, "ir_wsms_logs.txt"),
    "public.ir_steering_master":                os.path.join(RAG_BASE_DIR, "ir_steering_master.txt"),
    "public.ir_ntr_logs":                       os.path.join(RAG_BASE_DIR, "ir_ntr_logs.txt"),
}

TABLE_TO_NPY: Dict[str, str] = {
    "public.analytics_uniq_roamers_inbound":     "uniq_roamers_inbound.npy",
    "public.analytics_uniq_roamers":             "uniq_roamers_outbound.npy",
    "public.analytics_inbound_response_data_hr": "inbound_response_hr.npy",
    "public.analytics_outbound_response_data_hr":"outbound_response_hr.npy",
    "public.view_inbound_failure_rate":          "view_inbound_failure_rate.npy",
    "public.view_outbound_failure_rate":         "view_outbound_failure_rate.npy",
    "public.ir_wsms_logs":                       "ir_wsms_logs.npy",
    "public.ir_steering_master":                 "ir_steering_master.npy",
    "public.ir_ntr_logs":                        "ir_ntr_logs.npy",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1.5 — AUTH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ALLOW_VODAFONE_DOMAIN: bool = False

# ALLOWED_EMAILS: List[str] = [
#     "mohammed.shafique@vodafoneidea.com",
#     "harish.saragadam@vodafoneidea.com",
# ]

def get_allowed_emails() -> List[str]:
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT username FROM user_table")).fetchall()
            return [row[0].strip().lower() for row in rows if row[0]]
    except Exception as e:
        print(f"[AUTH] Failed to fetch allowed emails: {e}")
        return []

# SECURITY FIX (secrets-in-source): real account passwords used to be
# hardcoded here. Allow-listed emails and credentials are now loaded from
# the DB at startup; if the DB is unreachable both are empty and login
# fails closed rather than falling back to a hardcoded password. user_table
# should hold a proper per-user password hash column for a
# production-grade rollout — flagged separately since it's a schema
# change outside this application file.
ALLOWED_EMAILS: List[str] = get_allowed_emails()


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def get_user_credentials() -> Dict[str, str]:
    if engine is None:
        return {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT username FROM user_table")).fetchall()
            default_pw = required_secret("GRE_DEFAULT_USER_PASSWORD")
            return {row[0].strip().lower(): _hash(default_pw) for row in rows if row[0]}
    except Exception as e:
        print(f"[AUTH] Failed to fetch credentials: {e}")
        return {}


USER_CREDENTIALS: Dict[str, str] = get_user_credentials()

VODAFONE_DOMAIN_PASSWORD: str = _hash(required_secret("VODAFONE_DOMAIN_PASSWORD"))

# ── Session store: single active session per user + TTL/idle expiry ────────────
# (fixes Obs #15 — Concurrent Login)
_session_store = SessionStore(ttl_seconds=8 * 3600, idle_timeout_seconds=2 * 3600)


def auth_login(email: str, password: str) -> Tuple[bool, str]:
    email = (email or "").strip().lower()
    if not email or not password:
        return False, "Email and password are required."
    is_vi = email.endswith("@vodafoneidea.com")
    if ALLOW_VODAFONE_DOMAIN:
        if not is_vi:
            sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "domain_not_allowed"})
            return False, "Access is restricted to @vodafoneidea.com accounts."
    else:
        if email not in ALLOWED_EMAILS:
            sec_logger.log_event("auth_failure", "warning", user_email=email, detail={"reason": "not_on_allow_list"})
            return False, "Your email is not on the authorised access list."
    hashed   = _hash(password)
    expected = USER_CREDENTIALS.get(email)
    if expected is None:
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
    return _session_store.validate(token)

def auth_destroy(token: str):
    _session_store.destroy(token)

# ══════════════════════════════════════════════════════════════════════════════
#  NO-LOGIN MODE — auth stubs (login page is bypassed entirely)
# ══════════════════════════════════════════════════════════════════════════════

# def auth_validate(token: str) -> Optional[str]:
#     """Always returns 'anonymous' — login is disabled."""
#     return "anonymous"

# def auth_login(email: str, password: str) -> Tuple[bool, str]:
#     """No-op — always succeeds."""
#     return True, ""

# def auth_create_session(email: str) -> str:
#     """No-op — no session management needed."""
#     return "no-session"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COMPLETE SCHEMA DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_DOCS: Dict[str, str] = {

"public.view_inbound_failure_rate": """\
INBOUND FAILURE RATE VIEW — public.view_inbound_failure_rate
Context: Pre-aggregated monitoring view for tracking in-roamer failure rates by country and module.
         ONE ROW = A unique metric snapshot per Module, Country, and Timestamp.
         Use for: inbound failure rate monitoring, country-level threshold alerts, KPI trends.

TIME COLUMNS: time (interval timestamp)

COLUMNS:
  module_name         TEXT          Source system/feature identifier
                                    e.g. "Diameter Failure LU Rate per Country",
                                         "MAP LU Failure Rate per Country"
  country             TEXT          Country name of the in-roaming partner
                                    e.g. "Afghanistan", "United Arab Emirates"
  failure_percentage  DOUBLE PREC   Calculated failure rate (0-100)   e.g. 5.25
  total_volume        INTEGER       Total requests processed           e.g. 10000
  failure_volume      DOUBLE PREC   Count of failed requests           e.g. 525.0
  time                TIMESTAMP     Interval timestamp                 e.g. 2026-04-01 14:00

CRITICAL QUERY RULES:
  Use 'module_name' to isolate specific service failures (ILIKE for matching)
  Filter by 'country' for country-specific failure analysis
  To find high-failure alerts: WHERE failure_percentage > 10.0
  For hourly trend: SELECT time, failure_percentage WHERE time > NOW() - INTERVAL 'N hours'
  For top failing countries: ORDER BY failure_percentage DESC LIMIT 10

COMMON QUESTION PATTERNS:
  "Top failing countries"    → SELECT country, failure_percentage ORDER BY failure_percentage DESC
  "Inbound failure trend"    → SELECT time, failure_percentage WHERE module_name ILIKE '%Diameter%'
  "Hourly failure rate"      → SELECT time, failure_percentage WHERE time > NOW() - INTERVAL '1 hour'
  "Failure rate for UAE"     → SELECT time, failure_percentage WHERE country = 'United Arab Emirates'
""",

"public.view_outbound_failure_rate": """\
OUTBOUND FAILURE RATE VIEW — public.view_outbound_failure_rate
Context: Pre-aggregated monitoring view for tracking out-roamer failure rates by operator and module.
         ONE ROW = A unique metric snapshot per Module, Operator, and Timestamp.
         Use for: outbound failure rate monitoring, carrier-specific outage detection, KPI trends.

TIME COLUMNS: time (interval timestamp)

COLUMNS:
  module_name         TEXT          Source system/feature identifier
                                    e.g. "Diameter Failure LU Rate per Operator",
                                         "MAP LU Failure Rate per Operator"
  operater            TEXT          Network carrier name (NOTE: column spelling is 'operater')
                                    e.g. "ETISALAT (AE)", "du (AE)"
  failure_percentage  DOUBLE PREC   Calculated failure rate (0-100)   e.g. 5.25
  total_volume        INTEGER       Total requests processed           e.g. 10000
  failure_volume      DOUBLE PREC   Count of failed requests           e.g. 525.0
  time                TIMESTAMP     Interval timestamp                 e.g. 2026-04-01 14:00

CRITICAL QUERY RULES:
  Use 'module_name' to isolate specific service failures (ILIKE for matching)
  Filter by 'operater' (exact column name — NOT 'operator') for carrier-specific analysis
  To find high-failure alerts: WHERE failure_percentage > 10.0

COMMON QUESTION PATTERNS:
  "Top failing operators"    → SELECT operater, failure_percentage ORDER BY failure_percentage DESC
  "Outbound failure trend"   → SELECT time, failure_percentage WHERE module_name ILIKE '%Diameter%'
  "Volume vs failure"        → SELECT time, total_volume, failure_volume WHERE module_name ILIKE '%X%'
  "Total failed requests"    → SELECT SUM(failure_volume) WHERE operater ILIKE '%ETISALAT%'
""",

"public.analytics_uniq_roamers_inbound": """\
ANALYTICS UNIQUE ROAMERS INBOUND — public.analytics_uniq_roamers_inbound
Conext: Tracks unique IN-roamer counts segmented by country, partner, and circle.
         ONE ROW = Unique in-roamer count for a specific Country/Partner/Circle combination.
         Use for: in-roaming volumes, top countries, country/partner comparisons.

UNIQUE INDEX: ON (time, country_code, country_partner, circle, circle_provider)
TIME COLUMNS: time (Timestamp of data capture)

COLUMNS:
  module_name      TEXT      e.g. "Diameter Uniq Roamers"
  time             TIMESTAMP e.g. "2026-03-26 00:00:00"
  country_code     TEXT      Country identifier        e.g. "US", "UK", "IN", "AE"
  country_partner  TEXT      Roaming partner name      e.g. "AT&T", "Vodafone", "CHINA UNICOM GSM"
  circle           TEXT      Home/Visited circle name  e.g. "UPE", "KEL", "MUM"
  circle_provider  TEXT      Network provider          e.g. "IDEA", "VODAFONE"
  value            FLOAT8    Count of unique roamers   e.g. 450.0

CRITICAL QUERY RULES:
  For total in-roamers: SUM(value) WHERE country_code = '<code>'
  For PAN India: SUM(value) GROUP BY country_code
  For top countries: ORDER BY SUM(value) DESC LIMIT 10
  JOIN ON (time, country_code, circle) for cross-table roamer analysis.
  Use ILIKE for country_code or country_partner matching.
""",

"public.analytics_uniq_roamers": """\
ANALYTICS UNIQUE ROAMERS OUTBOUND — public.analytics_uniq_roamers
Context: Tracks unique OUT-roamer counts segmented by country, partner, and circle.
         ONE ROW = Unique out-roamer count for a specific Country/Partner/Circle combination.
         Use for: out-roaming volumes, top destinations, partner analysis.

UNIQUE INDEX: ON (time, country_code, country_partner, circle, circle_provider)
TIME COLUMNS: time (Timestamp of data capture)

COLUMNS:
  module_name      TEXT      e.g. "Diameter Uniq Roamers"
  time             TIMESTAMP e.g. "2026-03-26 00:00:00"
  country_code     TEXT      Country identifier        e.g. "US", "UK", "IN", "AE"
  country_partner  TEXT      Roaming partner name      e.g. "AT&T", "Vodafone"
  circle           TEXT      Home/Visited circle name  e.g. "UPE", "KEL", "MUM"
  circle_provider  TEXT      Network provider          e.g. "IDEA", "VODAFONE"
  value            FLOAT8    Count of unique roamers   e.g. 450.0

CRITICAL QUERY RULES:
  For total out-roamers: SUM(value) WHERE country_code = '<code>'
  For top out-roaming countries: ORDER BY SUM(value) DESC LIMIT 10
  For preferred partners: ORDER BY SUM(value) DESC WHERE country_code = 'UK'
  Use ILIKE for country_code or country_partner matching.
""",


"public.analytics_inbound_response_data_hr": """\
ANALYTICS RESPONSE DATA INBOUND (HOURLY) — public.analytics_inbound_response_data_hr
Context: Tracks granular hourly response metrics (MO level) for in-roamer traffic modules.
         ONE ROW = ONE response metric per Managed Object (MO) per module per timestamp.
         Use for: inbound error code distribution, IMSI troubleshooting, failure cause analysis.

UNIQUE INDEX: ON (module_name, time, mo)

COLUMNS:
  module_name  TEXT      e.g. "Diameter LU Failure Result Code Distribution",
                         "GTPv2 Create Session Failure Cause Distribution",
                         "SMS MT Failure Cause Distribution",
                         "Diameter Uniq Roamer count"
  time         TIMESTAMP e.g. "2026-03-26 13:00:00"
  mo           TEXT      Managed Object / Error code  e.g. "vodafone (IT)",
                         "CHINA UNICOM GSM (CN)",
                         "3002 DIAMETER_UNABLE_TO_DELIVER",
                         "3003 DIAMETER_REALM_NOT_SERVED"
  value        FLOAT8    High precision metric value  e.g. 950.0, 1227.0, 16613.0
  volume       INT4      Transaction count (can be NULL)  e.g. 49883

CRITICAL QUERY RULES:
  ALWAYS USE ILIKE for module matching: WHERE module_name ILIKE '%Distribution%'
  ALWAYS USE ILIKE for mo matching: WHERE mo ILIKE '%AE%'
  NEVER USE SUM — each row is the value at that timestamp; use AVG for aggregation
  NEVER GROUP BY without AVG when aggregating
  For today's metrics: WHERE DATE(time) = CURRENT_DATE
  For latest metrics: WHERE time = (SELECT MAX(time) FROM public.analytics_inbound_response_data_hr)

COMMON QUESTION PATTERNS:
  "Top inbound GTPv2 errors"   → SELECT mo, AVG(value) GROUP BY mo ORDER BY AVG(value) DESC LIMIT 20
  "In last 1 hour"             → WHERE time > NOW() - INTERVAL '1 hour'
  "Hourly trend"               → DATE_TRUNC('hour', time), AVG(value) GROUP BY 1 ORDER BY 1
  "Latest 10 records"          → ORDER BY time DESC LIMIT 10
""",

"public.analytics_outbound_response_data_hr": """\
ANALYTICS RESPONSE DATA OUTBOUND (HOURLY) — public.analytics_outbound_response_data_hr
Context: Tracks granular hourly response metrics (MO level) for out-roamer traffic modules.
         ONE ROW = ONE response metric per Managed Object (MO) per module per timestamp.
         Use for: outbound error analysis, operator-level error breakdown, failure cause distribution.

UNIQUE INDEX: ON (module_name, time, mo)

COLUMNS:
  module_name  TEXT      e.g. "Diameter Failure LU Rate per Operator",
                         "GTPv2 Create Session Failure Cause Distribution",
                         "Diameter LU Failure Result Code Distribution",
                         "Diameter Uniq Roamer count"
  time         TIMESTAMP e.g. "2026-03-26 13:00:00"
  mo           TEXT      Managed Object / Granular source  e.g. "vodafone (IT)",
                         "CHINA UNICOM GSM (CN)",
                         "3002 DIAMETER_UNABLE_TO_DELIVER"
  value        FLOAT8    High precision metric value  e.g. 950.0, 1227.0
  volume       INT4      Transaction count (can be NULL)

CRITICAL QUERY RULES:
  ALWAYS USE ILIKE for module matching: WHERE module_name ILIKE '%Distribution%'
  ALWAYS USE ILIKE for mo matching: WHERE mo ILIKE '%CN%'
  NEVER USE SUM — each row is the value at that timestamp; use AVG for aggregation
  NEVER GROUP BY without AVG when aggregating
  For today: WHERE DATE(time) = CURRENT_DATE

COMMON QUESTION PATTERNS:
  "Top outbound GTPv2 errors"  → SELECT mo, AVG(value) GROUP BY mo ORDER BY AVG(value) DESC LIMIT 10
  "In last 1 hour"             → WHERE time > NOW() - INTERVAL '1 hour'
  "Hourly trend"               → DATE_TRUNC('hour', time), AVG(value) GROUP BY 1 ORDER BY 1
  "Latest 10 records"          → ORDER BY time DESC LIMIT 10
""",

"public.ir_wsms_logs": """\
IR WSMS LOGS TABLE — public.ir_wsms_logs
Context: Auditing Welcome SMS (WSMS) history, dispatch status, and content.
         ONE ROW = ONE specific SMS dispatch attempt to a roaming customer.
         Use for: SMS delivery audits, failed SMS tracking, welcome pack analytics.

INDEX: "idx_ir_sms_logs" ON (date, time, msisdn, imsi)

COLUMNS:
  date            DATE        Date of SMS dispatch                e.g. 2026-02-16
  time            TIME(2)     Exact dispatch time (sec precision) e.g. 23:48:56.00
  msisdn          BIGINT      Customer Mobile Number (no +)       e.g. 919922308947
  imsi            BIGINT      International Mobile Sub ID         e.g. 404223004432085
  message_status  VARCHAR     Human-readable delivery status      e.g. "Dispatched to SMSC"
  message_code    VARCHAR     System status code                  e.g. "MesgSent"
  message_id      VARCHAR     ID of the dispatched pack/campaign  e.g. "Embassy_All_Dec2024"
  message_text    TEXT        The actual SMS body sent to user    e.g. "Hello! For any..."
  failure_code    VARCHAR     Error code (0 or NULL for success)  e.g. "0"

CRITICAL QUERY RULES:
  Use MSISDN/IMSI for specific customer tracking: WHERE msisdn = 919922308947
  For daily SMS volumes: SELECT COUNT(*) GROUP BY date
  To check failures: WHERE failure_code != '0'
  Join with pack table: JOIN ir_roaming_pack ON message_id = pack_name

COMMON QUESTION PATTERNS:
  "Check SMS sent to IMSI X"  -> SELECT * WHERE imsi = X ORDER BY date DESC, time DESC
  "Count failed SMS today"    -> SELECT COUNT(*) WHERE failure_code != '0' AND date = CURRENT_DATE
  "Last 5 messages for pack"  -> SELECT message_text WHERE message_id = 'PackName' LIMIT 5
  "Check delivery status"     -> SELECT message_status WHERE msisdn = X
""",

"public.ir_steering_master": """\
IR STEERING MASTER TABLE — public.ir_steering_master
Context: Master reference for traffic steering logic, partner priorities, and network configurations.
         ONE ROW = unique network operator per country.
         Use for: steering compliance checks, preferred partner lookup, MCC/MNC resolution.

UNIQUE CONSTRAINT: "unique_mcc_mnc" ON (mcc, mnc)
INDEX: "idx_ir_steering_master" ON (country, roaming_partner)

COLUMNS:
  country         VARCHAR     Country of the roaming partner      e.g. "USA", "Germany"
  roaming_partner VARCHAR     Official name of the operator       e.g. "AT&T", "T-Mobile"
  cos             VARCHAR     Class of Service                     e.g. "ZONEDEFAULT"
  mechanism       VARCHAR     Steering method applied             e.g. "LBTR", "SRDC"
  lbtr_distribution      VARCHAR     LBTR distribution — "F" (Forbidden), "P" (Preferred), or blank
  srdc_distribution       VARCHAR     SRDC distribution — numeric % for preferred (adds to 100), 0 for forbidden
  mcc             VARCHAR     Mobile Country Code (3 digits)      e.g. "310"
  mnc             VARCHAR     Mobile Network Code (2-3 digits)    e.g. "410"
  network_type    VARCHAR     Network preference type             e.g. "Forbidden", "Preferred", "Less-Preferred"

COMPLIANCE & ACCESS RULES (MUST FOLLOW):
  1. NEVER query across ALL countries in a single SQL (e.g. no WHERE clause missing country filter).
     EVERY query on this table MUST filter by a single country: WHERE country ILIKE '%<country>%'
  2. DEFAULT behaviour when listing partners for a country:
     - Always show Preferred partners with full details (roaming_partner, mechanism, lbtr_distribution/srdc_distribution, mcc, mnc).
     - DO NOT list Forbidden partners by default; only say "remaining partners are forbidden".
  3. If user explicitly asks for Forbidden list: return ONLY top 5, ordered by roaming_partner. LIMIT 5.
  4. MCC/MNC must always be returned TOGETHER. Never SELECT only mcc or only mnc — always SELECT mcc, mnc together.
  5. Do NOT generate queries that return all rows without a country filter (no full-table dumps).

CRITICAL QUERY RULES:
  Use 'mechanism' to determine why traffic is routed to a specific partner.
  For LBTR: lbtr_distribution = 'P' means Preferred, 'F' means Forbidden, blank means neutral.
  For SRDC: srdc_distribution > 0 means Preferred (% distribution), srdc_distribution = 0 means Forbidden.
  Search by Country: WHERE country ILIKE '%<name>%' — MANDATORY on every query.

COMMON QUESTION PATTERNS:
  "List all partners in [Country]"     -> SELECT roaming_partner, network_type, mechanism WHERE country ILIKE '%X%' AND network_type = 'Preferred' ORDER BY roaming_partner
  "Show preferred partners for UK"     -> SELECT roaming_partner, mechanism, lbtr_distribution, srdc_distribution, mcc, mnc WHERE country ILIKE '%UK%' AND network_type = 'Preferred'
  "What steering mechanism for AT&T"   -> SELECT country, roaming_partner, mechanism, lbtr_distribution, srdc_distribution, network_type WHERE roaming_partner ILIKE '%AT&T%'
  "Get MCC MNC for [operator]"         -> SELECT mcc, mnc, roaming_partner, country WHERE roaming_partner ILIKE '%<name>%'
  "Show forbidden partners in Germany" -> SELECT roaming_partner, mcc, mnc WHERE country ILIKE '%Germany%' AND network_type = 'Forbidden' ORDER BY roaming_partner LIMIT 5
""",

"public.ir_ntr_logs": """\
IR NTR LOGS TABLE — public.ir_ntr_logs
Context: Auditing Network Transaction Routing (NTR) events and steering decisions.
         ONE ROW = ONE specific protocol-level transaction (Update Location, Auth, etc.).
         Use for: attachment failure diagnosis, steering efficiency analysis, IMSI path tracing.

INDEX: "idx_ir_ntr_logs" ON (date, time, imsi)

COLUMNS:
  date              DATE        Date of network event               e.g. 2026-02-14
  time              TIME(2)     Exact event time (sec precision)    e.g. 21:13:33.00
  imsi              BIGINT      Customer IMSI (International ID)    e.g. 404277280153279
  txn_status_desc   VARCHAR     Protocol status description         e.g. "SS7_EVT_ALLWD"
  op_code           INTEGER     Numeric Operation Code              e.g. 2001
  op_code_desc      VARCHAR     Human-readable Operation name       e.g. "DIAMETER ULR"
  reason_code       INTEGER     Logic result code                   e.g. 83
  reason_code_desc  TEXT        Detail on why action was taken      e.g. "RDC allowed..."
  visited_nw_name   VARCHAR     Latched partner network name        e.g. "Telefonica Germany"
  visited_zone      VARCHAR     Country or Roaming Zone             e.g. "Germany"
  tr_mechanism      VARCHAR     Steering logic applied              e.g. "SRDC", "LBTR"

CRITICAL QUERY RULES:
  Identify why a user failed to attach: WHERE txn_status_desc != 'SS7_EVT_ALLWD'
  Monitor specific protocol trends: WHERE op_code_desc ILIKE '%DIAMETER%'
  Check steering efficiency: SELECT tr_mechanism, COUNT(*) GROUP BY 1
  Trace IMSI path: WHERE imsi = 404... ORDER BY date DESC, time DESC

COMMON QUESTION PATTERNS:
  "Why did IMSI X fail in Germany?" -> SELECT reason_code_desc WHERE imsi = X AND visited_zone ILIKE '%Germany%'
  "List all visitors in UK today"   -> SELECT DISTINCT imsi WHERE visited_zone ILIKE '%UK%' AND date = CURRENT_DATE
  "Count SRDC vs LBTR hits"         -> SELECT tr_mechanism, COUNT(*) GROUP BY 1
  "Last network latch for user"     -> SELECT visited_nw_name, visited_zone ORDER BY date DESC, time DESC LIMIT 1
""",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BASELINE SQL EXAMPLES (RAG SEED)
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_EXAMPLES = [
    # ── Category 1: Global Footprint & Traffic Oversight ──────────────────────
    {
        "question": "What is the total in-roamer count in RAJ today?",
        "sql": (
            "SELECT time, SUM(value) "
            "FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name ILIKE '%Uniq Roamers' AND CIRCLE = 'RAJ' AND DATE(time) = CURRENT_DATE "
            "GROUP BY time ORDER BY time DESC LIMIT 12"
        ),
        "category": "Footprint",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Show me the Top 10 countries by in-roamer count today",
        "sql": (
            "WITH latest AS ( "
                "SELECT country_code, country_partner, value, time "
                "FROM public.analytics_uniq_roamers_inbound "
                "WHERE module_name = 'Diameter Uniq Roamers' "
                "AND time = (SELECT MAX(time) FROM analytics_uniq_roamers_inbound)) "
            "select country_code, sum(value) from latest group by country_code order by 2 desc limit 10;"
        ),
        "category": "Footprint",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Top 10 out-roaming countries today",
        "sql": (
            "WITH latest AS ( "
                "SELECT country_code, country_partner, value, time "
                "FROM public.analytics_uniq_roamers "
                "WHERE module_name = 'Diameter Uniq Roamers' "
                "AND time = (SELECT MAX(time) FROM analytics_uniq_roamers)) "
            "select country_code, sum(value) from latest group by country_code order by 2 desc limit 10;"
        ),
        "category": "Footprint",
        "tables": ["public.analytics_uniq_roamers"],
    },
    {
        "question": "Total inroamer count for United States of America today",
        "sql": (
            "WITH latest AS ( "
                "SELECT country_code, country_partner, value, time "
                "FROM public.analytics_uniq_roamers "
                "WHERE module_name = 'Diameter Uniq Roamers' "
                "AND time = (SELECT MAX(time) FROM analytics_uniq_roamers)) "
            "select country_code, sum(value) from latest group by country_code having country_code = 'US';"
        ),
        "category": "Footprint",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    # ── Category 2: Network Performance & Diagnostics ─────────────────────────
    {
        "question": "Show inbound diameter failure rate in the last 4 hours for UAE",
        "sql": (
            "SELECT time, failure_percentage "
            "FROM public.view_inbound_failure_rate "
            "WHERE module_name ILIKE '%Diameter Failure LU Rate%' "
            "AND time > NOW() - INTERVAL '4 hours' "
            "AND country = 'United Arab Emirates';"
        ),
        "category": "Performance",
        "tables": ["public.view_inbound_failure_rate"],
    },
    {
        "question": "What are the top errors for GTPv2 inbound today?",
        "sql": (
            "WITH agg AS ( SELECT mo, time, SUM(value) AS total_errors, SUM(volume) AS total_volume,  "
            "ROW_NUMBER() OVER ( PARTITION BY mo ORDER BY time DESC ) AS rn "
            "FROM public.analytics_inbound_response_data_hr WHERE module_name ILIKE '%GTPv2%' AND time >= CURRENT_DATE AND time < CURRENT_DATE + INTERVAL '1 day' GROUP BY mo, time ) "
            "SELECT time, mo, total_errors FROM agg WHERE rn = 1 ORDER BY total_errors DESC LIMIT 20;"
        ),
        "category": "Error",
        "tables": ["public.analytics_inbound_response_data_hr"],
    },
    # ── Category 3: Customer Troubleshooting (examples use _hr tables) ────────
    {
        "question": "Why did users fail to attach — show top failure reasons inbound",
        "sql": (
            "SELECT mo AS reason_desc, AVG(value) AS avg_failure_count "
            "FROM public.analytics_inbound_response_data_hr "
            "WHERE module_name ILIKE '%Failure%' AND DATE(time) = CURRENT_DATE "
            "GROUP BY mo ORDER BY avg_failure_count DESC LIMIT 20;"
        ),
        "category": "Troubleshooting",
        "tables": ["public.analytics_inbound_response_data_hr"],
    },
    {
        "question": "Show top Diameter error codes inbound today",
        "sql": (
            "SELECT mo AS error_code, AVG(value) AS avg_error_count "
            "FROM public.analytics_inbound_response_data_hr "
            "WHERE module_name ILIKE '%Diameter%' AND DATE(time) = CURRENT_DATE "
            "GROUP BY mo ORDER BY avg_error_count DESC LIMIT 20;"
        ),
        "category": "Error",
        "tables": ["public.analytics_inbound_response_data_hr"],
    },
    {
        "question": "Show GTPv2 outbound failure trend last 4 hours",
        "sql": (
            "SELECT time, failure_percentage "
            "FROM public.view_outbound_failure_rate "
            "WHERE module_name ILIKE '%GTPv2%' "
            "AND time > NOW() - INTERVAL '4 hours' "
            "ORDER BY time ASC;"
        ),
        "category": "Performance",
        "tables": ["public.view_outbound_failure_rate"],
    },
    # ── Category 5: Advanced / Dip Analysis ──────────────────────────────────
    {
        "question": "Compare in-roamer volume for MUM and DEL circles today",
        "sql": (
            "SELECT circle, time, SUM(value) "
            "FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name = 'Diameter Uniq Roamers' AND (CIRCLE = 'MUM' OR CIRCLE = 'DEL') AND DATE(time) = CURRENT_DATE "
            "GROUP BY circle, time ORDER BY time DESC LIMIT 12"
        ),
        "category": "Circle",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Who are the preferred roaming partners in UK by in-roamer volume?",
        "sql": (
            "SELECT country_partner, SUM(value) AS roamer_count "
            "FROM public.analytics_uniq_roamers_inbound "
            "WHERE country_code ILIKE '%UK%' AND DATE(time) = CURRENT_DATE "
            "GROUP BY country_partner ORDER BY roamer_count DESC LIMIT 10;"
        ),
        "category": "Circle",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    # ── Category 5: Advanced / Dip Analysis ──────────────────────────────────
    # {
    #     "question": "Top 5 out-roaming countries for 4G today",
    #     "sql": (
    #         "SELECT country_code, SUM(value) AS roamer_count "
    #         "FROM public.analytics_uniq_roamers "
    #         "WHERE module_name ILIKE '%4G%' AND DATE(time) = CURRENT_DATE "
    #         "GROUP BY country_code ORDER BY roamer_count DESC LIMIT 5;"
    #     ),
    #     "category": "Trend",
    #     "tables": ["public.analytics_uniq_roamers"],
    # },
    {
        "question": "Top 5 in-roaming operators PAN India today",
        "sql": (
            "SELECT country_partner, SUM(value) AS roamer_count "
            "FROM public.analytics_uniq_roamers_inbound "
            "WHERE DATE(time) = CURRENT_DATE "
            "GROUP BY country_partner ORDER BY roamer_count DESC LIMIT 5;"
        ),
        "category": "Trend",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Show countries where in-roamer count dipped compared to yesterday",
        "sql": (
            "WITH today AS ( SELECT country_code, SUM(value) AS today_count FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name = 'Diameter Uniq Roamers' AND time >= CURRENT_DATE AND time <= NOW() GROUP BY country_code), "
            "yesterday AS ( SELECT country_code, SUM(value) AS yesterday_count FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name = 'Diameter Uniq Roamers' AND time >= CURRENT_DATE - INTERVAL '1 day' AND time <= NOW() - INTERVAL '1 day' GROUP BY country_code) "
            "SELECT t.country_code, t.today_count, y.yesterday_count, ROUND( (100.0 * (t.today_count - y.yesterday_count) / NULLIF(y.yesterday_count, 0))::numeric, 2 ) AS pct_change "
            "FROM today t JOIN yesterday y ON t.country_code = y.country_code WHERE (t.today_count - y.yesterday_count) / NULLIF(y.yesterday_count, 0) < -0.70 ORDER BY pct_change ASC LIMIT 20; "
        ),
        "category": "Trend",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Show countries with 55% in-roamer dip today vs yesterday",
        "sql": (
            "WITH today AS ( SELECT country_code, SUM(value) AS today_count FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name = 'Diameter Uniq Roamers' AND time >= CURRENT_DATE AND time <= NOW() GROUP BY country_code), "
            "yesterday AS ( SELECT country_code, SUM(value) AS yesterday_count FROM public.analytics_uniq_roamers_inbound "
            "WHERE module_name = 'Diameter Uniq Roamers' AND time >= CURRENT_DATE - INTERVAL '1 day' AND time <= NOW() - INTERVAL '1 day' GROUP BY country_code) "
            "SELECT t.country_code, t.today_count, y.yesterday_count, ROUND( (100.0 * (t.today_count - y.yesterday_count) / NULLIF(y.yesterday_count, 0))::numeric, 2 ) AS pct_change "
            "FROM today t JOIN yesterday y ON t.country_code = y.country_code WHERE (t.today_count - y.yesterday_count) / NULLIF(y.yesterday_count, 0) < -0.55 ORDER BY pct_change ASC LIMIT 20; "
        ),
        "category": "Trend",
        "tables": ["public.analytics_uniq_roamers_inbound"],
    },
    {
        "question": "Show hourly inbound failure rate trend today",
        "sql": (
            "SELECT time, module_name, failure_percentage "
            "FROM public.view_inbound_failure_rate "
            "WHERE DATE(time) = CURRENT_DATE "
            "ORDER BY time ASC;"
        ),
        "category": "Trend",
        "tables": ["public.view_inbound_failure_rate"],
    },

    # ── Category 6: Welcome SMS (WSMS) ────────────────────────────────────────
    {
        "question": "How many SMS were dispatched today?",
        "sql": (
            "SELECT COUNT(*) AS total_sms_dispatched "
            "FROM public.ir_wsms_logs "
            "WHERE date = CURRENT_DATE;"
        ),
        "category": "SMS",
        "tables": ["public.ir_wsms_logs"],
    },
    {
        "question": "Count failed SMS today",
        "sql": (
            "SELECT COUNT(*) AS failed_sms "
            "FROM public.ir_wsms_logs "
            "WHERE failure_code != '0' AND date = CURRENT_DATE;"
        ),
        "category": "SMS",
        "tables": ["public.ir_wsms_logs"],
    },
    {
        "question": "Check all SMS sent to MSISDN 919922308947",
        "sql": (
            "SELECT date, time, message_id, message_status, message_code, failure_code "
            "FROM public.ir_wsms_logs "
            "WHERE msisdn = 919922308947 "
            "ORDER BY date DESC, time DESC LIMIT 20;"
        ),
        "category": "SMS",
        "tables": ["public.ir_wsms_logs"],
    },
    {
        "question": "Show SMS delivery status breakdown today",
        "sql": (
            "SELECT message_status, message_code, COUNT(*) AS count "
            "FROM public.ir_wsms_logs "
            "WHERE date = CURRENT_DATE "
            "GROUP BY message_status, message_code ORDER BY count DESC;"
        ),
        "category": "SMS",
        "tables": ["public.ir_wsms_logs"],
    },
    {
        "question": "List all SMS sent for pack Embassy_All_Dec2024",
        "sql": (
            "SELECT date, time, msisdn, imsi, message_status, failure_code "
            "FROM public.ir_wsms_logs "
            "WHERE message_id ILIKE '%Embassy_All_Dec2024%' "
            "ORDER BY date DESC, time DESC LIMIT 50;"
        ),
        "category": "SMS",
        "tables": ["public.ir_wsms_logs"],
    },

    # ── Category 7: Steering Master ───────────────────────────────────────────
    {
        "question": "List all roaming partners in USA",
        "sql": (
            "SELECT roaming_partner, cos, network_type "
            "FROM public.ir_steering_master "
            "WHERE country ILIKE '%United%States%America%' "
            "ORDER BY CASE network_type WHEN 'Preferred' THEN 1 WHEN 'Less-Preferred' THEN 2 ELSE 3, roaming_partner "
            "LIMIT 500;"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },
    {
        "question": "List all preferred roaming partners in USA",
        "sql": (
            "SELECT roaming_partner, cos, network_type "
            "FROM public.ir_steering_master "
            "WHERE country ILIKE '%United%States%America%' "
            "AND network_type ILIKE '%Preferred%' "
            "ORDER BY CASE network_type WHEN 'Preferred' THEN 1 ELSE 2, roaming_partner "
            "LIMIT 500;"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },
    {
        "question": "Show all steering configurations for Germany",
        "sql": (
            "SELECT roaming_partner, mechanism, lbtr_distribution, srdc_distribution, network_type, mcc, mnc "
            "FROM public.ir_steering_master "
            "WHERE country ILIKE '%Germany%' "
            "ORDER BY network_type, roaming_partner;"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },
    {
        "question": "List all forbidden roaming partners in Germany",
        "sql": (
            "SELECT roaming_partner, cos, network_type "
            "FROM public.ir_steering_master "
            "WHERE country ILIKE '%Germany%' AND network_type ILIKE '%Forbidden%' "
            "ORDER BY roaming_partner LIMIT 5;"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },
    {
        "question": "What is the steering mechanism and distribution for AT&T?",
        "sql": (
            "SELECT country, roaming_partner, mechanism, lbtr_distribution, srdc_distribution, network_type "
            "FROM public.ir_steering_master "
            "WHERE roaming_partner ILIKE '%AT&T%';"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },
    {
        "question": "Show all non-preferred roaming partners in UK",
        "sql": (
            "SELECT roaming_partner, cos, network_type "
            "FROM public.ir_steering_master "
            "WHERE country ILIKE '%United%Kingdom%' AND network_type ILIKE '%Forbidden%' "
            "ORDER BY roaming_partner LIMIT 5;"
        ),
        "category": "Steering",
        "tables": ["public.ir_steering_master"],
    },

    # ── Category 8: NTR Logs ──────────────────────────────────────────────────
    {
        "question": "Why did IMSI 404277280153279 fail to attach?",
        "sql": (
            "SELECT date, time, txn_status_desc, op_code_desc, reason_code_desc, "
            "visited_nw_name, visited_zone, tr_mechanism "
            "FROM public.ir_ntr_logs "
            "WHERE imsi = 404277280153279 AND txn_status_desc != 'SS7_EVT_ALLWD' "
            "ORDER BY date DESC, time DESC LIMIT 20;"
        ),
        "category": "NTR",
        "tables": ["public.ir_ntr_logs"],
    },
    {
        "question": "List all visitors in Germany today",
        "sql": (
            "SELECT DISTINCT imsi, visited_nw_name, tr_mechanism "
            "FROM public.ir_ntr_logs "
            "WHERE visited_zone ILIKE '%Germany%' AND date = CURRENT_DATE;"
        ),
        "category": "NTR",
        "tables": ["public.ir_ntr_logs"],
    },
    {
        "question": "Count SRDC vs LBTR steering hits today",
        "sql": (
            "SELECT tr_mechanism, COUNT(*) AS hit_count "
            "FROM public.ir_ntr_logs "
            "WHERE date = CURRENT_DATE "
            "GROUP BY tr_mechanism ORDER BY hit_count DESC;"
        ),
        "category": "NTR",
        "tables": ["public.ir_ntr_logs"],
    },
    {
        "question": "Show the last network latch for IMSI 404277280153279",
        "sql": (
            "SELECT date, time, visited_nw_name, visited_zone, tr_mechanism, op_code_desc "
            "FROM public.ir_ntr_logs "
            "WHERE imsi = 404277280153279 "
            "ORDER BY date DESC, time DESC LIMIT 1;"
        ),
        "category": "NTR",
        "tables": ["public.ir_ntr_logs"],
    },
    {
        "question": "Show failed NTR transactions today with reason codes",
        "sql": (
            "SELECT txn_status_desc, reason_code, reason_code_desc, COUNT(*) AS count "
            "FROM public.ir_ntr_logs "
            "WHERE txn_status_desc != 'SS7_EVT_ALLWD' AND date = CURRENT_DATE "
            "GROUP BY txn_status_desc, reason_code, reason_code_desc "
            "ORDER BY count DESC LIMIT 20;"
        ),
        "category": "NTR",
        "tables": ["public.ir_ntr_logs"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3.5 — STATE TYPE
# ══════════════════════════════════════════════════════════════════════════════

class GREAgentState(TypedDict):
    user_query:       str
    model_name:       str
    route:            str
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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3.6 — GoT HELPER CLASSES (DSPy)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoTNode:
    sql:         str
    params:      Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    score:       float = 0.0


class GRESQLSignature(dspy.Signature):
    """Generate a correct PostgreSQL SELECT query for a GRE roaming analytics question."""
    question = dspy.InputField(desc="Natural language question about roaming data")
    schema   = dspy.InputField(desc="Relevant table schema and column details")
    plan     = dspy.InputField(desc="Chain-of-thought query plan")
    sql      = dspy.OutputField(desc="Complete PostgreSQL SELECT query")


class GraphOfThoughtRefiner:
    """Scores and selects the best SQL candidate from the beam."""
    def run(self, candidates: List[Dict], query: str, schema: str) -> Optional[GoTNode]:
        if not candidates:
            return None
        best: Optional[GoTNode] = None
        best_score = -1.0
        for c in candidates:
            sql = c.get("sql", "")
            if not sql:
                continue
            score = self._score(sql, query, schema)
            if score > best_score:
                best_score = score
                best = GoTNode(sql=sql, params=c.get("params", {}),
                               explanation=c.get("explanation", ""), score=score)
            if score >= GOT_EARLY_EXIT:
                break
        return best

    def _score(self, sql: str, query: str, schema: str) -> float:
        s = 0.0
        sl = sql.lower()
        ql = query.lower()
        # Structural correctness
        if re.search(r"\b(select|with)\b", sl):
            s += 2.0
        if sl.count("select") == 1:
            s += 1.0
        if re.search(r"\bfrom\s+public\.", sl):
            s += 2.0
        if sql.strip().endswith(";"):
            s += 0.5
        # GRE domain rewards
        for kw in ["sum(value)", "avg(value)", "group by", "order by", "date_trunc", "ilike"]:
            if kw in sl:
                s += 0.5
        for kw in ["roam", "country", "circle", "module_name", "volume", "value"]:
            if kw in sl:
                s += 0.3
        # Punish forbidden keywords
        for bad in DDL_DML_KEYWORDS:
            if bad in sl:
                s -= 5.0
        # Penalise cross-table JOINs beyond allowed
        if sl.count("join") > 1:
            s -= 1.5
        return s


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3.7 — RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGEntry:
    id:        str
    text:      str
    category:  str
    tables:    List[str]
    sql:       str = ""
    question:  str = ""
    doc_type:  str = "example"
    embedding: Optional[Any] = None


class GRERAG:
    def __init__(self):
        self.entries: List[RAGEntry] = []
        self._table_emb_cache: Dict[str, Tuple] = {}
        self.embedder = None
        self._init_embedder()
        self._ingest_baseline()
        self._load_csv()
        self._load_docs()
        self._load_table_embeddings()

    def _init_embedder(self):
        try:
            self.embedder = SentenceTransformer(EMBEDDER_PATH)
            print(f"[RAG] Embedder loaded from {EMBEDDER_PATH}")
        except Exception as e:
            print(f"[RAG] Embedder unavailable ({e}) — keyword fallback active")

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
        
        if best_score < 0.35: # confidence threshold of 0.35
            print(f"[TABLE_EMB] Best match: {best_table} not considered as (score={best_score:.4f}) is less than 0.35")
            return None
        return best_table


rag = GRERAG()


def build_rag_embeddings():
    try:
        _emb = SentenceTransformer(EMBEDDER_PATH)
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


# SECURITY FIX: see falcon_agent.py for the detailed rationale — the
# previous validator was a substring keyword blocklist that never
# restricted which SQL functions could be called (letting current_setting/
# version/inet_server_addr/session_user/pg_*() leak backend/config/TLS/
# version data), never stripped comments, and never rejected stacked
# ";"-separated statements. Replaced with the hardened allow-list validator.
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

# --- Country code → country name normalisation ---

COUNTRY_CODE_ALIASES = {
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "USA": "United States",
    "US": "United States",
    "UAE": "United Arab Emirates",
    "AE": "United Arab Emirates",
    "IN": "India",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "SG": "Singapore",
    "JP": "Japan",
    "AU": "Australia",
    "CA": "Canada",
}

try:
    import pycountry
except ImportError:
    pycountry = None


def country_name_from_code(code):
    if code is None or pd.isna(code):
        return code

    s = str(code).strip()
    if not s:
        return s

    key = s.upper().replace(".", "")
    if key in COUNTRY_CODE_ALIASES:
        return COUNTRY_CODE_ALIASES[key]

    if pycountry is not None:
        try:
            if len(key) == 2:
                c = pycountry.countries.get(alpha_2=key)
                if c:
                    return c.name
            elif len(key) == 3:
                c = pycountry.countries.get(alpha_3=key)
                if c:
                    return c.name
        except Exception:
            pass

    return s


def normalize_country_columns(df):
    if df is None or df.empty:
        return df

    out = df.copy()
    rename_map = {}

    for col in list(out.columns):
        col_l = str(col).strip().lower()
        if col_l == "country_code" or col_l.endswith("_country_code"):
            out[col] = out[col].map(country_name_from_code)
            if col_l == "country_code":
                rename_map[col] = "country_name"

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


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


_TIME_PATTERN = re.compile(
    r"\b(?:last|past|previous|in\s+the\s+last|in\s+past)\s+"
    r"(\d+)\s*"
    r"(hour|hr|hours|hrs|minute|min|minutes|mins|day|days|week|weeks|month|months)s?\b",
    re.IGNORECASE,
)
_UNIT_CANONICAL = {
    "hour":"hour","hr":"hour","hours":"hours","hrs":"hours",
    "minute":"minute","min":"minute","minutes":"minutes","mins":"minutes",
    "day":"day","days":"days","week":"week","weeks":"weeks",
    "month":"month","months":"months",
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
            rid  = data.get("request_id", "")
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
    
    # Inject current date so that the LLM always has temporal context
    current_date_str = datetime.now().strftime("%Y-%m-%d %A") # e.g. "2026-03-31 Tuesday"
    date_context = f"\nCURRENT DATE: {current_date_str}\n"
    system_with_date = system_prompt.strip() + date_context
    wrapped = (
        f"SYSTEM:\n{system_with_date}\n\n"
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
#  SECTION 6 — DOCUMENT SEARCH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def parse_uploaded_doc(contents: str, filename: str) -> Tuple[str, str]:
    """Parse base64-encoded document contents. Returns (text, error)."""
    import base64
    try:
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        # SECURITY FIX (Obs #11 — Improper File Type Validation): the
        # previous "else" branch accepted and UTF-8-decoded ANY file
        # regardless of extension (images, executables, anything) despite
        # the UI claiming CSV/TXT/JSON-only support. validate_upload()
        # enforces an extension allow-list AND checks the actual byte
        # signature/content, rejecting disguised binaries outright.
        ok, err = validate_upload(filename, decoded)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ok and ext in ("pdf", "docx"):
            ok, err = False, "This app's document search only supports .txt/.csv/.json/.tsv/.md/.log"
        if not ok:
            sec_logger.log_event("upload_rejected", "warning", detail={"filename": filename, "reason": err})
            return "", f"Upload rejected: {err}"

        if ext == "txt":
            return decoded.decode("utf-8", errors="replace"), ""
        elif ext == "csv":
            import io
            df = pd.read_csv(io.BytesIO(decoded))
            return df.to_string(), ""
        elif ext == "json":
            data = json.loads(decoded)
            return json.dumps(data, indent=2), ""
        else:
            # validate_upload() already restricted us to a known-safe text
            # extension at this point (tsv/md/log) — pdf/docx aren't handled
            # by this app's doc-search feature, so decode as plain text.
            return decoded.decode("utf-8", errors="replace"), ""
    except Exception as e:
        return "", f"Parse error: {str(e)[:100]}"


def run_doc_search(question: str, doc_text: str, doc_name: str) -> Dict[str, Any]:
    """Answer a question based on uploaded document content."""
    t0 = time.time()

    blocked_category = classify_blocked_request(question)
    if blocked_category:
        sec_logger.log_event("content_blocked", "warning",
                              detail={"stage": "doc_search_input", "category": blocked_category, "question": question[:500]})
        return {
            "answer": refusal_for(blocked_category),
            "route": "doc_search", "viz": "table",
            "sql": "", "rows": [], "cols": [],
            "fig": None, "error": False,
            "timing": {"total": round(time.time() - t0, 2)},
            "source_doc": doc_name,
        }

    sys_p = f"""\
{SAFETY_PREAMBLE}
You are a GRE Roaming Analytics assistant. Answer the user's question based ONLY on the provided document.
Be concise (3-5 sentences), cite specific values, and highlight any anomalies or key findings.
Never fabricate configuration, credentials, or values not literally present in the document.
Plain text only — no JSON, no markdown headers."""
    user_p = f"Document: {doc_name}\n\n{doc_text[:4000]}\n\nQuestion: {question}"
    answer = call_llm(sys_p, user_p, max_new_tokens=400,
                      decode_config=EXPLANATION_DECODE, stage_label="DOC_SEARCH").strip()
    answer = sanitize_output(answer, system_prompt_fragments=_SYSTEM_PROMPT_FRAGMENTS)
    elapsed = round(time.time() - t0, 2)
    return {
        "answer": answer or "I could not find a relevant answer in the document.",
        "route": "doc_search", "viz": "table",
        "sql": "", "rows": [], "cols": [],
        "fig": None,
        "error": not bool(answer),
        "timing": {"total": elapsed},
        "source_doc": doc_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — TABLE SUMMARIES (for router context)
# ══════════════════════════════════════════════════════════════════════════════

TABLE_SUMMARIES: Dict[str, str] = {
    "public.view_inbound_failure_rate":
        "Pre-aggregated inbound failure rates by country & module (failure_percentage, total_volume) — for inbound KPI monitoring & threshold alerts",
    "public.view_outbound_failure_rate":
        "Pre-aggregated outbound failure rates by operator & module (failure_percentage, total_volume) — for carrier-specific outage detection",
    "public.analytics_uniq_roamers_inbound":
        "Unique IN-roamer counts by country, partner, circle — for global footprint & in-roaming volumes",
    "public.analytics_uniq_roamers":
        "Unique OUT-roamer counts by country, partner, circle — for out-roaming volumes & partner analysis",
    "public.analytics_inbound_response_data_hr":
        "Hourly inbound response/error codes per MO — for IMSI troubleshooting & inbound error cause distribution (use AVG not SUM)",
    "public.analytics_outbound_response_data_hr":
        "Hourly outbound response/error codes per MO — for outbound error analysis & operator-level error distribution (use AVG not SUM)",
    "public.ir_wsms_logs":
        "Welcome SMS dispatch log (MSISDN/IMSI level) — for SMS delivery audits, failure tracking & pack analytics",
    "public.ir_steering_master":
        "Traffic steering master config (partner/MCC/MNC/mechanism) — for steering compliance & preferred or forbidden or all roaming partners lookup",
    "public.ir_ntr_logs":
        "Network Transaction Routing events (IMSI/protocol level) — for attachment failure diagnosis & steering efficiency",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_tbl_block = "\n".join(
    f"  {t.split('.')[-1]:50s} → {s}"
    for t, s in TABLE_SUMMARIES.items()
)

ROUTER_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are the GRE (Global Roamers Excellence) query routing agent for a telecom roaming analytics platform.

AVAILABLE TABLES:
{_tbl_block}

ROUTES:
  footprint_q   → global roamer counts, top countries, out/in-roaming volumes
  performance_q → inbound/outbound failure rates
  error_q       → granular error/response code distribution, failure reason analysis, IMSI lookup
  circle_q      → circle-level comparison, partner analysis, silent users
  trend_q       → time-series trend, hourly analysis, dip analysis (% change)
  sms_q         → Welcome SMS dispatch, delivery status, failed SMS, MSISDN/IMSI SMS history
  steering_q    → roaming partners, preferred/forbidden partners, LBTR/SRDC, MCC/MNC lookup
  ntr_q         → NTR attach events, SS7/Diameter ULR transactions, visited network, IMSI path trace

ROUTING RULES (apply in order):
  1. "SMS" / "WSMS" / "welcome message" / "dispatch" / "MSISDN" / "pack" → sms_q
  2. "steering" / "LBTR" / "SRDC" / "MCC" / "MNC" / "forbidden" / "preferred partner" → steering_q
  3. "NTR" / "latch" / "visited zone" / "SS7" / "ULR" / "attach" / "transaction routing" → ntr_q
  4. "IMSI" / "blacklist" / "reason" / "why failed" / "attach fail"    → error_q
  5. "dip" / "70%" / "% change" / "yesterday" / "hourly trend"   → trend_q
  6. "silent" / "compare" / "circle"      → circle_q
  7. "error" / "GTP" / "Diameter error" / "response code"         → error_q
  8. "threshold" / "breach" / "failure rate" / "Diameter trend"   → performance_q
  9. "roamer count" / "top countries" / "in-roaming" / "footprint"→ footprint_q

Return STRICT JSON only (no other text):
{{"route": "<route>", "confidence": "High|Medium|Low", "reasoning": "<one sentence>"}}
""".strip()


TABLE_ID_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a table identification expert for the GRE (Global Roamers Excellence) analytics database.
Identify the SINGLE best table to answer the query.

TABLE CAPABILITIES:
{_tbl_block}

ROUTE-TO-TABLE MAPPING (preferred — one table per route):
  footprint_q   → public.analytics_uniq_roamers_inbound  (or analytics_uniq_roamers for OUT)
  performance_q → public.view_inbound_failure_rate  (or view_outbound_failure_rate for OUT)
  error_q       → public.analytics_inbound_response_data_hr  (or analytics_outbound_response_data_hr for OUT)
  circle_q      → public.analytics_uniq_roamers_inbound  (or analytics_uniq_roamers)
  trend_q       → public.analytics_uniq_roamers_inbound  (or view_inbound_failure_rate for KPI trends)
  sms_q         → public.ir_wsms_logs
  steering_q    → public.ir_steering_master
  ntr_q         → public.ir_ntr_logs

DISAMBIGUATION:
  "inbound" / "in-roaming" / "in-roamers"         → use inbound tables
  "outbound" / "out-roaming" / "out-roamers"       → use outbound tables
  "error code" / "response" / "cause" / "reason"   → use analytics_inbound_response_data_hr or _outbound_hr
  "failure rate" / "failure %" / "KPI" / "breach"  → use view_inbound_failure_rate or view_outbound_failure_rate
  "Diameter" / "GTP" error distribution            → use analytics_inbound_response_data_hr
  "SMS" / "WSMS" / "message" / "MSISDN"            → use ir_wsms_logs
  "steering" / "LBTR" / "SRDC" / "MCC/MNC"        → use ir_steering_master
  "NTR" / "latch" / "visited" / "SS7" / "ULR"     → use ir_ntr_logs

Rules:
  - Always return exactly ONE table — avoid unnecessary JOINs when possible
  - Default to inbound unless the question explicitly mentions "outbound" / "out-roaming"

Return STRICT JSON only:
{{"tables": ["public.table1"],
  "primary_table": "<the one table>",
  "reasoning": "<why this table>"}}
""".strip()


SQL_RULES = f"""\
{SAFETY_PREAMBLE}
POSTGRESQL SQL GENERATION RULES (MUST FOLLOW ALL):
 1. SELECT only — never INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE
 2. FULLY QUALIFIED table names: public.<tablename>
 3. Use ONLY columns from the schema context provided
 4. LIMIT 500 for detail queries; no LIMIT for pure aggregates (GROUP BY only)
 5. Time filters: NOW() - INTERVAL '...' or CURRENT_DATE or DATE(time) = CURRENT_DATE
 6. String matching: ILIKE for case-insensitive (module_name, country, operater, mo, circle)
 7. For roamer counts: SUM(value) — value is FLOAT8 for analytics_uniq_roamers* tables
 8. For failure rate views (view_inbound_failure_rate / view_outbound_failure_rate):
    - Columns are: module_name, country (inbound) or operater (outbound), failure_percentage, total_volume, failure_volume, time
    - Use failure_percentage for rate queries — do NOT use AVG(value) on these views
    - NOTE: outbound view column is 'operater' (not 'operator') — exact spelling required
 9. For response data tables (analytics_inbound_response_data_hr / analytics_outbound_response_data_hr):
    - NEVER USE SUM — each row is already a discrete value at that timestamp
    - Use AVG(value) for aggregation across time periods
    - NEVER GROUP BY without AVG when aggregating value
10. For hourly trends: DATE_TRUNC('hour', time) GROUP BY 1 ORDER BY 1
11. End every query with a semicolon
12. For dip analysis: compare today vs yesterday using subqueries or CTEs
13. STEERING — "list all partners in <country>":
    SELECT roaming_partner, cos, network_type FROM public.ir_steering_master
    WHERE country ILIKE '%<country>%'
    ORDER BY CASE network_type WHEN 'Preferred' THEN 1 WHEN 'Less-Preferred' THEN 2 ELSE 3 END, roaming_partner
    LIMIT 500;
    The LIMIT for Forbidden rows must be enforced via a UNION:
      (SELECT roaming_partner, cos, network_type ... WHERE network_type IN ('Preferred','Less-Preferred') ORDER BY ...)
      UNION ALL
      (SELECT roaming_partner, cos, network_type ... WHERE network_type ILIKE '%Forbidden%' ORDER BY roaming_partner LIMIT 5);
14. STEERING — "list preferred/less-preferred partners in <country>":
    SELECT roaming_partner, cos, network_type FROM public.ir_steering_master
    WHERE country ILIKE '%<country>%' AND network_type ILIKE '%Preferred%'
    ORDER BY CASE network_type WHEN 'Preferred' THEN 1 ELSE 2 END, roaming_partner;
15. STEERING — "list forbidden / non-preferred partners in <country>":
    SELECT roaming_partner, cos, network_type FROM public.ir_steering_master
    WHERE country ILIKE '%<country>%' AND network_type ILIKE '%Forbidden%'
    ORDER BY roaming_partner LIMIT 5;
"""


EXPLANATION_SYSTEM = f"""\
{SAFETY_PREAMBLE}
You are a Senior Roaming Analytics Specialist for a major telecom operator (Vi — Vodafone Idea) GRE team.
Explain SQL query results clearly and concisely for roaming operations engineers.

Guidelines:
  • 3–5 sentences maximum
  • Highlight notable roamer counts, anomalies, top countries, partner volumes, error spikes
  • Use domain terms: in-roaming, out-roaming, circle, country partner, Diameter, GTPv2, IMSI
  • Mention the time window if this was a time-based query
  • If result is empty: say so clearly and suggest a possible reason (e.g. no data yet for today)
  • Do NOT repeat the SQL query text
  • Ground every statement strictly in the query result data — never invent
    counts, IDs, CVEs, credentials, or facts not present in the result set,
    and never state that a data-changing action occurred
  • Do not infer risk, suspicion, or intent from a person's race, ethnicity,
    or nationality if such data appears in results
  • End with one actionable insight if data warrants it

Abbrevations info:
  • LBTR - List-Based Traffic Redirection
  • SRDC - Subscriber-based Ratio Distribution Control
  
STEERING COMPLIANCE RULES (apply when route=steering_q):
  • When explaining results for "list all partners", describe counts per network_type
    (e.g. "X are Preferred, Y are Less-Preferred") but NEVER enumerate all Forbidden partner names.
  • When the result contains Forbidden partners, say:
    "<name1>, <name2>, <name3> are among the forbidden partners, to name a few.
     The full list cannot be disclosed due to compliance policy."
    Use ONLY the partner names present in the returned table rows — do NOT infer or add others.
  • NEVER state the total count of Forbidden partners. Only describe what is shown in the result.
  • For preferred/less-preferred queries: enumerate all partners freely — no restriction.

Plain text only — no JSON, no markdown headers, no bullet points."""

# Distinctive fragments used by sanitize_output() to detect system-prompt
# leakage in a model response (e.g. via translation/roleplay jailbreaks —
# this is exactly how the full system prompt was extracted in testing).
_SYSTEM_PROMPT_FRAGMENTS = [
    "GRE (Global Roamers Excellence) query routing agent",
    "ROUTE-TO-TABLE MAPPING",
    "POSTGRESQL SQL GENERATION RULES",
    "Senior Roaming Analytics Specialist for a major telecom operator",
    "STEERING COMPLIANCE RULES",
    "SAFETY & SCOPE RULES (non-negotiable",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — KEYWORD MAPS & ROUTE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

VALID_ROUTES = {"footprint_q", "performance_q", "error_q", "circle_q", "trend_q",
                "sms_q", "steering_q", "ntr_q"}

ROUTE_TABLE_DEFAULTS = {
    "footprint_q":   [ROAMERS_INBOUND_TABLE],
    "performance_q": [INBOUND_TABLE],
    "error_q":       [RESP_INBOUND_TABLE],
    "circle_q":      [ROAMERS_INBOUND_TABLE],
    "trend_q":       [ROAMERS_INBOUND_TABLE],
    "sms_q":         [WSMS_TABLE],
    "steering_q":    [STEERING_TABLE],
    "ntr_q":         [NTR_TABLE],
}

KW_TABLE_MAP = {
    # Inbound roamers
    "in-roam":            ROAMERS_INBOUND_TABLE,
    "in roam":            ROAMERS_INBOUND_TABLE,
    "in-roaming":         ROAMERS_INBOUND_TABLE,
    "inbound roam":       ROAMERS_INBOUND_TABLE,
    "roamer count":       ROAMERS_INBOUND_TABLE,
    "top countries":      ROAMERS_INBOUND_TABLE,
    "footprint":          ROAMERS_INBOUND_TABLE,
    "pan india":          ROAMERS_INBOUND_TABLE,
    # Outbound roamers
    "out-roam":           ROAMERS_OUTBOUND_TABLE,
    "out roam":           ROAMERS_OUTBOUND_TABLE,
    "out-roaming":        ROAMERS_OUTBOUND_TABLE,
    "outbound roam":      ROAMERS_OUTBOUND_TABLE,
    "silent":             ROAMERS_OUTBOUND_TABLE,
    # Inbound failure rate (view_inbound_failure_rate)
    "failure rate":       INBOUND_TABLE,
    "failure %":          INBOUND_TABLE,
    "threshold":          INBOUND_TABLE,
    "breach":             INBOUND_TABLE,
    "inbound kpi":        INBOUND_TABLE,
    "kpi":                INBOUND_TABLE,
    "inbound failure":    INBOUND_TABLE,
    # Outbound failure rate (view_outbound_failure_rate)
    "outbound kpi":       OUTBOUND_TABLE,
    "outbound failure":   OUTBOUND_TABLE,
    "outbound trend":     OUTBOUND_TABLE,
    "operator failure":   OUTBOUND_TABLE,
    # Inbound response/error (analytics_inbound_response_data_hr)
    "error code":         RESP_INBOUND_TABLE,
    "error":              RESP_INBOUND_TABLE,
    "response code":      RESP_INBOUND_TABLE,
    "diameter":           RESP_INBOUND_TABLE,
    "gtp":                RESP_INBOUND_TABLE,
    "gtpv1":              RESP_INBOUND_TABLE,
    "gtpv2":              RESP_INBOUND_TABLE,
    "imsi":               RESP_INBOUND_TABLE,
    "blacklist":          RESP_INBOUND_TABLE,
    "reason":             RESP_INBOUND_TABLE,
    "attach fail":        RESP_INBOUND_TABLE,
    "cause distribution": RESP_INBOUND_TABLE,
    "failure cause":      RESP_INBOUND_TABLE,
    "result code":        RESP_INBOUND_TABLE,
    # Outbound response/error (analytics_outbound_response_data_hr)
    "outbound error":     RESP_OUTBOUND_TABLE,
    "outbound response":  RESP_OUTBOUND_TABLE,
    # Trend / generic (roamers table for volume trends; failure rate view for KPI hourly)
    "dip":                ROAMERS_INBOUND_TABLE,
    "trend":              ROAMERS_INBOUND_TABLE,
    "hourly":             INBOUND_TABLE,
    "circle":             ROAMERS_INBOUND_TABLE,
    "country":            ROAMERS_INBOUND_TABLE,
    # Welcome SMS (WSMS)
    "sms":                WSMS_TABLE,
    "wsms":               WSMS_TABLE,
    "welcome sms":        WSMS_TABLE,
    "welcome message":    WSMS_TABLE,
    "dispatch":           WSMS_TABLE,
    "msisdn":             WSMS_TABLE,
    "message_id":         WSMS_TABLE,
    "failed sms":         WSMS_TABLE,
    "sms delivery":       WSMS_TABLE,
    "pack":               WSMS_TABLE,
    # Steering Master
    "steering":           STEERING_TABLE,
    "lbtr":               STEERING_TABLE,
    "srdc":               STEERING_TABLE,
    "mcc":                STEERING_TABLE,
    "mnc":                STEERING_TABLE,
    "steering master":    STEERING_TABLE,
    "forbidden":          STEERING_TABLE,
    "preferred partner":  STEERING_TABLE,
    "network_type":       STEERING_TABLE,
    # NTR Logs
    "ntr":                NTR_TABLE,
    "ntr log":            NTR_TABLE,
    "attach":             NTR_TABLE,
    "visited":            NTR_TABLE,
    "visited zone":       NTR_TABLE,
    "visited network":    NTR_TABLE,
    "latch":              NTR_TABLE,
    "ss7":                NTR_TABLE,
    "ulr":                NTR_TABLE,
    "tr_mechanism":       NTR_TABLE,
    "transaction":        NTR_TABLE,
}

_TABLE_TO_ROUTE_LABEL = {
    INBOUND_TABLE:          "inbound",
    OUTBOUND_TABLE:         "outbound",
    ROAMERS_INBOUND_TABLE:  "in-roamers",
    ROAMERS_OUTBOUND_TABLE: "out-roamers",
    RESP_INBOUND_TABLE:     "inbound-errors",
    RESP_OUTBOUND_TABLE:    "outbound-errors",
}

# UI label map
RLBL = {
    "footprint_q":   "🌍 FOOTPRINT",
    "performance_q": "📈 PERFORMANCE",
    "error_q":       "⚠️ ERRORS",
    "circle_q":      "🔄 CIRCLE",
    "trend_q":       "📉 TREND",
    "doc_search":    "📄 DOC SEARCH",
    "sms_q":         "💬 SMS",
    "steering_q":    "🧭 STEERING",
    "ntr_q":         "🔗 NTR",
}

RCLR = {
    "footprint_q":   "#1c7ed6",
    "performance_q": "#2f9e44",
    "error_q":       "#c92a2a",
    "circle_q":      "#7950f2",
    "trend_q":       "#e67700",
    "doc_search":    "#0c8599",
    "sms_q":         "#d6336c",
    "steering_q":    "#5c7cfa",
    "ntr_q":         "#087f5b",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — CORE PIPELINE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_cot_plan(query: str, schema: str, tables: List[str], few_shot: str) -> str:
    plan_sys = """\
You are a SQL planning expert for a telecom roaming analytics PostgreSQL database.
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
    gen_sys = f"""{SQL_RULES}

##IMPORTANT Return STRICT JSON only:
{{"sql": "<complete SELECT or WITH ... SELECT query>", "params": {{}}, "explanation": "<one line>"}}"""

    user_p = (
        f"## SCHEMA (need-to-know):\n{schema}\n\n"
        f"## FEW-SHOT EXAMPLES:\n{few_shot}\n\n"
        f"## SQL PLAN (Chain-of-Thought):\n{cot_plan}\n\n"
        f"## USER QUESTION:\n{query}\n\n"
        f"Generate a complete, correct PostgreSQL query against ONE table only in the required JSON format"
    )

    candidates: List[Dict] = []
    for beam_i in range(GOT_BEAM_WIDTH):
        temp = round(0.1 + beam_i * 0.1, 2)
        raw  = call_llm(gen_sys, user_p, max_new_tokens=700,
                        decode_config={**SQL_DECODE, "temperature": temp},
                        stage_label=f"GOT_GEN_{beam_i}")
        parsed = safe_json_parse(raw)
        if DEBUG:
            debug_print(f"GOT_GEN_{beam_i}_PARSED", parsed)
        if parsed.get("sql"):
            candidates.append(parsed)

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
    if engine is None:
        return {"df": None, "sql": sql, "exec_error": "DB not connected",
                "exec_valid": False, "exec_attempts": 0}
    exec_error = None
    for attempt in range(1, EXEC_MAX_ITERS + 1):
        try:
            validate_sql(sql)
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params or {})
                df = normalize_country_columns(df)
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
#  SECTION 11 — LLM DEEP-ANALYSIS FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_fallback(query: str, prior: GREAgentState) -> GREAgentState:
    print("\n🔁 LLM deep-analysis fallback triggered")
    tables   = prior.get("tables") or list(VALID_TABLES)
    schema   = rag.schema_context(tables)
    few_shot = rag.few_shot_block(query, tables_hint=tables, top_k=5)

    analyst_sys = f"""\
{SAFETY_PREAMBLE}
You are a GRE telecom roaming database Schema Analyst.
Analyse the query and identify the exact table and columns needed.

AVAILABLE TABLES AND SCHEMAS:
{schema}

Return STRICT JSON only:
{{
  "tables":      ["public.table1"],
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

    df      = exec_result.get("df")
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
#  SECTION 12 — LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] ROUTER")
    print("="*60)
    query  = state["user_query"]
    raw    = call_llm(ROUTER_SYSTEM, f"Query: {query}",
                      max_new_tokens=80, decode_config=ROUTER_DECODE, stage_label="ROUTER")
    parsed = safe_json_parse(raw)
    route  = parsed.get("route", "footprint_q")
    if route not in VALID_ROUTES:
        # keyword fallback
        ql = query.lower()
        if any(k in ql for k in ["sms", "wsms", "welcome sms", "msisdn", "dispatch", "pack", "message_id"]):
            route = "sms_q"
        elif any(k in ql for k in ["steering", "lbtr", "srdc", "mcc", "mnc", "forbidden", "network_type"]):
            route = "steering_q"
        elif any(k in ql for k in ["ntr", "latch", "visited", "ss7", "ulr", "transaction routing"]):
            route = "ntr_q"
        elif any(k in ql for k in ["error","imsi","blacklist","reason","fail","gtp","diameter"]):
            route = "error_q"
        elif any(k in ql for k in ["dip","trend","hourly","yesterday","% change"]):
            route = "trend_q"
        elif any(k in ql for k in ["circle","silent","compare","partner"]):
            route = "circle_q"
        elif any(k in ql for k in ["threshold","breach","failure rate","kpi"]):
            route = "performance_q"
        else:
            route = "footprint_q"
    log_trace({"node": "router", "route": route, "query": query,
               "confidence": parsed.get("confidence")})
    print(f"[ROUTER] → {route}  ({parsed.get('confidence','?')} confidence)")
    return {
        **state,
        "route":            route,
        "route_confidence": parsed.get("confidence", "Medium"),
        "route_reasoning":  parsed.get("reasoning", ""),
    }


def table_id_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] TABLE IDENTIFICATION")
    print("="*60)
    query  = state["user_query"]
    route  = state.get("route", "footprint_q")
    raw    = call_llm(
        TABLE_ID_SYSTEM,
        f"Valid tables: {sorted(VALID_TABLES)}\n\nRoute: {route}\n\nQuery: {query}",
        max_new_tokens=200, decode_config=INTENT_DECODE, stage_label="TABLE_ID",
    )
    parsed = safe_json_parse(raw)
    tables = [t for t in parsed.get("tables", []) if t in VALID_TABLES]
    primary_table = parsed.get("primary_table", [])

    if not tables:
        # Keyword-based fallback
        ql = query.lower()
        for kw, tbl in KW_TABLE_MAP.items():
            if kw in ql:
                tables = [tbl]
                break
        if not tables:
            tables = ROUTE_TABLE_DEFAULTS.get(route, [ROAMERS_INBOUND_TABLE])
    else:
        #tables = tables[:1]
        tables = [primary_table]

    print(f"[TABLE_ID] Identified: {tables}")
    return {**state, "tables": tables}


def rag_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] RAG RETRIEVAL")
    print("="*60)
    query  = state["user_query"]
    tables = state.get("tables", [])

    emb_table = rag.find_table_by_embedding(query)
    if emb_table and emb_table not in tables:
        print(f"[RAG] Embedding suggests table: {emb_table}")
        # Only override if LLM pick was the generic default
        if tables == ROUTE_TABLE_DEFAULTS.get(state.get("route", ""), []):
            tables = [emb_table]

    schema   = rag.schema_context(tables)
    few_shot = rag.few_shot_block(query, tables_hint=tables, top_k=5)
    debug_print("RAG_SCHEMA", schema[:1000])
    debug_print("RAG_FEW_SHOT", few_shot[:600])
    return {**state, "tables": tables, "schema_context": schema, "few_shot_block": few_shot}


def cot_plan_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] COT PLANNER")
    print("="*60)
    plan = run_cot_plan(
        state["user_query"],
        state.get("schema_context", ""),
        state.get("tables", []),
        state.get("few_shot_block", ""),
    )
    return {**state, "cot_plan": plan}


def sql_generation_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] SQL GENERATION (GoT)")
    print("="*60)
    retries = state.get("sql_retry_count", 0)
    if DEBUG:
        debug_print("SQL_GEN_INPUT", {
            "user_query":  state["user_query"],
            "tables":      state.get("tables", []),
            "cot_plan":    state.get("cot_plan", "")[:300],
            "few_shot":    state.get("few_shot_block", "")[:300],
        })
    best = run_got_gen(
        state["user_query"],
        state.get("schema_context", ""),
        state.get("cot_plan", ""),
        state.get("few_shot_block", ""),
        state.get("tables", []),
    )
    if best:
        print(f"[SQL_GEN] ✓ score={best.score:.2f}  sql={best.sql[:80]}")
        return {**state, "sql": best.sql, "sql_params": best.params,
                "sql_retry_count": retries + 1}
    # GoT produced no candidate — return empty SQL so validation triggers a retry
    print("[SQL_GEN] ✗ GoT returned no candidate")
    return {**state, "sql": "", "sql_params": {},
            "sql_retry_count": retries + 1,
            "sql_error": "GoT failed to produce a candidate"}


def sql_validation_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] SQL VALIDATION")
    print("="*60)
    sql = state.get("sql", "")
    if DEBUG:
        debug_print("SQL_VALIDATION_INPUT", sql[:500])
    try:
        validate_sql(sql)
        print("[VALIDATION] ✓ PASSED")
        return {**state, "sql_error": ""}
    except Exception as e:
        error = str(e)
        print(f"[VALIDATION] ✗ FAILED → {error}")
        sec_logger.log_event("sql_blocked", "warning", detail={"sql": sql[:500], "reason": error})
        return {**state, "sql_error": error}


def should_retry_sql(state: GREAgentState) -> str:
    error   = state.get("sql_error", "")
    retries = state.get("sql_retry_count", 0)
    if not error:
        return "execute"
    if retries <= SQL_MAX_RETRIES:
        print(f"[ROUTER] SQL invalid → retrying ({retries}/{SQL_MAX_RETRIES})")
        return "retry"
    print("[ROUTER] Max retries exhausted → failing")
    return "fail"


def sql_execution_node(state: GREAgentState) -> GREAgentState:
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


def explanation_node(state: GREAgentState) -> GREAgentState:
    print("\n" + "="*60)
    print("[NODE] EXPLANATION AGENT")
    print("="*60)
    query = state["user_query"]
    df    = state.get("df")
    error = state.get("exec_error")
    route = state.get("route", "unknown")

    if error or df is None:
        return {**state, "answer": f"Query could not be executed. Error: {error or 'Unknown'}"}

    summary = summarize_dataframe(df)
    
    # ── Steering compliance: cap Forbidden rows fed to the explanation LLM ──
    if route == "steering_q" and df is not None and "network_type" in df.columns:
        forbidden_df   = df[df["network_type"].str.contains("Forbidden", case=False, na=False)]
        non_forbidden  = df[~df["network_type"].str.contains("Forbidden", case=False, na=False)]
        # Only show up to 5 forbidden rows to the LLM
        capped_df      = pd.concat([non_forbidden, forbidden_df.head(5)], ignore_index=True)
        summary        = summarize_dataframe(capped_df)
        # Inject counts so the LLM can give accurate context without enumerating
        forbidden_count = len(forbidden_df)
        summary["_steering_note"] = (
            f"Total Forbidden partners in DB result: {forbidden_count}. "
            f"Only 5 are shown here per compliance policy. "
            f"Do NOT reveal the full list or the exact total count in your explanation."
        )

    user_p = (
        f"Question: {query}\n\n"
        f"Route: {route}\n"
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


def failure_node(state: GREAgentState) -> GREAgentState:
    print("\n[NODE] FAILURE TERMINAL")
    reason = state.get("sql_error") or state.get("failure_reason") or "Unknown SQL generation error"
    return {
        **state,
        "failed":         True,
        "failure_reason": f"SQL generation failed after {SQL_MAX_RETRIES + 1} attempts → {reason}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_gre_graph() -> StateGraph:
    graph = StateGraph(GREAgentState)
    graph.add_node("router",         router_node)
    graph.add_node("table_id",       table_id_node)
    graph.add_node("rag",            rag_node)
    graph.add_node("cot_planner",    cot_plan_node)
    graph.add_node("sql_generation", sql_generation_node)
    graph.add_node("sql_validation", sql_validation_node)
    graph.add_node("sql_execution",  sql_execution_node)
    graph.add_node("explanation",    explanation_node)
    graph.add_node("failure",        failure_node)

    graph.set_entry_point("router")
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


gre_graph = build_gre_graph()
print("[STARTUP] LangGraph compiled ✓")


def run_query(user_query: str, model_name: str = MODEL,
              use_fallback: bool = True) -> GREAgentState:
    initial: GREAgentState = {
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
    }
    final: GREAgentState = gre_graph.invoke(initial)
    if use_fallback and (final.get("failed") or final.get("df") is None):
        print("[ORCHESTRATOR] Graph result empty/failed — triggering fallback")
        final = run_llm_fallback(user_query, final)
    return final


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — CHART BUILDER
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
                    text=d[vl].apply(lambda v: f"{int(v):,}" if v >= 1 else f"{v:.4f}"), textposition="outside"))
                fig.update_layout(**_BL, height=max(240, len(d)*34+80), showlegend=False)
            else:
                fig = go.Figure(go.Bar(x=d[lb].astype(str), y=d[vl],
                    marker_color=clrs, opacity=.88,
                    text=d[vl].apply(lambda v: f"{int(v):,}" if v >= 1 else f"{v:.4f}"), textposition="outside"))
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
            n   = len(top)
            if n == 0:
                return None
            mx    = float(top[vc].max()) or 1
            sizes = ((top[vc] / mx) * 44 + 12).tolist()
            xp, yp = [0.0], [0.0]
            placed, ring = 1, 1
            while placed < n:
                r   = ring * 0.19
                cap = max(6, int(2 * math.pi * r / 0.15))
                rn  = min(cap, n - placed)
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
#  SECTION 15 — VIZ DETECTION & PIPELINE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _detect_viz(q: str, df) -> str:
    if df is None or df.empty:
        return "table"
    ql = q.lower()

    # Explicit chart requests
    if re.search(r"\bbar\s*chart\b|\bshow\s+(?:as\s+)?bar\b", ql):
        return "bar"
    if re.search(r"\bline\s*(?:chart|graph)\b|\btrend\b|\btime\s*series\b", ql):
        return "line"
    if re.search(r"\bpie\s*(?:chart|graph)\b|\bdistribution\b", ql):
        return "pie"
    if re.search(r"\bwordcloud\b|\bword\s*cloud\b|\bcloud\b", ql):
        return "wordcloud"

    # GRE domain implicit signals
    if re.search(r"\bhourly\b|\btrend\b|\bover\s+time\b|\blast\s+\d+\s+hours?\b|\btime\s*series\b", ql):
        return "line"
    if re.search(r"\btop\s+\d+\s+(?:countries|operators|partners|circles)\b", ql):
        return "bar"
    if re.search(r"\bdip\b|\bchange\b|\bpct\b|\bpercentage\b|\bcompare\b", ql):
        return "bar"
    if re.search(r"\bbreakdown\b|\bsplit\b|\bshare\b", ql):
        return "pie"
    if re.search(r"\berror\s+codes?\b|\bresponse\s+codes?\b", ql):
        return "wordcloud"

    # Data shape heuristics
    cols = list(df.columns)
    num  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat  = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    if not num:
        return "table"

    has_time = any(re.search(r"time|hour|date|ts|stamp", c, re.IGNORECASE) for c in cols)
    if has_time and len(num) >= 1:
        return "line"
    if len(cat) >= 1 and len(df) <= 20:
        return "bar"
    if len(cat) >= 1 and len(df) <= 8:
        return "pie"
    return "table"


def _steering_guard(query: str):
    """
    Pre-flight compliance guard for ir_steering_master queries.
    Returns (blocked: bool, message: str | None, restriction: str | None)
    restriction values: "preferred_only" | "forbidden_top5" | "mcc_mnc_pair" | None
    """
    ql = query.lower()

    # Only apply guard when this is clearly a steering/partner query
    is_steering = any(k in ql for k in [
        "steering", "roaming partner", "forbidden", "preferred partner",
        "lbtr", "srdc", "mcc", "mnc", "network_type", "ir_steering"
    ])
    if not is_steering:
        return False, None, None

    # ── Rule 1: Block cross-country dumps ("all countries" / no specific country) ──
    no_country_hints = [
    "all countries", "every country", "across countries",
    "for all countries", "all the countries", "each country",
    "list all preferred", "list all forbidden", "all forbidden", "all preferred",
    "preferred and forbidden", "forbidden and preferred",
    "all roaming partners", "full list", "complete list", "entire list",
    "give me all", "show all partners", "all partners",
    ]
    is_multi_country = (
        any(h in ql for h in no_country_hints)
        or bool(re.search(r"\ball\b.{0,30}\bcountries\b", ql))   # "for all countries", "partners for all countries"
        or bool(re.search(r"\bentire\b.{0,30}\blist\b", ql))     # "entire list of ..."
        or bool(re.search(r"\ball\b.{0,20}\bpartners?\b", ql))   # "all roaming partners", "all partners"
    )

    # Check if user specified exactly one country  
    # Positive signal: a known non-generic word follows in/for/of
    # Negative signal: generic words that are NOT countries
    _non_country_words = {
        "all", "every", "each", "the", "any", "roaming", "partners",
        "partner", "list", "countries", "country", "entire", "complete",
        "full", "data", "details", "info", "information", "me", "us",
    }
    _country_match = re.search(
        r"\b(in|for|of)\s+([A-Za-z]{3,})",
        query, re.IGNORECASE
    )
    country_specified = bool(
        _country_match and
        _country_match.group(2).lower() not in _non_country_words
    )

    if is_multi_country and not country_specified:
        msg = (
            "⚠️ **Compliance Restriction** — I cannot provide a full list of "
            "preferred/forbidden partners across all countries due to data compliance policy.\n\n"
            "Please ask about a **specific country** instead, for example:\n"
            "- *Show preferred roaming partners in Germany*\n"
            "- *What are the forbidden partners in USA?*\n"
            "- *List partners in Japan*"
        )
        return True, msg, None
    
    # ── Rule 1A: All partners requested → preferred + less-preferred + top 5 forbidden ──
    wants_all_partners = (
        "all" in ql and "partner" in ql
        ) or "all roaming partners" in ql
    
    if wants_all_partners and country_specified:
        return False, None, "all_partners"
    

    # ── Rule 2: Forbidden list explicitly requested → top 5 only ──
    wants_forbidden = any(k in ql for k in ["forbidden", "forbidden partner", "forbidden list", "forbidden network"])
    if wants_forbidden:
        return False, None, "forbidden_top5"

    # ── Rule 3: MCC/MNC must be returned as a pair ──
    wants_mcc_or_mnc = any(k in ql for k in ["mcc", "mnc", "mobile country code", "mobile network code"])
    if wants_mcc_or_mnc:
        return False, None, "mcc_mnc_pair"

    # ── Rule 4: Default — show preferred only ──
    return False, None, "preferred_only"


def _apply_steering_post_filter(df, restriction: str, query: str):
    """
    Post-execution filter for steering results.
    - preferred_only: if df has network_type column and query didn't ask for forbidden, keep preferred
    - forbidden_top5: cap forbidden rows to 5
    - mcc_mnc_pair: ensure both mcc+mnc are present
    """
    if df is None or df.empty:
        return df

    cols_lower = [c.lower() for c in df.columns]

    if restriction == "forbidden_top5":
        # Cap to top 5 forbidden rows
        if "network_type" in cols_lower:
            nt_col = df.columns[cols_lower.index("network_type")]
            mask = df[nt_col].str.lower() == "forbidden"
            forbidden_rows = df[mask].head(5)
            preferred_rows = df[~mask]
            df = pd.concat([preferred_rows, forbidden_rows], ignore_index=True)
        else:
            df = df.head(5)
            
    elif restriction == "all_partners":
        if "network_type" in cols_lower:
            nt_col = df.columns[cols_lower.index("network_type")]
            
            preferred = df[df[nt_col].str.lower().isin(["preferred","less-preferred"])]
            forbidden = df[df[nt_col].str.lower() == "forbidden"].head(5)
            
            df = pd.concat([preferred, forbidden], ignore_index=True)

    elif restriction == "preferred_only":
        if "network_type" in cols_lower:
            nt_col = df.columns[cols_lower.index("network_type")]
            ql = query.lower()
            if "forbidden" not in ql:
                df = df[df[nt_col].str.lower() != "forbidden"]

    elif restriction == "mcc_mnc_pair":
        # Ensure both mcc and mnc are present — add note if one is missing
        has_mcc = "mcc" in cols_lower
        has_mnc = "mnc" in cols_lower
        if not (has_mcc and has_mnc):
            # Can't fix SQL result retroactively, but flag it
            pass  # the schema rule prevents this; this is a fallback safety

    return df


def _build_steering_disclaimer(df, restriction: str) -> str:
    """Build the compliance note appended to steering answers."""
    if restriction == "forbidden_top5":
        forbidden_count = 0
        if df is not None and not df.empty:
            cols_lower = [c.lower() for c in df.columns]
            if "network_type" in cols_lower:
                nt_col = df.columns[cols_lower.index("network_type")]
                forbidden_count = (df[nt_col].str.lower() == "forbidden").sum()
        return (
            f"\n\n📋 *Showing top 5 forbidden partners only. "
            f"Full forbidden list is restricted per compliance policy.*"
        )
    
    elif restriction == "all_partners":
        return (
            "\n\n📋 *Showing all Preferred and Less-Preferred partners. "
            "Only top 5 Forbidden partners are displayed due to compliance policy.*"
            )
    
    elif restriction == "preferred_only":
        return (
            "\n\n📋 *Showing preferred partners only. "
            "Forbidden partners exist for this country but are not listed by default. "
            "Ask specifically for 'forbidden partners in [country]' to see up to 5.*"
        )
    return ""


def _is_steering_query(tables: list) -> bool:
    return "public.ir_steering_master" in (tables or [])


def pipeline(user_query: str, model_name: str = MODEL) -> Dict[str, Any]:
    t0 = time.time()

    # SECURITY FIX (Obs #21/#2 evidence — steering compliance bypassed by
    # asking in a non-English language): deterministic pre-model content
    # classifier runs first, before any LLM call or language-dependent
    # keyword matching.
    blocked_category = classify_blocked_request(user_query)
    if blocked_category:
        sec_logger.log_event("content_blocked", "warning",
                              detail={"stage": "input", "category": blocked_category, "query": user_query[:500]})
        return {
            "answer": refusal_for(blocked_category),
            "route": "blocked", "viz": "table",
            "sql": "", "rows": [], "cols": [],
            "fig": None, "error": False,
            "timing": {"total": round(time.time() - t0, 2)},
        }

    # ── Steering compliance pre-flight check ──
    blocked, block_msg, restriction = _steering_guard(user_query)
    if DEBUG:
        print(f"[STEERING_GUARD] restriction={restriction}  blocked={blocked}")

    if blocked:
        return {
            "answer": block_msg,
            "route": "steering_q", "viz": "table",
            "sql": "", "rows": [], "cols": [],
            "fig": None, "error": False,
            "timing": {"total": round(time.time() - t0, 3)},
            "tables": ["public.ir_steering_master"],
        }

    try:
        result = run_query(user_query, model_name=model_name)
    except Exception as ex:
        traceback.print_exc()
        return {
            "answer": f"Pipeline error: {str(ex)[:200]}",
            "route": "error", "viz": "table",
            "sql": "", "rows": [], "cols": [],
            "fig": None, "error": True,
            "timing": {"total": round(time.time() - t0, 2)},
        }

    df      = result.get("df")
    sql     = result.get("sql", "")
    route   = result.get("route", "footprint_q")
    answer  = result.get("answer", "")
    failed  = result.get("failed", False)
    fb_used = result.get("fallback_used", False)

    # ── Apply steering post-filter if applicable ──
    # SECURITY FIX (evidence: French-language request bypassed the English
    # keyword-only _steering_guard and returned the full unrestricted
    # forbidden-partner list — restriction came back None so this block used
    # to be skipped entirely). The compliance filter is now enforced
    # table-identity-first: any query that actually reached
    # ir_steering_master gets the most restrictive filter ("preferred_only")
    # by default whenever the free-text pre-flight check couldn't classify
    # it, regardless of the language/phrasing used to ask.
    tables_used = result.get("tables", [])
    if _is_steering_query(tables_used) and df is not None:
        effective_restriction = restriction or "preferred_only"
        df = _apply_steering_post_filter(df, effective_restriction, user_query)
        result["df"] = df
        disclaimer = _build_steering_disclaimer(df, effective_restriction)
        if disclaimer:
            answer = (answer or "") + disclaimer

    rows = safe_serialize_dataframe(df)
    cols = extract_columns(df)
    viz  = _detect_viz(user_query, df)
    fig  = build_chart(df, viz, title=user_query[:50]) if viz != "table" else None

    if not answer:
        if failed:
            answer = f"Could not retrieve data. {result.get('failure_reason','')}"
        elif df is not None and df.empty:
            answer = "Query executed successfully but returned no results. Try adjusting the time range or filters."
        else:
            answer = "Query executed successfully."

    sanitized = sanitize_output(answer, system_prompt_fragments=_SYSTEM_PROMPT_FRAGMENTS)
    if sanitized != answer:
        sec_logger.log_event("content_blocked", "warning", detail={"stage": "output", "query": user_query[:500]})
    answer = sanitized

    return {
        "answer": answer,
        "route":  route,
        "viz":    viz,
        "sql":    sql,
        "rows":   rows,
        "cols":   cols,
        "fig":    fig,
        "error":  failed,
        "timing": {"total": round(time.time() - t0, 2)},
        "fallback_used": fb_used,
        "tables": result.get("tables", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — QUICK CHIPS & DASHBOARD QUERIES
# ══════════════════════════════════════════════════════════════════════════════

QUICK = [
	"Show me the Top 10 countries by in-roamer count today",
    "Top 10 out-roaming countries today",
    "What is the total in-roamer count in RAJ today?",
    "Total roamer count for UK today",
    "Show inbound diameter failure rate in the last 4 hours for UAE",
    "What are the top errors for GTPv2 inbound today?",
    "Show GTPv2 outbound failure trend last 4 hours",
    "Count failed SMS today",
	"Compare in-roamer volume for MUM and DEL circles today",
    "List all preferred partners in USA",
    "Show top forbidden partners in Germany",
	"Show all steering configurations for Germany",
    "Show countries where in-roamer count dipped compared to yesterday",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — CSS & JS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@1&family=JetBrains+Mono:wght@400;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#f4f6fb;color:#0f2044;height:100vh;overflow:hidden}
.shell{display:grid;grid-template-rows:46px 1fr;grid-template-columns:220px 1fr;height:100vh}
.topbar{grid-column:1/-1;background:#0b1b3a;display:flex;align-items:center;gap:10px;padding:0 14px;border-bottom:1px solid rgba(255,255,255,.08)}
.t-brand{display:flex;align-items:center;gap:8px;flex-shrink:0}
.t-icon{width:26px;height:26px;border-radius:7px;background:linear-gradient(135deg,#1c7ed6,#0c8599);display:flex;align-items:center;justify-content:center;font-size:13px;color:#fff}
.t-name{font-family:'DM Serif Display',serif;font-style:italic;font-size:16px;color:#fff}
.t-sub{font-size:8px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3);margin-top:1px}
.vd{width:1px;height:20px;background:rgba(255,255,255,.1);flex-shrink:0}
.t-body{flex:1;display:flex;align-items:center;gap:10px;overflow:hidden}
.t-title{font-size:11.5px;font-weight:600;color:rgba(255,255,255,.55);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t-badges{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.t-right{display:flex;align-items:center;gap:8px;flex-shrink:0}
.clock{font-size:11px;font-weight:700;color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;letter-spacing:.04em}
.live-dot{width:7px;height:7px;border-radius:50%;background:#51cf66;animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(81,207,102,.5)}50%{opacity:.7;box-shadow:0 0 0 5px rgba(81,207,102,0)}}
.tbadge{padding:2px 7px;border-radius:10px;font-size:9.5px;font-weight:700;letter-spacing:.05em;border:1px solid}
.sidebar{background:#fff;border-right:1px solid #e4e7ec;display:flex;flex-direction:column;overflow:hidden}
.s-hdr{font-size:9.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af;padding:12px 14px 6px}
.s-new{all:unset;margin:0 10px 8px;padding:7px 12px;background:linear-gradient(135deg,#1c7ed6,#0c8599);color:#fff;border-radius:7px;font-size:11.5px;font-weight:600;cursor:pointer;text-align:center;transition:opacity .15s}
.s-new:hover{opacity:.88}
.s-list{flex:1;overflow-y:auto;padding:0 6px}
.s-item{padding:8px 10px;border-radius:7px;cursor:pointer;margin-bottom:2px;transition:background .12s}
.s-item:hover{background:#f1f3f5}
.s-act{background:#eef1fd!important}
.s-ttl{font-size:12px;font-weight:600;color:#0f2044;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.s-meta{font-size:10px;color:#9ca3af;margin-top:2px}
.s-foot{border-top:1px solid #e4e7ec;padding:10px}
.s-mrow{display:flex;align-items:center;gap:7px}
.s-mlbl{font-size:10px;font-weight:700;color:#9ca3af;white-space:nowrap}
.main{background:#f4f6fb;display:flex;flex-direction:column;overflow:hidden}
.scroll{flex:1;overflow-y:auto;padding:16px 18px 6px}
.stream{max-width:860px;margin:0 auto;padding-bottom:8px}
.welcome{max-width:640px;margin:0 auto;padding:32px 0 8px;text-align:center}
.w-eye{font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#9ca3af;margin-bottom:10px}
.w-h{font-size:22px;font-weight:700;color:#0f2044;margin-bottom:10px}
.w-p{font-size:13px;color:#6b7280;line-height:1.6;margin-bottom:20px}
.cap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px;text-align:left}
.cap{background:#fff;border:1.5px solid #e4e7ec;border-radius:10px;padding:14px;cursor:pointer;transition:all .15s}
.cap:hover{border-color:#1c7ed6;box-shadow:0 2px 8px rgba(28,126,214,.12);transform:translateY(-1px)}
.cap-ico{font-size:20px;margin-bottom:6px}
.cap-ttl{font-size:13px;font-weight:700;color:#0f2044;margin-bottom:3px}
.cap-dsc{font-size:11px;color:#6b7280;line-height:1.4}
.qlbl{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:6px;text-align:left}
.steps{display:flex;flex-wrap:wrap;gap:4px;padding:6px 12px;background:#f9fafb;border-bottom:1px solid #e4e7ec;align-items:center}
.step{padding:2px 7px;background:#eef1fd;color:#3b5bdb;border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap}
.step-sep{font-size:9px;color:#adb5bd}
.b{padding:2px 8px;border-radius:10px;font-size:9.5px;font-weight:700;letter-spacing:.04em;border:1px solid #e4e7ec;background:#f9fafb;color:#6b7280}
.br{border-radius:10px}
.bviz{color:#7950f2;background:rgba(121,80,242,.08);border-color:rgba(121,80,242,.2)}
.brows{color:#2f9e44;background:rgba(47,158,68,.08);border-color:rgba(47,158,68,.2)}
.btime{color:#9ca3af;background:#f9fafb}
.berr{color:#c92a2a;background:rgba(201,42,42,.08);border-color:rgba(201,42,42,.2)}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{padding:3px 10px;background:#fff;border:1px solid #e4e7ec;border-radius:16px;font-size:10.5px;color:#6b7280;cursor:pointer;transition:all .12s;white-space:nowrap;user-select:none}
.chip:hover{background:#eef1fd;border-color:rgba(28,126,214,.3);color:#1c7ed6}
.msg{margin-bottom:14px}
.mu{display:flex;justify-content:flex-end}
.ub{background:#0b1b3a;color:#fff;border-radius:9px 9px 3px 9px;padding:9px 13px;max-width:58%;font-size:13.5px;line-height:1.6;box-shadow:0 2px 5px rgba(0,0,0,.14)}
.mb{display:flex;align-items:flex-start;gap:8px}
.av{width:25px;height:25px;flex-shrink:0;border-radius:6px;background:#fff;display:flex;align-items:center;justify-content:center;padding:2px;margin-top:2px}
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
.iwrap:focus-within{border-color:#1c7ed6;box-shadow:0 0 0 3px rgba(28,126,214,.08)}
.ita{flex:1;background:none;border:none;outline:none;resize:none;max-height:100px;font-size:13.5px;line-height:1.5;color:#0f2044;font-family:'DM Sans',sans-serif;padding:2px 0}
.ita::placeholder{color:#adb5bd}
.sbtn{width:31px;height:31px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:#0b1b3a;border:none;border-radius:6px;cursor:pointer;color:#fff;font-size:15px;transition:background .15s,transform .1s}
.sbtn:hover{background:#1c7ed6;transform:translateY(-1px)}
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
.docbtn{width:31px;height:31px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:#f1f3f5;border:1.5px solid #e4e7ec;border-radius:6px;cursor:pointer;font-size:15px;transition:all .15s;line-height:1}
.docbtn:hover{background:#eef1fd;border-color:#1c7ed6}
.docbtn-active{background:#eef1fd!important;border-color:#1c7ed6!important;box-shadow:0 0 0 3px rgba(28,126,214,.1)!important}
.doc-strip{padding:7px 0 5px;margin-bottom:6px;border-bottom:1px solid #e4e7ec;display:flex;flex-direction:column;gap:5px}
.doc-strip-hidden{display:none!important}
.doc-upload-zone{border:1.5px dashed #c8d0e7;border-radius:8px;padding:9px 16px;text-align:center;cursor:pointer;font-size:12px;color:#6b7280;background:#f8f9ff;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:7px}
.doc-upload-zone:hover{border-color:#1c7ed6;background:#eef1fd;color:#1c7ed6}
.doc-info-row{display:flex;align-items:center;gap:6px;min-height:18px}
.doc-loaded{font-size:11px;color:#2f9e44;font-weight:600}
.doc-err{font-size:11px;color:#c92a2a;font-weight:500}
.doc-clear{all:unset;font-size:10px;color:#adb5bd;cursor:pointer;padding:2px 5px;border-radius:3px;border:1px solid #e4e7ec;background:#fff;transition:all .12s}
.doc-clear:hover{color:#c92a2a;border-color:#c92a2a;background:#fff5f5}
.bdoc{color:#0c8599;background:rgba(12,133,153,.08);border-color:rgba(12,133,153,.25)}
.bdocname{color:#495057;background:#f8f9fa;border-color:#e4e7ec;font-family:'JetBrains Mono',monospace;font-size:8.5px;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-steps{display:flex;align-items:center;gap:3px;padding:6px 12px;background:#f0fafe;border-bottom:1px solid #c8e6f0}
.doc-step{padding:2px 7px;background:#d3eef8;color:#0c8599;border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap}
.csv-btn{all:unset;display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:5px;font-size:10px;font-weight:600;cursor:pointer;border:1px solid #e4e7ec;background:#fff;color:#6b7280;transition:all .15s;margin-left:auto}
.csv-btn:hover{background:#ebfbee;border-color:#2f9e44;color:#2f9e44}
.login-shell{min-height:100vh;background:linear-gradient(135deg,#060f24 0%,#0b1b3a 55%,#0d2554 100%);display:flex;align-items:center;justify-content:center;padding:24px}
.login-card{width:100%;max-width:400px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:40px 36px 34px;box-shadow:0 24px 64px rgba(0,0,0,.45)}
.login-logo{display:flex;align-items:center;gap:11px;margin-bottom:28px}
.login-logo-icon{width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,#1c7ed6,#0c8599);display:flex;align-items:center;justify-content:center;font-size:16px;color:#fff;flex-shrink:0}
.login-logo-name{font-family:'DM Serif Display',serif;font-style:italic;font-size:21px;color:#fff}
.login-logo-sub{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3);margin-top:1px}
.login-h{font-size:18px;font-weight:700;color:#fff;margin-bottom:4px}
.login-sub{font-size:12px;color:rgba(255,255,255,.4);margin-bottom:26px;line-height:1.5}
.login-label{display:block;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:6px}
.login-input{width:100%;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:10px 13px;font-size:13.5px;color:#fff;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .15s,box-shadow .15s;margin-bottom:16px}
.login-input::placeholder{color:rgba(255,255,255,.25)}
.login-input:focus{border-color:rgba(28,126,214,.7);box-shadow:0 0 0 3px rgba(28,126,214,.2)}
.login-btn{width:100%;padding:11px;background:linear-gradient(135deg,#1c7ed6,#0c8599);border:none;border-radius:8px;color:#fff;font-size:13.5px;font-weight:600;cursor:pointer;font-family:'DM Sans',sans-serif;transition:opacity .15s,transform .1s;margin-top:4px}
.login-btn:hover{opacity:.88;transform:translateY(-1px)}
.login-btn:active{transform:translateY(0)}
.login-error{margin-top:12px;padding:9px 12px;background:rgba(201,42,42,.15);border:1px solid rgba(201,42,42,.35);border-radius:7px;color:#ff8787;font-size:12px;line-height:1.5}
.login-badge{display:inline-flex;align-items:center;gap:5px;margin-top:22px;padding:6px 11px;background:rgba(28,126,214,.1);border:1px solid rgba(28,126,214,.2);border-radius:20px;font-size:10px;color:rgba(28,126,214,.9)}
.login-divider{border:none;border-top:1px solid rgba(255,255,255,.08);margin:20px 0}
.user-chip{display:flex;align-items:center;gap:7px;padding:4px 10px 4px 5px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:20px;cursor:default}
.user-avatar{width:20px;height:20px;border-radius:50%;background:linear-gradient(135deg,#1c7ed6,#0c8599);display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;font-weight:700;flex-shrink:0}
.user-email{font-size:10px;color:rgba(255,255,255,.6);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.logout-btn{all:unset;padding:3px 9px;border-radius:5px;font-size:10px;font-weight:600;color:rgba(255,255,255,.35);border:1px solid rgba(255,255,255,.1);cursor:pointer;transition:all .15s;margin-left:2px}
.logout-btn:hover{color:#ff8787;border-color:rgba(255,100,100,.3);background:rgba(255,100,100,.08)}
.thinking-bubble{display:flex;align-items:center;gap:10px;padding:13px 15px;font-size:13px;color:#6b7280;background:#f9fafb}
.thinking-robot{font-size:20px;display:inline-block;animation:robot-bob .9s ease-in-out infinite}
@keyframes robot-bob{0%,100%{transform:translateY(0) rotate(-3deg)}50%{transform:translateY(-4px) rotate(3deg)}}
.thinking-label{font-weight:600;color:#374151;font-size:13px}
.thinking-dots{display:inline-flex;gap:2px;align-items:center;margin-left:1px}
.thinking-dots span{display:inline-block;width:5px;height:5px;border-radius:50%;background:#1c7ed6;animation:dot-pulse 1.4s ease-in-out infinite both}
.thinking-dots span:nth-child(2){animation-delay:.2s}
.thinking-dots span:nth-child(3){animation-delay:.4s}
@keyframes dot-pulse{0%,80%,100%{transform:scale(0.6);opacity:.4}40%{transform:scale(1);opacity:1}}
"""

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


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 18 — DASH APP & LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Vi Roamers AI Tech Assistant (VIRTA)", assets_folder="/srv/gre-assistant/assets")
app.index_string = f"""<!DOCTYPE html>
<html><head>
  {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
  <style>{CSS}</style>
</head><body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  <script>{JS}</script>
</body></html>"""


def _av():
    #return html.Div("🌐", className="av")
    return html.Div(
        html.Img(src="/assets/Vodafone_Idea_logo.png",
                 style={"width": "100%", "height": "100%", "objectFit": "contain"}),
        className="av"
        )


def _steps():
    labels = ["① Router","② TableID","③ RAG","④ CoT","⑤ GoT-SQL","⑥ Validate","⑦ Execute","⑧ Explain"]
    items = []
    for i, l in enumerate(labels):
        items.append(html.Span(l, className="step"))
        if i < len(labels)-1:
            items.append(html.Span("→", className="step-sep"))
    return html.Div(items, className="steps")


def make_login_page():
    domain_hint = (
        "All @vodafoneidea.com accounts are currently enabled."
        if ALLOW_VODAFONE_DOMAIN
        else f"{len(ALLOWED_EMAILS)} authorised user(s) only."
    )
    return html.Div(className="login-shell", children=[
        html.Div(className="login-card", children=[
            html.Div(className="login-logo", children=[
                html.Div("🌐", className="login-logo-icon"),
                html.Div([
                    html.Div("V!RTA", className="login-logo-name"),
                    html.Div("Vi Roamers AI Tech Assistant", className="login-logo-sub"),
                ]),
            ]),
            html.Div("Sign in to Vi GRE", className="login-h"),
            html.Div("SQM Roaming Intelligence Platform · LangGraph 9-Node Pipeline", className="login-sub"),
            html.Label("Email address", className="login-label", htmlFor="login-email"),
            dcc.Input(id="login-email", type="email", placeholder="you@vodafoneidea.com",
                      className="login-input", debounce=False, n_submit=0),
            html.Label("Password", className="login-label", htmlFor="login-password"),
            dcc.Input(id="login-password", type="password", placeholder="Enter your password",
                      className="login-input", debounce=False, n_submit=0),
            html.Button("Sign in →", id="login-btn", className="login-btn", n_clicks=0),
            html.Div(id="login-error", className="login-error", style={"display": "none"}),
            html.Hr(className="login-divider"),
            html.Div([
                html.Span("🔒 "),
                html.Span(domain_hint, style={"fontSize": "10px", "color": "rgba(255,255,255,.35)"}),
            ], className="login-badge"),
        ])
    ])


def make_welcome():
    return html.Div(className="welcome", children=[
        html.Div("Vi SQM — Global Roamers Excellence", className="w-eye"),
        html.Div("What would you like to analyse?", className="w-h"),
        html.P("Ask in plain English — GRE routes through 9 AI nodes (LangGraph + DSPy + GoT) and returns live roaming data.",
               className="w-p"),
        html.Div(className="cap-grid", children=[
            html.Div(className="cap", **{"data-query": "Show me the Top 10 countries by in-roamer count today"}, children=[
                html.Div("🌍", className="cap-ico"),
                html.Div("Global Footprint", className="cap-ttl"),
                html.Div("In/out roamer counts, top countries", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query": "Show inbound diameter failure rate in the last 4 hours for UAE"}, children=[
                html.Div("📈", className="cap-ico"),
                html.Div("Performance & KPIs", className="cap-ttl"),
                html.Div("Threshold breaches, failure rates, trends", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query": "What are the top errors for GTPv2 inbound today?"}, children=[
                html.Div("⚠️", className="cap-ico"),
                html.Div("Error Analysis", className="cap-ttl"),
                html.Div("GTPv2/Diameter error codes, response distribution", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query": "Countries with 70% in-roamer dip today vs yesterday"}, children=[
                html.Div("📉", className="cap-ico"),
                html.Div("Dip & Trend Analysis", className="cap-ttl"),
                html.Div("Roamer dips, hourly trends, circle comparisons", className="cap-dsc")]),
            # html.Div(className="cap", **{"data-query": "Count failed SMS today"}, children=[
            #     html.Div("💬", className="cap-ico"),
            #     html.Div("Welcome SMS (WSMS)", className="cap-ttl"),
            #     html.Div("SMS delivery audits, failed dispatch, pack tracking", className="cap-dsc")]),
            html.Div(className="cap", **{"data-query": "List all preferred partners in USA"}, children=[
                html.Div("🧭", className="cap-ico"),
                html.Div("Steering Master", className="cap-ttl"),
                html.Div("LBTR/SRDC config, preferred/forbidden partners, MCC/MNC", className="cap-dsc")]),
            # html.Div(className="cap", **{"data-query": "Count SRDC vs LBTR steering hits today"}, children=[
            #     html.Div("🔗", className="cap-ico"),
            #     html.Div("NTR Logs", className="cap-ttl"),
            #     html.Div("Attachment events, visited network, IMSI path trace", className="cap-dsc")]),
        ]),
        html.Div("Quick queries", className="qlbl"),
        html.Div(className="chips", children=[
            html.Span(q, className="chip", **{"data-query": q}) for q in QUICK
        ]),
    ])


def render_bot_doc(m, idx, voted=None):
    doc_name = m.get("source_doc", "Document")
    total    = m.get("timing", {}).get("total")
    is_err   = m.get("error", False)
    body     = []
    badges   = [
        html.Span("📄  DOC SEARCH", className="b bdoc"),
        html.Span(doc_name[:35], className="b bdocname"),
    ]
    if total:
        badges.append(html.Span(f"⏱ {total:.1f}s", className="b btime"))
    body.append(html.Div(badges, className="bm"))
    body.append(html.Div([
        html.Span("① Parse", className="doc-step"),
        html.Span("→", className="step-sep"),
        html.Span("② Extract", className="doc-step"),
        html.Span("→", className="step-sep"),
        html.Span("③ Analyse", className="doc-step"),
    ], className="doc-steps"))
    body.append(html.Div(m.get("content", ""), className="be" if is_err else "ba"))
    up_cls = "fbup fb-active" if voted == "up" else "fbup"
    dn_cls = "fbdn fb-active" if voted == "dn" else "fbdn"
    hint   = ("✓ Saved" if voted == "up" else "↻ Regenerating…" if voted == "dn" else "")
    body.append(html.Div([
        html.Span("Was this helpful?", className="fb-lbl"),
        html.Button("👍  Upvote",   n_clicks=0, id={"type": "upvote",   "index": idx},
                    className=up_cls, disabled=voted is not None),
        html.Button("👎  Downvote", n_clicks=0, id={"type": "downvote", "index": idx},
                    className=dn_cls, disabled=voted is not None),
        html.Span(hint, className="fb-hint") if hint else html.Span(),
    ], className="fb-bar"))
    return html.Div(className="msg mb", children=[_av(), html.Div(body, className="bb")])


def render_bot(m, idx, voted=None):
    rk    = m.get("route", "")
    col   = RCLR.get(rk, "#6b7280")
    rows  = m.get("rows", [])
    clst  = m.get("cols", [])
    fig_j = m.get("fig")
    sql   = m.get("sql", "")
    total = m.get("timing", {}).get("total")
    viz   = m.get("viz", "table")
    is_err = m.get("error", False)
    body  = []
    badges = []

    if m.get("route") == "doc_search":
        return render_bot_doc(m, idx, voted)

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
                             "maxWidth":"220px","overflow":"hidden"},
                style_header={"background":"#f9fafb","color":"#9ca3af","fontWeight":"700",
                               "fontSize":"9px","textTransform":"uppercase",
                               "letterSpacing":".06em","borderBottom":"1px solid #e4e7ec",
                               "padding":"6px 10px"},
                style_data_conditional=[
                    {"if":{"filter_query":'{value} > 0.5',"column_id":"value"},
                     "color":"#c92a2a","fontWeight":"700"},
                    {"if":{"filter_query":'{pct_change} < -50',"column_id":"pct_change"},
                     "color":"#c92a2a","fontWeight":"700"},
                ],
            )
        ]))

    if sql:
        sid = str(idx)
        body.append(html.Div([
            html.Button("⟨/⟩  View SQL", className="sq", **{"data-sql": sid}),
            html.Pre(sql, id=f"sqlpre-{sid}", className="sqc"),
        ]))

    up_cls = "fbup fb-active" if voted == "up" else "fbup"
    dn_cls = "fbdn fb-active" if voted == "dn" else "fbdn"
    hint   = ("✓ Saved to training set" if voted == "up"
              else "↻ Regenerating…"    if voted == "dn"
              else "")
    body.append(html.Div([
        html.Span("Was this helpful?", className="fb-lbl"),
        html.Button("👍  Upvote",   n_clicks=0, id={"type": "upvote",   "index": idx},
                    className=up_cls, disabled=voted is not None),
        html.Button("👎  Downvote", n_clicks=0, id={"type": "downvote", "index": idx},
                    className=dn_cls, disabled=voted is not None),
        html.Span(hint, className="fb-hint") if hint else html.Span(),
    ], className="fb-bar"))
    return html.Div(className="msg mb", children=[_av(), html.Div(body, className="bb")])


def render_stream(msgs, feedback=None):
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

def render_thinking():
    """Animated 🤖 Thinking... bubble shown while pipeline is running."""
    return html.Div(className="msg mb", id="thinking-bubble", children=[
        _av(),
        html.Div(className="bb", children=[
            html.Div(className="thinking-bubble", children=[
                #html.Span("🤖", className="thinking-robot"),
                html.Img(src="/assets/Vodafone_Idea_logo.png", className="thinking-robot", style={"width": "22px", "height": "22px", "objectFit": "contain"}),
                html.Span("Thinking", className="thinking-label"),
                html.Span(className="thinking-dots", children=[
                    html.Span(), html.Span(), html.Span(),
                ]),
            ])
        ])
    ])


def render_sidebar(convs, active):
    items = []
    for c in reversed(convs or []):
        cls = "s-item s-act" if c["id"] == active else "s-item"
        items.append(html.Div(className=cls,
            id={"type":"conv","index":c["id"]}, n_clicks=0, children=[
                html.Div(c.get("title","…")[:42], className="s-ttl"),
                html.Div(f"{c.get('count',0)} msg", className="s-meta")]))
    return items


def make_main_layout(email: str = ""):
    initials = (email[:2].upper() if email else "?")
    return html.Div(className="shell", children=[
        html.Header(className="topbar", children=[
            html.Div(className="t-brand", children=[
                html.Div("🌐", className="t-icon"),
                html.Div([html.Div("V!RTA", className="t-name"),
                          html.Div("Vi Roamers AI Tech Assistant", className="t-sub")])]),
            html.Div(className="t-body", children=[
                html.Span([html.B("VIRTA"), " — Roaming Intelligence Platform"], className="t-title"),
                html.Div(id="tbadges", className="t-badges")]),
            html.Div(className="t-right", children=[
                html.Div(className="vd"),
                html.Div([html.Div(className="live-dot"),
                          html.Span("Live", style={"fontSize":"10px","color":"rgba(255,255,255,.35)"})],
                         style={"display":"flex","alignItems":"center","gap":"5px"}),
                html.Div(className="vd"),
                html.Div("", className="clock", **{"data-clock":"1"}),
                html.Div(className="vd"),
                html.Div(className="user-chip", children=[
                    html.Div(initials, className="user-avatar"),
                    html.Span(email, className="user-email"),
                ]),
                html.Button("Sign out", id="logout-btn", className="logout-btn", n_clicks=0),
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
                    html.Div(id="doc-strip", className="doc-strip doc-strip-hidden", children=[
                        dcc.Upload(id="doc-upload", className="doc-upload-zone",
                            children=html.Div(["📎 Drop a file here or ", html.U("click to upload"),
                                               " (.txt, .csv, .json)"])),
                        html.Div(id="doc-info-row", className="doc-info-row"),
                    ]),
                    html.Div(className="iwrap", children=[
                        html.Button("📄", id="docbtn", className="docbtn", n_clicks=0),
                        dcc.Textarea(id="qin", className="ita", rows=1,
                            placeholder="Ask about roamers, KPIs, errors, circle analysis…"),
                        html.Button("↑", id="sbtn", className="sbtn", n_clicks=0),
                    ]),
                    html.Div("GRE · LangGraph 9-Node Pipeline · Press Enter to send",
                             className="ihint"),
                ])]),
        ]),
        # ── Hidden stores ─────────────────────────────────────────────────────
        #dcc.Store(id="sauth",     storage_type="session"),
        dcc.Store(id="smsgs",     data=[]),
        dcc.Store(id="sconvs",    data=[]),
        dcc.Store(id="scid",      data=None),
        dcc.Store(id="sfeedback", data={}),
        dcc.Store(id="sdoc",      data=None),
        dcc.Store(id="sdoc_mode", data=False),
        dcc.Store(id="spending",  data=None),
        dcc.Download(id="csv-dl"),
    ])


# ── Root layout ───────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="sauth", storage_type="session"),
    html.Div(id="page-content", children=make_login_page())
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 19 — AUTH CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

from urllib.parse import urlparse, parse_qs

@app.callback(
    Output("page-content", "children"),
    Output("sauth", "data"),
    Input("url", "href"),
    State("sauth", "data"),
    prevent_initial_call=False,
)
def on_url_load(href, sauth):
    """
    Reads ?user=<email> from the URL.
    If the email is in ALLOWED_EMAILS → load main layout directly.
    Otherwise → show login page with an appropriate message.
    """
    if sauth and sauth.get("email"):
        return make_main_layout(sauth["email"]), no_update
    
    if not href:
        return make_login_page(), None

    try:
        parsed   = urlparse(href)
        params   = parse_qs(parsed.query)
        # URL arrives as: ?user=someone%40vodafoneidea.com  (@ is encoded as %40)
        email    = params.get("user", [None])[0]
    except Exception:
        return make_login_page(), None

    if not email:
        # No ?user= param — show plain login page
        return make_login_page(), None

    email = email.strip().lower()
    #allowed_lower = [e.strip().lower() for e in ALLOWED_EMAILS]
    allowed_lower = get_allowed_emails()

    if email in allowed_lower:
        # ✅ Authorised — skip login and load the app directly
        return make_main_layout(email), {"email": email}
    else:
        # ❌ Email not in allowed list — show login page with error message
        return _make_denied_page(email), None


def _make_denied_page(email: str):
    """Login page variant shown when the URL email is not in the allowed list."""
    return html.Div(className="login-shell", children=[
        html.Div(className="login-card", children=[
            html.Div(className="login-logo", children=[
                html.Div("🌐", className="login-logo-icon"),
                html.Div([
                    html.Div("V!RTA", className="login-logo-name"),
                    html.Div("Vi Roamers AI Tech Assistant", className="login-logo-sub"),
                ]),
            ]),
            html.Div("Access Denied", className="login-h"),
            html.Div(
                f"The account  {email}  is not on the authorised access list. "
                "Please contact your GRE administrator to request access.",
                className="login-sub",
                style={"color": "#ff6b6b", "marginTop": "10px", "lineHeight": "1.6"},
            ),
            html.Hr(className="login-divider"),
            html.Div([
                html.Span("🔒 "),
                html.Span(
                    f"{len(ALLOWED_EMAILS)} authorised user(s) configured.",
                    style={"fontSize": "10px", "color": "rgba(255,255,255,.35)"}
                ),
            ], className="login-badge"),
        ])
    ])


# ── AUTH: Login button ────────────────────────────────────────────────────────
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
    if not (n_btn or _ns_email or _ns_pwd):
        return no_update, "", {"display": "none"}
    ok, err = auth_login(email or "", password or "")
    if not ok:
        return no_update, err, {"display": "block"}
    token = auth_create_session(email)
    print(f"[AUTH] Login OK: {(email or '').strip().lower()}")
    return {"email": (email or "").strip().lower(), "token": token}, "", {"display": "none"}
    
# ── AUTH: Render page on session change ───────────────────────────────────────
@app.callback(
    Output("page-content", "children", allow_duplicate=True),
    Input("sauth", "data"),
    prevent_initial_call=True,
)
def on_auth_change(sauth):
    if sauth and sauth.get("email"):
        return make_main_layout(sauth["email"])
    return make_login_page()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 20 — CONVERSATION CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("slist",  "children"),
    Input("sconvs",  "data"),
    State("scid",    "data"),
)
def refresh_sidebar(convs, cid):
    return render_sidebar(convs, cid)


@app.callback(
    Output("smsgs",  "data",     allow_duplicate=True),
    Output("sconvs", "data",     allow_duplicate=True),
    Output("scid",   "data",     allow_duplicate=True),
    Output("stream", "children", allow_duplicate=True),
    Input("newbtn",  "n_clicks"),
    prevent_initial_call=True,
)
def new_conv(n):
    if not n:
        return no_update, no_update, no_update, no_update
    cid = f"conv_{int(time.time()*1000)}"
    return [], no_update, cid, [make_welcome()]


@app.callback(
    Output("smsgs",     "data",     allow_duplicate=True),
    Output("sconvs",    "data",     allow_duplicate=True),
    Output("scid",      "data",     allow_duplicate=True),
    Output("stream",    "children", allow_duplicate=True),
    Output("sfeedback", "data",     allow_duplicate=True),
    Input({"type": "conv", "index": ALL}, "n_clicks"),
    State("sconvs", "data"),
    prevent_initial_call=True,
)
def switch_conv(n_clicks_list, convs):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return no_update, no_update, no_update, no_update, no_update
    cid = triggered.get("index")
    if not cid or not any(n and n > 0 for n in (n_clicks_list or [])):
        return no_update, no_update, no_update, no_update, no_update
    conv = next((c for c in (convs or []) if c["id"] == cid), None)
    if not conv:
        return no_update, no_update, no_update, no_update, no_update
    msgs = conv.get("msgs", [])
    fb   = conv.get("feedback", {})
    return msgs, no_update, cid, render_stream(msgs, fb), fb


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 21 — MAIN QUERY CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("stream",    "children", allow_duplicate=True),
    Output("smsgs",     "data",     allow_duplicate=True),
    Output("sconvs",    "data",     allow_duplicate=True),
    Output("scid",      "data",     allow_duplicate=True),
    Output("sfeedback", "data",     allow_duplicate=True),
    Output("qin",       "value",    allow_duplicate=True),
    Output("spending",   "data"),
    Input("sbtn",       "n_clicks"),
    State("qin",        "value"),
    State("smsgs",      "data"),
    State("sconvs",     "data"),
    State("scid",       "data"),
    State("sfeedback",  "data"),
    State("mdd",        "value"),
    State("sdoc",       "data"),
    State("sdoc_mode",  "data"),
    prevent_initial_call=True,
)
def on_send(n, q, msgs, convs, cid, feedback, model, doc_data, doc_mode):
    """Immediate: show user message + thinking bubble, queue pipeline via spending."""
    if not n or not (q or "").strip():
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    q    = q.strip()
    msgs = list(msgs or [])
    convs = list(convs or [])
    feedback = dict(feedback or {})

    if not cid:
        cid = f"conv_{int(time.time()*1000)}"
    conv = next((c for c in convs if c["id"] == cid), None)
    if not conv:
        convs.append({"id": cid, "title": q[:38], "msgs": [], "count": 0, "feedback": {}})
        conv = convs[-1]
    if conv.get("count", 0) == 0:
        conv["title"] = q[:38]
    conv["count"] = conv.get("count", 0) + 1
    
    msgs.append({"role": "user", "content": q})
    stream_children = render_stream(msgs, feedback) + [render_thinking()]
    
    pending = {
        "query": q,
        "doc_mode": doc_mode,
        "doc_data": doc_data,
        "model": model,
        "ts": time.time(),
        }
    
    return stream_children, msgs, convs, cid, feedback, "", pending


@app.callback(
    Output("stream",    "children", allow_duplicate=True),
    Output("smsgs",     "data",     allow_duplicate=True),
    Output("sconvs",    "data",     allow_duplicate=True),
    Output("sfeedback", "data",     allow_duplicate=True),
    Input("spending",   "data"),
    State("smsgs",      "data"),
    State("sconvs",     "data"),
    State("scid",  "data"),
    State("sfeedback",  "data"),
    State("sauth",  "data"),
    prevent_initial_call=True,
)
def on_submit(pending, msgs, convs, cid, feedback, sauth):
    """Pipeline execution: replaces the thinking bubble with the real response."""
    if not pending or not pending.get("query"):
        return no_update, no_update, no_update, no_update

    q    = pending["query"]
    doc_mode = pending.get("doc_mode", False)
    doc_data = pending.get("doc_data")
    model = pending.get("model") or MODEL
    
    msgs = list(msgs or [])
    convs = list(convs or [])
    feedback = dict(feedback or {})
    conv = next((c for c in convs if c["id"] == cid), None)

    if doc_mode and doc_data:
        res = run_doc_search(q, doc_data.get("text", ""), doc_data.get("name", "document"))
        bot_msg = {
            "role": "bot", "content": res["answer"],
            "route": "doc_search", "viz": "table",
            "sql": "", "rows": [], "cols":[], "fig": None,
            "timing": res.get("timing", {}), "error": res.get("error", False),
            "source_doc": res.get("source_doc", ""),
            }
    else:
        res = pipeline(q, model_name=model)
        bot_msg = {
            "role": "bot",
            "content": res["answer"],
            "route": res["route"],
            "viz": res["viz"],
            "sql": res["sql"],
            "rows": res["rows"],
            "cols": res["cols"],
            "fig": res["fig"],
            "timing": res["timing"],
            "error": bool(res["error"]),
            "tables": res.get("tables", []),
            }
        
    msgs.append(bot_msg)
    if conv:
        conv["msgs"] = msgs

    # ── Session logging ───────────────────────────────────────────────────────
    user_email = (sauth or {}).get("email", "unknown")
    log_session_event(
        user_email   = user_email,
        question     = q,
        llm_response = res.get("answer", ""),
        vote         = "",
    )
    # ─────────────────────────────────────────────────────────────────────────
        
    return render_stream(msgs, feedback), msgs, convs, feedback



# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 22 — FEEDBACK CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

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

    msgs     = list(msgs  or [])
    convs    = list(convs or [])
    feedback = dict(feedback or {})

    if bot_idx >= len(msgs):
        return no_update, no_update

    # Save to CSV for RAG
    bot_msg = msgs[bot_idx]
    sql = bot_msg.get("sql", "")
    user_msg = next((msgs[i] for i in range(bot_idx - 1, -1, -1)
                     if msgs[i]["role"] == "user"), None)
    q = user_msg.get("content", "").strip() if user_msg else ""

    if q and sql:
        try:
            rk  = bot_msg.get("route", "footprint_q")
            cat = RLBL.get(rk, "General").replace("🌍 ","").replace("📈 ","").replace("⚠️ ","").replace("🔄 ","").replace("📉 ","")
            # Fix: tables is now a list in bot_msg; convert to comma-separated string for CSV
            raw_tables = bot_msg.get("tables", [])
            if isinstance(raw_tables, list):
                tbl = ",".join(raw_tables)
            else:
                tbl = str(raw_tables)
            tables_list = [t.strip() for t in tbl.split(",") if t.strip() in VALID_TABLES]

            # ── Write to CSV ──────────────────────────────────────────────────
            file_exists = os.path.exists(QUESTIONS_CSV)
            with open(QUESTIONS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["question", "sql", "category", "tables"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"question": q, "sql": sql, "category": cat, "tables": tbl})
            print(f"[UPVOTE] ✓ Saved to CSV: {q[:60]}")

            # ── Fix: also add to in-memory RAG immediately so future similar ──
            # ── questions in this session benefit right away               ──
            new_entry = rag._make_entry(q, sql, cat, tables_list, "upvoted")
            existing_ids = {e.id for e in rag.entries}
            if new_entry.id not in existing_ids:
                rag.entries.insert(0, new_entry)  # front = highest priority
                print(f"[UPVOTE] ✓ Added to live RAG: {q[:60]}")
            else:
                print(f"[UPVOTE] Already in RAG (duplicate): {q[:60]}")

        except Exception as e:
            print(f"[UPVOTE] Error: {e}")

    feedback[str(bot_idx)] = "up"
    
    user_email = (sauth or {}).get("email","unknown")
    user_q = next((msgs[i].get("content","") for i in range(bot_idx-1,-1,-1)
                   if msgs[i]["role"] == "user"), "")
    bot_answer = msgs[bot_idx].get("content", "") if bot_idx < len(msgs) else ""
    log_session_event(
        user_email   = user_email,
        question     = user_q,
        llm_response = bot_answer,
        vote         = "up",
    )
    
    conv = next((c for c in convs if c["id"] == cid), None)
    if conv:
        conv["feedback"] = feedback
    return feedback, render_stream(msgs, feedback)


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

    user_msg = next((msgs[i] for i in range(bot_idx - 1, -1, -1)
                     if msgs[i]["role"] == "user"), None)
    if not user_msg:
        return no_update, no_update, no_update, no_update

    original_q = user_msg.get("content", "").strip()
    retry_q = (
        f"{original_q}\n\n"
        f"[SYSTEM NOTE: The previous answer to this question was marked as unhelpful "
        f"by the GRE analyst. Re-analyse carefully — choose the correct table and "
        f"columns, verify time filters and ILIKE patterns, and generate a more accurate "
        f"SQL query. Do NOT repeat the prior response.]"
    )

    feedback[str(bot_idx)] = "dn"
    
    user_email = (sauth or {}).get("email","unknown")
    log_session_event(
        user_email = user_email,
        question = original_q,
        llm_response = msgs[bot_idx].get("content","") if bot_idx < len(msgs) else "",
        vote = "dn",
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
        "tables":  res.get("tables", []),
    }
    msgs.append({"role": "user",  "content": f"↻ Retry: {original_q}"})
    msgs.append(new_bot)

    conv = next((c for c in convs if c["id"] == cid), None)
    if conv:
        conv["msgs"]     = msgs
        conv["feedback"] = feedback

    return feedback, render_stream(msgs, feedback), msgs, convs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 23 — CSV DOWNLOAD CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

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
    bot_idx = triggered.get("index")
    if bot_idx is None or not any(n and n > 0 for n in (n_clicks_list or [])):
        return no_update
    msgs = list(msgs or [])
    if bot_idx >= len(msgs):
        return no_update
    bot_msg = msgs[bot_idx]
    rows = bot_msg.get("rows", [])
    cols = bot_msg.get("cols", [])
    if not rows:
        return no_update
    import io
    df  = pd.DataFrame(rows, columns=cols)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return dcc.send_string(buf.getvalue(), filename=f"gre_export_{int(time.time())}.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 24 — DOC SEARCH CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("sdoc_mode",  "data",      allow_duplicate=True),
    Output("doc-strip",  "className"),
    Output("docbtn",     "className"),
    Output("qin",        "placeholder"),
    Input("docbtn",      "n_clicks"),
    State("sdoc_mode",   "data"),
    prevent_initial_call=True,
)
def on_doc_toggle(n, currently_on):
    if not n:
        return no_update, no_update, no_update, no_update
    new_mode   = not bool(currently_on)
    strip_cls  = "doc-strip"            if new_mode else "doc-strip doc-strip-hidden"
    btn_cls    = "docbtn docbtn-active" if new_mode else "docbtn"
    placeholder = (
        "Ask a question about the uploaded document…"
        if new_mode else
        "Ask about roamers, KPIs, errors, circle analysis…"
    )
    return new_mode, strip_cls, btn_cls, placeholder


@app.callback(
    Output("sdoc",        "data",     allow_duplicate=True),
    Output("doc-info-row","children"),
    Input("doc-upload",   "contents"),
    State("doc-upload",   "filename"),
    prevent_initial_call=True,
)
def on_doc_upload(contents, filename):
    if not contents or not filename:
        return no_update, no_update
    text, err = parse_uploaded_doc(contents, filename)
    if err:
        return None, html.Span(f"❌  {err}", className="doc-err")
    word_count = len(text.split())
    info = html.Div([
        html.Span(f"✅  {filename}  ·  {word_count:,} words", className="doc-loaded"),
        html.Button("✕  Clear", id="doc-clear-btn", n_clicks=0, className="doc-clear"),
    ], className="doc-info-row")
    print(f"[DOC_UPLOAD] '{filename}' parsed — {word_count:,} words")
    return {"text": text, "name": filename, "words": word_count}, info


@app.callback(
    Output("sdoc",         "data",     allow_duplicate=True),
    Output("doc-info-row", "children", allow_duplicate=True),
    Input("doc-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_doc_clear(n):
    if not n:
        return no_update, no_update
    return None, html.Span()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_rag_embeddings()  # Run once to pre-build .npy embedding cache
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  GRE Analytics  ·  Vi SQM  ·  LangGraph 9-Node Pipeline (V2.4)  ║
╠══════════════════════════════════════════════════════════════════╣
║  http://0.0.0.0:18003                                            ║
║  Model  : {MODEL:<54}║
║  DB     : {PG['host']}:{PG['port']}/{PG['dbname']:<38}║
║  RAG    : {len(rag.entries)} entries loaded                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Tables :                                                        ║
║   view_inbound_failure_rate / view_outbound_failure_rate         ║
║   analytics_uniq_roamers_inbound / analytics_uniq_roamers        ║
║   analytics_inbound_response_data_hr / _outbound_hr              ║
║   ir_wsms_logs / ir_steering_master / ir_ntr_logs                ║
╠══════════════════════════════════════════════════════════════════╣
║  Routes : footprint_q · performance_q · error_q                  ║
║           circle_q · trend_q · sms_q · steering_q · ntr_q        ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline: Router → TableID → RAG → CoT → GoT-SQL               ║
║            → Validate → Execute → Explain  (+LLM Fallback)       ║
╚══════════════════════════════════════════════════════════════════╝""")
    # SECURITY FIX (Obs #6 — Absence of Secure Transport Controls): see
    # falcon_agent.py for details. Serve TLS directly if certs are provided,
    # otherwise this MUST run behind a TLS-terminating reverse proxy.
    _ssl_cert = os.environ.get("SSL_CERTFILE")
    _ssl_key  = os.environ.get("SSL_KEYFILE")
    _ssl_ctx  = (_ssl_cert, _ssl_key) if _ssl_cert and _ssl_key else None
    if _ssl_ctx is None:
        print("[SECURITY WARNING] SSL_CERTFILE/SSL_KEYFILE not set — serving "
              "plain HTTP. Deploy behind a TLS-terminating reverse proxy.")
    app.run(host="0.0.0.0", port=19000, debug=False, dev_tools_hot_reload=False, ssl_context=_ssl_ctx)
