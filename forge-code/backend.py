"""
Forge Code — backend: configuration, database, auth, LLM, memory and business logic.

The Dash UI lives in app.py and calls into this module. Nothing here imports Dash, so
everything below can be exercised from a REPL or a test without starting a web server.

Sections:
  1. Configuration (environment only — no secrets in source)
  2. Database (pooled connections)
  3. Auth (bcrypt + JWT in an httpOnly cookie, admin approval, login throttling)
  4. Embeddings & similarity search (pgvector)
  5. LLM — Mistral via the GPU proxy: prompts, generation, response parsing
  6. Memory (short-term per conversation / long-term per user)
  7. Conversations (sidebar history, ownership-checked)
  8. Business logic (ask, feedback, admin actions)
  9. Health
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import psycopg2
import requests
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

# sentence_transformers drags in torch, which takes seconds to import and is only needed
# once an embedding is actually requested — so it is imported inside get_embedding_model().

# ============================================================
# 1. CONFIGURATION
# ============================================================


class ConfigError(RuntimeError):
    """Raised at startup when a required environment variable is missing or unusable."""


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"Required environment variable {name} is not set. "
            f"See README.md for the full list of variables."
        )
    return value or ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Database -----------------------------------------------------------
POSTGRES_USER = _env("POSTGRES_USER", required=True)
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", required=True)
POSTGRES_DB = _env("POSTGRES_DB", required=True)
POSTGRES_HOST = _env("POSTGRES_HOST", required=True)
POSTGRES_PORT = _env("POSTGRES_PORT", "5432")
DB_POOL_MIN = _env_int("DB_POOL_MIN", 1)
DB_POOL_MAX = _env_int("DB_POOL_MAX", 10)

# --- Mistral (internal GPU proxy: raw prompt-completion, not OpenAI-compatible) ---
GPU_PROXY_URL = _env("GPU_PROXY_URL", required=True)
GPU_API_KEY = _env("GPU_API_KEY")
GPU_TIMEOUT = _env_int("GPU_TIMEOUT", 180)
MISTRAL_MODEL = _env("MISTRAL_MODEL_NAME", "mistral")
GPU_MAX_NEW_TOKENS = _env_int("GPU_MAX_NEW_TOKENS", 1024)
GPU_RETRIES = _env_int("GPU_RETRIES", 2)

# --- Auth ---------------------------------------------------------------
SECRET_KEY = _env("SECRET_KEY", required=True)
if len(SECRET_KEY.encode()) < 32:
    raise ConfigError(
        "SECRET_KEY must be at least 32 bytes (PyJWT warns below that for HS256). "
        "Generate one with:  python3 -c 'import secrets; print(secrets.token_hex(32))'"
    )
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_HOURS = _env_int("TOKEN_LIFETIME_HOURS", 12)
COOKIE_NAME = "forge_session"
# Set COOKIE_SECURE=true once the app is behind HTTPS. Left off by default because the
# current deployment is plain http on an internal VM, where a Secure cookie is never sent.
COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)

MIN_PASSWORD_LENGTH = _env_int("MIN_PASSWORD_LENGTH", 10)
MAX_PASSWORD_LENGTH = 128  # bcrypt only reads the first 72 bytes; reject early rather than silently truncating
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

LOGIN_MAX_ATTEMPTS = _env_int("LOGIN_MAX_ATTEMPTS", 8)
LOGIN_WINDOW_SECONDS = _env_int("LOGIN_WINDOW_SECONDS", 300)
REGISTER_MAX_ATTEMPTS = _env_int("REGISTER_MAX_ATTEMPTS", 5)
REGISTER_WINDOW_SECONDS = _env_int("REGISTER_WINDOW_SECONDS", 3600)

# --- Embeddings ---------------------------------------------------------
# Must produce 384-dimensional vectors to match the vector(384) columns in init.sql.
EMBEDDING_MODEL_PATH = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# --- Behaviour ----------------------------------------------------------
SHORT_TERM_LIMIT = _env_int("SHORT_TERM_LIMIT", 5)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))
CONVERSATION_RETENTION_DAYS = _env_int("CONVERSATION_RETENTION_DAYS", 7)
TITLE_MAX_LEN = 48
MAX_QUESTION_CHARS = _env_int("MAX_QUESTION_CHARS", 8000)
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 200_000)
SUPPORTED_UPLOAD_EXTENSIONS = (".py", ".sql")


class AppError(Exception):
    """Raised by business-logic functions with a message safe to show the user directly."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# ============================================================
# 2. DATABASE
# ============================================================

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ThreadedConnectionPool:
    """Lazily open the connection pool so importing this module never touches the network."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadedConnectionPool(
                    DB_POOL_MIN,
                    DB_POOL_MAX,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    dbname=POSTGRES_DB,
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    cursor_factory=RealDictCursor,
                )
    return _pool


@contextmanager
def db_cursor(commit: bool = False):
    """Borrow a pooled connection and yield a cursor.

    Commits on clean exit when commit=True, always rolls back on error, and always
    returns the connection to the pool. The old code opened a fresh connection per
    query — a single question cost eight connect/TLS handshakes.
    """
    pool = get_pool()
    try:
        conn = pool.getconn()
    except psycopg2.OperationalError as exc:
        raise AppError("The database is unreachable. Try again shortly.") from exc

    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
        else:
            conn.rollback()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ============================================================
# 3. AUTH
# ============================================================


class _RateLimiter:
    """Fixed-window attempt counter, keyed by IP or username. In-process by design:
    the app runs as a single Python process, so a shared store would be overkill."""

    def __init__(self, max_attempts: int, window_seconds: int):
        self._max = max_attempts
        self._window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Record an attempt. Returns False once the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False
            hits.append(now)
            return True

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


_login_limiter = _RateLimiter(LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS)
_register_limiter = _RateLimiter(REGISTER_MAX_ATTEMPTS, REGISTER_WINDOW_SECONDS)


def create_token(user_id: int, username: str, is_admin: bool) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "username": username,
        "is_admin": bool(is_admin),
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_LIFETIME_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Verify a JWT's signature and expiry. Returns its claims, or None if invalid."""
    if not token:
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def resolve_user(token: str) -> dict | None:
    """Turn a cookie value into a live user record.

    The signature check alone is not enough: it would keep honouring tokens issued to
    accounts that have since been deleted or deactivated, for as long as the token
    lives. So the row is re-read on every request and the DB, not the token, decides
    whether the session is still valid.
    """
    claims = decode_token(token)
    if not claims:
        return None
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, username, is_admin, is_active FROM coding_agent_schema.users WHERE id = %s",
            (claims.get("user_id"),),
        )
        row = cur.fetchone()
    if not row or not row["is_active"]:
        return None
    return {"user_id": row["id"], "username": row["username"], "is_admin": row["is_admin"]}


def validate_credentials(username: str, password: str, email: str | None = None) -> None:
    """Raise AppError describing the first problem with a proposed set of credentials."""
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise AppError(
            "Username must be 3-32 characters, using letters, digits, dot, underscore or hyphen only."
        )
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise AppError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode()) > MAX_PASSWORD_LENGTH:
        raise AppError(f"Password must be at most {MAX_PASSWORD_LENGTH} bytes.")
    if password.lower() == username.lower():
        raise AppError("Password must not be the same as the username.")
    if email and not EMAIL_RE.match(email.strip()):
        raise AppError("That does not look like a valid email address.")


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def register_user(
    username: str,
    password: str,
    email: str | None = None,
    *,
    client_ip: str = "unknown",
    is_admin: bool = False,
    is_active: bool = False,
) -> dict:
    """Create an account. New accounts are inactive until an admin approves them."""
    if not _register_limiter.check(client_ip):
        raise AppError("Too many registration attempts. Try again later.")

    username = (username or "").strip()
    email = (email or "").strip() or None
    validate_credentials(username, password, email)

    password_hash = _hash_password(password)
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO coding_agent_schema.users (username, email, password_hash, is_active, is_admin) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (username, email, password_hash, is_active, is_admin),
            )
            user_id = cur.fetchone()["id"]
    except psycopg2.errors.UniqueViolation:
        # Relies on the unique index rather than a SELECT-then-INSERT, which raced.
        raise AppError("That username is already taken.")

    return {"user_id": user_id, "username": username, "is_active": is_active, "is_admin": is_admin}


def login_user(username: str, password: str, *, client_ip: str = "unknown") -> dict:
    username = (username or "").strip()
    if not username or not password:
        raise AppError("Username and password are required.")

    # Throttle on both the source address and the account, so neither a single host
    # spraying many usernames nor many hosts targeting one account gets unlimited tries.
    if not _login_limiter.check(f"ip:{client_ip}") or not _login_limiter.check(f"user:{username.lower()}"):
        raise AppError("Too many failed login attempts. Try again in a few minutes.")

    with db_cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash, is_active, is_admin "
            "FROM coding_agent_schema.users WHERE LOWER(username) = LOWER(%s)",
            (username,),
        )
        row = cur.fetchone()

    # Hash against a dummy value when the user does not exist so that a missing account
    # and a wrong password take the same amount of time.
    stored_hash = row["password_hash"] if row else "$2b$12$" + "." * 53
    password_ok = bcrypt.checkpw(password.encode(), stored_hash.encode())

    if not row or not password_ok:
        raise AppError("Invalid username or password.")

    if not row["is_active"]:
        raise AppError("Your account is awaiting administrator approval.")

    _login_limiter.reset(f"ip:{client_ip}")
    _login_limiter.reset(f"user:{username.lower()}")

    token = create_token(row["id"], row["username"], row["is_admin"])
    return {
        "token": token,
        "user_id": row["id"],
        "username": row["username"],
        "is_admin": row["is_admin"],
    }


# --- Admin actions ------------------------------------------------------

def _require_admin(admin_id: int) -> None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT is_admin, is_active FROM coding_agent_schema.users WHERE id = %s",
            (admin_id,),
        )
        row = cur.fetchone()
    if not row or not row["is_active"] or not row["is_admin"]:
        raise AppError("Administrator privileges are required for that action.")


def list_users(admin_id: int) -> list[dict]:
    """All accounts, pending first, for the admin screen."""
    _require_admin(admin_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, username, email, is_active, is_admin, created_at, approved_at "
            "FROM coding_agent_schema.users ORDER BY is_active ASC, created_at DESC"
        )
        return cur.fetchall()


def set_user_active(admin_id: int, user_id: int, active: bool) -> None:
    """Approve (active=True) or suspend (active=False) an account."""
    _require_admin(admin_id)
    if admin_id == user_id and not active:
        raise AppError("You cannot suspend your own account.")
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE coding_agent_schema.users "
            "SET is_active = %s, approved_by = %s, approved_at = CASE WHEN %s THEN NOW() ELSE approved_at END "
            "WHERE id = %s",
            (active, admin_id, active, user_id),
        )
        if cur.rowcount == 0:
            raise AppError("No such user.")


def set_user_admin(admin_id: int, user_id: int, is_admin: bool) -> None:
    _require_admin(admin_id)
    if admin_id == user_id and not is_admin:
        raise AppError("You cannot remove your own administrator role.")
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE coding_agent_schema.users SET is_admin = %s WHERE id = %s",
            (is_admin, user_id),
        )
        if cur.rowcount == 0:
            raise AppError("No such user.")


def count_pending_users() -> int:
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM coding_agent_schema.users WHERE is_active = FALSE")
        return cur.fetchone()["n"]


def bootstrap_admin(username: str, password: str, email: str | None = None) -> dict:
    """Create the first administrator (used by `python app.py create-admin`)."""
    return register_user(username, password, email, client_ip="cli", is_admin=True, is_active=True)


# ============================================================
# 4. EMBEDDINGS & SIMILARITY SEARCH
# ============================================================

_embedding_model = None  # lazily-loaded SentenceTransformer
_embedding_lock = threading.Lock()


def get_embedding_model():
    """Load the sentence-transformer on first use, and verify its output width matches
    the vector(384) columns — a mismatched model otherwise fails deep inside a query."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(EMBEDDING_MODEL_PATH, device="cpu")
                dim = model.get_sentence_embedding_dimension()
                if dim != EMBEDDING_DIM:
                    raise ConfigError(
                        f"EMBEDDING_MODEL {EMBEDDING_MODEL_PATH!r} produces {dim}-dimensional "
                        f"vectors but the schema declares vector({EMBEDDING_DIM}). Use a "
                        f"{EMBEDDING_DIM}-dimensional model, or change the embedding columns "
                        f"in init.sql to match and re-index."
                    )
                _embedding_model = model
    return _embedding_model


def embed(text: str) -> list[float]:
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def _to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# Each example table stores a slightly different row shape; the reason note only
# exists on flagged answers.
_EXAMPLE_COLUMNS = {
    "golden_examples": "question, answer",
    "flagged_answers": "question, answer, reason",
}


def _find_best_match(table: str, user_id: int, question: str) -> dict | None:
    """Nearest stored example for this user's question, or None if nothing is close enough."""
    columns = _EXAMPLE_COLUMNS.get(table)
    if columns is None:
        raise ValueError(f"unsupported example table: {table}")

    with db_cursor() as cur:
        # Cheap existence check first. Embedding means loading a sentence-transformer
        # (and torch) into memory, which is pure waste for a user who has not rated any
        # answers yet — which is every user until they click a thumb.
        cur.execute(
            f"SELECT 1 FROM coding_agent_schema.{table} WHERE user_id = %s LIMIT 1",
            (user_id,),
        )
        if cur.fetchone() is None:
            return None

    vec = _to_pgvector(embed(question))
    with db_cursor() as cur:
        # Scoped to the asking user: examples are built from that developer's own code,
        # and must never be pulled into somebody else's prompt.
        cur.execute(
            f"SELECT {columns}, 1 - (embedding <=> %s::vector) AS similarity "
            f"FROM coding_agent_schema.{table} "
            f"WHERE user_id = %s AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> %s::vector ASC LIMIT 1",
            (vec, user_id, vec),
        )
        row = cur.fetchone()

    if row and row["similarity"] is not None and row["similarity"] >= SIMILARITY_THRESHOLD:
        return row
    return None


def find_golden_example(user_id: int, question: str) -> dict | None:
    return _find_best_match("golden_examples", user_id, question)


def find_flagged_answer(user_id: int, question: str) -> dict | None:
    return _find_best_match("flagged_answers", user_id, question)


def store_golden_example(user_id: int, question: str, answer: str) -> None:
    vec = _to_pgvector(embed(question))
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.golden_examples (user_id, question, answer, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            (user_id, question, answer, vec),
        )


def store_flagged_answer(user_id: int, question: str, answer: str, reason: str | None) -> None:
    vec = _to_pgvector(embed(question))
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.flagged_answers (user_id, question, answer, reason, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)",
            (user_id, question, answer, reason, vec),
        )


# ============================================================
# 5. LLM — Mistral via the GPU proxy
# ============================================================

# The tool is deliberately scoped to Python and SQL only, so the prompts say so
# explicitly rather than leaving the model to guess what is in scope.
SCOPE_RULE = (
    "You only support Python and SQL. If the developer asks for code in any other language, "
    "say plainly that only Python and SQL are supported and offer the Python or SQL equivalent "
    "if one makes sense. Never emit code in another language."
)

CODE_SYSTEM_PROMPT = (
    "You are a coding assistant for Python and SQL. Respond with executable Python or SQL code "
    "in a fenced code block tagged with the language (```python or ```sql), plus a brief, "
    "plain-English instruction of one or two short sentences describing what it does or how to "
    "run/use it. Keep the instruction minimal - no long explanations, no markdown prose beyond "
    "that one instruction, no restating the question.\n" + SCOPE_RULE
)

FILE_EDIT_SYSTEM_PROMPT = (
    "You are a coding assistant for Python and SQL, helping a developer with a code file they "
    "have uploaded, given in the [UPLOADED FILE] section below. That file is the ONLY source of "
    "truth for its contents - base every answer strictly on what is actually written in it, never "
    "on similarly-themed code from earlier questions in [LONG-TERM MEMORY] or [SHORT-TERM MEMORY]. "
    "Those sections are prior conversation context only, not part of this file.\n"
    "If the developer asks for a code change, respond with the necessary change(s) in a fenced "
    "code block tagged with the language, plus brief plain-English instructions on where in the "
    "file to place it or how to apply it (e.g. \"replace the body of function X\" or \"add this "
    "after the imports\"). Keep instructions short, and do not rewrite parts of the file that do "
    "not need to change.\n"
    "If the developer instead asks a question about the file (e.g. to summarize or explain it), "
    "answer in plain English describing what is actually in the file - do not include a code block "
    "unless a code change was specifically requested.\n"
    "When referring to a location in the file, cite the line or function by name so the developer "
    "can find it.\n" + SCOPE_RULE
)

CLARIFY_ADDENDUM = (
    "\n\nIf the request does not give you enough information to produce a correct, runnable answer "
    "(for example, a SQL query without knowing the exact table or column names), do not guess. "
    "Instead, respond in plain English asking exactly what additional information you need, and do "
    "not include a code block in that response."
)

SUMMARIZER_SYSTEM_PROMPT = (
    "You maintain a rolling summary of a developer's past questions and the code they were given. "
    "Given the existing summary (if any) and one new question/answer pair, write an updated summary "
    "in plain English, a few sentences long, capturing what the developer has been working on. "
    "Return only the summary text."
)

# A fenced block, capturing the optional language tag and the body. Tolerates \r\n and a
# missing trailing newline before the closing fence.
CODE_FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)

_LANGUAGE_ALIASES = {
    "py": "Python",
    "python": "Python",
    "python3": "Python",
    "sql": "SQL",
    "postgres": "SQL",
    "postgresql": "SQL",
    "psql": "SQL",
    "plpgsql": "SQL",
    "mysql": "SQL",
    "sqlite": "SQL",
}

_SQL_HINT_RE = re.compile(
    r"\b(SELECT\s|INSERT\s+INTO|UPDATE\s|DELETE\s+FROM|CREATE\s+(TABLE|INDEX|VIEW|SCHEMA)|ALTER\s+TABLE|WITH\s+\w+\s+AS)\b",
    re.IGNORECASE,
)
_PYTHON_HINT_RE = re.compile(r"^\s*(def |class |import |from \w+ import |@|print\()", re.MULTILINE)


class GPUApiClient:
    """Client for the internal GPU proxy serving Mistral (raw prompt completion, not chat).

    The proxy takes a single flat prompt and returns {"text": ...}, so chat-style
    message lists are flattened into SYSTEM/USER/ASSISTANT turns before sending.
    """

    def __init__(self, api_key: str = GPU_API_KEY, proxy_url: str = GPU_PROXY_URL, timeout: int = GPU_TIMEOUT):
        self._url = proxy_url
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._session = requests.Session()

    def infer(
        self,
        prompt: str,
        model: str = MISTRAL_MODEL,
        max_new_tokens: int = GPU_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 3,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
        }

        last_exc: Exception | None = None
        for attempt in range(GPU_RETRIES + 1):
            try:
                response = self._session.post(
                    self._url, json=payload, headers=self._headers, timeout=self._timeout
                )
                response.raise_for_status()
                return response.json().get("text", "")
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Transient: the proxy was restarting or the GPU was saturated. Retry.
                last_exc = exc
                if attempt < GPU_RETRIES:
                    time.sleep(2 ** attempt)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (429, 502, 503, 504) and attempt < GPU_RETRIES:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Flatten an OpenAI-style message list into the proxy's single-prompt format."""
        role_labels = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
        parts = [
            f"{role_labels.get(m['role'], m['role'].upper())}:\n{(m.get('content') or '').strip()}"
            for m in messages
        ]
        parts.append("ASSISTANT:\n")
        return self.infer(prompt="\n\n".join(parts), **kwargs)


_gpu_client: GPUApiClient | None = None
_gpu_lock = threading.Lock()


def get_gpu_client() -> GPUApiClient:
    global _gpu_client
    if _gpu_client is None:
        with _gpu_lock:
            if _gpu_client is None:
                _gpu_client = GPUApiClient()
    return _gpu_client


def build_prompt(
    question: str,
    long_term_summary: str | None,
    short_term: list[dict],
    golden_example: dict | None = None,
    flagged_answer: dict | None = None,
    uploaded_file: dict | None = None,
) -> list[dict]:
    base_prompt = FILE_EDIT_SYSTEM_PROMPT if uploaded_file else CODE_SYSTEM_PROMPT
    system_prompt = base_prompt + CLARIFY_ADDENDUM

    sections = [
        "[LONG-TERM MEMORY]",
        f"Summary of this user's past work:\n{long_term_summary or 'No long-term history yet.'}",
    ]

    if short_term:
        sections.append("\n[SHORT-TERM MEMORY]")
        sections.append(f"Last {len(short_term)} questions and answers in this conversation:")
        for qa in short_term:
            sections.append(f"Q: {qa['question']}\nA: {qa['answer']}")

    if golden_example:
        sections.append("\n[GOOD EXAMPLE - USE AS REFERENCE]")
        sections.append("For a similar question before, this answer was rated good:")
        sections.append(f"Q: {golden_example['question']}\nA: {golden_example['answer']}")

    if flagged_answer:
        sections.append("\n[BAD EXAMPLE - AVOID THIS PATTERN]")
        sections.append("For a similar question before, this answer was rated bad:")
        sections.append(f"Q: {flagged_answer['question']}\nA: {flagged_answer['answer']}")
        sections.append(f"Reason: {flagged_answer.get('reason') or 'Not specified.'}")

    if uploaded_file:
        sections.append(f"\n[UPLOADED FILE: {uploaded_file['name']}]")
        sections.append(uploaded_file["content"])

    sections.append("\n[CURRENT QUESTION]")
    sections.append(question)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(sections)},
    ]


def generate_code(
    question: str,
    long_term_summary: str | None = None,
    short_term: list[dict] | None = None,
    golden_example: dict | None = None,
    flagged_answer: dict | None = None,
    uploaded_file: dict | None = None,
) -> str:
    messages = build_prompt(
        question, long_term_summary, short_term or [], golden_example, flagged_answer, uploaded_file
    )
    return get_gpu_client().chat(messages)


def summarize_qa(existing_summary: str | None, question: str, answer: str) -> str:
    user_content = (
        f"Existing summary:\n{existing_summary or 'None yet.'}\n\n"
        f"New Q&A to fold in:\nQ: {question}\nA: {answer}"
    )
    return get_gpu_client().chat(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_new_tokens=256,
    )


def guess_language(code: str) -> str:
    """Best-effort language label for a code block whose fence carried no tag."""
    code = code or ""
    if _PYTHON_HINT_RE.search(code):
        return "Python"
    if _SQL_HINT_RE.search(code):
        return "SQL"
    return "Python"


def parse_response(raw: str) -> dict:
    """Split a raw LLM answer into its fenced code blocks plus the surrounding prose.

    Returns `blocks` as a list so a multi-part answer (e.g. a schema change and the
    query that uses it) renders every block instead of silently dropping all but the
    first, which is what the single-match version did.
    """
    raw = raw or ""
    matches = list(CODE_FENCE_RE.finditer(raw))

    if not matches:
        # An answer truncated mid-block has an opening fence and no closing one. Treat
        # everything after that fence as code rather than showing the raw backticks.
        opening = re.search(r"```([A-Za-z0-9_+#.-]*)[ \t]*\r?\n", raw)
        if opening:
            code = raw[opening.end():].strip()
            if code:
                language = _LANGUAGE_ALIASES.get(opening.group(1).lower()) or guess_language(code)
                return {
                    "response_type": "code",
                    "instructions": raw[: opening.start()].strip() or None,
                    "code": code,
                    "blocks": [{"language": language, "code": code, "truncated": True}],
                }
        return {"response_type": "message", "instructions": None, "code": None, "blocks": []}

    blocks = []
    for match in matches:
        code = match.group(2).strip()
        if not code:
            continue
        language = _LANGUAGE_ALIASES.get(match.group(1).lower()) or guess_language(code)
        blocks.append({"language": language, "code": code, "truncated": False})

    if not blocks:
        return {"response_type": "message", "instructions": None, "code": None, "blocks": []}

    # Prose is whatever sits outside the fences, stitched back together in order.
    prose_parts, cursor = [], 0
    for match in matches:
        prose_parts.append(raw[cursor:match.start()])
        cursor = match.end()
    prose_parts.append(raw[cursor:])
    instructions = "\n".join(part.strip() for part in prose_parts if part.strip()).strip()

    return {
        "response_type": "code",
        "instructions": instructions or None,
        "code": blocks[0]["code"],
        "blocks": blocks,
    }


# ============================================================
# 6. MEMORY — short-term (per conversation) / long-term (per user)
# ============================================================

def get_long_term_summary(user_id: int) -> str | None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT summary FROM coding_agent_schema.long_term_memory WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    return row["summary"] if row else None


def get_short_term_history(user_id: int, conversation_id: int | None) -> list[dict]:
    """The recent Q&A window used as prompt context.

    Scoped to one conversation: the previous version pulled the user's last answers
    regardless of which chat they came from, so starting a "New Chat" still fed the
    model context from an unrelated thread.
    """
    if conversation_id is None:
        return []  # a brand-new chat has no history yet
    with db_cursor() as cur:
        cur.execute(
            "SELECT question, answer FROM coding_agent_schema.short_term_memory "
            "WHERE user_id = %s AND conversation_id = %s AND answer IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, conversation_id, SHORT_TERM_LIMIT),
        )
        rows = cur.fetchall()
    return list(reversed(rows))  # oldest first, so the prompt reads chronologically


def enforce_memory_limit(user_id: int, conversation_id: int | None) -> None:
    """Fold anything past the sliding window into the rolling long-term summary.

    Loops rather than handling a single row, so a window that somehow grew past the
    limit (e.g. an earlier crash between insert and fold) drains fully instead of
    staying permanently over budget.
    """
    while True:
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, question, answer FROM coding_agent_schema.short_term_memory "
                "WHERE user_id = %s AND conversation_id IS NOT DISTINCT FROM %s AND answer IS NOT NULL "
                "ORDER BY created_at ASC",
                (user_id, conversation_id),
            )
            rows = cur.fetchall()

        if len(rows) <= SHORT_TERM_LIMIT:
            return

        oldest = rows[0]
        existing_summary = get_long_term_summary(user_id)

        try:
            new_summary = summarize_qa(existing_summary, oldest["question"], oldest["answer"])
        except requests.RequestException:
            # The answer is already saved and shown; losing the summary update is not
            # worth failing the user's request over. It will be retried next turn.
            return

        if not new_summary or not new_summary.strip():
            return

        with db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO coding_agent_schema.long_term_memory (user_id, summary) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW()",
                (user_id, new_summary.strip()),
            )
            cur.execute(
                "DELETE FROM coding_agent_schema.short_term_memory WHERE id = %s",
                (oldest["id"],),
            )


# ============================================================
# 7. CONVERSATIONS — sidebar history, ownership-checked
# ============================================================

def make_title(question: str) -> str:
    question = " ".join((question or "").split())
    if not question:
        return "New conversation"
    return question if len(question) <= TITLE_MAX_LEN else question[: TITLE_MAX_LEN - 1].rstrip() + "…"


def create_conversation(user_id: int, title: str) -> int:
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.conversations (user_id, title) VALUES (%s, %s) RETURNING id",
            (user_id, title),
        )
        return cur.fetchone()["id"]


def assert_conversation_owner(user_id: int, conversation_id: int) -> None:
    """Guard every conversation-scoped read/write.

    Conversation ids arrive from the browser, so without this check any logged-in user
    could read or append to another user's chat just by guessing a sequential id.
    """
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM coding_agent_schema.conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        if cur.fetchone() is None:
            raise AppError("That conversation is not available.")


def save_message(conversation_id: int, question: str, answer: str, file_name: str | None = None) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.messages (conversation_id, question, answer, file_name) "
            "VALUES (%s, %s, %s, %s)",
            (conversation_id, question, answer, file_name),
        )


def list_conversations_for_user(user_id: int) -> list[dict]:
    with db_cursor(commit=True) as cur:
        # Build the interval from an integer parameter instead of interpolating into the
        # literal, which only worked by accident.
        cur.execute(
            "DELETE FROM coding_agent_schema.conversations "
            "WHERE user_id = %s AND created_at < NOW() - make_interval(days => %s)",
            (user_id, CONVERSATION_RETENTION_DAYS),
        )
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at FROM coding_agent_schema.conversations "
            "WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        return cur.fetchall()


def get_conversation_messages(user_id: int, conversation_id: int) -> list[dict]:
    assert_conversation_owner(user_id, conversation_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT question, answer, file_name FROM coding_agent_schema.messages "
            "WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "question": row["question"],
            "answer": row["answer"],
            "file_name": row["file_name"],
            **parse_response(row["answer"] or ""),
        }
        for row in rows
    ]


# ============================================================
# 8. BUSINESS LOGIC
# ============================================================

def validate_upload(file_name: str | None, file_content: str | None) -> dict | None:
    """Check an attachment against the supported languages and the size cap.

    Enforced here as well as in the browser, because the browser check is only a
    convenience — the store it writes to is client-side and can be set to anything.
    """
    if not file_name or file_content is None:
        return None

    name = os.path.basename(file_name.strip())
    if not name.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        supported = " or ".join(SUPPORTED_UPLOAD_EXTENSIONS)
        raise AppError(f"Only {supported} files are supported.")

    size = len(file_content.encode("utf-8", errors="ignore"))
    if size > MAX_UPLOAD_BYTES:
        raise AppError(
            f"{name} is {size // 1024} KB, over the {MAX_UPLOAD_BYTES // 1024} KB limit. "
            f"Attach the relevant part of the file instead."
        )
    if not file_content.strip():
        raise AppError(f"{name} is empty.")

    return {"name": name, "content": file_content}


def ask_logic(
    user_id: int,
    question: str,
    conversation_id: int | None = None,
    file_name: str | None = None,
    file_content: str | None = None,
) -> dict:
    """Answer one question. Mistral is the only model, so there is no model argument.

    Nothing is persisted until the model has actually answered: a failed or rejected
    request used to leave an empty conversation in the sidebar and an answerless row in
    short-term memory behind it.
    """
    question = (question or "").strip()
    if not question:
        raise AppError("Ask a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise AppError(f"Question is too long (limit {MAX_QUESTION_CHARS} characters).")

    uploaded_file = validate_upload(file_name, file_content)

    # A conversation id arriving from the browser is verified before it is used for
    # anything, including as a key to read this user's prompt context.
    if conversation_id is not None:
        assert_conversation_owner(user_id, conversation_id)

    long_term_summary = get_long_term_summary(user_id)
    short_term = get_short_term_history(user_id, conversation_id)
    golden_example = find_golden_example(user_id, question)
    flagged_answer = find_flagged_answer(user_id, question)

    try:
        answer = generate_code(
            question, long_term_summary, short_term, golden_example, flagged_answer, uploaded_file
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in (401, 403):
            raise AppError("The coding assistant rejected this server's credentials. Contact your administrator.")
        raise AppError("The coding assistant returned an error. Try again shortly.")
    except requests.RequestException:
        raise AppError("The coding assistant model is unreachable. Try again shortly.")
    except ValueError:
        # The proxy answered with something that was not JSON.
        raise AppError("The coding assistant returned an unreadable response. Try again shortly.")

    if not answer or not answer.strip():
        raise AppError("The coding assistant returned an empty response.")

    if conversation_id is None:
        conversation_id = create_conversation(user_id, make_title(question))

    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.short_term_memory "
            "(user_id, conversation_id, question, answer) VALUES (%s, %s, %s, %s)",
            (user_id, conversation_id, question, answer),
        )

    save_message(conversation_id, question, answer, uploaded_file["name"] if uploaded_file else None)
    enforce_memory_limit(user_id, conversation_id)

    parsed = parse_response(answer)
    return {
        "question": question,
        "answer": answer,
        "conversation_id": conversation_id,
        "file_name": uploaded_file["name"] if uploaded_file else None,
        **parsed,
    }


def submit_feedback(user_id: int, question: str, answer: str, vote: str, reason: str | None) -> None:
    if vote not in ("up", "down"):
        raise AppError("Invalid vote.")
    if not question or not answer:
        raise AppError("Nothing to rate.")

    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO coding_agent_schema.feedback (user_id, question, answer, vote, reason) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, question, answer, vote, reason),
        )

    if vote == "up":
        store_golden_example(user_id, question, answer)
    else:
        store_flagged_answer(user_id, question, answer, reason)


# ============================================================
# 9. HEALTH
# ============================================================

def health_check() -> dict:
    """Liveness for monitoring. Reports each dependency separately; never raises."""
    status = {"app": "ok", "db": "unreachable", "mistral": "unreachable", "embeddings": "unloaded"}

    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        status["db"] = "ok"
    except Exception as exc:
        status["db_error"] = str(exc)

    try:
        # A tiny real completion is the only honest check: the proxy's /v1/infer is a
        # POST-only endpoint, so a GET against it proves nothing about whether the
        # model behind it can actually answer.
        text = get_gpu_client().infer(prompt="SYSTEM:\nReply with OK.\n\nUSER:\nping\n\nASSISTANT:\n", max_new_tokens=4)
        status["mistral"] = "ok" if text is not None else "unreachable"
    except Exception as exc:
        status["mistral_error"] = str(exc)

    status["embeddings"] = "ok" if _embedding_model is not None else "unloaded"
    return status
