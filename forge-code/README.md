# Forge Code

A coding agent for **Python and SQL**. Developers upload a `.py` or `.sql` file and ask
questions about the code in it, ask for edit suggestions, or ask for code from scratch.
Answers come from **Mistral**, served through your internal GPU proxy.

Everything runs as two Python files in one process — no Docker, no build step.

| File | What it is |
| --- | --- |
| `app.py` | Dash UI, callbacks, `/health`, the `create-admin` command |
| `backend.py` | Config, database, auth, LLM, memory, business logic — no Dash imports |
| `init.sql` | Full database schema (run once) |
| `migrations/` | Upgrade scripts for databases created by an earlier version |
| `tests/` | Test suite (see `tests/README.md`) |
| `assets/` | Icons and the self-hosted DM Sans font — no CDN needed at runtime |

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch` to the CPU-only build from `download.pytorch.org` — this
app only ever runs it on CPU, for the embedding model, and the default build pulls ~2.5 GB
of unneeded CUDA libraries. **If your proxy can't reach `download.pytorch.org`**, delete the
`--extra-index-url` line and the `torch` line, then run `pip install torch` separately.

## 2. Create the database schema

Against your PostgreSQL + pgvector database:

```bash
psql -h <host> -U <user> -d <database> -f init.sql
```

Every statement is idempotent, so re-running it is safe.

**Upgrading an existing deployment** (one created by the previous eight-table `init.sql`)?
Run the migration instead — it adds the approval columns, scopes the example tables per
user, and activates the accounts you already created so nobody is locked out:

```bash
psql -h <host> -U <user> -d <database> -f migrations/001_auth_and_scoping.sql
```

## 3. Set environment variables

Nothing is hardcoded — the app refuses to start if a required variable is missing.

### Required

```bash
export POSTGRES_USER=...
export POSTGRES_PASSWORD=...
export POSTGRES_DB=...
export POSTGRES_HOST=...            # your Postgres server's hostname/IP
export POSTGRES_PORT=5432           # optional, defaults to 5432

export GPU_PROXY_URL=http://127.0.0.1:8071/v1/infer   # your Mistral GPU proxy
export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

`SECRET_KEY` signs session tokens. It must be **at least 32 bytes** — a shorter key makes
PyJWT warn and weakens the signature. Keep it stable: changing it logs everyone out.

### Optional

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_API_KEY` | *(none)* | Sent as `X-API-Key`; the header is omitted when unset |
| `GPU_TIMEOUT` | `180` | Seconds to wait for a completion |
| `GPU_RETRIES` | `2` | Retries on connection errors and 429/502/503/504 |
| `GPU_MAX_NEW_TOKENS` | `1024` | Output cap per answer |
| `MISTRAL_MODEL_NAME` | `mistral` | Model name sent to the proxy |
| `PORT` | `8054` | HTTP port |
| `FORGE_DEBUG` | off | Dash debug mode — **never enable on a shared host** |
| `COOKIE_SECURE` | `false` | Set `true` once the app is behind HTTPS |
| `TOKEN_LIFETIME_HOURS` | `12` | Session length |
| `MIN_PASSWORD_LENGTH` | `10` | Minimum password length at registration |
| `LOGIN_MAX_ATTEMPTS` / `LOGIN_WINDOW_SECONDS` | `8` / `300` | Login throttle |
| `REGISTER_MAX_ATTEMPTS` / `REGISTER_WINDOW_SECONDS` | `5` / `3600` | Registration throttle |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Path or name of the sentence-transformer |
| `SHORT_TERM_LIMIT` | `5` | Q&A pairs kept as prompt context per conversation |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum similarity for a stored example to be reused |
| `CONVERSATION_RETENTION_DAYS` | `7` | How long sidebar history is kept |
| `MAX_UPLOAD_BYTES` | `200000` | Attachment size cap |
| `MAX_QUESTION_CHARS` | `8000` | Question length cap |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `10` | Connection pool size |

The embedding model must produce **384-dimensional** vectors, matching the `vector(384)`
columns in `init.sql`; the app checks this at load time and says so if it doesn't.

## 4. Create the first administrator

Registration is self-service, but new accounts are inactive until an admin approves them —
so the first admin has to be made from the command line:

```bash
python app.py create-admin
```

It prompts for a username, an optional email, and a password (twice), and creates an
account that is active and admin from the start.

## 5. Run it

```bash
python app.py
```

Open `http://<this-machine>:8054`.

Werkzeug's built-in server is fine for a small internal team. For anything larger, put a
real WSGI server in front:

```bash
gunicorn --workers 4 --timeout 300 app:server
```

Use a `--timeout` comfortably above `GPU_TIMEOUT`, or long completions get killed mid-answer.
Note that the login throttle is per process, so it becomes per-worker under multiple workers.

## 6. Verify

```bash
curl http://localhost:8054/health
```

Returns `200` when both the database and Mistral answer, `503` otherwise, with a per-
dependency breakdown:

```json
{"app": "ok", "db": "ok", "mistral": "ok", "embeddings": "unloaded"}
```

`embeddings: unloaded` is normal — the embedding model loads on first use, which only
happens once somebody has rated an answer.

## How accounts work

1. A developer registers at `/register`. The account is created **inactive**.
2. An admin opens `/admin`, sees it under "Pending approval", and clicks **Approve**.
3. The developer can now log in.

Admins can also **Suspend** an account or promote someone to **Make admin**. Suspending
takes effect immediately, including for a session that is already open — every request
re-reads the account, so a valid token alone is not enough to keep working.

An admin cannot suspend their own account or drop their own admin role, so the last
administrator can't lock themselves out.

## Security notes

- Session tokens live in an **httpOnly, SameSite=Strict** cookie, so page JavaScript can't
  read them and cross-site requests don't carry them.
- The user id is derived from that cookie **on the server for every callback**. Nothing the
  browser sends about identity is trusted.
- Conversation ids are ownership-checked on every read and write.
- Good/bad examples replayed into prompts are scoped per user — one developer's code is
  never shown to another.
- Passwords are bcrypt-hashed. Logins are throttled per source address *and* per account,
  and a non-existent user costs the same time as a wrong password.
- Set `COOKIE_SECURE=true` and terminate TLS in front of the app for a real deployment.

## Scope

Python and SQL only. Attachments are limited to `.py` and `.sql` (checked in the browser
*and* on the server), and the system prompt tells the model to decline other languages
rather than answer in them.
