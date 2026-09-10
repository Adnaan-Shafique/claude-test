# Tests

```bash
pip install pytest
```

## Unit tests — no services needed

```bash
pytest tests/test_backend.py
```

Covers response parsing, credential and upload validation, prompt construction, JWT
round-trips, and the GPU proxy's prompt flattening. `tests/conftest.py` supplies dummy
environment variables because `backend.py` validates its configuration at import time.

## Integration tests — needs PostgreSQL

These exercise the approval workflow, login throttling, conversation ownership and the
memory window against a real database. Point them at a **scratch** database: the suite
truncates every table between tests.

```bash
createdb forge_test
psql -d forge_test -f init.sql

FORGE_TEST_DB=1 \
POSTGRES_USER=... POSTGRES_PASSWORD=... POSTGRES_DB=forge_test \
POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
GPU_PROXY_URL=http://127.0.0.1:8071/v1/infer \
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
pytest tests/test_integration.py
```

Without `FORGE_TEST_DB` set, they skip.

No GPU proxy is required — the tests stub `generate_code` and `summarize_qa`. pgvector is
not required either: the similarity-search helpers are the only callers of the vector
operators and they are stubbed, since these tests are about ownership and scoping rather
than nearest-neighbour search. If your test database lacks pgvector, drop the
`CREATE EXTENSION` line and change `vector(384)` to `text` when applying `init.sql` to it.
