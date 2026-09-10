"""Test fixtures.

These tests cover the pure logic — prompt building, response parsing, credential
validation, upload checks, JWT round-trips — so they need neither a database nor the
GPU proxy. The required environment variables are set here because backend.py validates
its configuration at import time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("GPU_PROXY_URL", "http://127.0.0.1:8071/v1/infer")
os.environ.setdefault("SECRET_KEY", "0" * 64)
