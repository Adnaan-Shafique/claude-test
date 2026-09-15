# proxy/llm_proxy_v3.py
#
# Forwards requests to the GPU server with:
#   - API key authentication  (X-API-Key header)
#   - client_id namespacing   (derived from the API key)
#   - request_id tracking     (UUID, round-tripped through GPU server)
#
# ── CHANGES FROM v2 ──────────────────────────────────────────────────────────
#
#  [1] MULTIMODAL PASSTHROUGH.  v2's InferRequest had no `images` field, so a
#      VLM request was silently stripped of its images before reaching the GPU
#      server and the model answered as if blind.  v3 forwards `images` and
#      `system`, and enforces a per-model image cap that mirrors the GPU
#      server's `limit_mm_per_prompt`.
#
#  [2] VLM MODELS ALLOWLISTED.  v2's AVAILABLE_MODELS listed neither
#      "qwen3-vl" nor "internvl", so those requests were rejected with a 422 by
#      the pydantic validator before any network call happened.  Both are now
#      in the default list, alongside the text models v2 shipped with.
#
#  [3] ASYNC UPSTREAM.  v2 used a *synchronous* `def infer` calling
#      `requests.post`.  Starlette runs sync endpoints on a bounded worker
#      thread pool (40 threads by default), so a load test at 30 images/sec
#      against a multi-second VLM would queue on the *proxy's* thread pool and
#      measure Starlette rather than the H200.  v3 is `async def` over
#      httpx.AsyncClient with an explicit connection pool.
#
#  [4] TIMEOUT RAISED OFF THE COLD-LOAD CLIFF.  v2's default was 120 s.  A
#      lazy VLM load (30B weights off disk + CUDA graph capture) routinely
#      exceeds that, so the very first request of a run would 504 while the
#      GPU server happily kept loading.  Default is now 600 s, and a separate
#      shorter connect timeout still fails fast on a genuinely dead upstream.
#
#  [5] OBSERVABILITY PASSTHROUGH.  /v1/metrics, /v1/client-metrics and
#      /v1/gpu-health expose the GPU server's counters through the proxy, so a
#      benchmark client that can only reach the proxy can still read
#      server-side truth (VRAM, per-model throughput, per-client tokens).
#
#  [6] The response now echoes `images` and `queue_wait_s` so a caller can
#      separate proxy overhead from GPU time.
#
# ── Quick-start ──────────────────────────────────────────────────────────────
#
#   export API_KEYS="alice:secret-abc123,bench:secret-bench"
#   export GPU_HOST=10.66.98.137
#   uvicorn llm_proxy_v3:app --host 0.0.0.0 --port 8071
#
# ── Client usage ─────────────────────────────────────────────────────────────
#
#   curl -X POST http://proxy:8071/v1/infer \
#        -H "X-API-Key: secret-bench" \
#        -H "Content-Type: application/json" \
#        -d '{"model":"qwen3-vl","prompt":"Extract all fields as JSON.",
#             "images":["data:image/png;base64,iVBOR..."],"max_new_tokens":768}'
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# pydantic v1 / v2 compatibility for the model-name validator.
try:                                    # pydantic v2
    from pydantic import field_validator

    def _model_validator(fn):
        return field_validator("model")(classmethod(fn))
except ImportError:                     # pydantic v1
    from pydantic import validator

    def _model_validator(fn):
        return validator("model", allow_reuse=True)(classmethod(fn))


# ── Configuration ─────────────────────────────────────────────────────────────

LOCAL_BIND_HOST     = os.environ.get("LOCAL_BIND_HOST", "0.0.0.0")
LOCAL_BIND_PORT     = int(os.environ.get("LOCAL_BIND_PORT", "8071"))
GPU_HOST            = os.environ.get("GPU_HOST", "10.66.98.137")
GPU_PORT            = os.environ.get("GPU_PORT", "5432")
GPU_INFER_PATH      = os.environ.get("GPU_INFER_PATH", "/infer")
GPU_BASE_URL        = os.environ.get("GPU_BASE_URL", f"http://{GPU_HOST}:{GPU_PORT}")
GPU_API_URL         = os.environ.get("GPU_API_URL", f"{GPU_BASE_URL}{GPU_INFER_PATH}")

# [4] A lazy 30B VLM load can take several minutes. The read timeout has to
# clear that; the connect timeout stays short so a dead box still fails fast.
GPU_TIMEOUT_SECONDS = float(os.environ.get("GPU_TIMEOUT_SECONDS", "600"))
GPU_CONNECT_TIMEOUT = float(os.environ.get("GPU_CONNECT_TIMEOUT", "10"))

# [3] Connection pool must be at least as deep as the concurrency you intend to
# offer, or the proxy itself becomes the queue you are trying to measure.
MAX_CONNECTIONS     = int(os.environ.get("MAX_CONNECTIONS", "512"))

ALLOW_ORIGINS       = os.environ.get("ALLOW_ORIGINS", "*")

# [2] qwen3-vl and internvl added — v2 omitted them and the validator rejected
# every VLM request with a 422 before it left the proxy.
AVAILABLE_MODELS    = [
    m.strip() for m in os.environ.get(
        "AVAILABLE_MODELS",
        "mistral,qwen3,codestral,qwen3-vl,internvl",
    ).split(",") if m.strip()
]

# Mirrors limit_mm_per_prompt in the GPU server's registry. Kept here so an
# over-limit request is rejected at the edge instead of after a multi-megabyte
# base64 upload has crossed the wire.
MODEL_IMAGE_CAPS: dict[str, int] = {
    "qwen3-vl": int(os.environ.get("QWEN3_VL_IMAGE_CAP", "4")),
    "internvl": int(os.environ.get("INTERNVL_IMAGE_CAP", "4")),
}
VISION_MODELS = set(MODEL_IMAGE_CAPS)

MAX_PROMPT_CHARS    = int(os.environ.get("MAX_PROMPT_CHARS", "32000"))
MAX_ALLOWED_TOKENS  = int(os.environ.get("MAX_ALLOWED_TOKENS", "8192"))
# Guards against a client shipping a 100 MB base64 page and stalling the proxy.
MAX_IMAGE_CHARS     = int(os.environ.get("MAX_IMAGE_CHARS", str(24 * 1024 * 1024)))

# ── API key registry ──────────────────────────────────────────────────────────
#
#  Set API_KEYS as a comma-separated list of  client_id:secret  pairs.
#  If API_KEYS is empty / unset, authentication is DISABLED and all requests
#  are labelled as client "anonymous".  Set at least one key in production.
#
#  "bench" is included so capacity-test traffic lands in its own bucket in
#  /client-metrics instead of polluting a real project's numbers.

_raw_keys = os.environ.get(
    "API_KEYS",
    "falcon:secret-falcon1,falcon2:secret-falcon2,zabbix:secret-zabbix,"
    "gentrace:secret-gentrace,gre:secret-gre,service360:secret-service360,"
    "npm:secret-npm,bench:secret-bench",
)

API_KEY_MAP: dict[str, str] = {}
for entry in _raw_keys.split(","):
    entry = entry.strip()
    if ":" not in entry:
        continue
    client_id, secret = entry.split(":", 1)
    API_KEY_MAP[secret.strip()] = client_id.strip()

AUTH_ENABLED = bool(API_KEY_MAP)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("llm-proxy")

if not AUTH_ENABLED:
    logger.warning(
        "API_KEYS is not set — authentication is DISABLED. "
        "Set API_KEYS=client_id:secret,... to enable per-client isolation."
    )

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="LLM Proxy", version="3.0.0")

origins = ["*"] if ALLOW_ORIGINS == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(GPU_TIMEOUT_SECONDS, connect=GPU_CONNECT_TIMEOUT),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_CONNECTIONS,
        ),
    )
    logger.info(
        "proxy v3 up — upstream=%s timeout=%ss pool=%d models=%s vision=%s",
        GPU_API_URL, GPU_TIMEOUT_SECONDS, MAX_CONNECTIONS,
        AVAILABLE_MODELS, sorted(VISION_MODELS),
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


# ── Schemas ───────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    model:          str
    prompt:         str
    # [1] Vision models only. Each entry may be a data URI, bare base64, an
    # http(s) URL, or a path on the GPU box — the GPU server decodes all four.
    images:         Optional[list[str]] = None
    system:         Optional[str]       = None
    max_new_tokens: Optional[int]   = Field(256,  ge=1,   le=MAX_ALLOWED_TOKENS)
    temperature:    Optional[float] = Field(0.7,  ge=0.0, le=2.0)
    top_p:          Optional[float] = Field(0.9,  ge=0.0, le=1.0)
    top_k:          Optional[int]   = Field(50,   ge=0)

    @_model_validator
    def model_allowed(cls, v):
        if v not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{v}'. Allowed: {AVAILABLE_MODELS}")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_client(x_api_key: Optional[str]) -> str:
    """
    Returns the client_id for the given API key.
    Raises HTTP 401/403 if auth is enabled and the key is missing/invalid.
    """
    if not AUTH_ENABLED:
        return "anonymous"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    client_id = API_KEY_MAP.get(x_api_key)
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return client_id


def _validate_multimodal(req: InferRequest) -> list[str]:
    """
    Enforce at the edge what the GPU server would otherwise reject after the
    upload: text models take no images, vision models take at most their
    configured cap, and no single image may be absurdly large.
    """
    images = req.images or []

    if images and req.model not in VISION_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{req.model}' is text-only. Vision models: {sorted(VISION_MODELS)}",
        )

    cap = MODEL_IMAGE_CAPS.get(req.model, 0)
    if len(images) > cap:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{req.model}' accepts at most {cap} image(s) per request; got {len(images)}.",
        )

    for i, src in enumerate(images):
        if not isinstance(src, str) or not src.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"images[{i}] is empty.",
            )
        if len(src) > MAX_IMAGE_CHARS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"images[{i}] is {len(src)} chars; max is {MAX_IMAGE_CHARS}.",
            )

    if len(req.prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt is {len(req.prompt)} chars; max is {MAX_PROMPT_CHARS}.",
        )

    return images


async def _get_upstream_json(path: str) -> Any:
    """GET a read-only endpoint on the GPU server and return its JSON."""
    assert _client is not None
    try:
        resp = await _client.get(f"{GPU_BASE_URL}{path}", timeout=30.0)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot reach GPU server: {exc}",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"GPU server returned {resp.status_code} for {path}",
        )
    return resp.json()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    return {
        "status":         "ok",
        "version":        "3.0.0",
        "gpu_api_url":    GPU_API_URL,
        "auth_enabled":   AUTH_ENABLED,
        "clients":        sorted(set(API_KEY_MAP.values())) if AUTH_ENABLED else ["anonymous"],
        "vision_models":  sorted(VISION_MODELS),
        "timeout_s":      GPU_TIMEOUT_SECONDS,
        "max_connections": MAX_CONNECTIONS,
    }


@app.get("/v1/models")
async def models():
    return {"models": AVAILABLE_MODELS, "image_caps": MODEL_IMAGE_CAPS}


# [5] Observability passthrough — lets a benchmark client that can only reach
# the proxy still read the GPU server's own counters.

@app.get("/v1/gpu-health")
async def gpu_health():
    return await _get_upstream_json("/health")


@app.get("/v1/metrics")
async def gpu_metrics():
    return await _get_upstream_json("/metrics")


@app.get("/v1/client-metrics")
async def gpu_client_metrics():
    return await _get_upstream_json("/client-metrics")


@app.post("/v1/infer")
async def infer(
    req: InferRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    client_id  = _resolve_client(x_api_key)
    request_id = str(uuid.uuid4())
    images     = _validate_multimodal(req)

    logger.info(
        "[%s] client=%s model=%s max_tokens=%s images=%d",
        request_id, client_id, req.model, req.max_new_tokens, len(images),
    )

    payload: dict[str, Any] = {
        "model":          req.model,
        "prompt":         req.prompt,
        "max_new_tokens": int(req.max_new_tokens or 256),
        "temperature":    float(req.temperature  or 0.7),
        "top_p":          float(req.top_p        or 0.9),
        "top_k":          int(req.top_k          or 50),
        # Passed through so the GPU server echoes them back, enabling
        # end-to-end response verification and cross-client contamination detection.
        "request_id":     request_id,
        "client_id":      client_id,
    }
    # [1] Only send the multimodal keys when they carry something — a text
    # model receiving images=[] from the proxy would still be a valid call,
    # but keeping the payload identical to v2 for text traffic means the text
    # path is provably unchanged by this upgrade.
    if images:
        payload["images"] = images
    if req.system:
        payload["system"] = req.system

    assert _client is not None
    t0 = time.perf_counter()
    try:
        resp = await _client.post(GPU_API_URL, json=payload)
    except httpx.ConnectTimeout:
        logger.error("[%s] Connect timeout to GPU server", request_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Cannot connect to GPU server")
    except httpx.ReadTimeout:
        logger.error("[%s] Read timeout after %ss", request_id, GPU_TIMEOUT_SECONDS)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="GPU server timed out")
    except httpx.ConnectError:
        logger.error("[%s] Connection error to GPU server", request_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Cannot connect to GPU server")
    except httpx.HTTPError:
        logger.exception("[%s] Unexpected error calling GPU server", request_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Upstream request failed")

    proxy_elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        logger.error("[%s] GPU error status=%s body=%s",
                     request_id, resp.status_code, resp.text[:200])
        # 503 from the GPU server means "all max_concurrent slots busy". That is
        # a capacity signal, not a proxy fault, so it is surfaced verbatim
        # rather than being flattened into a 502 the way v2 did — a load test
        # cannot distinguish saturation from breakage otherwise.
        if resp.status_code == 503:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"GPU server saturated: {resp.text[:200]}",
            )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"GPU server error: {resp.status_code}")

    try:
        body = resp.json()
    except Exception:
        logger.exception("[%s] Invalid JSON from GPU API", request_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Invalid response from GPU server")

    # ── Sanity-check: verify the response belongs to this request ─────────────
    returned_rid = body.get("request_id") if isinstance(body, dict) else None
    if returned_rid and returned_rid != request_id:
        logger.error(
            "[%s] request_id MISMATCH — sent %s, got back %s  "
            "(cross-client contamination detected!)",
            request_id, request_id, returned_rid,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Response mismatch detected. Please retry.",
        )

    gpu_elapsed = float(body.get("elapsed_s") or 0.0)

    logger.info(
        "[%s] done client=%s model=%s new_tokens=%s gpu=%.2fs proxy=%.2fs",
        request_id, client_id,
        body.get("model", req.model),
        body.get("new_tokens", "?"),
        gpu_elapsed, proxy_elapsed,
    )

    return {
        "text":            body.get("text", "") if isinstance(body, dict) else "",
        "request_id":      request_id,
        "client_id":       client_id,
        "model":           body.get("model", req.model),
        "images":          body.get("images", len(images)),
        "prompt_tokens":   body.get("prompt_tokens"),
        "new_tokens":      body.get("new_tokens"),
        "elapsed_s":       body.get("elapsed_s"),
        "tokens_per_sec":  body.get("tokens_per_sec"),
        # proxy_elapsed_s - elapsed_s is time spent queueing for a semaphore
        # slot plus network; it is the single most useful saturation signal.
        "proxy_elapsed_s": round(proxy_elapsed, 3),
        "queue_wait_s":    round(max(0.0, proxy_elapsed - gpu_elapsed), 3),
        "upstream_status": resp.status_code,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_proxy_v3:app", host=LOCAL_BIND_HOST, port=LOCAL_BIND_PORT,
                log_level="info")
