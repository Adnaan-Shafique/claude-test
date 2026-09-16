# Reaching the model from a host that cannot see the GPU

The GPU server (`gpu_api_server_v6` on `10.66.98.137:5432`) is not reachable
from every demo host. The field-ops VM `10.19.75.122` can reach **FALCONPRD**
and nothing else; FALCONPRD runs `llm_proxy_v3` and forwards on its behalf.

Stage 3 therefore has two transports. They end at the same model — only the
routing differs.

| | `direct` | `proxy` |
|---|---|---|
| Inference | `POST /infer` | `POST /v1/infer` |
| Auth | none | `X-API-Key:` header |
| Health | `GET /health` | `GET /v1/gpu-health` (GPU) · `GET /v1/health` (proxy) |
| Registry | `GET /models` | `GET /v1/gpu-models`, falling back to `/v1/models` |
| Prompt debug | `POST /debug/prompt` | **not available** |
| Use it on | FALCONPRD, AISERVER | the field-ops VM |

## Selecting it

Three ways, in increasing precedence:

```bash
# 1. Environment — set once per host, picked up by every entry point.
export FIELDOPS_VLM_TRANSPORT=proxy
export FIELDOPS_GPU_URL=http://10.19.71.246:8071
export FIELDOPS_VLM_API_KEY=secret-falcon1

# 2. Command line, for the smoke tools.
python tools/smoke_stage3.py <photos> --transport proxy \
    --gpu-url http://10.19.71.246:8071 --api-key secret-falcon1

# 3. The UI's "Route to the model" radio, which wins over both.
```

Keep the key in the environment rather than the command line — a shell history
holding a live API key is a small problem that only ever becomes a bigger one.
The UI's key box is a password field for the same reason.

## Four things that behave differently through the proxy

**`repetition_penalty` does not exist there.** `llm_proxy_v3`'s `InferRequest`
is a pydantic model, and pydantic *ignores* fields it does not declare rather
than rejecting them — so the value would be dropped in silence. At the demo
default of `1.0` (the server's own floor, i.e. no penalty) that changes
nothing, which is why the payload is otherwise identical on both routes.
`build_payload()` strips it for the proxy anyway, so a recorded run never
claims a setting that was not honoured. Raise it above 1.0 and `validate()`
warns you.

**`request_id` and `client_id` move rather than vanish.** The proxy mints its
own UUID per request and derives `client_id` from the API key, then round-trips
both through the GPU server and checks them on the way back. That check is
cross-client contamination detection, and it is stricter than anything the
direct path does.

**`max_new_tokens` is capped at 8192.** Over that is a 422 at the edge, before
any GPU time is spent. The demo asks for 300.

**You cannot warm the model from behind the proxy.** It forwards `/infer` and
the read-only endpoints; there is no passthrough for the GPU server's model-load
route. If `qwen3-vl` is not resident, someone with a direct route has to load it:

```bash
curl -XPOST http://10.66.98.137:5432/models/qwen3-vl/load
```

`preflight.py --transport proxy` says this instead of printing a curl that
would 404.

## When it breaks

The failure worth naming in advance: pointing `gpu_url` at the proxy while
leaving the transport on `direct`. The proxy serves `/v1/infer`, not `/infer`,
so you get a **404** — which on stage reads as "the model is down" rather than
"wrong path". `describe_error()` spells it out rather than leaving you to guess.

Each proxy status carries its own explanation:

| Status | What it actually means |
|---|---|
| 401 | The proxy wants an API key and none was sent |
| 403 | The key was sent and rejected |
| 404 | Wrong path — almost always the transport mix-up above |
| 413 | The image exceeded the proxy's `MAX_IMAGE_CHARS` |
| 422 | Bad body — unknown model, or `max_new_tokens` over the ceiling |
| 502 | **The proxy is up and the GPU behind it is not.** Not this host's fault |
| 503 | The GPU is saturated. Load, not breakage — retry |
| 504 | The GPU did not answer in time; often a cold model load |

The health check makes the same distinction. If `/v1/gpu-health` fails it asks
`/v1/health` before reporting, so the status line says either *"the proxy IS up,
so the break is between the proxy and the GPU"* or *"the proxy itself is not
answering either"* — the difference between restarting a service on FALCONPRD
and walking to the GPU box.

**Registry fallback.** If the proxy cannot reach the GPU, `/v1/gpu-models`
fails but `/v1/models` still lists the names and image caps the proxy itself
enforces. That is enough to build a valid payload, so the client falls back to
it and flags that the registry is the allowlist rather than the GPU's own. A
model carrying an image cap is treated as a vision model — in `llm_proxy_v3`'s
`MODEL_IMAGE_CAPS`, having a cap is precisely what makes it one.

## Files

| Path | Role |
|---|---|
| `app/pipeline/config.py` | `vlm_transport`, `vlm_api_key`, the env overrides, the `validate()` guards |
| `app/pipeline/stage3_vlm.py` | `ENDPOINTS`, `auth_headers()`, the per-status help, the registry fallback |
| `tools/preflight.py` | `--transport` / `--api-key` |
| `tools/smoke_stage3.py` | same two flags |
| `tests/test_transport.py` | 53 assertions |
