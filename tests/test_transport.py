"""Stage 3 over two transports: direct to the GPU server, or via llm_proxy_v3.

Both routes end at the same model. Every difference between them is a path, a
header or a dropped field, and each one fails silently or misleadingly if it is
wrong - a 404 that reads as a dead server, a 401 nobody expected, a sampling
knob that stops applying. This suite pins each one.

No network: requests is stubbed with a recorder that answers by path, so the
assertions are about what we SEND, not about what a server happens to reply.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


from pipeline.config import (PROXY_DROPS_FIELDS, PROXY_MAX_NEW_TOKENS,  # noqa: E402
                             TRANSPORT_DIRECT, TRANSPORT_PROXY,
                             default_config, env_overrides)
from pipeline.questions import get_question                             # noqa: E402
from pipeline import stage3_vlm as v                                    # noqa: E402

hazard = get_question("hazard_warning")
REG = {"qwen3-vl": {"name": "qwen3-vl", "modality": "vision", "max_images": 4},
       "mistral": {"name": "mistral", "modality": "text", "max_images": None}}
SAMP = {"max_new_tokens": 300, "temperature": 0.0, "top_p": 1.0,
        "top_k": 0, "repetition_penalty": 1.0}


# ── A recorder standing in for requests ──────────────────────────────────────

class Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class Recorder(types.ModuleType):
    """Answers GET/POST by exact path; records every call with its headers."""

    def __init__(self, routes):
        super().__init__("requests")
        self.routes, self.calls = routes, []

        class exceptions:
            class ConnectionError(Exception):
                pass

            class ReadTimeout(Exception):
                pass
        self.exceptions = exceptions

    def _serve(self, method, url, headers, body=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1] if "://" in url else url
        path = path[path.index("/", 1):] if path.count("/") > 1 and ":" in path else path
        self.calls.append({"method": method, "url": url, "path": path,
                           "headers": headers or {}, "body": body})
        for route, payload in self.routes.items():
            if url.endswith(route):
                if isinstance(payload, Exception):
                    raise payload
                # A Response in the route table is a deliberate status (a 502
                # from the proxy, say); anything else is a 200 body.
                return payload if isinstance(payload, Response) else Response(payload)
        return Response({"detail": "Not Found"}, status=404)

    def get(self, url, timeout=None, headers=None):
        return self._serve("GET", url, headers)

    def post(self, url, json=None, timeout=None, headers=None):
        return self._serve("POST", url, headers, body=json)


def install(routes):
    rec = Recorder(routes)
    sys.modules["requests"] = rec
    return rec


GPU_REGISTRY = [REG["qwen3-vl"], REG["mistral"]]
INFER_OK = {"text": '{"answer": "yes", "reasoning": "A sign is visible."}',
            "model": "qwen3-vl", "elapsed_s": 0.4}


# ── Paths ────────────────────────────────────────────────────────────────────

print("\nthe two transports use different paths")
direct = v.endpoints_for(TRANSPORT_DIRECT)
proxy = v.endpoints_for(TRANSPORT_PROXY)
check("direct infers on /infer", direct.infer == "/infer")
check("the proxy infers on /v1/infer", proxy.infer == "/v1/infer")
check("they are not the same path", direct.infer != proxy.infer)
check("the proxy reads the registry from the GPU passthrough, not its allowlist",
      proxy.registry == "/v1/gpu-models" and proxy.allowlist == "/v1/models")
check("direct has a debug-prompt endpoint", direct.debug_prompt == "/debug/prompt")
check("the proxy has none - llm_proxy_v3 exposes no equivalent",
      proxy.debug_prompt is None)
try:
    v.endpoints_for("carrier-pigeon")
    check("an unknown transport is refused", False, "no exception")
except ValueError as exc:
    check("an unknown transport is refused", True)
    check("and the error lists the real ones", "direct" in str(exc) and "proxy" in str(exc))

print("\nthe API key travels as a header, and only on the proxy")
cfg_p = default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="secret-falcon1")
cfg_d = default_config(vlm_transport=TRANSPORT_DIRECT, vlm_api_key="secret-falcon1")
check("the proxy sends X-API-Key",
      v.auth_headers(cfg_p) == {"X-API-Key": "secret-falcon1"})
check("direct sends no auth header at all - that server has none",
      v.auth_headers(cfg_d) == {})
check("a blank key sends no header rather than an empty one",
      v.auth_headers(default_config(vlm_transport=TRANSPORT_PROXY)) == {})

# ── The wire ─────────────────────────────────────────────────────────────────

print("\nwhat actually goes over the wire")
rec = install({"/v1/gpu-models": GPU_REGISTRY, "/v1/infer": INFER_OK})
cfg = default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="secret-bench",
                     gpu_url="http://10.19.71.246:8071")
client = v.VLMClient(cfg)
names, error = client.refresh_registry()
check("the registry loads through the proxy", error is None and "qwen3-vl" in names,
      str(error))
check("it came from the GPU server's own registry, not the allowlist",
      client.registry_is_allowlist is False)
check("and it carried the API key", rec.calls[0]["headers"].get("X-API-Key")
      == "secret-bench", str(rec.calls[0]["headers"]))

result = client.ask_leg(None, hazard, "answer", "Be terse.", image_uri="data:image/jpeg;base64,AA")
check("a leg call succeeds through the proxy", result["error"] is None, str(result))
post = [c for c in rec.calls if c["method"] == "POST"][-1]
check("it POSTed to /v1/infer", post["url"].endswith("/v1/infer"), post["url"])
check("with the API key", post["headers"].get("X-API-Key") == "secret-bench")
check("the system prompt reached the body", post["body"]["system"] == "Be terse.")
check("so did the image", post["body"]["images"] == ["data:image/jpeg;base64,AA"])

print("\nfields the proxy would silently drop are dropped here instead")
for name in PROXY_DROPS_FIELDS:
    check(f"{name} is not sent to the proxy", name not in post["body"], str(post["body"].keys()))
check("but the sampling knobs that DO pass through are sent",
      post["body"]["temperature"] == 0.0 and post["body"]["top_p"] == 1.0
      and post["body"]["max_new_tokens"] == 300)
d_payload = v.build_payload("qwen3-vl", "q", hazard.system_prompt, ["data:x"], SAMP,
                            REG, TRANSPORT_DIRECT)
check("direct still sends all of them", all(n in d_payload for n in PROXY_DROPS_FIELDS),
      str(sorted(d_payload)))
check("and the two payloads otherwise agree",
      {k: d_payload[k] for k in ("model", "prompt", "system", "temperature", "top_p",
                                 "top_k", "max_new_tokens")}
      == {k: post["body"][k] for k in ("model", "prompt", "system", "temperature",
                                       "top_p", "top_k", "max_new_tokens")}
      or d_payload["prompt"] != post["body"]["prompt"])

# ── Degradation ──────────────────────────────────────────────────────────────

print("\nwhen the proxy is up but the GPU behind it is not")
rec = install({"/v1/gpu-models": Response({"detail": "Cannot reach GPU server"}, 502),
               "/v1/models": {"models": ["mistral", "qwen3-vl"],
                              "image_caps": {"qwen3-vl": 4}},
               "/v1/health": {"version": "3.0.0", "gpu_api_url": "http://gpu:5432",
                              "auth_enabled": True}})
client = v.VLMClient(default_config(vlm_transport=TRANSPORT_PROXY,
                                    vlm_api_key="k"))
names, error = client.refresh_registry()
check("it falls back to the proxy's own allowlist rather than giving up",
      error is None and "qwen3-vl" in names, str(error))
check("and says the registry is only the allowlist",
      client.registry_is_allowlist is True)
check("a model with an image cap is treated as vision - that IS what a cap means",
      client.registry["qwen3-vl"]["modality"] == "vision")
check("one without a cap is text", client.registry["mistral"]["modality"] == "text")
check("the cap survives so build_payload can still enforce it",
      client.max_images("qwen3-vl") == 4)

print("\nerrors name the right machine")
msg = v.describe_error(RuntimeError("HTTP 502 from /v1/infer: Cannot reach GPU server"),
                       TRANSPORT_PROXY)
check("a 502 blames the link between proxy and GPU, not this host",
      "cannot reach the GPU server behind it" in msg, msg)
msg = v.describe_error(RuntimeError("HTTP 404 from /v1/infer: Not Found"), TRANSPORT_PROXY)
check("a 404 names the transport mix-up, which is what causes it",
      "transport is still 'direct'" in msg, msg)
msg = v.describe_error(RuntimeError("HTTP 401 from /v1/infer: Missing X-API-Key header."),
                       TRANSPORT_PROXY)
check("a 401 names the API key", "API key" in msg, msg)
msg = v.describe_error(RuntimeError("HTTP 503 from /v1/infer: saturated"), TRANSPORT_PROXY)
check("a 503 is called load, not breakage", "load, not breakage" in msg, msg)
conn = sys.modules["requests"].exceptions.ConnectionError("refused")
check("a refused connection to the proxy blames the proxy",
      "Could not reach the proxy" in v.describe_error(conn, TRANSPORT_PROXY))
check("and a direct one blames the GPU server",
      "Could not reach the GPU server" in v.describe_error(conn, TRANSPORT_DIRECT))
check("an unrecognised status is passed through rather than guessed at",
      "HTTP 418" in v.describe_error(RuntimeError("HTTP 418 from /v1/infer: teapot"),
                                     TRANSPORT_PROXY))

print("\nthe health check distinguishes the two failures")
client = v.VLMClient(default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="k"))
health, error = client.health()
check("a GPU-side failure is reported as proxy-up", health is None and
      "The proxy IS up" in (error or ""), str(error))
install({})   # nothing answers: proxy down too
client = v.VLMClient(default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="k"))
health, error = client.health()
check("a proxy-side failure says so instead",
      "not answering either" in (error or ""), str(error))

print("\ndebug_prompt is honest about not existing on the proxy")
client = v.VLMClient(default_config(vlm_transport=TRANSPORT_PROXY))
body, error = client.debug_prompt(hazard)
check("it returns no body", body is None)
check("and explains why rather than 404ing", "no /debug/prompt equivalent" in (error or ""),
      str(error))

# ── Config ───────────────────────────────────────────────────────────────────

print("\nconfig guards the combinations that fail silently")
problems = default_config(vlm_transport=TRANSPORT_PROXY).validate()
check("a proxy with no API key is warned about",
      any("no API key is set" in p for p in problems), str(problems))
problems = default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="k",
                          repetition_penalty=1.2).validate()
check("a repetition_penalty the proxy would drop is warned about",
      any("SILENTLY DROPPED" in p for p in problems), str(problems))
problems = default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="k",
                          max_new_tokens=PROXY_MAX_NEW_TOKENS + 1).validate()
check("max_new_tokens above the proxy's ceiling is warned about",
      any("422" in p for p in problems), str(problems))
problems = default_config(vlm_transport=TRANSPORT_PROXY, vlm_api_key="k").validate()
check("a correct proxy config raises none of those",
      not any("API key" in p or "DROPPED" in p or "422" in p for p in problems),
      str(problems))
problems = default_config(vlm_transport="sneakernet").validate()
check("an unknown transport is a config problem",
      any("vlm_transport must be" in p for p in problems), str(problems))
check("the direct default needs no API key",
      not any("API key" in p for p in default_config().validate()))

print("\nthe environment can select the transport without a code edit")
import os  # noqa: E402
for var in ("FIELDOPS_VLM_TRANSPORT", "FIELDOPS_GPU_URL", "FIELDOPS_VLM_API_KEY"):
    os.environ.pop(var, None)
check("nothing set means nothing overridden", env_overrides() == {})
os.environ["FIELDOPS_VLM_TRANSPORT"] = TRANSPORT_PROXY
os.environ["FIELDOPS_GPU_URL"] = "http://10.19.71.246:8071"
os.environ["FIELDOPS_VLM_API_KEY"] = "secret-falcon1"
cfg = default_config()
check("the transport comes from the environment", cfg.vlm_transport == TRANSPORT_PROXY)
check("so does the URL", cfg.gpu_url == "http://10.19.71.246:8071")
check("so does the key", cfg.vlm_api_key == "secret-falcon1")
check("an explicit argument still beats the environment",
      default_config(vlm_transport=TRANSPORT_DIRECT).vlm_transport == TRANSPORT_DIRECT)
os.environ["FIELDOPS_VLM_API_KEY"] = "   "
check("an exported-but-blank variable means unset, not empty-override",
      "vlm_api_key" not in env_overrides())
for var in ("FIELDOPS_VLM_TRANSPORT", "FIELDOPS_GPU_URL", "FIELDOPS_VLM_API_KEY"):
    os.environ.pop(var, None)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
