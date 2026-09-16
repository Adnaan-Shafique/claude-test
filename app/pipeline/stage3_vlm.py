"""Stage 3 - VLM question answering against gpu_api_server_v6.

Self-contained rather than importing from vlm_test_client_v2.py: that client
does not run on the demo host, so there is nothing to import from. The HTTP
contract is lifted, not shared.

Three things here are load-bearing and easy to get wrong.

REGISTRY BOOTSTRAP. The reference client's _build_payload() reads a module-level
_registry that only _refresh_registry() fills, and defaults an unknown model's
modality to "text". Lift that logic without the bootstrap and EVERY image
request raises "'qwen3-vl' is text-only" - which on stage reads as a broken
server rather than a missing initialisation. build_payload() below refuses to
guess: an empty registry is an error that names the fix.

ARRAYS, NEVER PATHS. The reference client re-opens the original file with PIL
and does not apply EXIF transposition, while quality_check.load_image_bgr() does.
A box computed on the transposed array is wrong on the untransposed file. This
module only ever accepts the BGR array the rest of the pipeline is already
holding, so the question cannot arise.

THE SYSTEM PROMPT FAILS SILENTLY. payload["system"] is only set when non-empty,
and both of the server's prompt builders guard on `if system:`. A blank system
prompt produces an unframed answer that looks like a model-quality problem.
ask() asserts it is non-empty before sending, and debug_prompt() lets you verify
it actually lands in the templated prompt.
"""
from __future__ import annotations

import base64
import json
import re
import time
import uuid
from typing import Iterable, Optional

from .config import VLM_MODE_MOCK, SEND_FULL, SEND_FULL_CROP
from .questions import ANSWER_NO, ANSWER_UNKNOWN, ANSWER_YES, \
    LEG_ANSWER, LEG_PRESENCE, LEG_QUALITY, default_leg_system, render_leg_user, \
    render_user_prompt, sampling_for, select_relevant
from .schemas import VLMAnswer

CLIENT_ID = "fieldops-demo-pipeline"

# The server caps decoded images at 2048px anyway; shrinking here keeps the
# POST body small over the wire.
MAX_UPLOAD_SIDE_PX = 2048
JPEG_QUALITY = 92

CONNECT_TIMEOUT_S = 10


# ─────────────────────────────── HTTP ────────────────────────────────────────

def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _get(base: str, path: str, timeout: int = 30):
    import requests
    r = requests.get(_url(base, path), timeout=(CONNECT_TIMEOUT_S, timeout))
    r.raise_for_status()
    return r.json()


def _post(base: str, path: str, payload: Optional[dict] = None, timeout: int = 180):
    import requests
    r = requests.post(_url(base, path), json=payload if payload is not None else {},
                      timeout=(CONNECT_TIMEOUT_S, timeout))
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)   # FastAPI puts it in "detail"
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code} from {path}: {detail}")
    return r.json()


def describe_error(exc: Exception) -> str:
    import requests
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Could not reach the GPU server. {exc}"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"Timed out waiting for the GPU server. {exc}"
    return f"{type(exc).__name__}: {exc}"


# ─────────────────────────────── Image encoding ──────────────────────────────

def array_to_data_uri(image_bgr, max_side: int = MAX_UPLOAD_SIDE_PX) -> str:
    """BGR array -> JPEG data URI.

    Takes the array the pipeline already holds - the EXIF-corrected one from
    quality_check.load_image_bgr(). Never re-reads the file, so the crop the VLM
    sees is in the same frame the boxes were computed in.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed while preparing the image for the VLM")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def crop_for_detection(image_bgr, box, min_frame_frac: float = 0.20):
    """Crop around an absolute-pixel [x1,y1,x2,y2] box, padded.

    Padding is a fraction of the FRAME, not of the box. The real annotations run
    as small as 0.27% of the frame by area; padding such a box by 15% of itself
    yields a ~230x190px sliver on a 4000x3000 photo, which is far too little
    context to judge "is the sky above this obstructed?". Growing it to at least
    min_frame_frac of the shorter side keeps the surroundings in view.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    target = max(x2 - x1, y2 - y1, min(h, w) * min_frame_frac)
    half = target / 2.0
    nx1, ny1 = int(round(cx - half)), int(round(cy - half))
    nx2, ny2 = int(round(cx + half)), int(round(cy + half))

    # Shift back inside the frame rather than shrinking, so the crop keeps its
    # size when the box sits near an edge.
    if nx1 < 0:
        nx2, nx1 = nx2 - nx1, 0
    if ny1 < 0:
        ny2, ny1 = ny2 - ny1, 0
    if nx2 > w:
        nx1, nx2 = max(0, nx1 - (nx2 - w)), w
    if ny2 > h:
        ny1, ny2 = max(0, ny1 - (ny2 - h)), h
    nx1, ny1 = max(0, nx1), max(0, ny1)

    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        raise ValueError(f"crop collapsed to {nx2 - nx1}x{ny2 - ny1}px")
    return image_bgr[ny1:ny2, nx1:nx2]


# ─────────────────────────────── Payload ─────────────────────────────────────

def build_payload(model: str, prompt: str, system: str, image_uris: list[str],
                  sampling: dict, registry: dict) -> dict:
    if not model:
        raise ValueError("No VLM model selected.")
    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")
    if not (system or "").strip():
        # Trap 9. Sending this would produce an unframed answer that looks like
        # a model-quality problem rather than the wiring bug it is.
        raise ValueError(
            f"System prompt for this question is empty - refusing to send. "
            f"payload['system'] is only set when non-empty and the server's "
            f"prompt builders skip a falsy system turn, so this would fail "
            f"silently and the model would answer with no framing at all."
        )
    if not registry:
        raise RuntimeError(
            "The model registry is empty - call refresh_registry() before building a "
            "payload. Without it every model looks text-only and every image request "
            "is rejected with \"'<model>' is text-only\", which looks like a server "
            "fault rather than a missing bootstrap."
        )

    info = registry.get(model)
    if info is None:
        raise ValueError(f"'{model}' is not in the server registry. Available: "
                         f"{sorted(registry)}")
    if image_uris and info.get("modality") != "vision":
        vision = [n for n, i in registry.items() if i.get("modality") == "vision"]
        raise ValueError(f"'{model}' is text-only and will reject images. "
                         f"Vision models: {vision}")

    cap = info.get("max_images")
    if image_uris and cap and len(image_uris) > cap:
        raise ImageCapExceeded(
            f"'{model}' accepts at most {cap} image(s) per prompt; this request "
            f"would send {len(image_uris)}.", cap=cap)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system.strip(),
        "max_new_tokens": int(sampling["max_new_tokens"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "repetition_penalty": float(sampling["repetition_penalty"]),
        "request_id": str(uuid.uuid4()),
        "client_id": CLIENT_ID,
    }
    if image_uris:
        payload["images"] = image_uris
    return payload


class ImageCapExceeded(ValueError):
    def __init__(self, message, cap):
        super().__init__(message)
        self.cap = cap


# ─────────────────────────────── Answer parsing ──────────────────────────────

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(yes|no|unknown)"', re.IGNORECASE)
_REASON_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b(yes|no|unknown)\b", re.IGNORECASE)


def parse_vlm_answer(text: str) -> tuple[str, str]:
    """Tolerant, in four descending tiers. Returns (answer, reasoning).

    The answer is always one of yes / no / unknown, so the UI chip never has to
    handle a surprise value. raw_text is kept separately by the caller.
    """
    raw = (text or "").strip()
    if not raw:
        return ANSWER_UNKNOWN, ""

    # 1. Clean JSON, with or without a markdown fence.
    stripped = _FENCE.sub("", raw).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            answer = str(data.get("answer", "")).strip().lower()
            reasoning = str(data.get("reasoning", "")).strip()
            if answer in (ANSWER_YES, ANSWER_NO, ANSWER_UNKNOWN):
                return answer, reasoning or raw
    except (ValueError, TypeError):
        pass

    # 2. The right keys inside prose, or inside malformed JSON.
    m = _ANSWER_RE.search(raw)
    if m:
        reason_match = _REASON_RE.search(raw)
        reasoning = reason_match.group(1) if reason_match else raw
        try:
            reasoning = json.loads(f'"{reasoning}"')   # unescape \n, \" etc.
        except ValueError:
            pass
        return m.group(1).lower(), reasoning.strip() or raw

    # 3. A bare yes/no in the opening sentence.
    first = re.split(r"(?<=[.!?])\s", raw, maxsplit=1)[0]
    m = _TOKEN_RE.search(first)
    if m:
        return m.group(1).lower(), raw

    # 4. Give up honestly rather than guessing.
    return ANSWER_UNKNOWN, raw


# ─────────────────────────────── Mock ────────────────────────────────────────

MOCK_REASONING = {
    "hazard_warning": "MOCK RESPONSE - the GPU server was not reached, so no model "
                      "examined this image. This text is canned.",
    "gps_antenna": "MOCK RESPONSE - the GPU server was not reached, so no model "
                   "examined this image. This text is canned.",
}


def mock_answer(question, model: str, error: Optional[str] = None) -> VLMAnswer:
    """A clearly-labelled canned answer, so legs 1 and 2 can be demonstrated with
    the VLM panel populated rather than blank. Always 'unknown': asserting yes or
    no without a model having looked would be a lie on screen."""
    return VLMAnswer(
        answer=ANSWER_UNKNOWN,
        reasoning=MOCK_REASONING.get(question.id, "MOCK RESPONSE - no model was called."),
        raw_text="", model=model, elapsed_s=0.0, error=error, is_mock=True,
    )


# ─────────────────────────────── Client ──────────────────────────────────────

class VLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg.gpu_url
        self.registry: dict = {}
        self.registry_error: Optional[str] = None

    # ── Server state ─────────────────────────────────────────────────────────
    def refresh_registry(self) -> tuple[list[str], Optional[str]]:
        """Populate the registry from /models. MUST run before any payload is
        built - see the module docstring."""
        try:
            models = _get(self.base, "/models")
        except Exception as exc:
            self.registry, self.registry_error = {}, describe_error(exc)
            return [], self.registry_error
        self.registry = {m["name"]: m for m in models}
        self.registry_error = None
        return list(self.registry), None

    def health(self) -> tuple[Optional[dict], Optional[str]]:
        try:
            return _get(self.base, "/health"), None
        except Exception as exc:
            return None, describe_error(exc)

    def vision_models(self) -> list[str]:
        return [n for n, i in self.registry.items() if i.get("modality") == "vision"]

    def max_images(self, model: str) -> Optional[int]:
        return (self.registry.get(model) or {}).get("max_images")

    def ensure_registry(self) -> Optional[str]:
        if not self.registry:
            _, error = self.refresh_registry()
            return error
        return None

    def debug_prompt(self, question, n_images: int = 1) -> tuple[Optional[dict], Optional[str]]:
        """Render the templated prompt server-side without touching the GPU.

        This is how you confirm the per-question system prompt actually lands -
        trap 9's failure is silent, so verify it rather than assuming.
        /debug/prompt only counts images, so cheap placeholders suffice.
        """
        payload = {
            "model": self.cfg.vlm_model,
            "prompt": render_user_prompt(question),
            "system": question.system_prompt,
            "images": ["x" * 4 for _ in range(n_images)] or None,
        }
        try:
            return _post(self.base, "/debug/prompt", payload, timeout=120), None
        except Exception as exc:
            return None, describe_error(exc)

    # ── The question ─────────────────────────────────────────────────────────
    def ask(self, image_bgr, question, detections: Optional[Iterable] = None) -> VLMAnswer:
        """Answer one question about one image. Never raises: a failure becomes a
        mock answer carrying the error, so one bad image cannot end the batch."""
        model = self.cfg.vlm_model
        if self.cfg.vlm_mode == VLM_MODE_MOCK:
            return mock_answer(question, model, error=None)

        error = self.ensure_registry()
        if error:
            return mock_answer(question, model,
                               error=f"registry unavailable: {error}")

        relevant = select_relevant(detections or [], question)
        prompt = render_user_prompt(question, relevant)
        sampling = sampling_for(question, self.cfg)

        try:
            uris = self._images_for(image_bgr, question, relevant)
        except Exception as exc:
            return mock_answer(question, model, error=f"image encoding failed: {exc}")

        try:
            payload = build_payload(model, prompt, question.system_prompt, uris,
                                    sampling, self.registry)
        except ImageCapExceeded:
            # Downgrade to the full image rather than erroring out mid-demo.
            try:
                uris = [array_to_data_uri(image_bgr)]
                payload = build_payload(model, prompt, question.system_prompt, uris,
                                        sampling, self.registry)
            except Exception as exc:
                return mock_answer(question, model, error=describe_error(exc))
        except Exception as exc:
            return mock_answer(question, model, error=str(exc))

        t0 = time.time()
        try:
            response = _post(self.base, "/infer", payload,
                             timeout=self.cfg.request_timeout_s)
        except Exception as exc:
            return mock_answer(question, model, error=describe_error(exc))

        elapsed = float(response.get("elapsed_s") or (time.time() - t0))
        text = response.get("text", "") or ""
        answer, reasoning = parse_vlm_answer(text)
        return VLMAnswer(
            answer=answer, reasoning=reasoning, raw_text=text,
            model=response.get("model", model), elapsed_s=elapsed,
            error=None, is_mock=False,
        )

    def ask_leg(self, image_bgr, question, leg: str, system: str,
                image_uri: Optional[str] = None) -> dict:
        """One leg of mode 3: one system prompt, one question, one answer.

        `system` comes from the prompt store, so an edit in the UI reaches the
        model without a restart. `image_uri` lets the caller encode the image
        once and reuse it across all three legs - re-encoding a 2048px JPEG
        three times per photograph is pure waste.
        """
        model = self.cfg.vlm_model
        prompt = render_leg_user(question, leg)
        sampling = sampling_for(question, self.cfg)

        try:
            uris = [image_uri or array_to_data_uri(image_bgr)]
            payload = build_payload(model, prompt, system, uris, sampling,
                                    self.registry)
        except Exception as exc:
            return {"text": "", "error": describe_error(exc), "elapsed_s": 0.0,
                    "model": model}

        t0 = time.time()
        try:
            response = _post(self.base, "/infer", payload,
                             timeout=self.cfg.request_timeout_s)
        except Exception as exc:
            return {"text": "", "error": describe_error(exc), "elapsed_s": 0.0,
                    "model": model}
        return {"text": response.get("text", "") or "", "error": None,
                "elapsed_s": float(response.get("elapsed_s") or (time.time() - t0)),
                "model": response.get("model", model)}

    def ask_vlm_only(self, image_bgr, question, prompts=None) -> dict:
        """Mode 3: quality, subject presence and the inspection answer, as three
        separate calls with three separate system prompts.

        Never raises. Any leg that fails degrades the whole result to a
        clearly-labelled mock rather than reporting two real judgements and one
        silent default - a partial answer that looks complete is worse than an
        obvious mock.
        """
        model = self.cfg.vlm_model
        if self.cfg.vlm_mode == VLM_MODE_MOCK:
            return mock_vlm_only(question, model)

        error = self.ensure_registry()
        if error:
            return mock_vlm_only(question, model,
                                 error=f"registry unavailable: {error}")

        def system_for(leg):
            if prompts is not None:
                return prompts.get(question.id, leg)
            return default_leg_system(question, leg)

        try:
            image_uri = array_to_data_uri(image_bgr)   # encoded once, used thrice
        except Exception as exc:
            return mock_vlm_only(question, model,
                                 error=f"image encoding failed: {exc}")

        legs = {}
        for leg in (LEG_QUALITY, LEG_PRESENCE, LEG_ANSWER):
            legs[leg] = self.ask_leg(image_bgr, question, leg, system_for(leg),
                                     image_uri=image_uri)

        failed = [leg for leg, r in legs.items() if r["error"]]
        if failed:
            first = legs[failed[0]]["error"]
            return mock_vlm_only(question, model,
                                 error=f"{', '.join(failed)} leg(s) failed: {first}")

        quality, quality_reasoning = parse_quality(legs[LEG_QUALITY]["text"])
        presence, presence_reasoning = parse_presence(legs[LEG_PRESENCE]["text"])
        answer, reasoning = parse_vlm_answer(legs[LEG_ANSWER]["text"])

        return {
            "quality": quality, "quality_reasoning": quality_reasoning,
            "subject_present": presence, "subject_reasoning": presence_reasoning,
            "answer": answer, "reasoning": reasoning,
            # The answer leg's raw text is what the card's "Raw model output"
            # accordion shows; every leg's raw text is kept for the record.
            "raw_text": legs[LEG_ANSWER]["text"],
            "model": legs[LEG_ANSWER]["model"],
            "elapsed_s": round(sum(r["elapsed_s"] for r in legs.values()), 3),
            "error": None, "is_mock": False,
            "legs": {leg: {"raw": r["text"], "elapsed_s": r["elapsed_s"]}
                     for leg, r in legs.items()},
        }

    def _images_for(self, image_bgr, question, relevant) -> list[str]:
        """Full image, optionally plus a padded crop of the best relevant box."""
        uris = [array_to_data_uri(image_bgr)]
        if self.cfg.vlm_send_mode != SEND_FULL_CROP or not relevant:
            return uris
        best = max(relevant, key=lambda d: d.confidence)
        try:
            crop = crop_for_detection(image_bgr, best.box, self.cfg.crop_min_frame_frac)
        except ValueError:
            return uris   # too small to be useful; the full image still answers
        cap = self.max_images(self.cfg.vlm_model)
        if cap and cap < 2:
            return uris
        uris.append(array_to_data_uri(crop))
        return uris


# ─────────────────── Mode 3: three legs, three calls ─────────────────────────

_QUALITY_RE = re.compile(r'"quality"\s*:\s*"(good|poor)"', re.IGNORECASE)
_PRESENT_RE = re.compile(r'"present"\s*:\s*"(yes|no|unknown)"', re.IGNORECASE)


def _reasoning_from(raw: str, parsed: Optional[dict]) -> str:
    if parsed and str(parsed.get("reasoning", "")).strip():
        return str(parsed["reasoning"]).strip()
    return raw


def _loads(text: str) -> Optional[dict]:
    try:
        data = json.loads(_FENCE.sub("", (text or "").strip()).strip())
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def parse_quality(text: str) -> tuple[str, str]:
    """-> ("good" | "poor", reasoning). Defaults to "poor": calling a
    photograph usable on no evidence is the failure that matters here."""
    raw = (text or "").strip()
    if not raw:
        return "poor", ""
    data = _loads(raw)
    if data:
        value = str(data.get("quality", "")).strip().lower()
        if value in ("good", "poor"):
            return value, _reasoning_from(raw, data)
    m = _QUALITY_RE.search(raw)
    if m:
        return m.group(1).lower(), _reasoning_from(raw, data)
    # Last resort: an unhedged "good" in the opening sentence.
    first = re.split(r"(?<=[.!?])\s", raw, maxsplit=1)[0].lower()
    if re.search(r"\bgood\b", first) and not re.search(r"\bpoor\b", first):
        return "good", raw
    return "poor", raw


def parse_presence(text: str) -> tuple[str, str]:
    """-> ("yes" | "no" | "unknown", reasoning). Defaults to "unknown"."""
    raw = (text or "").strip()
    if not raw:
        return ANSWER_UNKNOWN, ""
    data = _loads(raw)
    if data:
        value = str(data.get("present", data.get("subject_present", ""))).strip().lower()
        if value in (ANSWER_YES, ANSWER_NO, ANSWER_UNKNOWN):
            return value, _reasoning_from(raw, data)
    m = _PRESENT_RE.search(raw)
    if m:
        return m.group(1).lower(), _reasoning_from(raw, data)
    answer, reasoning = parse_vlm_answer(raw)
    return answer, reasoning


def mock_vlm_only(question, model: str, error: Optional[str] = None) -> dict:
    """A canned mode-3 result. Quality "poor" and presence "unknown" on purpose:
    a mock must never assert that a photograph is fine or a subject visible when
    nothing looked at it."""
    note = "MOCK - no model examined this image."
    return {
        "quality": "poor", "quality_reasoning": note,
        "subject_present": ANSWER_UNKNOWN, "subject_reasoning": note,
        "answer": ANSWER_UNKNOWN,
        "reasoning": MOCK_REASONING.get(question.id, "MOCK - no model was called."),
        "raw_text": "", "model": model, "elapsed_s": 0.0,
        "error": error, "is_mock": True,
        "legs": {LEG_QUALITY: {"raw": "", "elapsed_s": 0.0},
                 LEG_PRESENCE: {"raw": "", "elapsed_s": 0.0},
                 LEG_ANSWER: {"raw": "", "elapsed_s": 0.0}},
    }
# ─────────────────────── Mode 3: one call, three judgements ──────────────────

_QUALITY_RE = re.compile(r'"quality"\s*:\s*"(good|poor)"', re.IGNORECASE)
_SUBJECT_RE = re.compile(r'"subject_present"\s*:\s*"(yes|no|unknown)"', re.IGNORECASE)
