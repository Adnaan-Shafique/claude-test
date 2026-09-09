"""Phase 3 tests - VLM payload building, answer parsing, mock fallback.

No network and no cv2: requests is stubbed per test and the image helpers are
exercised separately where cv2 exists.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from pipeline.config import default_config, VLM_MODE_MOCK, SEND_FULL_CROP  # noqa: E402
from pipeline.questions import get_question                                # noqa: E402
from pipeline.schemas import Detection                                     # noqa: E402
from pipeline import stage3_vlm as v                                       # noqa: E402

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


hazard = get_question("hazard_warning")
antenna = get_question("gps_antenna")
REG = {"qwen3-vl": {"name": "qwen3-vl", "modality": "vision", "max_images": 4},
       "mistral": {"name": "mistral", "modality": "text", "max_images": None}}
SAMP = {"max_new_tokens": 300, "temperature": 0.0, "top_p": 1.0,
        "top_k": 0, "repetition_penalty": 1.0}

print("\npayload - the registry bootstrap that would otherwise break every call")
try:
    v.build_payload("qwen3-vl", "q", hazard.system_prompt, ["data:x"], SAMP, {})
    check("an empty registry is refused", False, "no exception")
except RuntimeError as exc:
    check("an empty registry is refused", True)
    check("the error names refresh_registry as the fix", "refresh_registry" in str(exc),
          str(exc))
    check("the error explains the misleading symptom", "text-only" in str(exc), str(exc))

print("\npayload - the system prompt cannot be dropped silently (trap 9)")
try:
    v.build_payload("qwen3-vl", "q", "   ", ["data:x"], SAMP, REG)
    check("a blank system prompt is refused", False, "no exception")
except ValueError as exc:
    check("a blank system prompt is refused", True)
    check("the error explains the silent failure", "silently" in str(exc), str(exc))

p = v.build_payload("qwen3-vl", "q", hazard.system_prompt, ["data:x"], SAMP, REG)
check("system lands in the payload", p["system"] == hazard.system_prompt)
check("images included", p["images"] == ["data:x"])
check("request_id generated", bool(p["request_id"]))
check("client_id set", p["client_id"] == "fieldops-demo-pipeline")
check("sampling forwarded", p["temperature"] == 0.0 and p["top_p"] == 1.0)
check("repetition_penalty respects the server's ge=1.0", p["repetition_penalty"] >= 1.0)

print("\npayload - modality and image cap")
try:
    v.build_payload("mistral", "q", hazard.system_prompt, ["data:x"], SAMP, REG)
    check("a text model is refused images", False)
except ValueError as exc:
    check("a text model is refused images", "text-only" in str(exc), str(exc))
try:
    v.build_payload("qwen3-vl", "q", hazard.system_prompt, ["d"] * 5, SAMP, REG)
    check("over the per-model cap raises ImageCapExceeded", False)
except v.ImageCapExceeded as exc:
    check("over the per-model cap raises ImageCapExceeded", exc.cap == 4)
p = v.build_payload("qwen3-vl", "q", hazard.system_prompt, [], SAMP, REG)
check("no images -> no images key", "images" not in p)

print("\nanswer parsing - tier 1, clean JSON")
a, r = v.parse_vlm_answer('{"answer": "yes", "reasoning": "A red danger placard is visible."}')
check("answer read", a == "yes")
check("reasoning read", r == "A red danger placard is visible.")
a, r = v.parse_vlm_answer('```json\n{"answer":"no","reasoning":"Nothing visible."}\n```')
check("markdown fence stripped", (a, r) == ("no", "Nothing visible."))

print("\nanswer parsing - tier 2, keys inside prose or broken JSON")
a, r = v.parse_vlm_answer('Sure! {"answer": "unknown", "reasoning": "Too blurry."} hope that helps')
check("answer found in surrounding prose", a == "unknown")
check("reasoning found in surrounding prose", r == "Too blurry.")
a, r = v.parse_vlm_answer('{"answer": "yes", "reasoning": "Line one.\\nLine two."')
check("unterminated JSON still yields the answer", a == "yes")
check("escapes unescaped in the reasoning", "\n" in r, repr(r))

print("\nanswer parsing - tier 3, a bare token in the first sentence")
a, r = v.parse_vlm_answer("Yes, there is a high-voltage warning sign on the cabinet door.")
check("leading yes recognised", a == "yes")
check("whole text kept as reasoning", r.startswith("Yes, there is"))
a, _ = v.parse_vlm_answer("No. The antenna sits under a metal canopy.")
check("leading no recognised", a == "no")

print("\nanswer parsing - tier 4, honest fallback")
a, r = v.parse_vlm_answer("The image shows a telecom cabinet in a compound.")
check("no verdict -> unknown", a == "unknown")
check("text preserved as reasoning", r.startswith("The image shows"))
a, r = v.parse_vlm_answer("")
check("empty input -> unknown", (a, r) == ("unknown", ""))
a, _ = v.parse_vlm_answer("I'm not confident enough to answer that.")
check("the server's own canned fallback parses to unknown", a == "unknown")

for text in ['{"answer":"YES","reasoning":"x"}', "YES it is present."]:
    a, _ = v.parse_vlm_answer(text)
    check(f"case-insensitive: {text[:24]!r}", a == "yes", a)

print("\nmock mode - clearly labelled, never asserts a verdict")
cfg = default_config(vlm_mode=VLM_MODE_MOCK)
client = v.VLMClient(cfg)
ans = client.ask(None, hazard)
check("mock returns without touching the network", ans.is_mock is True)
check("mock never claims yes or no", ans.answer == "unknown", ans.answer)
check("mock says so in the reasoning", "MOCK" in ans.reasoning, ans.reasoning)
check("provenance is unmistakable in the UI",
      ans.provenance == "MOCK - GPU server unreachable", ans.provenance)

print("\nlive mode - an unreachable server degrades instead of failing the batch")
class Boom:
    class exceptions:
        class ConnectionError(Exception): pass
        class ReadTimeout(Exception): pass
    @staticmethod
    def get(*a, **k):
        raise Boom.exceptions.ConnectionError("connection refused")
    @staticmethod
    def post(*a, **k):
        raise Boom.exceptions.ConnectionError("connection refused")

sys.modules["requests"] = Boom
client = v.VLMClient(default_config())
ans = client.ask(None, hazard)
check("unreachable server -> mock, not an exception", ans.is_mock is True)
check("the error is carried on the answer", ans.error and "registry" in ans.error, ans.error)
check("still a valid chip value", ans.answer == "unknown")

print("\ncrop geometry - a tiny box grows to carry context")
class FakeArr:
    def __init__(self, h, w):
        self.shape = (h, w, 3)
        self.sliced = None
    def __getitem__(self, key):
        ys, xs = key
        out = FakeArr(ys.stop - ys.start, xs.stop - xs.start)
        self.sliced = (xs.start, ys.start, xs.stop, ys.stop)
        return out

# The real annotation: 0.27% of a 1200x1600 frame.
img = FakeArr(1600, 1200)
crop = v.crop_for_detection(img, [616, 945, 676, 1032], min_frame_frac=0.20)
x1, y1, x2, y2 = img.sliced
check("crop grows past the 60x87px box", (x2 - x1) >= 240 and (y2 - y1) >= 240,
      f"{x2 - x1}x{y2 - y1}")
check("crop stays centred on the box", abs((x1 + x2) / 2 - 646) < 2, f"cx={(x1 + x2) / 2}")
check("crop stays inside the frame", x1 >= 0 and y1 >= 0 and x2 <= 1200 and y2 <= 1600,
      str(img.sliced))

# A box in the corner must shift, not shrink.
img2 = FakeArr(1600, 1200)
v.crop_for_detection(img2, [0, 0, 30, 30], min_frame_frac=0.20)
x1, y1, x2, y2 = img2.sliced
check("a corner box shifts inward rather than shrinking",
      (x2 - x1) >= 240 and x1 == 0 and y1 == 0, str(img2.sliced))

# A large box is not shrunk to the minimum.
img3 = FakeArr(1600, 1200)
v.crop_for_detection(img3, [100, 100, 900, 900], min_frame_frac=0.20)
x1, y1, x2, y2 = img3.sliced
check("a large box keeps its own size", (x2 - x1) >= 800, f"{x2 - x1}")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
