"""Dash UI tests - the pure render functions.

Stubs dash so this runs without it installed; only layout construction and the
provenance requirements are exercised, never a server.
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


# ── A dash stub that records what was built ─────────────────────────────────
class Node:
    def __init__(self, children=None, **kw):
        self.children = children
        self.kw = kw

    def text(self) -> str:
        """Flatten this tree to searchable text: class names, titles and strings."""
        # id included: component ids are how callbacks find things, so a test
        # asserting a control exists needs to see them.
        bits = [str(self.kw.get("className", "")), str(self.kw.get("title", "")),
                str(self.kw.get("alt", "")), str(self.kw.get("id", ""))]
        c = self.children
        if isinstance(c, (list, tuple)):
            for item in c:
                bits.append(item.text() if isinstance(item, Node) else str(item))
        elif isinstance(c, Node):
            bits.append(c.text())
        elif c is not None:
            bits.append(str(c))
        return " ".join(bits)


_html = types.ModuleType("dash.html")
for _n in ("Div", "Span", "H1", "H4", "P", "Ul", "Li", "Img", "Pre", "Details",
           "Summary", "Button", "Label", "B", "A"):
    setattr(_html, _n, Node)

_dcc = types.ModuleType("dash.dcc")
for _n in ("Dropdown", "Input", "Upload", "Checklist", "RadioItems", "Tabs", "Tab",
           "Loading", "Store", "Download"):
    setattr(_dcc, _n, Node)
_dcc.send_file = lambda p: {"file": p}

_dt = types.ModuleType("dash.dash_table")
_dt.DataTable = Node

_dash = types.ModuleType("dash")
_dash.html, _dash.dcc, _dash.dash_table = _html, _dcc, _dt
class _Dep:
    def __init__(self, *a, **k):
        self.args = a


_dash.Input = _dash.Output = _dash.State = _Dep
_dash.no_update = object()
_dash.callback_context = types.SimpleNamespace(triggered=[])


class _App:
    def __init__(self, *a, **k):
        self.server = object()
        self.layout = None

    def callback(self, *a, **k):
        return lambda fn: fn

    def run(self, *a, **k):
        pass


_dash.Dash = _App
sys.modules["dash"] = _dash
sys.modules["dash.html"] = _html
sys.modules["dash.dcc"] = _dcc
sys.modules["dash.dash_table"] = _dt

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("demo_dash", ROOT / "app" / "demo_dash.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

from pipeline.questions import get_question           # noqa: E402
from pipeline.schemas import (Detection, DetectionStageResult, PipelineRecord,  # noqa: E402
                              QualityStageResult, STOPPED_COMPLETE, STOPPED_QUALITY,
                              VLMAnswer)

q = get_question("hazard_warning")


def quality(passed_=True, score=84.2, resolution_ok=True):
    return QualityStageResult(
        passed=passed_, score=score, threshold=65.0, resolution_ok=resolution_ok,
        ignore_resolution_used=False, failure_reasons=[] if passed_ else ["blur", "haze"],
        retake_instructions=[] if passed_ else ["Hold the camera steady and retake."],
        foreground_box=(341, 152, 719, 1524), segmentation_used=True,
        whole_frame_score=63.2, width=1200, height=1600, foreground_score=score)


def record(stem="00002_task_328989", stopped=STOPPED_COMPLETE, answer="yes",
           passed_=True, mock=False, note="class names from question defaults"):
    det = DetectionStageResult(
        detections=[Detection(label="hazard_sign", confidence=0.95,
                              box=[616, 945, 676, 1032], source="annotation_file")],
        annotated_path=None, model_name="human annotation (no model loaded)",
        is_stub=True, note=note)
    vlm = VLMAnswer(answer=answer, reasoning="A green sign with a yellow triangular "
                    "warning symbol is clearly visible.",
                    raw_text='{"answer":"%s"}' % answer, model="qwen3-vl",
                    elapsed_s=0.32, is_mock=mock)
    return PipelineRecord(
        filename=f"{stem}.jpg", source_path=f"/photos/{stem}.jpg", stem=stem,
        question_id="hazard_warning", quality=quality(passed_=passed_),
        detection=None if stopped == STOPPED_QUALITY else det,
        vlm=None if stopped == STOPPED_QUALITY else vlm, stopped_at=stopped)


print("\nlayout builds without a server")
check("app.layout constructed", ui.app.layout is not None)
check("stylesheet is external, not inline", "assets" not in str(ui.FONTS))

print("\nresult card - the three legs")
card = ui.result_card(record(), q).text()
check("quality column present", "1 · Quality gate" in card)
check("detection column present", "2 · Detection" in card)
check("answer column present", "3 · Model answer" in card)
check("quality headline shown", "PASS 84.2 / 65" in card, card[:200])
check("whole-frame to cropped delta explained", "moved the score" in card, "")
check("answer chip styled by verdict", "chip chip-yes" in card)
check("detection listed with confidence", "hazard_sign — 0.95" in card)
check("raw output kept behind a disclosure", "Raw model output" in card)

print("\nhonesty requirements")
check("stub provenance stated verbatim", "human annotation (no model loaded)" in card)
check("inferred class names surfaced as a warning",
      "banner banner-warn" in card and "question defaults" in card)
mock_card = ui.result_card(record(answer="unknown", mock=True), q).text()
check("mock answers labelled MOCK", "MOCK" in mock_card)
check("mock provenance in the meta line", "MOCK - GPU server unreachable" in mock_card)

print("\nquality-gate stop")
stopped = ui.result_card(record(stopped=STOPPED_QUALITY, passed_=False), q).text()
check("says the downstream legs did not run", "Stopped at the quality gate" in stopped)
check("names the override toggle", "Run downstream legs on FAIL" in stopped)
check("detection column explains itself", "Not run" in stopped)

print("\nresolution failure reads honestly, not as a scoring bug")
r = record()
r.quality = QualityStageResult(
    passed=False, score=82.1, threshold=65.0, resolution_ok=False,
    ignore_resolution_used=False, failure_reasons=["resolution_too_low"],
    retake_instructions=["Use a higher-resolution camera setting."],
    foreground_box=None, segmentation_used=False, whole_frame_score=82.1,
    width=320, height=240)
res = ui.result_card(r, q).text()
check("the headline names the resolution floor", "resolution floor" in res, res[:300])
check("the frame size is shown", "320x240" in res or "320×240" in res)

print("\nviews")
check("empty results state is explanatory",
      "No results yet" in ui.results_view([], q).text())
view = ui.results_view([record(), record("b", answer="no"),
                        record("c", stopped=STOPPED_QUALITY, passed_=False)], q).text()
check("stat tiles rendered", "tiles" in view)
check("yes tile styled", "tile-yes" in view)
check("no tile styled", "tile-no" in view)
summary = ui.summary_view([record(), record("b", answer="no", mock=True)], q,
                          "/runs/x").text()
check("summary states the answer semantics", "YES =" in summary)
check("summary flags mock answers", "MOCK" in summary)
check("summary shows the run folder", "/runs/x" in summary)

print("\nprompt view")
text = ui.prompt_text("gps_antenna")
check("system prompt shown", "SYSTEM PROMPT" in text)
check("user prompt shown", "USER PROMPT" in text)
check("the gps system prompt is the one rendered", "GPS antenna" in text)
check("switching question switches the pair",
      "site-safety inspector" in ui.prompt_text("hazard_warning"))

print("\nYOLOX UI - upload works here, unlike the annotation version")
# The annotation UI deliberately has no upload: it needs a .txt sidecar beside
# each photo and a browser upload cannot carry one. A trained detector reads the
# image, so uploading is fully supported - and is the better demo.
spec_y = importlib.util.spec_from_file_location("demo_dash_yolox",
                                                ROOT / "app" / "demo_dash_yolox.py")
yolox_ui = importlib.util.module_from_spec(spec_y)
spec_y.loader.exec_module(yolox_ui)
layout_text = yolox_ui.layout().text()

check("a dropzone is present", "dropzone" in layout_text)
check("the upload component is wired", "uploads" in layout_text)
check("staged-file feedback has a home", "upload-note" in layout_text)
check("the folder path is still offered", "folder" in layout_text)

src = (ROOT / "app" / "demo_dash_yolox.py").read_text()
check("uploads are staged to disk, not passed as base64 downstream",
      "dest.write_bytes(base64.b64decode(b64))" in src)
check("a folder path takes precedence over uploads",
      src.index("if folder and folder.strip():") < src.index("if upload_contents:"))
check("the empty state names both inputs",
      "Drop images above, or point at a photo folder" in src)
check("run_id is set before staging, so uploads land in this run's folder",
      "cfg.run_id = new_run_id()" in src)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
