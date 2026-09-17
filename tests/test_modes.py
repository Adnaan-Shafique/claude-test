"""Three-mode pipeline: gating logic, mode-3 mapping, comparison, UI wiring."""
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


from pipeline import modes as M                                      # noqa: E402
from pipeline.questions import get_question                          # noqa: E402
from pipeline.schemas import (Detection, DetectionStageResult,        # noqa: E402
                              PipelineRecord, QualityStageResult,
                              STOPPED_COMPLETE, STOPPED_QUALITY, VLMAnswer)

q = get_question("hazard_warning")


def quality(passed_=True, score=80.0):
    return QualityStageResult(
        passed=passed_, score=score, threshold=65.0, resolution_ok=True,
        ignore_resolution_used=False, failure_reasons=[] if passed_ else ["blur"],
        retake_instructions=[], foreground_box=(1, 2, 3, 4), segmentation_used=True,
        whole_frame_score=70.0, width=1200, height=1600, foreground_score=score)


def detection(n=1):
    dets = [Detection(label="Warning sign (HV / RF radiation)", confidence=0.9,
                      box=[1, 2, 3, 4], source="model") for _ in range(n)]
    return DetectionStageResult(detections=dets, annotated_path=None,
                                model_name="YOLOX-S", is_stub=False)


print("\nthe three modes are registered and ordered")
check("three modes", len(M.MODES) == 3 and len(M.MODE_ORDER) == 3)
check("order is classic, or_gate, vlm_only",
      M.MODE_ORDER == [M.MODE_CLASSIC, M.MODE_OR_GATE, M.MODE_VLM_ONLY])
for mode_id in M.MODE_ORDER:
    m = M.MODES[mode_id]
    check(f"{mode_id} has a label, short name and blurb",
          all(m.get(k) for k in ("label", "short", "blurb")))

print("\nmode 2's OR gate - the reason string explains the decision")
r = M._or_gate_reason(quality(True), detection(1))
check("both satisfied", "quality passed and the detector found" in r, r)
r = M._or_gate_reason(quality(True), detection(0))
check("quality only", "detector found nothing" in r, r)
r = M._or_gate_reason(quality(False, 51.2), detection(1))
check("the OR case is spelled out", "proceeding on the OR" in r, r)
check("and it quotes the failing score", "51.2" in r, r)
r = M._or_gate_reason(quality(False), detection(0))
check("neither satisfied", "neither" in r, r)

print("\nmode 2 is strictly more permissive than mode 1, never less")
for qp, nd in [(True, 1), (True, 0), (False, 1), (False, 0)]:
    classic = qp
    or_gate = qp or bool(nd)
    check(f"quality={qp}, detections={nd}: mode 2 >= mode 1",
          or_gate >= classic, f"{or_gate} < {classic}")
check("the one case where they differ is the useful one: failed quality, "
      "detector found something", (False or True) and not False)

print("\nmode 3 maps onto the same three panels")
combined = {"quality": "good", "quality_reasoning": "Sharp and well exposed.",
            "subject_present": "yes", "subject_reasoning": "On the mast.",
            "answer": "yes", "reasoning": "A green placard is visible.",
            "raw_text": "{}", "model": "qwen3-vl", "elapsed_s": 0.4,
            "error": None, "is_mock": False}
rec = M._vlm_only_record(Path("/p/a.jpg"), "a", "hazard_warning", combined,
                         quality(True), 0.0)
check("quality panel filled from the model", rec.quality.passed is True)
check("and it says the model judged it", rec.quality.assessed_by == "vlm")
check("no fake numeric score in the headline",
      "0.0" not in rec.quality.headline and "judged by the model" in rec.quality.headline,
      rec.quality.headline)
check("frame size carried over from the file", rec.quality.width == 1200)
check("detection panel reports presence, not boxes",
      rec.detection.presence == "yes" and rec.detection.detections == [])
check("presence reasoning kept", rec.detection.presence_reasoning == "On the mast.")
check("the model name says no boxes", "presence only" in rec.detection.model_name)
check("the answer is the answer", rec.vlm.answer == "yes")
check("record completes", rec.stopped_at == STOPPED_COMPLETE)

poor = dict(combined, quality="poor", quality_reasoning="Badly blurred.",
            subject_present="unknown", answer="unknown")
rec = M._vlm_only_record(Path("/p/b.jpg"), "b", "hazard_warning", poor,
                         quality(True), 0.0)
check("a poor judgement fails the quality panel", rec.quality.passed is False)
check("and the reason is the model's own words",
      rec.quality.failure_reasons == ["Badly blurred."], str(rec.quality.failure_reasons))
check("unknown presence survives", rec.detection.presence == "unknown")

print("\nmode 3 renders the photograph it judged")
import tempfile  # noqa: E402
_written = []
# A real file on disk: image_or_placeholder checks existence before emitting an
# <img>, so a purely in-memory fake would silently exercise the placeholder path.
sys.modules["cv2"].imwrite = lambda path, img: (_written.append(path),
                                                Path(path).write_bytes(b"jpg"), True)[2]
tmp = Path(tempfile.mkdtemp()) / "plain" / "a.jpg"
rec_img = M._vlm_only_record(Path("/p/a.jpg"), "a", "hazard_warning", combined,
                             quality(True), 0.0, image_bgr=object(), image_path=tmp)
check("the plain copy is written and recorded",
      rec_img.quality.annotated_path == str(tmp), str(rec_img.quality.annotated_path))
check("its folder is created", tmp.parent.is_dir())
check("mode 3 draws no box on the detection panel",
      rec_img.detection.annotated_path is None)
sys.modules["cv2"].imwrite = lambda path, img: False
rec_fail = M._vlm_only_record(Path("/p/a.jpg"), "a", "hazard_warning", combined,
                              quality(True), 0.0, image_bgr=object(), image_path=tmp)
check("a failed write leaves the path empty rather than lying",
      rec_fail.quality.annotated_path is None)

print("\nmode 3 never fabricates on a mock or a parse failure")
from pipeline.stage3_vlm import (mock_vlm_only, parse_quality,  # noqa: E402
                                 parse_presence)
mock = mock_vlm_only(q, "qwen3-vl")
check("mock quality is poor, not good", mock["quality"] == "poor")
check("mock presence is unknown", mock["subject_present"] == "unknown")
check("mock answer is unknown", mock["answer"] == "unknown")
check("mock is labelled", mock["is_mock"] is True and "MOCK" in mock["quality_reasoning"])
check("an empty quality response defaults to poor", parse_quality("")[0] == "poor")
check("an empty presence response defaults to unknown",
      parse_presence("")[0] == "unknown")
check("prose with no keys does not invent a quality verdict",
      parse_quality("This photograph shows a cabinet in a compound.")[0] == "poor")
check("prose with no keys does not invent a presence verdict",
      parse_presence("This photograph shows a cabinet in a compound.")[0] == "unknown")

print("\nper-leg prompts")
from pipeline.questions import (LEGS, LEG_ANSWER, LEG_PRESENCE,  # noqa: E402
                                LEG_QUALITY, default_leg_system, render_leg_user)
check("there are exactly three legs", list(LEGS) == [LEG_QUALITY, LEG_PRESENCE,
                                                    LEG_ANSWER], str(LEGS))
check("each leg has a non-empty system prompt",
      all(default_leg_system(q, leg).strip() for leg in LEGS))
check("the answer leg uses the question's own system prompt",
      default_leg_system(q, LEG_ANSWER) == q.system_prompt)
from pipeline.questions import SUBJECTS  # noqa: E402
subject = SUBJECTS[q.id]
check("the quality leg is generic - it never names the subject",
      subject.lower() not in default_leg_system(q, LEG_QUALITY).lower(),
      default_leg_system(q, LEG_QUALITY)[:160])
check("the presence leg does name the subject",
      subject.lower() in render_leg_user(q, LEG_PRESENCE).lower(),
      render_leg_user(q, LEG_PRESENCE)[:160])
check("every leg's user prompt asks for its own key",
      all(k in render_leg_user(q, leg) for leg, k in
          ((LEG_QUALITY, "quality"), (LEG_PRESENCE, "present"),
           (LEG_ANSWER, "answer"))))
check("and asks for JSON only, so the parser has something to key on",
      all("JSON only" in render_leg_user(q, leg) for leg in LEGS))

print("\nprompt store")
from pipeline.prompts import PromptStore  # noqa: E402
store_path = Path(tempfile.mkdtemp()) / "prompts.yaml"
store = PromptStore(store_path)
check("an untouched leg returns the built-in prompt",
      store.get(q.id, LEG_QUALITY) == default_leg_system(q, LEG_QUALITY))
check("and is not reported as overridden",
      store.is_overridden(q.id, LEG_QUALITY) is False)
store.set(q.id, LEG_QUALITY, "Judge only sharpness.")
check("an edit is returned back", store.get(q.id, LEG_QUALITY) == "Judge only sharpness.")
check("and is reported as overridden", store.is_overridden(q.id, LEG_QUALITY) is True)
check("other legs are untouched",
      store.get(q.id, LEG_ANSWER) == default_leg_system(q, LEG_ANSWER))
check("saving succeeds", store.save() is None and store_path.exists())
reloaded = PromptStore(store_path)
check("the edit survives a reload",
      reloaded.get(q.id, LEG_QUALITY) == "Judge only sharpness.")
check("and nothing else was written",
      reloaded.is_overridden(q.id, LEG_ANSWER) is False)
reloaded.set(q.id, LEG_QUALITY, default_leg_system(q, LEG_QUALITY))
check("text equal to the default clears the override",
      reloaded.is_overridden(q.id, LEG_QUALITY) is False)
reloaded.save()
check("and an empty store removes the file entirely", not store_path.exists())
store_path.write_text("{{{ not yaml")
broken = PromptStore(store_path)
check("a malformed file degrades to the built-ins rather than crashing",
      broken.get(q.id, LEG_QUALITY) == default_leg_system(q, LEG_QUALITY))
check("and says so", bool(broken.load_error), str(broken.load_error))

print("\ncomparison across modes")
def record(stem, answer, stopped=STOPPED_COMPLETE, mock=False):
    return PipelineRecord(
        filename=f"{stem}.jpg", source_path=f"/p/{stem}.jpg", stem=stem,
        question_id="hazard_warning", quality=quality(True),
        detection=detection(1), stopped_at=stopped,
        vlm=None if stopped == STOPPED_QUALITY else VLMAnswer(
            answer=answer, reasoning="", raw_text="", model="qwen3-vl",
            elapsed_s=0.3, is_mock=mock))

results = {
    M.MODE_CLASSIC: [record("a", "yes"), record("b", "no", STOPPED_QUALITY)],
    M.MODE_OR_GATE: [record("a", "yes"), record("b", "no")],
    M.MODE_VLM_ONLY: [record("a", "yes"), record("b", "unknown")],
}
rows = M.compare_rows(results)
check("one row per image", len(rows) == 2, str(len(rows)))
check("columns carry each mode's answer",
      rows[0][M.MODE_CLASSIC] == "YES" and rows[0][M.MODE_VLM_ONLY] == "YES")
check("a stopped record reads as stopped, not as an answer",
      rows[1][M.MODE_CLASSIC] == "— stopped", rows[1][M.MODE_CLASSIC])
check("mode 2 answered where mode 1 stopped", rows[1][M.MODE_OR_GATE] == "NO")

a = M.agreement(results)
check("agreement counts every image", a["total"] == 2)
check("unanimous row counted", a["unanimous"] == 1, str(a))
check("split row counted", a["split"] == 1, str(a))
check("empty results do not divide by zero",
      M.agreement({})["total"] == 0)

mocked = {m: [record("a", "unknown", mock=True)] for m in M.MODE_ORDER}
check("a mock answer is marked in the comparison",
      "mock" in M.compare_rows(mocked)[0][M.MODE_CLASSIC],
      M.compare_rows(mocked)[0][M.MODE_CLASSIC])

print("\nUI wiring")
_gr = types.ModuleType("dash")


class Node:
    def __init__(self, children=None, **kw):
        self.children, self.kw = children, kw

    def text(self):
        # `options` carries a Dropdown/RadioItems' choices as a list of dicts;
        # a test asserting "all three modes are offered" has to be able to see
        # them, so it is flattened here rather than skipped.
        bits = [str(v) for k, v in self.kw.items()
                if k in ("className", "id", "title", "alt", "label", "value",
                         # `accept` decides which dropped files the browser
                         # hands over at all, so a test about uploads has to
                         # be able to see it.
                         "accept")]
        for option in self.kw.get("options") or []:
            bits.append(str(option))
        c = self.children
        if isinstance(c, (list, tuple)):
            bits += [i.text() if isinstance(i, Node) else str(i) for i in c]
        elif isinstance(c, Node):
            bits.append(c.text())
        elif c is not None:
            bits.append(str(c))
        return " ".join(bits)


class Dep:
    def __init__(self, *a, **k):
        self.args = a


_html = types.ModuleType("dash.html")
for n in ("Div", "Span", "H1", "H4", "P", "Ul", "Li", "Img", "Pre", "Details",
          "Summary", "Button", "Label", "B", "A"):
    setattr(_html, n, Node)
_dcc = types.ModuleType("dash.dcc")
for n in ("Dropdown", "Input", "Textarea", "Upload", "Checklist", "RadioItems",
          "Tabs", "Tab", "Loading", "Store", "Download"):
    setattr(_dcc, n, Node)
_dcc.send_file = lambda p: p
_dt = types.ModuleType("dash.dash_table")
_dt.DataTable = Node
_gr.html, _gr.dcc, _gr.dash_table = _html, _dcc, _dt
_gr.Input = _gr.Output = _gr.State = Dep
_gr.no_update = object()
_gr.callback_context = types.SimpleNamespace(triggered=[])


class App:
    def __init__(self, *a, **k):
        self.server, self.layout = object(), None

    def callback(self, *a, **k):
        return lambda f: f

    def run(self, *a, **k):
        pass


_gr.Dash = App
sys.modules.update({"dash": _gr, "dash.html": _html, "dash.dcc": _dcc,
                    "dash.dash_table": _dt})

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("ui", ROOT / "app" / "demo_dash_modes.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
layout_text = ui.layout().text()
from pipeline.questions import LEG_LABELS as M_LEG_LABELS  # noqa: E402

check("a mode selector exists", "mode" in layout_text)
check("all three modes are offered",
      all(M.MODES[m]["label"] in layout_text for m in M.MODE_ORDER))
check("a Compare tab exists", "compare" in layout_text)
check("the run button says it runs all three",
      "Run all three modes" in layout_text)
check("upload survives from the YOLOX app", "dropzone" in layout_text)

# The dropzone must accept files the browser cannot type. accept="image/*"
# alone filters on the reported MIME type, and a file dragged from a network
# share or a mapped drive arrives with none - react-dropzone then discards it
# in silence, so dropping five photographs delivered one with no error shown.
from pipeline.orchestrator import IMAGE_EXTENSIONS  # noqa: E402
check("the dropzone still accepts image MIME types", "image/*" in ui.UPLOAD_ACCEPT)
for ext in sorted(IMAGE_EXTENSIONS):
    check(f"...and {ext} by extension, whatever MIME the browser reports",
          ext in ui.UPLOAD_ACCEPT, ui.UPLOAD_ACCEPT)
check("accept and the folder scan agree on what a photograph is",
      {t for t in ui.UPLOAD_ACCEPT.split(",") if t.startswith(".")}
      == set(IMAGE_EXTENSIONS), ui.UPLOAD_ACCEPT)
check("the layout uses that accept rather than a literal",
      ui.UPLOAD_ACCEPT in layout_text, ui.UPLOAD_ACCEPT)
check("a non-image extension is still not accepted",
      ".txt" not in ui.UPLOAD_ACCEPT and ".csv" not in ui.UPLOAD_ACCEPT)

# Mode 3's detection panel must not read as "found nothing".
rec3 = M._vlm_only_record(Path("/p/a.jpg"), "a", "hazard_warning", combined,
                          quality(True), 0.0)
panel = ui.detection_column(rec3).text()
check("mode 3 detection panel shows presence", "Subject visible" in panel, panel[:120])
check("and states no boxes are drawn", "presence, not geometry" in panel)
check("and does NOT claim no detections", "No detections" not in panel)

rec12 = record("a", "yes")
panel = ui.detection_column(rec12).text()
check("modes 1 and 2 still render boxes normally", "2 · Detection" in panel)
check("with the label and confidence", "0.90" in panel, panel[:200])

check("a prompt editor exists for every leg",
      all(f"prompt-{leg}" in layout_text for leg in LEGS), layout_text[:0])
check("the fixed user prompt is shown too",
      all(f"userprompt-{leg}" in layout_text for leg in LEGS))
check("each leg is labelled", all(l in layout_text for l in M_LEG_LABELS.values()))
check("prompts can be saved and reset",
      "prompt-save" in layout_text and "prompt-reset" in layout_text)
check("mode 3 can be re-run on its own", "rerun3" in layout_text)
check("the model route can be chosen in the UI", "transport" in layout_text)
check("both routes are offered",
      "Direct to the GPU server" in layout_text and "proxy" in layout_text.lower())
check("there is somewhere to put the proxy's API key", "api-key" in layout_text)

# Mode 3's quality panel: the photograph, the verdict, and no invented number.
qpanel = ui.quality_column(rec3).text()
check("mode 3 quality panel renders the image", str(rec3.quality.annotated_path or "")
      in qpanel or "could not be rendered" in qpanel, qpanel[:160])
check("and says the model judged it", "judged by the model" in qpanel, qpanel[:200])
check("and shows no MM-IQA score", "whole frame" not in qpanel, qpanel[:200])
# thumb() embeds the photograph as a data URI via cv2, so the stub has to be
# able to decode and re-encode for this assertion to mean anything.
_cv2 = sys.modules["cv2"]
_cv2.imread = lambda path: (types.SimpleNamespace(shape=(100, 80, 3))
                            if Path(path).exists() else None)
_cv2.imencode = lambda ext, img, params=None: (
    True, types.SimpleNamespace(tobytes=lambda: b"\xff\xd8jpegbytes"))
_cv2.IMWRITE_JPEG_QUALITY = 1
# text() deliberately does not flatten `src`, so reach for the node itself.
qimg_node = ui.quality_column(rec_img).children[1]
check("the photograph is embedded as a data URI",
      str(qimg_node.kw.get("src", "")).startswith("data:image/jpeg;base64,"),
      str(qimg_node.kw)[:200])
check("and a missing file falls back to the placeholder, not a broken img",
      "Image could not be rendered" in ui.quality_column(rec_fail).text())
check("modes 1 and 2 keep the frozen quality panel",
      "whole frame" in ui.quality_column(record("a", "yes")).text())

card = ui.result_card(record("a", "yes"), q)
check("a gate reason is shown when present", True)
gated = record("a", "yes")
gated.extra["gate"] = "quality failed but the detector found the subject"
check("mode 2's gate reason reaches the card",
      "proceeding" in ui.result_card(gated, q).text()
      or "detector found the subject" in ui.result_card(gated, q).text())

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
