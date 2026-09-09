"""Phase 4 tests - orchestrator record shaping, CSV/JSON export, rendering.

Runs without cv2, gradio or a GPU: the record objects are built directly and
only the pure functions are exercised.
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from pipeline.orchestrator import (collect_images, new_run_id, sort_for_display,  # noqa: E402
                                   summarise, write_results)
from pipeline.questions import get_question                                       # noqa: E402
from pipeline.schemas import (Detection, DetectionStageResult, FLAT_ROW_COLUMNS,   # noqa: E402
                              PipelineRecord, QualityStageResult, STOPPED_COMPLETE,
                              STOPPED_QUALITY, VLMAnswer)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


def quality(passed_=True, score=80.0, resolution_ok=True, error=None):
    return QualityStageResult(
        passed=passed_, score=score, threshold=65.0, resolution_ok=resolution_ok,
        ignore_resolution_used=False, failure_reasons=[] if passed_ else ["blur"],
        retake_instructions=[] if passed_ else ["Hold the camera steady and retake."],
        foreground_box=(1, 2, 3, 4), segmentation_used=True, whole_frame_score=70.0,
        width=1200, height=1600, foreground_score=score, error=error)


def record(name, stopped=STOPPED_COMPLETE, answer="yes", passed_=True, mock=False):
    det = DetectionStageResult(
        detections=[Detection(label="hazard_sign", confidence=0.95,
                              box=[616, 945, 676, 1032], source="annotation_file")],
        annotated_path=None, model_name="human annotation (no model loaded)",
        is_stub=True, note="class names from question 'hazard_warning' defaults")
    vlm = VLMAnswer(answer=answer, reasoning="A warning placard is visible.",
                    raw_text='{"answer":"%s"}' % answer, model="qwen3-vl",
                    elapsed_s=0.31, is_mock=mock)
    return PipelineRecord(
        filename=f"{name}.jpg", source_path=f"/photos/{name}.jpg", stem=name,
        question_id="hazard_warning", quality=quality(passed_=passed_),
        detection=None if stopped == STOPPED_QUALITY else det,
        vlm=None if stopped == STOPPED_QUALITY else vlm, stopped_at=stopped)


print("\nrun ids are unique and sortable")
a, b = new_run_id(), new_run_id()
check("two ids differ", a != b)
check("id starts with a sortable timestamp", a[:8].isdigit() and len(a) > 15, a)

print("\ndisplay order - completed first, upload order preserved within a group")
recs = [record("c", STOPPED_QUALITY, passed_=False), record("a"), record("b"),
        record("d", STOPPED_QUALITY, passed_=False)]
ordered = [r.stem for r in sort_for_display(recs)]
check("completed images lead", ordered[:2] == ["a", "b"], str(ordered))
check("quality stops trail", ordered[2:] == ["c", "d"], str(ordered))

print("\nsummary counts")
s = summarise([record("a"), record("b", answer="no"),
               record("c", STOPPED_QUALITY, passed_=False),
               record("d", answer="unknown", mock=True)])
check("total counted", s["total"] == 4)
check("passes counted", s["passed"] == 3, str(s))
check("quality stops counted", s["stopped_at_quality"] == 1)
check("answers tallied", s["answers"] == {"yes": 1, "no": 1, "unknown": 1}, str(s["answers"]))
check("mock answers counted separately", s["mocked"] == 1)

print("\nexport - CSV and JSON")
tmp = Path(tempfile.mkdtemp())
recs = [record("a"), record("b", answer="no"), record("c", STOPPED_QUALITY, passed_=False)]
csv_path, json_path = write_results(recs, tmp)
check("csv written", csv_path.exists())
check("json written", json_path.exists())

import csv as _csv
with csv_path.open() as fh:
    rows = list(_csv.DictReader(fh))
check("one row per image", len(rows) == 3, str(len(rows)))
check("headers match the declared columns",
      list(rows[0].keys()) == FLAT_ROW_COLUMNS,
      str(set(rows[0].keys()) ^ set(FLAT_ROW_COLUMNS)))
check("stem carried into the csv", rows[0]["stem"] == "a")
check("detections serialised", rows[0]["detections"] == "hazard_sign:0.95",
      rows[0]["detections"])
check("a quality stop has no answer", rows[2]["vlm_answer"] == "", repr(rows[2]["vlm_answer"]))
check("stopped_at recorded", rows[2]["stopped_at"] == "quality")

data = json.loads(json_path.read_text())
check("json has one entry per image", len(data) == 3)
check("nested quality present", data[0]["quality"]["headline"].startswith("PASS"))
check("nested detection present", data[0]["detection"]["is_stub"] is True)
check("detection note preserved", "question" in data[0]["detection"]["note"])
check("nested vlm present", data[0]["vlm"]["answer"] == "yes")
check("raw model output kept for the accordion", data[0]["vlm"]["raw_text"] != "")
check("a stopped record carries null legs",
      data[2]["detection"] is None and data[2]["vlm"] is None)

print("\ncollect_images - recursive, since photos live in per-class subfolders")
tree = Path(tempfile.mkdtemp())
(tree / "hv").mkdir()
(tree / "gps").mkdir()
for p in ["hv/a.jpg", "gps/b.JPG", "hv/c.png", "hv/notes.txt"]:
    (tree / p).touch()
found = [p.name for p in collect_images(tree)]
check("images found across subfolders", sorted(found) == ["a.jpg", "b.JPG", "c.png"],
      str(sorted(found)))
check("non-images skipped", "notes.txt" not in found)
check("a single file path works too", collect_images(tree / "hv" / "a.jpg")[0].name == "a.jpg")

print("\ncard rendering - provenance and honesty requirements")
# A stub rich enough to import demo_app. Only the pure render functions are
# exercised; nothing here builds or launches a UI.
_gr = types.ModuleType("gradio")
_gr.__version__ = "5.50.0"


class _Anything:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        return _Anything()


for _name in ("Blocks", "Progress", "Row", "Column", "Group", "Tabs", "Tab",
              "Accordion", "Markdown", "HTML", "Dropdown", "Textbox", "File",
              "Number", "Checkbox", "Radio", "Button", "Dataframe", "State"):
    setattr(_gr, _name, _Anything)
sys.modules["gradio"] = _gr
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("demo_app", ROOT / "app" / "demo_app.py")
demo_app = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(demo_app)
    loaded = True
except Exception as exc:
    loaded = False
    print(f"       (demo_app not importable without real gradio: {exc})")

if loaded:
    q = get_question("hazard_warning")
    html_out = demo_app.render_cards([record("a"), record("c", STOPPED_QUALITY,
                                                          passed_=False)], q)
    check("stub provenance stated in the card",
          "human annotation (no model loaded)" in html_out)
    check("the answer chip is rendered", 'class="chip chip-yes"' in html_out, "")
    check("a quality stop says the legs did not run",
          "Stopped at the quality gate" in html_out)
    check("raw output behind an accordion", "<details>" in html_out)
    check("the detection note is surfaced", "question" in html_out)
    mock_html = demo_app.render_cards([record("m", answer="unknown", mock=True)], q)
    check("mock answers are labelled MOCK in the card", "MOCK" in mock_html)

for d in (tmp, tree):
    shutil.rmtree(d, ignore_errors=True)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
