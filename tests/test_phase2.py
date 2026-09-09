"""Phase 2 tests - AnnotationFileDetector.

Parsing, class-name resolution, deterministic confidence and the missing /
malformed paths run with a stubbed numpy-free image, so this needs no cv2.
Drawing is covered separately where cv2 exists.
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

# stage2_detect imports .detect_draw, which imports cv2 only inside functions,
# but the module object must exist for the package import to succeed.
for name in ("cv2",):
    sys.modules.setdefault(name, types.ModuleType(name))

from pipeline.config import default_config           # noqa: E402
from pipeline.stage2_detect import (                 # noqa: E402
    AnnotationFileDetector, _stub_confidence, load_class_names, parse_annotation_line)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


class FakeImage:
    """Just enough of a numpy array for the detector: .shape."""
    def __init__(self, h, w):
        self.shape = (h, w, 3)


tmp = Path(tempfile.mkdtemp())


def detector(**over):
    cfg = default_config(annotation_dir=tmp, **over)
    return AnnotationFileDetector(cfg)


print("\nline parsing - YOLO normalized (the real sample's format)")
# The actual annotation supplied for this demo, against its actual photo size.
p = parse_annotation_line("0 0.538250 0.617906 0.050500 0.054312", 1200, 1600)
check("class id parsed", p["class_id"] == 0 and p["label_token"] is None)
check("no confidence column detected", p["confidence"] is None)
x1, y1, x2, y2 = p["box"]
check("centre x lands where expected", abs((x1 + x2) / 2 - 0.53825 * 1200) < 0.01,
      f"centre={(x1 + x2) / 2}")
check("centre y lands where expected", abs((y1 + y2) / 2 - 0.617906 * 1600) < 0.01)
check("width scales to px", abs((x2 - x1) - 0.0505 * 1200) < 0.01, f"w={x2 - x1}")
check("height scales to px", abs((y2 - y1) - 0.054312 * 1600) < 0.01, f"h={y2 - y1}")

print("\nline parsing - absolute pixel xyxy, auto-detected")
p = parse_annotation_line("1 100 200 340 560", 1200, 1600)
check("coords passed through unscaled", p["box"] == [100.0, 200.0, 340.0, 560.0], str(p["box"]))
p = parse_annotation_line("0 0.5 0.5 0.2 0.2 0.77", 1000, 1000)
check("6th column read as confidence", p["confidence"] == 0.77)
p = parse_annotation_line("hazard_sign 0.5 0.5 0.2 0.2", 1000, 1000)
check("string label in column 0 kept as the label",
      p["label_token"] == "hazard_sign" and p["class_id"] is None)

print("\nline parsing - malformed input raises rather than corrupting")
for bad, why in [
    ("0 0.5 0.5", "too few fields"),
    ("0 a b c d", "non-numeric coords"),
    ("0 0.5 0.5 0 0.2", "zero width"),
    ("0 900 200 100 560", "x2 < x1 in pixel mode"),
    ("0 0.5 0.5 0.2 0.2 zzz", "non-numeric confidence"),
]:
    try:
        parse_annotation_line(bad, 1000, 1000)
        check(f"rejects {why}", False, f"{bad!r} was accepted")
    except ValueError:
        check(f"rejects {why}", True)

print("\ndeterministic confidence (re-running live must not change the number)")
a = _stub_confidence("00002_task_328989", 0, (0.88, 0.97))
b = _stub_confidence("00002_task_328989", 0, (0.88, 0.97))
check("same stem+index gives the same value every call", a == b, f"{a} vs {b}")
check("value sits inside the configured band", 0.88 <= a <= 0.97, str(a))
check("a different index gives a different value",
      _stub_confidence("00002_task_328989", 1, (0.88, 0.97)) != a)
check("a different stem gives a different value",
      _stub_confidence("00005_task_329433", 0, (0.88, 0.97)) != a)
narrow = _stub_confidence("x", 0, (0.5, 0.6))
check("the band is configurable", 0.5 <= narrow <= 0.6, str(narrow))

print("\ndetect() - a real annotation file")
(tmp / "00002_task_328989.txt").write_text("0 0.538250 0.617906 0.050500 0.054312\n")
d = detector()
r = d.detect(FakeImage(1600, 1200), "00002_task_328989")
check("one detection returned", len(r.detections) == 1)
check("provenance marked as annotation", r.detections[0].source == "annotation_file")
check("result flagged is_stub", r.is_stub is True)
check("model_name never implies a model ran",
      r.model_name == "human annotation (no model loaded)", r.model_name)
check("falls back to class_<id> with no classes.txt", r.detections[0].label == "class_0",
      r.detections[0].label)
check("the missing-class-names situation is noted", "classes.txt" in r.note, r.note)

print("\ndetect() - missing file must not raise and must not stop the VLM")
r = d.detect(FakeImage(1600, 1200), "no_such_photo")
check("no exception, zero detections", r.detections == [])
check("note names the file that was looked for", "no_such_photo.txt" in r.note, r.note)
check("still flagged is_stub", r.is_stub is True)

print("\ndetect() - malformed lines are skipped, the good ones survive")
(tmp / "mixed.txt").write_text(
    "# a comment\n"
    "\n"
    "0 0.5 0.5 0.2 0.2\n"
    "0 0.5 0.5\n"                 # too few fields
    "1 nope nope nope nope\n"     # non-numeric
    "1 0.25 0.25 0.1 0.1 0.42\n"
)
r = d.detect(FakeImage(1000, 1000), "mixed")
check("both valid lines kept", len(r.detections) == 2, f"{len(r.detections)}")
check("both malformed lines noted", r.note.count("line") >= 2, r.note)
check("note cites the offending line numbers", "line 4" in r.note and "line 5" in r.note, r.note)
check("explicit confidence preserved", r.detections[1].confidence == 0.42)
check("missing confidence filled deterministically", 0.88 <= r.detections[0].confidence <= 0.97)

print("\nclass names - resolution order")
check("nothing present -> empty", load_class_names(tmp) == [])
(tmp / "dataset.yaml").write_text("names:\n  - gps_antenna\n  - hazard_sign\n")
check("dataset.yaml block list read", load_class_names(tmp) == ["gps_antenna", "hazard_sign"],
      str(load_class_names(tmp)))
(tmp / "classes.txt").write_text("gps_antenna\nhazard_sign\n")
check("classes.txt wins over dataset.yaml",
      load_class_names(tmp) == ["gps_antenna", "hazard_sign"])
r = detector().detect(FakeImage(1600, 1200), "00002_task_328989")
check("label resolves through classes.txt", r.detections[0].label == "gps_antenna",
      r.detections[0].label)
check("no class-names complaint once they exist", "classes.txt" not in r.note, r.note)

(tmp / "classes.txt").unlink()
(tmp / "dataset.yaml").write_text("names: {0: gps_antenna, 1: hazard_sign}\n")
check("dataset.yaml dict form ordered by index",
      load_class_names(tmp) == ["gps_antenna", "hazard_sign"], str(load_class_names(tmp)))
(tmp / "dataset.yaml").unlink()
(tmp / "obj.names").write_text("gps_antenna\nhazard_sign\n")
check("CVAT obj.names read", load_class_names(tmp) == ["gps_antenna", "hazard_sign"])

print("\nout-of-range class id degrades rather than crashing")
(tmp / "high.txt").write_text("7 0.5 0.5 0.2 0.2\n")
r = detector().detect(FakeImage(1000, 1000), "high")
check("unknown index falls back to class_7", r.detections[0].label == "class_7",
      r.detections[0].label)

print("\ntwo single-class exports both numbering their class 0")
from pipeline.questions import get_question   # noqa: E402
from pipeline.stage2_detect import get_detector  # noqa: E402

hv = Path(tempfile.mkdtemp())      # the HV Hazardous Radiations export
gps = Path(tempfile.mkdtemp())     # a GPS Antenna export, added later
(hv / "hv_photo.txt").write_text("0 0.5 0.5 0.2 0.2\n")
(gps / "gps_photo.txt").write_text("0 0.5 0.5 0.2 0.2\n")

hazard_q = get_question("hazard_warning")
antenna_q = get_question("gps_antenna")

d_hv = get_detector(default_config(annotation_dir=hv), question=hazard_q)
r_hv = d_hv.detect(FakeImage(1000, 1000), "hv_photo")
check("class 0 reads as hazard_sign under the hazard question",
      r_hv.detections[0].label == "hazard_sign", r_hv.detections[0].label)

d_gps = get_detector(default_config(annotation_dir=gps), question=antenna_q)
r_gps = d_gps.detect(FakeImage(1000, 1000), "gps_photo")
check("the SAME class id 0 reads as gps_antenna under the antenna question",
      r_gps.detections[0].label == "gps_antenna", r_gps.detections[0].label)

check("the note says the names came from the question, not the data",
      "question" in r_hv.note, r_hv.note)

# A real classes.txt must beat the question's guess.
(hv / "classes.txt").write_text("HV_warning_placard\n")
d_hv2 = get_detector(default_config(annotation_dir=hv), question=hazard_q)
r_hv2 = d_hv2.detect(FakeImage(1000, 1000), "hv_photo")
check("a classes.txt in the folder overrides the question default",
      r_hv2.detections[0].label == "HV_warning_placard", r_hv2.detections[0].label)
check("no question-provenance note once the folder supplies names",
      "question" not in r_hv2.note, r_hv2.note)

# Per-question folders, so both exports can coexist.
both = default_config()
both.annotation_dirs = {"hazard_warning": hv, "gps_antenna": gps}
check("per-question folder chosen for hazard_warning",
      both.resolve_annotation_dir(question_id="hazard_warning") == hv)
check("per-question folder chosen for gps_antenna",
      both.resolve_annotation_dir(question_id="gps_antenna") == gps)
check("an unmapped question still falls back",
      both.resolve_annotation_dir(question_id="something_else")
      == both.project_root / "data" / "labels")

# With no question at all, ids stay honest rather than guessing.
d_bare = get_detector(default_config(annotation_dir=gps))
r_bare = d_bare.detect(FakeImage(1000, 1000), "gps_photo")
check("no question -> class_0, never an invented name",
      r_bare.detections[0].label == "class_0", r_bare.detections[0].label)

for d in (hv, gps):
    shutil.rmtree(d, ignore_errors=True)

print("\nlabel-directory resolution - labels usually sit beside the photos")
side = Path(tempfile.mkdtemp())
(side / "a.jpg").touch()
(side / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
cfg = default_config()
check("photo folder holding .txt files is chosen",
      cfg.resolve_annotation_dir(side) == side, str(cfg.resolve_annotation_dir(side)))

bare = Path(tempfile.mkdtemp())
(bare / "a.jpg").touch()
check("photo folder with no labels falls back to data/labels",
      cfg.resolve_annotation_dir(bare) == cfg.project_root / "data" / "labels",
      str(cfg.resolve_annotation_dir(bare)))

only_classes = Path(tempfile.mkdtemp())
(only_classes / "a.jpg").touch()
(only_classes / "classes.txt").write_text("gps_antenna\n")
check("a folder holding only classes.txt is not mistaken for a label folder",
      cfg.resolve_annotation_dir(only_classes) != only_classes,
      str(cfg.resolve_annotation_dir(only_classes)))

explicit = default_config(annotation_dir=tmp)
check("an explicit annotation_dir always wins",
      explicit.resolve_annotation_dir(side) == tmp, str(explicit.resolve_annotation_dir(side)))
check("resolution is safe with no images_dir at all",
      cfg.resolve_annotation_dir(None) == cfg.project_root / "data" / "labels")

for d in (side, bare, only_classes):
    shutil.rmtree(d, ignore_errors=True)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
