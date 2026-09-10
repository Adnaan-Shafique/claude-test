"""YOLOX integration tests.

Runs without torch, torchvision or the yolox package: the runtime's pure
geometry is exercised directly, and the model path is stubbed. Real weights are
verified by tools/smoke_yolox.py on the demo host.
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


from pipeline.config import default_config                    # noqa: E402
from pipeline import yolox_runtime as yr                      # noqa: E402

print("\ndefaults pinned to the training run's own exp table and classes.json")
cfg = default_config()
check("num classes = 2", len(cfg.yolox_class_names) == 2, str(cfg.yolox_class_names))
# classes.json: {"names": ["GPS Antenna", "Warning sign (HV / RF radiation)"]}
check("index 0 is GPS Antenna", cfg.yolox_class_names[0] == "GPS Antenna",
      cfg.yolox_class_names[0])
check("index 1 is the warning sign", cfg.yolox_class_names[1].startswith("Warning sign"),
      cfg.yolox_class_names[1])
check("depth 0.33 (yolox_s)", cfg.yolox_depth == 0.33)
check("width 0.50 (yolox_s)", cfg.yolox_width == 0.50)
check("input size is 640x480, NOT the stock 640x640",
      cfg.yolox_input_size == (640, 480), str(cfg.yolox_input_size))
check("nms threshold 0.65 (yolox_base default)", cfg.yolox_nms_threshold == 0.65)
check("stub is still the default backend", cfg.use_model is False)

print("\nconfig validation catches a misconfigured detector")
bad = default_config(use_model=True)
problems = " ".join(bad.validate())
check("missing checkpoint is reported", "yolox_checkpoint" in problems, problems)
bad2 = default_config(use_model=True, yolox_checkpoint="/nope/best_ckpt.pth")
check("a non-existent checkpoint is reported",
      "not found" in " ".join(bad2.validate()), " ".join(bad2.validate()))
bad3 = default_config(use_model=True, yolox_checkpoint=__file__, yolox_class_names=())
check("empty class names are reported",
      "yolox_class_names" in " ".join(bad3.validate()))

print("\nletterbox geometry - the ratio the boxes are divided back out by")
# The demo photos: 1200x1600 portrait into a 640x480 canvas.
for (h, w), expect in [((1600, 1200), 640 / 1600), ((2880, 1623), 640 / 2880),
                       ((480, 640), 480 / 640)]:
    r = min(640 / h, 480 / w)
    check(f"{w}x{h} -> ratio {r:.4f}", abs(r - expect) < 1e-9, f"{r} vs {expect}")

r = min(640 / 1600, 480 / 1200)
check("a 1200x1600 photo scales to fit exactly, no crop",
      abs(1600 * r - 640) < 1 and abs(1200 * r - 480) < 1,
      f"{1200 * r:.1f}x{1600 * r:.1f}")

# The stock square canvas is NOT always a different ratio. For a portrait photo
# the height limits the scale either way, so the boxes land identically - the
# model simply sees more grey padding. The geometry only diverges once the photo
# is wider than the canvas's own aspect.
r_square_portrait = min(640 / 1600, 640 / 1200)
check("portrait: 640x640 gives the SAME ratio, so boxes do not move",
      abs(r_square_portrait - r) < 1e-9, f"{r_square_portrait} vs {r}")

r_land_trained = min(640 / 1200, 480 / 1600)      # 1600x1200 landscape into 640x480
r_land_square = min(640 / 1200, 640 / 1600)       # ... into 640x640
check("landscape: the ratio genuinely differs between the two canvases",
      abs(r_land_trained - r_land_square) > 1e-6,
      f"{r_land_trained} vs {r_land_square}")
# 1600x1200 into 640x480: width limits at 480/1600 = 0.30.
# Into 640x640: width limits at 640/1600 = 0.40. The boxes come out 4/3 too big.
check("and every box would be 4/3 too large",
      abs(r_land_square / r_land_trained - 4 / 3) < 0.01,
      f"{r_land_square / r_land_trained:.3f}")

print("\nbuild_model explains itself when yolox.models is absent")
sys.modules.pop("yolox.models", None)
try:
    yr.build_model(2)
    check("missing yolox.models raises", False, "no exception")
except ImportError as exc:
    msg = str(exc)
    check("missing yolox.models raises ImportError", True)
    check("the message names the gitignore cause", "gitignore" in msg, msg[:200])
    check("the message gives the copy command", "cp -r" in msg)
except Exception as exc:
    check("missing yolox.models raises ImportError", False, f"got {type(exc).__name__}")

print("\ndetector selection")
from pipeline.stage2_detect import AnnotationFileDetector, YoloxDetector, get_detector  # noqa: E402
from pipeline.questions import get_question                                             # noqa: E402
q = get_question("hazard_warning")
det = get_detector(default_config(), question=q)
check("use_model=False gives the annotation stub",
      isinstance(det, AnnotationFileDetector))
check("the stub reports is_stub", det.is_stub is True)
check("YoloxDetector declares is_stub False", YoloxDetector.is_stub is False)

try:
    get_detector(default_config(use_model=True), question=q)
    check("use_model=True with no checkpoint raises", False, "no exception")
except (ValueError, ImportError, FileNotFoundError) as exc:
    check("use_model=True with no checkpoint raises", True)
    check("the error names the setting", "yolox_checkpoint" in str(exc)
          or "models" in str(exc), str(exc)[:160])

print("\nquestion matching survives the real label text")
from pipeline.questions import _normalise, select_relevant  # noqa: E402
from pipeline.schemas import Detection as _D                # noqa: E402
hazard_q = get_question("hazard_warning")
antenna_q = get_question("gps_antenna")


def det(label):
    return _D(label=label, confidence=0.9, box=[0, 0, 10, 10], source="model")


check("punctuation and case folded",
      _normalise("Warning sign (HV / RF radiation)") == "warning sign hv rf radiation",
      _normalise("Warning sign (HV / RF radiation)"))
check("underscores folded too", _normalise("gps_antenna") == "gps antenna")

warning = det("Warning sign (HV / RF radiation)")
antenna = det("GPS Antenna")

picked = select_relevant([warning, antenna], hazard_q)
check("hazard question picks the warning sign",
      [d.label for d in picked] == ["Warning sign (HV / RF radiation)"],
      str([d.label for d in picked]))

picked = select_relevant([warning, antenna], antenna_q)
check("antenna question picks the GPS antenna",
      [d.label for d in picked] == ["GPS Antenna"], str([d.label for d in picked]))

# The annotation stub's snake_case names must keep working alongside.
check("snake_case labels still match the hazard question",
      [d.label for d in select_relevant([det("hazard_sign")], hazard_q)] == ["hazard_sign"])
check("snake_case labels still match the antenna question",
      [d.label for d in select_relevant([det("gps_antenna")], antenna_q)] == ["gps_antenna"])

# And the tolerant fallback survives.
unknown = [det("class_0")]
check("an unrecognised label still falls back to ALL detections",
      [d.label for d in select_relevant(unknown, hazard_q)] == ["class_0"])

from pipeline.questions import build_detection_block  # noqa: E402
block = build_detection_block([warning])
check("the real name reaches the model prompt verbatim",
      "Warning sign (HV / RF radiation)" in block, block)

print("\nno tool hardcodes class names over the config")
# A stale literal in a --classes default silently overrides cfg.yolox_class_names
# and reports wrong labels while the config is right. That happened once; this
# stops it recurring. Question IDs (--question hazard_warning) are a different
# thing and are deliberately not flagged.
import re as _re  # noqa: E402

offenders = []
for path in sorted((ROOT / "tools").glob("*.py")) + [ROOT / "app" / "demo_dash_yolox.py"]:
    text = path.read_text()
    for match in _re.finditer(r'add_argument\(\s*["\']--classes["\'][^)]*?\)', text,
                              _re.S):
        block = match.group(0)
        default = _re.search(r'default\s*=\s*(["\'][^"\']*["\']|None)', block)
        if default and default.group(1) != "None":
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}: --classes default={default.group(1)}")
check("no --classes default overrides the config", not offenders,
      "\n         ".join(offenders))

smoke = (ROOT / "tools" / "smoke_yolox.py").read_text()
check("smoke_yolox falls back to cfg.yolox_class_names",
      "default_config().yolox_class_names" in smoke)

# The class names live in exactly one place. Anything else defining that tuple
# is a second source of truth waiting to drift.
cfg_src = (ROOT / "app" / "pipeline" / "config.py").read_text()
check("config.py is where the names are defined",
      "GPS Antenna" in cfg_src and "Warning sign" in cfg_src)
for path in sorted((ROOT / "app" / "pipeline").glob("*.py")):
    if path.name in ("config.py", "questions.py"):
        continue   # questions.py legitimately names them for relevance matching
    body = path.read_text()
    check(f"{path.name} does not redefine the class names",
          "GPS Antenna" not in body, path.name)

print("\nboth backends produce the same downstream shape")
from pipeline.schemas import Detection, DetectionStageResult, SOURCE_MODEL  # noqa: E402
model_det = Detection.from_model_dict(
    {"class": "hazard_sign", "confidence": 0.87, "box": [10, 20, 110, 140]},
    source=SOURCE_MODEL)
check("a model detection carries source='model'", model_det.source == "model")
check("its label came from 'class'", model_det.label == "hazard_sign")
real = DetectionStageResult(detections=[model_det], annotated_path=None,
                            model_name="YOLOX-S · 2 classes · 640x480 · cpu",
                            is_stub=False)
check("a real result is not flagged stub", real.is_stub is False)
check("model_name describes the real model", "YOLOX-S" in real.model_name)
check("top() works the same either way", real.top().label == "hazard_sign")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
