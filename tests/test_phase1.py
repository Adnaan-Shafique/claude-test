"""Phase 1 acceptance tests.

Covers only the stdlib-pure modules (schemas, config, questions) - stage1_quality
needs cv2/numpy/rembg and is verified separately on a machine that has them.
Run: python tests/test_phase1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from pipeline import config as pcfg
from pipeline import questions as pq
from pipeline import schemas as ps

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


print("\nschemas - the resolution-failure case the plan's schema could not express")
# A high-scoring image that fails purely on the resolution floor. This is the
# case that rendered as "FAIL 82.1 / 65" and read like a scoring bug.
r = ps.QualityStageResult(
    passed=False, score=82.1, threshold=65.0,
    resolution_ok=False, ignore_resolution_used=False,
    failure_reasons=["resolution_too_low"], retake_instructions=["Use a higher-resolution camera setting."],
    foreground_box=(10, 20, 300, 400), segmentation_used=True,
    whole_frame_score=71.3, width=320, height=240, foreground_score=82.1,
)
check("verdict is FAIL", r.verdict == ps.FAIL)
check("fail_kind is 'resolution', not 'score'", r.fail_kind == "resolution", f"got {r.fail_kind!r}")
check("headline explains the real reason",
      "resolution floor" in r.headline and "320x240" in r.headline, r.headline)

# Same score, above the floor, below the threshold.
r2 = ps.QualityStageResult(
    passed=False, score=51.2, threshold=65.0, resolution_ok=True, ignore_resolution_used=False,
    failure_reasons=["blur"], retake_instructions=["Hold the camera steady and retake."],
    foreground_box=None, segmentation_used=False, whole_frame_score=51.2,
    width=4000, height=3000,
)
check("plain score failure reports fail_kind 'score'", r2.fail_kind == "score", f"got {r2.fail_kind!r}")
check("plain score failure headline stays terse", r2.headline == "FAIL 51.2 / 65", r2.headline)

r3 = ps.QualityStageResult(
    passed=True, score=78.3, threshold=65.0, resolution_ok=True, ignore_resolution_used=False,
    failure_reasons=[], retake_instructions=[], foreground_box=(1, 2, 3, 4),
    segmentation_used=True, whole_frame_score=64.0, width=4000, height=3000, foreground_score=78.3,
)
check("pass has no fail_kind", r3.fail_kind is None)
check("pass headline is 'PASS 78.3 / 65'", r3.headline == "PASS 78.3 / 65", r3.headline)

print("\nschemas - Detection key mapping (inference returns 'class', not 'label')")
d = ps.Detection.from_model_dict(
    {"class": "gps_antenna", "confidence": 0.913, "box": [10, 20, 110, 140]})
check("'class' maps to .label", d.label == "gps_antenna")
check("source defaults to 'model'", d.source == ps.SOURCE_MODEL)
check("box coerced to floats", all(isinstance(v, float) for v in d.box))
check("area computed from xyxy", d.area == 100.0 * 120.0, f"got {d.area}")

print("\nschemas - provenance is explicit, never sniffed from the name")
stub = ps.DetectionStageResult(detections=[], annotated_path=None,
                               model_name="human annotation (no model loaded)", is_stub=True)
real = ps.DetectionStageResult(detections=[d], annotated_path=None,
                               model_name="yolox_s_field_ops", is_stub=False)
check("empty stub result still reports is_stub", stub.is_stub is True)
check("real result reports is_stub False", real.is_stub is False)
check("top() returns the highest-confidence detection", real.top().label == "gps_antenna")
check("top() on empty returns None", stub.top() is None)

print("\nquestions - every question carries a non-empty system prompt (trap 9)")
check("two questions registered", set(pq.QUESTIONS) == {"hazard_warning", "gps_antenna"},
      f"got {sorted(pq.QUESTIONS)}")
for qid, q in pq.QUESTIONS.items():
    check(f"{qid}: system_prompt non-empty", bool(q.system_prompt.strip()))
    check(f"{qid}: system_prompt is question-specific", q.system_prompt != pq.OUTPUT_CONTRACT)
check("the two system prompts differ from each other",
      pq.QUESTIONS["hazard_warning"].system_prompt != pq.QUESTIONS["gps_antenna"].system_prompt)
check("dropdown choices are (label, id) pairs",
      pq.QUESTION_CHOICES == [(q.label, q.id) for q in pq.QUESTIONS.values()])

# A Question with a blank system prompt must be rejected at construction, not
# silently sent with no framing.
try:
    pq.Question(id="bad", label="bad", system_prompt="   ",
                user_template="{detection_block}{output_contract}",
                answer_semantics="", relevant_classes=[])
    check("blank system_prompt raises", False, "no exception raised")
except ValueError:
    check("blank system_prompt raises ValueError", True)

print("\nquestions - prompt rendering with and without detections")
no_det = pq.render_user_prompt(pq.QUESTIONS["gps_antenna"])
check("no detections: no detector line", "Object detector output" not in no_det)
check("no detections: output contract present", "Respond with JSON only" in no_det)
check("no detections: no stray blank run", "\n\n\n" not in no_det)

with_det = pq.render_user_prompt(pq.QUESTIONS["gps_antenna"], [d])
check("with detections: detector line present", "Object detector output" in with_det)
check("with detections: marked advisory only", "advisory only" in with_det)
check("with detections: label and confidence rendered", "gps_antenna (confidence 0.91)" in with_det,
      with_det)
check("with detections: box rendered as ints", "[10, 20, 110, 140]" in with_det, with_det)

print("\nquestions - relevant_classes matching degrades safely (class names unconfirmed)")
matched = pq.select_relevant([d], pq.QUESTIONS["gps_antenna"])
check("matching label is selected", [x.label for x in matched] == ["gps_antenna"])
unknown = ps.Detection(label="class_0", confidence=0.9, box=[0, 0, 1, 1],
                       source=ps.SOURCE_ANNOTATION)
fallback = pq.select_relevant([unknown], pq.QUESTIONS["gps_antenna"])
check("unrecognised label falls back to ALL detections, not none",
      [x.label for x in fallback] == ["class_0"],
      "an unmatched label must not silently empty the detection block")
check("no detections stays empty", pq.select_relevant([], pq.QUESTIONS["gps_antenna"]) == [])

print("\nconfig - one explicit root, and server-side validation mirrored")
cfg = pcfg.default_config()
check("project_root resolves to the repo root",
      (cfg.project_root / "app" / "pipeline" / "config.py").exists(), str(cfg.project_root))
check("annotation_dir defaults under the root", cfg.annotation_dir == cfg.project_root / "data" / "labels")
check("models_dir agrees with foreground_segmentation's U2NET_HOME target",
      cfg.models_dir == cfg.project_root / "models")
check("default vlm model is the eager-loaded one", cfg.vlm_model == "qwen3-vl")
check("sampling defaults are repeatable", (cfg.temperature, cfg.top_p) == (0.0, 1.0))
check("repetition_penalty respects the server's ge=1.0", cfg.repetition_penalty >= 1.0)
check("stub mode is the default", cfg.use_model is False)
check("crop padding is a frame fraction, not a box fraction", cfg.crop_min_frame_frac == 0.20)

over = pcfg.default_config(quality_threshold=70.0, annotation_dir="/tmp/labels", vlm_mode="mock")
check("overrides apply", over.quality_threshold == 70.0 and over.vlm_mode == "mock")
check("path overrides route through the property setter",
      over.annotation_dir == Path("/tmp/labels"), str(over.annotation_dir))
try:
    pcfg.default_config(nonexistent_field=1)
    check("unknown config field raises", False, "no exception")
except AttributeError:
    check("unknown config field raises AttributeError", True)

bad = pcfg.default_config(vlm_mode="bogus", repetition_penalty=0.5, stub_conf_range=(0.9, 0.5))
problems = " ".join(bad.validate())
check("validate() catches a bad vlm_mode", "vlm_mode" in problems)
check("validate() catches repetition_penalty < 1.0", "repetition_penalty" in problems)
check("validate() catches an inverted stub_conf_range", "stub_conf_range" in problems)

print("\nschemas - flat CSV row")
rec = ps.PipelineRecord(
    filename="00002_task_328989.jpg", source_path="/photos/00002_task_328989.jpg",
    stem="00002_task_328989", question_id="gps_antenna", quality=r3,
    detection=ps.DetectionStageResult(detections=[d], annotated_path=None,
                                      model_name="human annotation (no model loaded)", is_stub=True),
    vlm=ps.VLMAnswer(answer="yes", reasoning="The antenna sits on an open mast.",
                     raw_text='{"answer":"yes"}', model="qwen3-vl", elapsed_s=3.21),
    stopped_at=ps.STOPPED_COMPLETE,
)
row = rec.to_flat_row()
check("row covers every declared column", set(row) == set(ps.FLAT_ROW_COLUMNS),
      f"missing={set(ps.FLAT_ROW_COLUMNS) - set(row)} extra={set(row) - set(ps.FLAT_ROW_COLUMNS)}")
check("stem is carried, not re-derived", row["stem"] == "00002_task_328989")
check("detections serialise as 'label:conf'", row["detections"] == "gps_antenna:0.91", row["detections"])
check("vlm answer present", row["vlm_answer"] == "yes")
check("mock flag surfaces through provenance", row["vlm_model"] == "qwen3-vl")

mock = ps.VLMAnswer(answer="unknown", reasoning="", raw_text="", model="qwen3-vl",
                    elapsed_s=0.0, is_mock=True, error="connection refused")
check("mock provenance is unmistakable", mock.provenance == "MOCK - GPU server unreachable")
check("answer chip uppercases", mock.chip == "UNKNOWN")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
