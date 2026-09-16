"""The demo's question registry - each question owns its own prompt pair.

There is deliberately no single global system prompt. "Is a warning sign
present" is a presence/legibility judgement; "is this antenna open to the sky"
is a spatial-occlusion judgement about the scene ABOVE the object. One generic
persona does both worse than two specific ones, so swapping the dropdown swaps
the entire prompt pair with it.

Adding a third question tomorrow = adding one Question entry below. No other
file changes.

The prompt text lives here as plain string constants rather than scattered
f-strings so a non-engineer can tune wording between rehearsals without reading
pipeline code.

Pure stdlib - importable and testable with nothing installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ─────────────────────────────── Shared contract ─────────────────────────────
# Appended to every question so the parser only ever sees one shape.
OUTPUT_CONTRACT = (
    'Respond with JSON only, no markdown fence:\n'
    '{"answer": "yes" | "no" | "unknown", '
    '"reasoning": "<2-3 sentences citing the specific visual evidence in the image>"}'
)

# Injected only when detections exist, otherwise the empty string. The
# "advisory only" wording is load-bearing: a confident-looking 0.91 presented as
# ground truth will anchor the model's answer, and in stub mode that number is a
# deterministic hash, not a measurement.
DETECTION_BLOCK_PREFIX = (
    "Object detector output (advisory only - it may be incomplete or wrong; "
    "trust the image over this list): "
)

ANSWER_YES = "yes"
ANSWER_NO = "no"
ANSWER_UNKNOWN = "unknown"
VALID_ANSWERS = (ANSWER_YES, ANSWER_NO, ANSWER_UNKNOWN)


# ─────────────────────────────── System prompts ──────────────────────────────

HAZARD_WARNING_SYSTEM = (
    "You are a site-safety inspector reviewing photographs of telecom equipment "
    "installations. Your task is to determine whether a hazardous-warning sign, "
    "label, or placard is visible in the photograph - for example high-voltage "
    "warnings, electrical-hazard symbols, RF-radiation warnings, danger/caution "
    "placards, or equivalent safety signage. Judge only from what is actually "
    "visible; do not infer that signage exists because the equipment type would "
    "normally require it. If the image is too unclear, cropped, or distant to "
    'tell, answer "unknown". Respond with JSON only.'
)

GPS_ANTENNA_SYSTEM = (
    "You are a telecom installation inspector reviewing photographs of GPS "
    "antennas mounted at cell sites. A GPS antenna functions correctly only when "
    "it has a clear, unobstructed view of the open sky above it. Your task is to "
    "judge, from the photograph, whether the antenna's upward view is clear. Pay "
    "attention to what sits directly above and around the antenna: roof "
    "overhangs, canopies, walls, tree canopy, cable trays, other antennas or "
    "equipment, or an indoor/sheltered location. Do not judge the antenna's "
    "cabling, cosmetic condition, or brand. If the region above the antenna is "
    'not visible in the frame, answer "unknown". Respond with JSON only.'
)


# ─────────────────────────────── User templates ──────────────────────────────

HAZARD_WARNING_TEMPLATE = """Does this photograph contain a hazardous-warning sign or label?
yes = at least one hazard/warning/danger sign or safety placard is visible.
no  = no such signage is visible anywhere in the frame.
{detection_block}
{output_contract}"""

GPS_ANTENNA_TEMPLATE = """Is the GPS antenna in this photograph properly mounted with an open view of the sky?
yes = the antenna is open to the sky; nothing significant obstructs its upward view.
no  = the antenna's view of the sky is blocked or partially blocked (overhang, canopy,
      wall, foliage, other equipment, or an indoor/sheltered mounting).
{detection_block}
{output_contract}"""


@dataclass(frozen=True)
class Question:
    id: str
    label: str                        # shown in the dropdown
    system_prompt: str                # PER QUESTION - sent as payload["system"]
    user_template: str                # PER QUESTION - .format(**ctx)
    answer_semantics: str             # human-readable yes/no meaning, shown in the UI
    relevant_classes: list[str]       # detector labels to surface / crop to
    # Class names for THIS question's annotation export, indexed by class_id.
    #
    # Each CVAT export is a separate single-class task, so every one of them
    # numbers its only class 0. "0" therefore means a hazard sign in the HV
    # export and a GPS antenna in the antenna export - the same id, two
    # meanings. A single shared classes.txt cannot express that, and guessing
    # wrong injects a false label into {detection_block}, which the prompt
    # explicitly tells the model to weigh as evidence.
    #
    # A real classes.txt or dataset.yaml in the label folder always wins; this
    # is the fallback that makes a bare single-class export readable.
    default_class_names: list[str] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)   # optional per-question overrides

    def __post_init__(self) -> None:
        # Trap 9: _build_payload only sets payload["system"] when the string is
        # non-empty, and both of the server's prompt builders guard on
        # `if system:`. A blank or mis-keyed system prompt therefore fails
        # SILENTLY - the model answers with no framing at all, which looks like a
        # model-quality problem rather than the wiring bug it is. Fail loudly
        # here instead, at import time.
        if not (self.system_prompt or "").strip():
            raise ValueError(f"Question {self.id!r} has an empty system_prompt")
        if "{output_contract}" not in self.user_template:
            raise ValueError(f"Question {self.id!r} template is missing {{output_contract}}")
        if "{detection_block}" not in self.user_template:
            raise ValueError(f"Question {self.id!r} template is missing {{detection_block}}")


QUESTIONS: dict[str, Question] = {
    "hazard_warning": Question(
        id="hazard_warning",
        label="Is a hazardous-warning sign present?",
        system_prompt=HAZARD_WARNING_SYSTEM,
        user_template=HAZARD_WARNING_TEMPLATE,
        answer_semantics=(
            "YES = a hazard/warning/danger sign or safety placard is visible.  "
            "NO = no such signage anywhere in the frame."
        ),
        # The HV export labels its single class 0; default_class_names below
        # resolves that to "hazard_sign" without a classes.txt. select_relevant()
        # still degrades safely if a folder uses different names.
        relevant_classes=["Warning sign (HV / RF radiation)", "hazard_sign",
                          "warning_sign", "hazard", "sign", "placard"],
        default_class_names=["Warning sign (HV / RF radiation)"],
    ),
    "gps_antenna": Question(
        id="gps_antenna",
        label="Is the GPS antenna open to the sky?",
        system_prompt=GPS_ANTENNA_SYSTEM,
        user_template=GPS_ANTENNA_TEMPLATE,
        answer_semantics=(
            "YES = the antenna's upward view is clear.  "
            "NO = its view of the sky is blocked or partially blocked."
        ),
        relevant_classes=["GPS Antenna", "gps_antenna", "gps", "antenna"],
        default_class_names=["GPS Antenna"],
    ),
}

# Feeds the Gradio dropdown: [(visible label, value), ...]
QUESTION_CHOICES = [(q.label, q.id) for q in QUESTIONS.values()]


def get_question(question_id: str) -> Question:
    try:
        return QUESTIONS[question_id]
    except KeyError:
        raise KeyError(
            f"Unknown question id {question_id!r}. Known: {sorted(QUESTIONS)}"
        ) from None


def _normalise(label: str) -> str:
    """Fold a class name to comparable words.

    Real labels are written for humans, not for matching: the trained detector's
    own classes.json says "GPS Antenna" and "Warning sign (HV / RF radiation)",
    while annotation folders use "gps_antenna". Exact string equality would miss
    every one of those, so punctuation, case and separators are flattened.
    """
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


def select_relevant(detections: Iterable, question: Question) -> list:
    """Detections worth surfacing for this question.

    Matches on normalised substrings in either direction, so "GPS Antenna"
    matches the gps_antenna question and "Warning sign (HV / RF radiation)"
    matches hazard_warning via "warning sign".

    Still deliberately TOLERANT: if nothing matches, return ALL detections
    rather than none. A label this function does not recognise silently
    emptying both the detection block and the crop would look exactly like
    "the detector found nothing" - the wrong story to tell on stage.
    """
    dets = list(detections)
    if not dets:
        return []
    wanted = [_normalise(c) for c in question.relevant_classes if c and c.strip()]
    wanted = [w for w in wanted if w]
    matched = []
    for d in dets:
        label = _normalise(str(getattr(d, "label", "")))
        if label and any(w in label or label in w for w in wanted):
            matched.append(d)
    return matched or dets


def build_detection_block(detections: Iterable) -> str:
    """The {detection_block} substitution. Empty string when there is nothing to
    report, so the template collapses cleanly rather than saying "detector found:"
    followed by nothing."""
    dets = list(detections)
    if not dets:
        return ""
    parts = []
    for d in dets:
        box = [int(round(v)) for v in d.box]
        parts.append(f"{d.label} (confidence {d.confidence:.2f}) at {box}")
    return DETECTION_BLOCK_PREFIX + "; ".join(parts) + "."


def render_user_prompt(question: Question, detections: Optional[Iterable] = None) -> str:
    """Format a question's user_template with its detection block and the shared
    output contract. Collapses the blank line left behind when there are no
    detections, so the prompt never carries a stray empty line."""
    block = build_detection_block(detections or [])
    text = question.user_template.format(
        detection_block=block,
        output_contract=OUTPUT_CONTRACT,
    )
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def sampling_for(question: Question, cfg) -> dict:
    """Per-question sampling overrides layered over the config defaults."""
    base = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repetition_penalty": cfg.repetition_penalty,
    }
    base.update(question.sampling or {})
    return base


# ─────────────────────── Mode 3: one call, three judgements ──────────────────
# Mode 3 replaces the classical quality gate and the trained detector with the
# model itself. One call returns all three judgements, so they are mutually
# consistent - the same pass that decided the photo is usable also decided
# whether the subject is visible and answered the inspection question.
#
# The question's OWN system prompt is reused verbatim rather than rewritten, so
# there is still exactly one place where each question's framing lives. Only the
# two extra judgements are described here.

COMBINED_SYSTEM_PREFIX = (
    "You are reviewing a single photograph from a telecom site inspection and "
    "must make THREE separate judgements about it: whether the photograph is of "
    "usable quality, whether the subject of the inspection is actually visible "
    "in it, and the inspection question itself. Judge each independently - a "
    "blurry photograph can still clearly show its subject, and a sharp "
    "photograph can be of the wrong thing. Answer only from what is visible, "
    "and prefer \"unknown\" over guessing. Respond with JSON only.\n\n"
    "The inspection question you are answering is framed as follows."
)

COMBINED_OUTPUT_CONTRACT = (
    'Respond with JSON only, no markdown fence:\n'
    '{"quality": "good" | "poor",\n'
    ' "quality_reasoning": "<1 sentence: sharpness, exposure, framing>",\n'
    ' "subject_present": "yes" | "no" | "unknown",\n'
    ' "subject_reasoning": "<1 sentence: is the subject visible, and where>",\n'
    ' "answer": "yes" | "no" | "unknown",\n'
    ' "reasoning": "<2-3 sentences citing the specific visual evidence>"}'
)

COMBINED_TEMPLATE = """Make three judgements about this photograph.

1. QUALITY - is this photograph usable for an inspection decision?
good = sharp enough, exposed well enough and framed well enough to judge from.
poor = blurred, too dark or bright, hazy, or the subject is too small or cut off.

2. SUBJECT - is {subject} actually visible in the frame?
yes = it is visible and identifiable.
no  = it is not in this photograph.
unknown = something may be there but it cannot be identified.

3. THE INSPECTION QUESTION - {question_text}
{semantics}

Judge each independently: a poor-quality photograph can still answer the
question, and a good-quality photograph of the wrong subject cannot.

{output_contract}"""

# What each question is looking for, in words the model can match against the
# image. Kept beside the questions themselves rather than in the mode code.
SUBJECTS = {
    "hazard_warning": "a hazard, warning or danger sign, label or placard",
    "gps_antenna": "a GPS antenna",
}

QUESTION_TEXT = {
    "hazard_warning": "does this photograph contain a hazardous-warning sign or label?",
    "gps_antenna": ("is the GPS antenna properly mounted with an open view of the "
                    "sky?"),
}


def combined_system_prompt(question: Question) -> str:
    """The mode-3 system prompt: the shared three-judgement framing, then the
    question's own system prompt verbatim."""
    return f"{COMBINED_SYSTEM_PREFIX}\n\n{question.system_prompt}"


def render_combined_prompt(question: Question) -> str:
    """The mode-3 user prompt. No detection block: in mode 3 nothing has run
    before this call, so there is no detector output to advise with."""
    return COMBINED_TEMPLATE.format(
        subject=SUBJECTS.get(question.id, "the subject of the inspection"),
        question_text=QUESTION_TEXT.get(question.id, question.label),
        semantics=question.answer_semantics,
        output_contract=COMBINED_OUTPUT_CONTRACT,
    )
