"""Editable system prompts for mode 3, persisted to config/prompts.yaml.

Each of mode 3's three legs has its own system prompt, per question. The
built-in text lives in questions.py; anything edited in the UI is saved here and
loaded at startup, so a wording found during a rehearsal survives a restart and
can be reviewed, diffed and committed rather than living in someone's browser.

Only SYSTEM prompts are editable. The user prompts carry the JSON contract the
parsers depend on, and a demo is not the place to discover that someone removed
it - render_leg_user() in questions.py stays code.

File shape:

    hazard_warning:
      quality:  "..."
      presence: "..."
      answer:   "..."
    gps_antenna:
      ...

Only overridden legs are written. Anything absent falls back to the built-in
default, so deleting a key is a valid way to reset it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .questions import LEGS, QUESTIONS, default_leg_system, get_question

CONFIG_NAME = "prompts.yaml"


class PromptStore:
    """Per-question, per-leg system prompts with the built-ins underneath."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.overrides: dict = {}
        self.load_error: Optional[str] = None
        if self.path is not None:
            self.load()

    # ── Reading ──────────────────────────────────────────────────────────────
    def get(self, question_id: str, leg: str) -> str:
        if leg not in LEGS:
            raise KeyError(f"Unknown leg {leg!r}; expected one of {LEGS}")
        override = (self.overrides.get(question_id) or {}).get(leg)
        if override and override.strip():
            return override
        return default_leg_system(get_question(question_id), leg)

    def is_overridden(self, question_id: str, leg: str) -> bool:
        value = (self.overrides.get(question_id) or {}).get(leg)
        return bool(value and value.strip())

    def for_question(self, question_id: str) -> dict:
        return {leg: self.get(question_id, leg) for leg in LEGS}

    def overridden_legs(self) -> list[tuple[str, str]]:
        """Every (question_id, leg) currently differing from the built-in, so
        the UI can say plainly what has been changed."""
        return [(qid, leg) for qid in QUESTIONS for leg in LEGS
                if self.is_overridden(qid, leg)]

    # ── Writing ──────────────────────────────────────────────────────────────
    def set(self, question_id: str, leg: str, text: str) -> None:
        """Store an override. Text equal to the built-in, or blank, clears it
        rather than storing a redundant copy - so a file only ever contains
        genuine differences."""
        if leg not in LEGS:
            raise KeyError(f"Unknown leg {leg!r}; expected one of {LEGS}")
        text = (text or "").strip()
        default = default_leg_system(get_question(question_id), leg)
        bucket = self.overrides.setdefault(question_id, {})
        if not text or text == default.strip():
            bucket.pop(leg, None)
            if not bucket:
                self.overrides.pop(question_id, None)
        else:
            bucket[leg] = text

    def reset(self, question_id: Optional[str] = None,
              leg: Optional[str] = None) -> None:
        """Drop overrides: one leg, one question, or everything."""
        if question_id is None:
            self.overrides.clear()
        elif leg is None:
            self.overrides.pop(question_id, None)
        else:
            bucket = self.overrides.get(question_id) or {}
            bucket.pop(leg, None)
            if not bucket:
                self.overrides.pop(question_id, None)

    # ── Persistence ──────────────────────────────────────────────────────────
    def load(self) -> None:
        self.load_error = None
        if self.path is None or not self.path.exists():
            self.overrides = {}
            return
        try:
            import yaml
        except ImportError:
            self.load_error = ("pyyaml is not installed - saved prompts cannot be "
                               "read, using the built-in defaults")
            self.overrides = {}
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            # A malformed file must not take the demo down. Fall back to the
            # built-ins and say why, rather than raising at import time.
            self.load_error = f"could not read {self.path}: {exc}"
            self.overrides = {}
            return
        clean: dict = {}
        for question_id, legs in (data or {}).items():
            if question_id not in QUESTIONS or not isinstance(legs, dict):
                continue
            bucket = {leg: str(text) for leg, text in legs.items()
                      if leg in LEGS and str(text or "").strip()}
            if bucket:
                clean[question_id] = bucket
        self.overrides = clean

    def save(self) -> Optional[str]:
        """Write the overrides. Returns an error string, or None on success."""
        if self.path is None:
            return "no prompts file configured"
        try:
            import yaml
        except ImportError:
            return "pyyaml is not installed - cannot save prompts"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.overrides:
                # Everything is back to the built-ins: remove the file rather
                # than leaving an empty one that looks like a configuration.
                if self.path.exists():
                    self.path.unlink()
                return None
            header = (
                "# Mode 3 system prompts, edited in the demo UI.\n"
                "# Only legs that differ from the built-in defaults appear here;\n"
                "# deleting a key restores that leg's default.\n"
                "# The user prompts are NOT editable - they carry the JSON\n"
                "# contract the parsers depend on. See pipeline/questions.py.\n\n")
            body = yaml.safe_dump(self.overrides, sort_keys=True,
                                  allow_unicode=True, default_flow_style=False)
            self.path.write_text(header + body, encoding="utf-8")
            return None
        except Exception as exc:
            return f"could not write {self.path}: {exc}"


def default_store(cfg=None) -> PromptStore:
    """The store at <project_root>/config/prompts.yaml."""
    if cfg is None:
        from .config import default_config
        cfg = default_config()
    return PromptStore(Path(cfg.project_root) / "config" / CONFIG_NAME)
