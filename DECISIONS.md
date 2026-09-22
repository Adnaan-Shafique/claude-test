# Decisions

Why things are the way they are. Each entry records the choice, the reason, and
what it costs — so a later reader can tell a deliberate trade from an accident,
and knows what to re-examine if the constraint changes.

Newest first. Dates are when the decision was made, not when it was written up.

---

## 2026-09-18 · Run the demo under systemd, on Werkzeug

**Decision.** A systemd unit (`deploy/fieldops-demo-modes.service`) running
Dash's built-in server, not gunicorn.

**Why.** The demo had been started by hand on every host, so it died with the
SSH session. Werkzeug is a development server, but for a closed network and a
known audience it is the better trade: no extra dependency, no extra failure
mode, and it is threaded, so a multi-second pipeline run does not block the
page.

**Cost.** Not suitable if this outlives the demo. `SERVICE.md` has the gunicorn
line, pinned to `--workers 1` — results live in a module-level dict, so a
second worker would answer half the requests from an empty set.

**Related.** The API key lives in `/etc/fieldops-demo.env` at `chmod 600`, not
in the unit: unit files are world-readable and `systemctl cat` prints them on
request.

---

## 2026-09-17 · Uploads accumulate instead of replacing

**Decision.** Photographs stage to disk on arrival and accumulate in a
`dcc.Store`; a second drop adds to the set rather than replacing it.

**Why.** Six files selected in Explorer, dragged in one gesture, arrived as
one. Six correctly-typed JPEGs dropped in a real browser against this exact
layout all arrive, so the desktop handed the browser a single-file
`DataTransfer` — not something Python can fix. What *can* be fixed is the UI
depending on that gesture succeeding.

**Cost.** More moving parts than reading `contents` off the component: a Store,
a Clear button, and staged files that outlive a page reload.

**Also decided here.** The note reports what is *staged*, not what the last drop
carried, so the number on screen is the number that will run. Clearing the
component's `contents` after each merge releases browser memory *and* lets the
same file be dropped twice — an unchanged prop would not fire the callback
again.

---

## 2026-09-17 · `accept` lists extensions, not just `image/*`

**Decision.** The dropzone accepts `image/*` **and** every extension
`collect_images()` takes, from one `IMAGE_EXTENSIONS` set.

**Why.** `image/*` filters on the MIME type the *browser* reports. A file
dragged from a network share or a mapped drive arrives with no type, and
react-dropzone discards it silently — no message, no count, nothing in the
server log. Reproduced in Chromium: three files dropped, one accepted.

**Cost.** None found. Non-images are still refused, now server-side as well —
a filter that runs only in the browser is one that can be bypassed.

---

## 2026-09-16 · A second transport rather than a second client

**Decision.** `vlm_transport` selects between the GPU server's paths and
`llm_proxy_v3`'s. One `ENDPOINTS` table holds every difference.

**Why.** The field-ops VM has no route to the GPU box at all; it reaches
FALCONPRD, which forwards. That is a different HTTP contract, not just a
different address. A second client class would have duplicated the payload
building, the registry bootstrap and the parsers — the parts most expensive to
get wrong twice.

**Cost.** Every call site passes a transport. Worth it: nothing outside
`stage3_vlm.py` knows which route is in use.

**Deliberate.** `build_payload()` strips `repetition_penalty`, `request_id` and
`client_id` for the proxy. Pydantic *ignores* undeclared fields, so they would
be dropped in silence; dropping them here means a recorded run cannot claim a
setting that was never honoured. At the demo default of 1.0 — the server's own
floor — the behaviour is identical either way, and `validate()` warns if it is
raised.

---

## 2026-09-16 · Mode 3 is three calls, not one

**Decision.** Mode 3 asks quality, presence and the answer as three separate
calls, each with its own editable system prompt.

**Why.** One call keeps the three judgements mutually consistent but makes a
wording change to one criterion move the other two with it. Three calls let
each be tuned — and blamed — independently.

**Cost.** Three calls per photograph instead of one; roughly 6.5s per photo for
all three modes against 5.5s for one. The image is encoded once and reused, so
the cost is tokens, not another decode.

**Deliberate.** Each leg's *user* prompt is shown but fixed: it carries the JSON
contract the parser keys on, and an edit there would silently turn every answer
into `unknown`.

---

## 2026-09-16 · Mode 3 renders the plain photograph, with no box

**Decision.** Mode 3 writes an unannotated EXIF-corrected copy and renders it;
no green u2netp foreground box.

**Why.** u2netp does not run in that mode. Drawing its output would credit a
component for work it did not do. For the same reason the quality panel is
re-implemented rather than reused from the frozen one, which would print
`whole frame 0.0` — a number the model never produced.

---

## 2026-09-16 · Tuned prompts are per-machine, and absence is the default

**Decision.** `config/prompts.yaml` holds only the legs that differ from the
built-ins. Text equal to a default clears the override; an empty store deletes
the file. It is gitignored.

**Why.** The repo default should be the *absence* of a file, not a stale copy
of the defaults that drifts from the code. Committing one machine's wording
would silently change what every other machine sends.

**Cost.** Tuning does not travel between hosts. Deliberate.

---

## 2026-09-15 · A fourth UI rather than a fourth mode of one UI

**Decision.** `demo_dash_modes.py` is a separate file that imports renderers
from the frozen `demo_dash.py`.

**Why.** `demo_dash.py` is the fallback that is known to work end to end on
FALCONPRD against the live GPU server. Keeping it runnable is worth the
duplication of two renderers.

**Cost.** `quality_column` and `detection_column` exist twice, with the copies
handling mode 3's cases. `FROZEN.md` records the SHA to restore from.

---

## 2026-09-10 · Class names live in config, read from the data

**Decision.** `yolox_class_names` is pinned in `config.py`, index 0 =
`GPS Antenna`, taken from the training run's own `classes.json`.

**Why.** YOLOX stores no class names in a checkpoint. A wrong order does not
error — it puts a confident wrong label on screen *and* into the VLM prompt as
evidence the model is told to weigh.

**How it went wrong.** The detector was reported as swapping the two classes.
The config was correct; `smoke_yolox.py` carried
`--classes default="hazard_sign,gps_antenna"`, overriding it. Two guard tests
now assert no `--classes` string default exists anywhere and that no pipeline
module besides `config.py`/`questions.py` defines the names.

**Lesson recorded.** Trace to the source before flipping a constant. Flipping
it would have "fixed" the smoke tool and broken the UI.

---

## 2026-09-10 · `yolox_input_size` is `(640, 480)`

**Decision.** Pinned to the training run's exp table, not the stock 640×640.

**Why.** `test_size` is not stored in a checkpoint; it has to come from the
training config.

**Correction.** An earlier comment here claimed a wrong size "shifts every
box". The test proved otherwise: for a portrait photograph the height limits
the scale either way, so the boxes land identically. Only landscape diverges,
by 4/3. The comment and the test were corrected.

---

## 2026-09-09 · Vendor the HTTP contract, do not import it

**Decision.** `stage3_vlm.py` is self-contained rather than importing from
`vlm_test_client_v2.py`.

**Why.** That client does not run on the demo host, so there is nothing to
import. The contract is lifted, and the three traps in it are guarded
explicitly: the registry bootstrap, arrays-never-paths, and the silent blank
system prompt.

---

## 2026-09-09 · One question registry, per-question class names

**Decision.** Each `Question` carries its own `default_class_names`.

**Why.** Each CVAT export is a separate single-class task, so every one numbers
its only class `0`. The same id means a hazard sign in one export and a GPS
antenna in another. A single shared `classes.txt` cannot express that, and
guessing injects a false label into the VLM prompt.

---

## Standing rules

**Never let a mock assert.** Mock and parse-failure paths return `unknown`, and
`poor` for quality, always labelled as mock. A partial answer that looks
complete is worse than an obvious mock.

**A silent failure gets a named error.** Where a wrong configuration would
produce plausible-looking output instead of an error, the code refuses and the
message names the cause — the empty registry, the blank system prompt, the
missing `yolox/models`, the proxy 404 that means the wrong transport.

**Tests run without heavy dependencies.** `cv2`, `dash` and `requests` are
stubbed; `torch`, `rembg` and `pandas` are never imported. Verified by running
every suite with all six blocked. `pyyaml` is the single exception — two suites
assert saving and parsing that genuinely need it.

**Tests must not depend on their host.** Clear the environment you read;
force the absence you assert.
