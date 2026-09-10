# Frozen versions

## `dash-ui-v1` — annotation-file detector

**Commit:** `f978b7df790f312a096977a4bd103b9c66f13948`
**Files:** `app/demo_dash.py`, `app/assets/demo.css`
**Detector:** `use_model=False` — human annotation `.txt` sidecars
**Tests:** 229 assertions across six suites

The demo-ready state, verified end to end on FALCONPRD against the live GPU
server. Do not change `app/demo_dash.py` — the YOLOX version is a separate
file so this one stays runnable as a fallback.

Restore with:

    git checkout f978b7df790f312a096977a4bd103b9c66f13948 -- app/demo_dash.py app/assets/demo.css

A local git tag `dash-ui-v1` also marks this commit. It is not on the remote:
this environment's proxy rejects tag pushes with HTTP 403, and pushing a
separate branch was not authorised. The SHA above is the durable reference.
