"""Regression test for the rembg new_session() signature change.

rembg 2.0.69 builds its own SessionOptions inside new_session() and passes it
positionally to the session class, so forwarding sess_opts= as a keyword raises
"BaseSession.__init__() got multiple values for argument 'sess_opts'". That made
the segmenter unloadable, even for sess_opts=None.

Runs without rembg, cv2 or numpy installed: all three are stubbed, and both
rembg signatures are simulated.
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

# ── Stub the module-scope imports so this runs anywhere ──────────────────────
for name in ("cv2", "numpy"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


class FakeOpts:
    def __init__(self, intra=0, inter=0):
        self.intra_op_num_threads = intra
        self.inter_op_num_threads = inter


def install_rembg(style: str):
    """Install a fake rembg whose new_session mimics one of the two signatures."""
    mod = types.ModuleType("rembg")
    calls = []

    if style == "old":
        # Pre-2.0.6x: accepts sess_opts as a keyword.
        def new_session(model_name, sess_opts=None, *a, **k):
            calls.append({"model": model_name, "sess_opts": sess_opts})
            return f"session<{model_name}>"
    else:
        # 2.0.69: no sess_opts parameter; passing it duplicates a positional.
        def new_session(model_name="u2net", providers=None, *a, **k):
            if "sess_opts" in k:
                raise TypeError(
                    "BaseSession.__init__() got multiple values for argument 'sess_opts'")
            calls.append({"model": model_name,
                          "omp": os.environ.get("OMP_NUM_THREADS")})
            return f"session<{model_name}>"

    mod.new_session = new_session
    sys.modules["rembg"] = mod
    return calls


def fresh_module():
    """Re-import foreground_segmentation with its session singleton cleared."""
    sys.modules.pop("foreground_segmentation", None)
    import foreground_segmentation as fs
    fs._session = None
    fs._session_model_name = None
    return fs


print("\nrembg WITHOUT sess_opts support (2.0.69) - the version on the demo box")
calls = install_rembg("new")
os.environ.pop("OMP_NUM_THREADS", None)
fs = fresh_module()

# The demo path: preload() with no thread capping. This is what raised
# TypeError before the fix, despite asking for nothing.
try:
    fs.preload(model_name="u2netp")
    check("preload() with no sess_opts succeeds", True)
except TypeError as exc:
    check("preload() with no sess_opts succeeds", False, str(exc))
check("new_session called with just the model name", calls and calls[-1]["model"] == "u2netp",
      str(calls))

# The CLI path: a real SessionOptions asking for one thread per worker.
fs._session = None
fs._session_model_name = None
try:
    fs.preload(model_name="u2netp", sess_opts=FakeOpts(intra=1))
    check("preload() with sess_opts falls back instead of raising", True)
except TypeError as exc:
    check("preload() with sess_opts falls back instead of raising", False, str(exc))
check("thread cap translated to OMP_NUM_THREADS on the fallback",
      os.environ.get("OMP_NUM_THREADS") == "1",
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r}")

print("\nrembg WITH sess_opts support (older) - must keep working")
os.environ.pop("OMP_NUM_THREADS", None)
calls = install_rembg("old")
fs = fresh_module()
opts = FakeOpts(intra=1)
fs.preload(model_name="u2netp", sess_opts=opts)
check("sess_opts is forwarded when the signature accepts it",
      calls and calls[-1]["sess_opts"] is opts, str(calls))
check("OMP_NUM_THREADS is NOT set when forwarding worked",
      os.environ.get("OMP_NUM_THREADS") is None,
      f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r}")

print("\nsession caching")
fs._session = None
fs._session_model_name = None
calls = install_rembg("new")
fs = fresh_module()
fs.preload(model_name="u2netp")
n_after_first = len(calls)
fs.preload(model_name="u2netp")
check("second preload of the same model reuses the cached session",
      len(calls) == n_after_first, f"{len(calls)} calls, expected {n_after_first}")
fs.preload(model_name="u2net")
check("a different model name builds a new session", len(calls) == n_after_first + 1)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
