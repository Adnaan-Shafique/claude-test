#!/usr/bin/env python3
"""
loadgen.py — open-loop arrival-rate load generator. RUNS ON THE RHEL VM.

Drives the VLM through llm_proxy_v3 at a *fixed arrival rate*, not a fixed
concurrency. That distinction is the whole point of the exercise:

  A closed-loop test ("keep 16 requests in flight") can never fail. Offered
  load silently throttles itself to whatever the server can do, so you measure
  throughput but learn nothing about whether 30 images/sec is achievable — the
  arrival rate was never 30/sec, it was "as fast as the box allowed".

  An open-loop test fires requests on a schedule regardless of whether earlier
  ones have come back. If the server cannot keep up, in-flight count climbs,
  latency climbs, and the run produces 503s. That failure IS the capacity
  answer, and it is what this script records.

Phases run back to back in one process so the sampler on the GPU box can be
joined to them by timestamp:

  idle_before → warmup → scenario(s), each followed by a cooldown → idle_after

Outputs (into --out):
  requests.csv      one row per request, including failures
  phases.csv        phase name, kind, start_ts, end_ts, target rate
  run_meta.json     configuration, corpus stats, clock reference

Usage:
    python3 loadgen.py --corpus ./corpus --out ./results \\
        --proxy http://127.0.0.1:8071 --api-key secret-bench
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Each run must be self-describing. An A/B between serving configurations is
# worthless if you cannot later prove which directory was which, and a
# filename is not proof — the server's own /models output is.

# ── The production unit of work ──────────────────────────────────────────────
# One page in, structured JSON out, with OCR'd values. Output length is the
# dominant cost term in a VLM serving workload, so this prompt is written to
# produce a realistically bounded answer rather than an open-ended description.

DEFAULT_SYSTEM = (
    "You are a document extraction engine. You reply with a single valid JSON "
    "object and nothing else — no prose, no markdown fences, no explanation."
)

DEFAULT_PROMPT = (
    "Extract the following fields from this document and return them as one "
    "JSON object:\n"
    '  vendor_name, vendor_city, gstin, invoice_number, po_reference, '
    'invoice_date, currency, subtotal, tax_amount, total_due, '
    'payment_terms, bank_account, ifsc,\n'
    '  line_items: an array of {description, quantity, unit_price, amount}\n'
    "Transcribe values exactly as printed. Use null for any field that is not "
    "present. Return only the JSON object."
)


@dataclass
class Phase:
    name:     str
    kind:     str            # "idle" | "warmup" | "load" | "cooldown"
    rate:     float          # target images/sec (0 for idle/cooldown)
    duration: float          # seconds


@dataclass
class Result:
    scenario:       str
    seq:            int
    scheduled_ts:   float
    send_ts:        float
    recv_ts:        float
    latency_s:      float
    status:         str
    http_code:      int   = 0
    prompt_tokens:  int   = 0
    new_tokens:     int   = 0
    gpu_elapsed_s:  float = 0.0
    queue_wait_s:   float = 0.0
    tokens_per_sec: float = 0.0
    image:          str   = ""
    error:          str   = ""


class Corpus:
    """Images pre-encoded once, then cycled. Encoding inside the request loop
    would charge base64 CPU time to the measured latency."""

    def __init__(self, path: Path, mode: str, limit: int | None = None):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
        if limit:
            files = files[:limit]
        if not files:
            raise SystemExit(f"No images found under {path}")

        self.files = files
        self.mode = mode
        self.total_bytes = sum(f.stat().st_size for f in files)
        self.payloads: list[str] = []

        if mode == "base64":
            for f in files:
                raw = f.read_bytes()
                suffix = f.suffix.lower().lstrip(".")
                mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
                self.payloads.append(
                    f"data:image/{mime};base64," + base64.b64encode(raw).decode()
                )
        else:                       # "path" — corpus must live on the GPU box
            self.payloads = [str(f) for f in files]

        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> tuple[str, str]:
        """Round-robin, so every request in a phase uses a different page.
        Reusing one image would let vLLM's prefix cache serve vision tokens
        from cache and overstate throughput."""
        with self._lock:
            i = self._i
            self._i += 1
        idx = i % len(self.files)
        return self.payloads[idx], self.files[idx].name

    def __len__(self) -> int:
        return len(self.files)


class LoadGen:
    def __init__(self, args):
        self.args = args
        self.url = args.proxy.rstrip("/") + "/v1/infer"
        self.corpus = Corpus(Path(args.corpus), args.image_mode, args.corpus_limit)
        self.results: list[Result] = []
        self.res_lock = threading.Lock()
        self.inflight = 0
        self.inflight_lock = threading.Lock()
        self.peak_inflight = 0
        self.session_local = threading.local()
        # Set if any response carries mock_gpu_server's marker. Propagated into
        # run_meta.json so analyze.py stamps the report SYNTHETIC and a dry-run
        # can never be mistaken for a capacity measurement.
        self.mock_seen = False
        # A VLM answering blind still returns HTTP 200. Throughput alone cannot
        # distinguish 'processed 300 images' from 'returned 300 responses to a
        # prompt it could not see', so a sample of real answers is kept for
        # inspection and the report's credibility rests on it.
        self.samples: list[dict] = []
        self.samples_lock = threading.Lock()

    def _session(self) -> requests.Session:
        """One HTTP session per worker thread — sharing one Session across
        hundreds of threads serialises on its connection pool."""
        s = getattr(self.session_local, "s", None)
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            self.session_local.s = s
        return s

    def capture_server_config(self) -> dict:
        """Snapshot the GPU server's live registry and VRAM state before load.

        This is what makes an A/B honest: tensor_parallel_size,
        gpu_memory_utilization, max_concurrent and max_model_len are recorded
        from the running server, not from whatever the operator believed was
        deployed.
        """
        base = self.args.proxy.rstrip("/")
        out: dict = {}
        # /v1/gpu-models is the GPU server's real registry. /v1/models is only
        # the proxy's allowlist of names and cannot describe a configuration;
        # it is kept as a fallback for a proxy too old to have the passthrough.
        for label, path in (("gpu_models", "/v1/gpu-models"),
                            ("models", "/v1/models"),
                            ("gpu_health", "/v1/gpu-health")):
            try:
                resp = requests.get(f"{base}{path}",
                                    headers={"X-API-Key": self.args.api_key}, timeout=30)
                out[label] = resp.json() if resp.status_code == 200 else {
                    "error": f"HTTP {resp.status_code}"}
            except Exception as exc:
                out[label] = {"error": f"{type(exc).__name__}: {exc}"}

        def _find(payload) -> Optional[dict]:
            """The registry may arrive as a bare list or wrapped in {"models": ...}."""
            items = payload
            if isinstance(payload, dict):
                items = payload.get("models")
            if not isinstance(items, list):
                return None
            for m in items:
                if isinstance(m, dict) and m.get("name") == self.args.model:
                    return m
            return None

        entry = _find(out.get("gpu_models")) or _find(out.get("models"))
        out["model_entry"] = entry

        if entry:
            print(f"Server config for '{self.args.model}': "
                  f"TP={entry.get('tensor_parallel_size')} "
                  f"gpu_mem_util={entry.get('gpu_memory_utilization')} "
                  f"max_concurrent={entry.get('max_concurrent')} "
                  f"max_model_len={entry.get('max_model_len')}")
        else:
            print(f"!! Could not read server config for '{self.args.model}' — the A/B "
                  f"comparison will not be able to label this run.\n"
                  f"   Check that the proxy exposes /v1/gpu-models (llm_proxy_v3) and "
                  f"that it can reach the GPU server's /models.")
        return out

    def _record(self, r: Result) -> None:
        with self.res_lock:
            self.results.append(r)

    def _one_request(self, scenario: str, seq: int, scheduled_ts: float) -> None:
        image, name = self.corpus.next()
        payload = {
            "model":          self.args.model,
            "prompt":         self.args.prompt,
            "system":         self.args.system,
            "images":         [image],
            "max_new_tokens": self.args.max_new_tokens,
            "temperature":    self.args.temperature,
            "top_p":          0.9,
            "top_k":          50,
        }
        headers = {"X-API-Key": self.args.api_key, "Content-Type": "application/json"}

        with self.inflight_lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)

        send_ts = time.time()
        try:
            resp = self._session().post(
                self.url, json=payload, headers=headers, timeout=self.args.timeout
            )
            recv_ts = time.time()
            if resp.status_code == 200:
                body = resp.json()
                if body.get("mock"):
                    self.mock_seen = True
                if self.args.save_samples:
                    with self.samples_lock:
                        if len(self.samples) < self.args.save_samples:
                            self.samples.append({
                                "image":         name,
                                "prompt_tokens": body.get("prompt_tokens"),
                                "new_tokens":    body.get("new_tokens"),
                                "images_echoed": body.get("images"),
                                "text":          (body.get("text") or "")[:4000],
                            })
                self._record(Result(
                    scenario, seq, scheduled_ts, send_ts, recv_ts,
                    round(recv_ts - send_ts, 4), "ok", 200,
                    int(body.get("prompt_tokens") or 0),
                    int(body.get("new_tokens") or 0),
                    float(body.get("elapsed_s") or 0.0),
                    float(body.get("queue_wait_s") or 0.0),
                    float(body.get("tokens_per_sec") or 0.0),
                    name,
                ))
            else:
                # 503 = the GPU server's per-model semaphore was full for longer
                # than QUEUE_TIMEOUT_S. This is the capacity ceiling announcing
                # itself, and is counted separately from a genuine error.
                status = "saturated" if resp.status_code == 503 else "http_error"
                self._record(Result(
                    scenario, seq, scheduled_ts, send_ts, recv_ts,
                    round(recv_ts - send_ts, 4), status, resp.status_code,
                    image=name, error=resp.text[:200],
                ))
        except requests.exceptions.Timeout:
            recv_ts = time.time()
            self._record(Result(scenario, seq, scheduled_ts, send_ts, recv_ts,
                                round(recv_ts - send_ts, 4), "timeout",
                                image=name, error=f"client timeout {self.args.timeout}s"))
        except Exception as exc:
            recv_ts = time.time()
            self._record(Result(scenario, seq, scheduled_ts, send_ts, recv_ts,
                                round(recv_ts - send_ts, 4), "error",
                                image=name, error=f"{type(exc).__name__}: {exc}"))
        finally:
            with self.inflight_lock:
                self.inflight -= 1

    # ── Phase drivers ────────────────────────────────────────────────────────

    def run_idle(self, phase: Phase) -> None:
        print(f"  [{phase.name}] idle for {phase.duration:.0f}s "
              f"(baseline for the before/after charts)")
        time.sleep(phase.duration)

    def run_warmup(self, phase: Phase, pool: ThreadPoolExecutor) -> None:
        """Sequential requests that force the lazy VLM load and CUDA graph
        capture. Without this the first scenario absorbs a multi-minute model
        load and its latency percentiles are meaningless."""
        print(f"  [{phase.name}] {self.args.warmup_requests} sequential warmup "
              f"requests (forces lazy model load)")
        t0 = time.time()
        for i in range(self.args.warmup_requests):
            self._one_request(phase.name, i, time.time())
        ok = sum(1 for r in self.results if r.scenario == phase.name and r.status == "ok")
        print(f"  [{phase.name}] {ok}/{self.args.warmup_requests} ok in {time.time() - t0:.1f}s")
        if ok == 0:
            raise SystemExit(
                "Warmup produced zero successful responses — fix connectivity, the "
                "API key, or the model name before running the capacity test.\n"
                f"  last error: {self.results[-1].error if self.results else 'n/a'}"
            )

    def run_load(self, phase: Phase, pool: ThreadPoolExecutor) -> float:
        """Fire requests on a schedule set by the arrival process, never
        waiting for earlier ones to return.

        Returns the instant firing stopped. That, not the post-drain time, is
        the denominator for the achieved rate: counting drain time as offered
        time would understate throughput by the tail latency of the phase."""
        rng = random.Random(self.args.seed)
        start = time.time()
        deadline = start + phase.duration
        seq = 0
        next_ts = start
        shed = 0

        print(f"  [{phase.name}] target {phase.rate:g} img/s for {phase.duration:.0f}s "
              f"(~{int(phase.rate * phase.duration)} requests, {self.args.arrival} arrivals)")

        while True:
            now = time.time()
            if next_ts >= deadline:
                break
            if next_ts > now:
                time.sleep(min(next_ts - now, 0.25))
                continue

            with self.inflight_lock:
                current = self.inflight
            if current >= self.args.max_inflight:
                # The client refused to add more load. Recorded explicitly so it
                # is never mistaken for server saturation in the analysis.
                self._record(Result(phase.name, seq, next_ts, next_ts, next_ts, 0.0,
                                    "shed_client", error="max_inflight reached"))
                shed += 1
            else:
                pool.submit(self._one_request, phase.name, seq, next_ts)

            seq += 1
            gap = (rng.expovariate(phase.rate) if self.args.arrival == "poisson"
                   else 1.0 / phase.rate)
            next_ts += gap

        # The offered window is the phase's nominal duration. Under Poisson
        # gaps the scheduler can step past the deadline while real time is
        # still short of it; returning early would shrink the denominator and
        # overstate the achieved rate. Wait out the remainder.
        time.sleep(max(0.0, deadline - time.time()))
        fire_end = time.time()

        # Let in-flight requests land before the phase is marked finished.
        drain_until = time.time() + self.args.drain
        while time.time() < drain_until:
            with self.inflight_lock:
                if self.inflight == 0:
                    break
            time.sleep(0.5)

        rows = [r for r in self.results if r.scenario == phase.name]
        ok = sum(1 for r in rows if r.status == "ok")
        sat = sum(1 for r in rows if r.status == "saturated")
        print(f"  [{phase.name}] fired={seq} ok={ok} saturated={sat} "
              f"shed={shed} peak_inflight={self.peak_inflight}")
        return fire_end

    # ── Orchestration ────────────────────────────────────────────────────────

    def run(self, phases: list[Phase], out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        marks: list[dict] = []
        run_start = time.time()

        print(f"\nCorpus: {len(self.corpus)} images, "
              f"{self.corpus.total_bytes / 1e6:.1f} MB, mode={self.args.image_mode}")
        print(f"Target: {self.url}  model={self.args.model} "
              f"max_new_tokens={self.args.max_new_tokens}")
        server_config = self.capture_server_config()
        print()

        with ThreadPoolExecutor(max_workers=self.args.max_inflight + 8) as pool:
            for phase in phases:
                self.peak_inflight = 0
                t0 = time.time()
                fire_end = None
                if phase.kind in ("idle", "cooldown"):
                    self.run_idle(phase)
                elif phase.kind == "warmup":
                    self.run_warmup(phase, pool)
                else:
                    fire_end = self.run_load(phase, pool)
                marks.append({
                    "phase": phase.name, "kind": phase.kind, "rate": phase.rate,
                    "start_ts": round(t0, 3), "end_ts": round(time.time(), 3),
                    "fire_end_ts": round(fire_end if fire_end else time.time(), 3),
                    "peak_inflight": self.peak_inflight,
                })

        # ── write outputs ────────────────────────────────────────────────────
        req_path = out / "requests.csv"
        with req_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "seq", "scheduled_ts", "send_ts", "recv_ts",
                        "latency_s", "status", "http_code", "prompt_tokens",
                        "new_tokens", "gpu_elapsed_s", "queue_wait_s",
                        "tokens_per_sec", "image", "error"])
            for r in sorted(self.results, key=lambda x: x.send_ts):
                w.writerow([r.scenario, r.seq, f"{r.scheduled_ts:.3f}",
                            f"{r.send_ts:.3f}", f"{r.recv_ts:.3f}", r.latency_s,
                            r.status, r.http_code, r.prompt_tokens, r.new_tokens,
                            r.gpu_elapsed_s, r.queue_wait_s, r.tokens_per_sec,
                            r.image, r.error])

        ph_path = out / "phases.csv"
        with ph_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["phase", "kind", "rate", "start_ts",
                                               "fire_end_ts", "end_ts", "peak_inflight"])
            w.writeheader()
            w.writerows(marks)

        meta = {
            "run_start_ts":   round(run_start, 3),
            "run_end_ts":     round(time.time(), 3),
            "loadgen_host_time_at_finish": time.time(),
            "proxy_url":      self.url,
            "model":          self.args.model,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature":    self.args.temperature,
            "arrival":        self.args.arrival,
            "image_mode":     self.args.image_mode,
            "max_inflight":   self.args.max_inflight,
            "client_timeout_s": self.args.timeout,
            "corpus_images":  len(self.corpus),
            "corpus_bytes":   self.corpus.total_bytes,
            "corpus_avg_kb":  round(self.corpus.total_bytes / len(self.corpus) / 1e3, 1),
            "mock_detected":  self.mock_seen,
            "run_label":      self.args.label,
            "server_config":  server_config,
            "prompt":         self.args.prompt,
            "system":         self.args.system,
            "phases":         marks,
        }
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

        if self.samples:
            # Identical answers across different pages is the signature of a model
            # that never received the pixels — the single most important thing to
            # rule out before any of these numbers are quoted.
            distinct = len({s_["text"] for s_ in self.samples})
            payload = {
                "note": "Read these. Different invoices must produce different "
                        "values. Identical answers mean the model did not see the "
                        "images and every throughput figure in the report is "
                        "measuring the wrong workload.",
                "samples_captured": len(self.samples),
                "distinct_responses": distinct,
                "images_echoed_by_server": sorted(
                    {s_.get("images_echoed") for s_ in self.samples}),
                "samples": self.samples,
            }
            (out / "samples.json").write_text(json.dumps(payload, indent=2))
            print(f"\nSaved {len(self.samples)} response samples "
                  f"({distinct} distinct) → {out / 'samples.json'}")
            if distinct <= 1 and len(self.samples) > 1:
                print("  !! ALL SAMPLED RESPONSES ARE IDENTICAL. Verify the model is "
                      "actually receiving images before trusting this run.")
            echoed = {s_.get("images_echoed") for s_ in self.samples}
            if echoed and echoed <= {0, None}:
                print("  !! The server echoed images=0 for every request — the images "
                      "are not reaching the model.")

        print(f"\nWrote:\n  {req_path}\n  {ph_path}\n  {out / 'run_meta.json'}")
        print(f"Total wall time: {(time.time() - run_start) / 60:.1f} min")


def build_phases(args) -> list[Phase]:
    """idle_before → warmup → cooldown → (scenario → cooldown)* → idle_after"""
    phases = [
        Phase("idle_before", "idle", 0.0, args.idle),
        Phase("warmup", "warmup", 0.0, 0.0),
        Phase("cooldown_warmup", "cooldown", 0.0, args.cooldown),
    ]
    for rate in args.rates:
        # The steady-state scenario runs long enough to consume the whole
        # corpus once; burst scenarios are short because saturation, if it is
        # going to happen, happens within seconds.
        duration = args.steady_duration if rate < 1.0 else args.burst_duration
        phases.append(Phase(f"rate_{rate:g}ps", "load", rate, duration))
        phases.append(Phase(f"cooldown_{rate:g}ps", "cooldown", 0.0, args.cooldown))
    phases.append(Phase("idle_after", "idle", 0.0, args.idle))
    return phases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus",   required=True, help="directory of images")
    ap.add_argument("--out",      default="./results")
    ap.add_argument("--proxy",    default="http://127.0.0.1:8071")
    ap.add_argument("--api-key",  default="secret-bench")
    ap.add_argument("--model",    default="qwen3-vl")
    ap.add_argument("--rates",    type=float, nargs="+",
                    default=[0.67, 10.0, 20.0, 30.0],
                    help="target arrival rates in images/sec")
    ap.add_argument("--steady-duration", type=float, default=450.0,
                    help="seconds for sub-1/s scenarios (450s x 0.67/s = 300 images)")
    ap.add_argument("--burst-duration",  type=float, default=60.0,
                    help="seconds for the high-rate burst scenarios")
    ap.add_argument("--idle",     type=float, default=60.0,
                    help="idle baseline before and after the run")
    ap.add_argument("--cooldown", type=float, default=45.0,
                    help="idle gap between scenarios, so VRAM/power settle")
    ap.add_argument("--drain",    type=float, default=180.0,
                    help="max seconds to wait for in-flight requests after a phase")
    ap.add_argument("--arrival",  choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--max-inflight", type=int, default=400,
                    help="client-side safety cap on concurrent requests")
    ap.add_argument("--timeout",  type=float, default=300.0, help="client HTTP timeout")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature",    type=float, default=0.1,
                    help="low, because extraction should be deterministic")
    ap.add_argument("--warmup-requests", type=int, default=3)
    ap.add_argument("--corpus-limit", type=int, default=None)
    ap.add_argument("--image-mode", choices=["base64", "path"], default="base64",
                    help="'path' only works if the corpus is on the GPU box")
    ap.add_argument("--prompt",  default=DEFAULT_PROMPT)
    ap.add_argument("--system",  default=DEFAULT_SYSTEM)
    ap.add_argument("--save-samples", type=int, default=5,
                    help="keep this many response bodies for verification (0 = none)")
    ap.add_argument("--label",   default="",
                    help="name for this run in an A/B comparison, e.g. 'TP1-0.60-c2'")
    ap.add_argument("--seed",    type=int, default=20250915)
    args = ap.parse_args()

    gen = LoadGen(args)
    gen.run(build_phases(args), Path(args.out))


if __name__ == "__main__":
    main()
