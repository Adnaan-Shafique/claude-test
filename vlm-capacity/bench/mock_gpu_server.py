#!/usr/bin/env python3
"""
mock_gpu_server.py — a stand-in that speaks llm_proxy_v3's /v1/infer contract.

PURPOSE: dry-run the harness end to end — loadgen → results → analyze → report —
without booking H200 time. Run it, point loadgen at it, confirm you get a
report, and only then do the real run.

IT IS NOT A PERFORMANCE MODEL. The latency it returns comes from a crude
arithmetic sketch, not from Qwen3-VL. Any number produced against this server
is a plumbing check, never a capacity finding. The report generator stamps
runs made against it as SYNTHETIC.

What it does reproduce faithfully is the *shape* of the real system's
backpressure, which is what the harness needs to be exercised against:
  - a semaphore of `--max-concurrent` slots (the GPU server's per-model cap)
  - HTTP 503 once a request has waited `--queue-timeout` for a slot
  - decode throughput per sequence degrading as the batch grows

Stdlib only — no fastapi, no httpx.

Usage:
    python3 mock_gpu_server.py --port 8071 --max-concurrent 2
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {
    "max_concurrent":   2,
    "queue_timeout":    60.0,
    "vision_prefill_s": 0.35,    # per image, once the sequence is scheduled
    "base_decode_tps":  62.0,    # tokens/sec for a single resident sequence
    "batch_penalty":    0.16,    # per-seq slowdown for each extra sequence
    "jitter":           0.06,
}

_sem: threading.Semaphore
_active = 0
_active_lock = threading.Lock()
_rng = random.Random(7)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # noqa: ARG002  — quiet
        pass

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/health":
            self._json(200, {"status": "ok", "mock": True,
                             "max_concurrent": CFG["max_concurrent"]})
        elif self.path in ("/v1/metrics", "/metrics"):
            self._json(200, {"models": {"qwen3-vl": {"requests_active": _active}},
                             "mock": True})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        global _active
        if self.path != "/v1/infer":
            self._json(404, {"detail": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json(400, {"detail": "bad json"})
            return

        t_queue = time.perf_counter()
        if not _sem.acquire(timeout=CFG["queue_timeout"]):
            self._json(503, {"detail": f"Model '{req.get('model')}' is busy — all "
                                       f"{CFG['max_concurrent']} slots occupied."})
            return

        try:
            with _active_lock:
                _active += 1
                batch = _active

            queue_wait = time.perf_counter() - t_queue
            n_images   = len(req.get("images") or [])
            max_tok    = int(req.get("max_new_tokens") or 256)

            # Extraction answers rarely run to the token cap; sample a plausible
            # completion length so the report's tokens/image is not just max_tok.
            new_tokens = max(48, min(max_tok, int(_rng.gauss(max_tok * 0.62, max_tok * 0.13))))

            per_seq_tps = CFG["base_decode_tps"] / (1.0 + CFG["batch_penalty"] * (batch - 1))
            gpu_s = (CFG["vision_prefill_s"] * max(1, n_images)
                     + new_tokens / per_seq_tps)
            gpu_s *= 1.0 + _rng.uniform(-CFG["jitter"], CFG["jitter"])
            time.sleep(gpu_s)

            self._json(200, {
                "text":            '{"vendor_name": "Mock Vendor", "total_due": 1234.56}',
                "request_id":      str(uuid.uuid4()),
                "client_id":       "bench",
                "model":           req.get("model"),
                "images":          n_images,
                # ~1350 vision tokens for a page at Qwen3-VL's default max_pixels,
                # plus the text prompt. Sketch, not measurement.
                "prompt_tokens":   1350 * max(1, n_images) + 180,
                "new_tokens":      new_tokens,
                "elapsed_s":       round(gpu_s, 3),
                "tokens_per_sec":  round(new_tokens / gpu_s, 1),
                "proxy_elapsed_s": round(gpu_s + queue_wait, 3),
                "queue_wait_s":    round(queue_wait, 3),
                "upstream_status": 200,
                "mock":            True,
            })
        finally:
            with _active_lock:
                _active -= 1
            _sem.release()


def main() -> None:
    global _sem
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-concurrent", type=int, default=2,
                    help="mirrors MODEL_CONFIGS['qwen3-vl']['max_concurrent']")
    ap.add_argument("--queue-timeout", type=float, default=60.0,
                    help="mirrors QUEUE_TIMEOUT_S in the GPU server")
    ap.add_argument("--decode-tps", type=float, default=62.0)
    args = ap.parse_args()

    CFG["max_concurrent"] = args.max_concurrent
    CFG["queue_timeout"]  = args.queue_timeout
    CFG["base_decode_tps"] = args.decode_tps
    _sem = threading.Semaphore(args.max_concurrent)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"MOCK server on http://{args.host}:{args.port}  "
          f"max_concurrent={args.max_concurrent} queue_timeout={args.queue_timeout}s")
    print("*** SYNTHETIC LATENCY — plumbing checks only, never capacity findings ***")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
