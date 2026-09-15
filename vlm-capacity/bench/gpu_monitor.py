#!/usr/bin/env python3
"""
gpu_monitor.py — 1 Hz observability sampler. RUNS ON THE GPU BOX (H200 host).

Writes three CSVs that analyze.py joins by wall-clock timestamp:

  gpu_samples.csv     per-GPU: utilization, VRAM, power, temperature, clocks
  host_samples.csv    host:    RAM, swap, load average, model-store disk usage
  server_metrics.csv  vLLM:    per-model requests/tokens/images from /metrics

Why all three: "GPU utilization" from nvidia-smi is a *time-occupancy* metric —
it reports the fraction of the sample window during which at least one kernel
was resident, not how much of the SM array that kernel used. A VLM decoding one
sequence at batch size 1 can read 100% while leaving most of the H200 idle. The
only way to tell saturation from occupancy is to read it next to achieved
throughput and VRAM, which is why the load generator's request log and this
sampler are analysed together rather than separately.

Start it BEFORE the load generator and stop it AFTER, so the run captures idle
baselines on both sides of the load. run_all.sh does this for you.

Usage:
    python3 gpu_monitor.py --out ./results --interval 1.0
    python3 gpu_monitor.py --out ./results --duration 1800     # auto-stop
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GPU_FIELDS = [
    "index", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "memory.reserved",
    "temperature.gpu", "power.draw", "power.limit",
    "clocks.current.sm", "clocks.current.memory",
]

_stop = False


def _handle_signal(signum, frame):        # noqa: ARG001
    global _stop
    _stop = True


def sample_gpus() -> list[dict]:
    """One nvidia-smi poll across every visible GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        print(f"[warn] nvidia-smi failed: {exc}", file=sys.stderr)
        return []

    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            continue
        row: dict = {}
        for key, val in zip(GPU_FIELDS, parts):
            # "[N/A]" shows up for memory.reserved and some clocks on certain
            # driver versions; keep the row rather than dropping the sample.
            if val in ("[N/A]", "N/A", ""):
                row[key] = ""
                continue
            try:
                row[key] = float(val) if "." in val or key.startswith(("power", "utilization")) else int(val)
            except ValueError:
                row[key] = val
        rows.append(row)
    return rows


def sample_host(model_path: str) -> dict:
    """Host RAM, swap, load average and free space on the model store."""
    mem: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                mem[key] = int(rest.strip().split()[0])   # kB
    except OSError:
        pass

    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)

    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0

    disk_total = disk_used = disk_free = 0
    probe = model_path if os.path.exists(model_path) else "/"
    try:
        usage = shutil.disk_usage(probe)
        disk_total, disk_used, disk_free = usage.total, usage.used, usage.free
    except OSError:
        pass

    return {
        "ram_total_gb":  round(total_kb / 1024 / 1024, 3),
        "ram_used_gb":   round((total_kb - avail_kb) / 1024 / 1024, 3),
        "ram_avail_gb":  round(avail_kb / 1024 / 1024, 3),
        "ram_used_pct":  round((total_kb - avail_kb) / total_kb * 100, 2) if total_kb else 0.0,
        "swap_used_gb":  round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024 / 1024, 3),
        "load1":         round(load1, 2),
        "load5":         round(load5, 2),
        "load15":        round(load15, 2),
        "disk_path":     probe,
        "disk_total_gb": round(disk_total / 1e9, 2),
        "disk_used_gb":  round(disk_used / 1e9, 2),
        "disk_free_gb":  round(disk_free / 1e9, 2),
        "disk_used_pct": round(disk_used / disk_total * 100, 2) if disk_total else 0.0,
    }


def sample_server(url: str) -> dict | None:
    """GET the vLLM server's own /metrics. Returns None if it is unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",        default="./results")
    ap.add_argument("--interval",   type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--duration",   type=float, default=0.0,
                    help="auto-stop after N seconds (0 = run until SIGINT/SIGTERM)")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:5432/metrics",
                    help="vLLM server /metrics (localhost — this runs on the GPU box)")
    ap.add_argument("--model-path", default="/data01/llm_models",
                    help="model store, for disk-usage sampling")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gpu_path    = out / "gpu_samples.csv"
    host_path   = out / "host_samples.csv"
    server_path = out / "server_metrics.csv"

    gpu_cols = ["ts", "gpu", "util_gpu_pct", "util_mem_pct", "mem_used_mb",
                "mem_total_mb", "mem_used_pct", "temp_c", "power_w",
                "power_limit_w", "sm_clock_mhz", "mem_clock_mhz"]
    host_cols = ["ts", "ram_total_gb", "ram_used_gb", "ram_avail_gb", "ram_used_pct",
                 "swap_used_gb", "load1", "load5", "load15", "disk_path",
                 "disk_total_gb", "disk_used_gb", "disk_free_gb", "disk_used_pct"]
    srv_cols = ["ts", "model", "requests_total", "requests_active", "requests_queued",
                "errors", "tokens_generated", "images_processed",
                "avg_tokens_per_req", "avg_latency_s", "overall_tokens_per_s", "loaded"]

    f_gpu  = gpu_path.open("w",  newline="")
    f_host = host_path.open("w", newline="")
    f_srv  = server_path.open("w", newline="")
    w_gpu, w_host, w_srv = csv.writer(f_gpu), csv.writer(f_host), csv.writer(f_srv)
    w_gpu.writerow(gpu_cols)
    w_host.writerow(host_cols)
    w_srv.writerow(srv_cols)

    started = time.time()
    n = 0
    print(f"Sampling every {args.interval}s → {out}  (Ctrl-C to stop)")

    try:
        while not _stop:
            ts = time.time()
            if args.duration and (ts - started) >= args.duration:
                break

            for g in sample_gpus():
                used  = g.get("memory.used") or 0
                total = g.get("memory.total") or 0
                w_gpu.writerow([
                    f"{ts:.3f}", g.get("index"),
                    g.get("utilization.gpu"), g.get("utilization.memory"),
                    used, total,
                    round(used / total * 100, 2) if total else "",
                    g.get("temperature.gpu"), g.get("power.draw"), g.get("power.limit"),
                    g.get("clocks.current.sm"), g.get("clocks.current.memory"),
                ])

            h = sample_host(args.model_path)
            w_host.writerow([f"{ts:.3f}"] + [h[c] for c in host_cols[1:]])

            srv = sample_server(args.metrics_url)
            if srv:
                for name, m in (srv.get("models") or {}).items():
                    w_srv.writerow([
                        f"{ts:.3f}", name,
                        m.get("requests_total"), m.get("requests_active"),
                        m.get("requests_queued"), m.get("errors"),
                        m.get("tokens_generated"), m.get("images_processed"),
                        m.get("avg_tokens_per_req"), m.get("avg_latency_s"),
                        m.get("overall_tokens_per_s"), m.get("loaded"),
                    ])

            n += 1
            if n % 30 == 0:
                f_gpu.flush(); f_host.flush(); f_srv.flush()
                print(f"  {n} samples ({ts - started:.0f}s elapsed)")

            # Drift-free pacing: sleep to the next interval boundary rather than
            # for a fixed interval, so a slow nvidia-smi poll does not slowly
            # skew every later timestamp.
            time.sleep(max(0.0, args.interval - (time.time() - ts)))

    finally:
        f_gpu.close(); f_host.close(); f_srv.close()
        print(f"\nStopped after {n} samples / {time.time() - started:.0f}s")
        print(f"  {gpu_path}\n  {host_path}\n  {server_path}")


if __name__ == "__main__":
    main()
