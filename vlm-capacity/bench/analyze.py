#!/usr/bin/env python3
"""
analyze.py — turn a benchmark run into the infra & compute sourcing report.

Reads the CSVs written by loadgen.py (on the VM) and gpu_monitor.py (on the GPU
box), joins them on wall-clock time, and emits:

  report/REPORT.md      the sourcing report: tables, verdicts, sizing
  report/summary.json   every computed number, for spreadsheets
  report/chart_*.png    trend charts, including idle-vs-load before/after

CLOCK ALIGNMENT: the two samplers run on different hosts. If their clocks are
not NTP-synced, pass --clock-offset-s to shift GPU-box timestamps onto the load
generator's clock. Check with `date +%s.%N` on both hosts; the script warns when
the GPU sample window does not cover the load window.

Usage:
    python3 analyze.py --results ./results --out ./report \\
        --daily-images 24000 --window-hours 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

# ── Palette (validated: slots 1–3 pass all-pairs CVD and normal-vision gates) ──
C_S1      = "#2a78d6"   # categorical slot 1 — blue
C_S2      = "#eb6834"   # categorical slot 2 — orange
C_S3      = "#1baf7a"   # categorical slot 3 — aqua (sub-3:1 → always direct-labelled)
SEQ       = ["#86b6ef", "#5598e7", "#256abf", "#104281"]   # ordinal ramp, >= step 250
ST_GOOD   = "#0ca30c"
ST_WARN   = "#fab219"
ST_CRIT   = "#d03b3b"
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

STATUS_COLOR = {"SUSTAINABLE": ST_GOOD, "DEGRADED": ST_WARN, "FAILED": ST_CRIT}
STATUS_ICON  = {"SUSTAINABLE": "✔", "DEGRADED": "▲", "FAILED": "✖"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "figure.dpi": 130,
    "legend.frameon": False,
})


# ═══════════════════════════════ loading ══════════════════════════════════════

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in ("", None, "[N/A]", "N/A"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pct(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def describe(values) -> dict:
    """mean / percentiles / max for a sample, with empty-safe zeros."""
    if not len(values):
        return {k: 0.0 for k in ("n", "mean", "p50", "p90", "p95", "p99", "min", "max", "std")}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size), "mean": float(a.mean()),
        "p50": pct(a, 50), "p90": pct(a, 90), "p95": pct(a, 95), "p99": pct(a, 99),
        "min": float(a.min()), "max": float(a.max()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


# ═══════════════════════════════ analysis ═════════════════════════════════════

def peak_concurrency(rows: list[dict]) -> int:
    """Max simultaneously in-flight, from send/recv interval overlaps."""
    events = []
    for r in rows:
        if r["status"] == "shed_client":
            continue
        events.append((fnum(r, "send_ts"), 1))
        events.append((fnum(r, "recv_ts"), -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def concurrency_series(rows: list[dict], step: float = 1.0):
    """In-flight count sampled on a grid, for the concurrency chart."""
    live = [r for r in rows if r["status"] != "shed_client"]
    if not live:
        return np.array([]), np.array([])
    t0 = min(fnum(r, "send_ts") for r in live)
    t1 = max(fnum(r, "recv_ts") for r in live)
    if t1 <= t0:
        return np.array([]), np.array([])
    grid = np.arange(t0, t1 + step, step)
    starts = np.array([fnum(r, "send_ts") for r in live])
    ends   = np.array([fnum(r, "recv_ts") for r in live])
    counts = np.array([int(((starts <= t) & (ends > t)).sum()) for t in grid])
    return grid, counts


def analyse_scenario(name: str, phase: dict, rows: list[dict], sla_p95: float) -> dict:
    # The offered window is start → fire_end, NOT start → end. end_ts includes
    # the drain, and charging drain seconds against offered time would
    # understate the achieved rate by exactly the phase's tail latency.
    fire_end = fnum(phase, "fire_end_ts") or fnum(phase, "end_ts")
    duration = fire_end - fnum(phase, "start_ts")
    wall     = fnum(phase, "end_ts") - fnum(phase, "start_ts")
    target   = fnum(phase, "rate")

    by_status: dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1

    ok_rows   = [r for r in rows if r["status"] == "ok"]
    fired     = len(rows)
    n_ok      = len(ok_rows)
    n_sat     = by_status.get("saturated", 0)
    n_shed    = by_status.get("shed_client", 0)
    n_timeout = by_status.get("timeout", 0)
    n_err     = by_status.get("error", 0) + by_status.get("http_error", 0)

    # Every request counted here was FIRED during the offered window, so
    # completions are divided by that window even when some of them landed
    # during the drain. That is the throughput the offered load actually got.
    achieved = n_ok / duration if duration > 0 else 0.0

    lat   = describe([fnum(r, "latency_s")     for r in ok_rows])
    gpu_t = describe([fnum(r, "gpu_elapsed_s") for r in ok_rows])
    queue = describe([fnum(r, "queue_wait_s")  for r in ok_rows])
    p_tok = describe([fnum(r, "prompt_tokens") for r in ok_rows])
    n_tok = describe([fnum(r, "new_tokens")    for r in ok_rows])

    total_new    = sum(fnum(r, "new_tokens")    for r in ok_rows)
    total_prompt = sum(fnum(r, "prompt_tokens") for r in ok_rows)

    success_rate = n_ok / fired if fired else 0.0
    rate_ratio   = achieved / target if target else 0.0

    # A scenario is only SUSTAINABLE if it kept up, stayed healthy, AND met the
    # latency objective. Any one of the three failing is a real capacity limit.
    if success_rate >= 0.99 and rate_ratio >= 0.95 and lat["p95"] <= sla_p95:
        verdict, why = "SUSTAINABLE", "kept pace, no shedding, p95 within SLA"
    elif success_rate >= 0.90 and rate_ratio >= 0.80:
        reasons = []
        if lat["p95"] > sla_p95:
            reasons.append(f"p95 {lat['p95']:.1f}s over {sla_p95:.0f}s SLA")
        if success_rate < 0.99:
            reasons.append(f"{(1 - success_rate) * 100:.1f}% requests lost")
        if rate_ratio < 0.95:
            reasons.append(f"delivered {rate_ratio * 100:.0f}% of offered rate")
        verdict, why = "DEGRADED", "; ".join(reasons) or "marginal"
    else:
        reasons = []
        if rate_ratio < 0.80:
            reasons.append(f"delivered only {rate_ratio * 100:.0f}% of offered rate")
        if success_rate < 0.90:
            reasons.append(f"{(1 - success_rate) * 100:.1f}% requests lost")
        verdict, why = "FAILED", "; ".join(reasons) or "did not keep pace"

    return {
        "scenario": name, "target_rate": target, "duration_s": round(duration, 1),
        "wall_s": round(wall, 1),
        "fired": fired, "ok": n_ok, "saturated": n_sat, "shed_client": n_shed,
        "timeout": n_timeout, "errors": n_err,
        "success_rate": round(success_rate, 4),
        "achieved_rate": round(achieved, 3),
        "rate_ratio": round(rate_ratio, 3),
        "peak_concurrency": peak_concurrency(rows),
        "latency_s": lat, "gpu_time_s": gpu_t, "queue_wait_s": queue,
        "prompt_tokens": p_tok, "new_tokens": n_tok,
        "total_new_tokens": int(total_new), "total_prompt_tokens": int(total_prompt),
        "output_tokens_per_s": round(total_new / duration, 1) if duration else 0.0,
        "verdict": verdict, "verdict_reason": why,
    }


def gpu_window_stats(gpu_rows: list[dict], t0: float, t1: float, offset: float) -> dict:
    """Per-GPU and aggregate stats over one phase window."""
    per_gpu: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # Total board power has to be summed per *timestamp* first; averaging each
    # GPU then adding would be right only if both GPUs were sampled identically.
    power_by_ts: dict[float, float] = defaultdict(float)

    for r in gpu_rows:
        ts = fnum(r, "ts") + offset
        if not (t0 <= ts <= t1):
            continue
        g = r.get("gpu", "?")
        per_gpu[g]["util"].append(fnum(r, "util_gpu_pct"))
        per_gpu[g]["mem_util"].append(fnum(r, "util_mem_pct"))
        per_gpu[g]["mem_used_mb"].append(fnum(r, "mem_used_mb"))
        per_gpu[g]["mem_used_pct"].append(fnum(r, "mem_used_pct"))
        per_gpu[g]["temp"].append(fnum(r, "temp_c"))
        per_gpu[g]["power"].append(fnum(r, "power_w"))
        per_gpu[g]["sm_clock"].append(fnum(r, "sm_clock_mhz"))
        per_gpu[g]["mem_total_mb"].append(fnum(r, "mem_total_mb"))
        power_by_ts[round(ts, 1)] += fnum(r, "power_w")

    out: dict = {"gpus": {}, "samples": 0}
    for g, cols in sorted(per_gpu.items()):
        out["gpus"][g] = {
            "util_pct":     describe(cols["util"]),
            "mem_util_pct": describe(cols["mem_util"]),
            "mem_used_gb":  describe([v / 1024 for v in cols["mem_used_mb"]]),
            "mem_used_pct": describe(cols["mem_used_pct"]),
            "temp_c":       describe(cols["temp"]),
            "power_w":      describe(cols["power"]),
            "sm_clock_mhz": describe(cols["sm_clock"]),
            "mem_total_gb": round(max(cols["mem_total_mb"]) / 1024, 1) if cols["mem_total_mb"] else 0.0,
        }
        out["samples"] = max(out["samples"], len(cols["util"]))

    totals = list(power_by_ts.values())
    out["total_power_w"] = describe(totals)
    return out


def host_window_stats(host_rows: list[dict], t0: float, t1: float, offset: float) -> dict:
    sel = [r for r in host_rows if t0 <= fnum(r, "ts") + offset <= t1]
    if not sel:
        return {}
    return {
        "ram_used_gb":   describe([fnum(r, "ram_used_gb")  for r in sel]),
        "ram_used_pct":  describe([fnum(r, "ram_used_pct") for r in sel]),
        "ram_total_gb":  round(max(fnum(r, "ram_total_gb") for r in sel), 1),
        "swap_used_gb":  describe([fnum(r, "swap_used_gb") for r in sel]),
        "load1":         describe([fnum(r, "load1")        for r in sel]),
        "disk_used_gb":  round(max(fnum(r, "disk_used_gb") for r in sel), 1),
        "disk_free_gb":  round(min(fnum(r, "disk_free_gb") for r in sel), 1),
        "disk_total_gb": round(max(fnum(r, "disk_total_gb") for r in sel), 1),
        "disk_used_pct": round(max(fnum(r, "disk_used_pct") for r in sel), 1),
        "disk_path":     sel[-1].get("disk_path", ""),
    }


# ═══════════════════════════════ charts ═══════════════════════════════════════

def _titles(ax, title: str, subtitle: str, pad: float = 34.0):
    """Title above subtitle above the axes, with enough room for both.
    Setting them independently is what collided them."""
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=pad, fontweight="bold")
    ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom", ha="left")


def _band_phases(ax, phases: list[dict], t_origin: float):
    """Shade load phases and label them, so every timeline reads as
    idle → load → idle without the reader counting gridlines."""
    for ph in phases:
        if ph["kind"] != "load":
            continue
        a = fnum(ph, "start_ts") - t_origin
        b = fnum(ph, "end_ts")   - t_origin
        ax.axvspan(a, b, color=C_S1, alpha=0.07, zorder=0)
        # Inside the plot area, not above it — above, these collided with the
        # subtitle line on every timeline.
        ax.annotate(f"{fnum(ph, 'rate'):g}/s", xy=((a + b) / 2, 0.97),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=8.5, color=INK_2,
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="none", alpha=0.82))


def chart_timeline(path: Path, gpu_rows, phases, offset, t_origin, value_key,
                   ylabel, title, subtitle, ylim=None, scale=1.0):
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in gpu_rows:
        series[r.get("gpu", "?")].append(
            ((fnum(r, "ts") + offset) - t_origin, fnum(r, value_key) * scale)
        )
    if not series:
        return False

    fig, ax = plt.subplots(figsize=(11, 4.0))
    colors = [C_S1, C_S2, C_S3]
    for i, (g, pts) in enumerate(sorted(series.items())):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=2.0, color=colors[i % len(colors)], label=f"GPU {g}",
                solid_capstyle="round")
    _band_phases(ax, phases, t_origin)

    ax.set_xlabel("seconds since run start")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    _titles(ax, title, subtitle)
    if len(series) >= 2:
        # Anchored above the plot area — inside it, the legend sat on top of a
        # line pinned near 100%.
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  ncol=len(series), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_host_timeline(path: Path, host_rows, phases, offset, t_origin):
    if not host_rows:
        return False
    xs = [(fnum(r, "ts") + offset) - t_origin for r in host_rows]
    ys = [fnum(r, "ram_used_gb") for r in host_rows]
    total = max((fnum(r, "ram_total_gb") for r in host_rows), default=0)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(xs, ys, lw=2.0, color=C_S1, solid_capstyle="round", label="RAM in use")
    if total:
        ax.axhline(total, color=BASELINE, lw=1.5, ls="--")
        ax.annotate(f"installed {total:.0f} GB", xy=(xs[-1] if xs else 0, total),
                    xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8.5, color=MUTED)
    _band_phases(ax, phases, t_origin)
    ax.set_xlabel("seconds since run start")
    ax.set_ylabel("GB")
    ax.set_ylim(0, max(total * 1.12, max(ys) * 1.2) if ys else 1)
    _titles(ax, "Host RAM during the run",
            "Pinned host memory for image decode and the vLLM process")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_before_after(path: Path, phase_gpu: dict, scenarios: list[dict],
                       idle_before_key: str, idle_after_key: str):
    """The explicit before/during/after comparison, per metric."""
    order = [idle_before_key] + [s["scenario"] for s in scenarios] + [idle_after_key]
    order = [p for p in order if p in phase_gpu and phase_gpu[p].get("gpus")]
    if len(order) < 2:
        return False

    def agg(phase: str, metric: str, stat: str = "mean") -> float:
        gpus = phase_gpu[phase]["gpus"]
        vals = [g[metric][stat] for g in gpus.values()]
        return float(np.mean(vals)) if vals else 0.0

    def total(phase: str, metric: str, stat: str = "mean") -> float:
        gpus = phase_gpu[phase]["gpus"]
        return float(sum(g[metric][stat] for g in gpus.values()))

    # One hue for every load bar. Colour here encodes idle-vs-load, and the
    # panel title already says which metric it is — giving each panel its own
    # hue made the shared legend's "under load" swatch a lie in three of four.
    panels = [
        ("GPU utilization", "mean across GPUs, %",    lambda p: agg(p, "util_pct"),      "%.0f%%"),
        ("VRAM in use",     "summed across GPUs, GB", lambda p: total(p, "mem_used_gb"), "%.0f"),
        ("Board power",     "summed across GPUs, W",  lambda p: total(p, "power_w"),     "%.0f"),
        ("Temperature",     "mean across GPUs, °C",   lambda p: agg(p, "temp_c"),        "%.0f"),
    ]

    labels = [p.replace("rate_", "").replace("ps", "/s").replace("idle_", "idle ")
              for p in order]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, (title, sub, fn, fmt) in zip(axes.flat, panels):
        vals = [fn(p) for p in order]
        # Idle bars are drawn in muted ink so the eye reads baseline-vs-load,
        # not five equally weighted categories.
        cols = [BASELINE if p.startswith("idle") else C_S1 for p in order]
        bars = ax.bar(range(len(order)), vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.annotate(fmt % v, xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=24, fontweight="bold")
        ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="bottom", ha="left")
        ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)

    fig.legend(handles=[Patch(facecolor=BASELINE, label="idle baseline"),
                        Patch(facecolor=C_S1, label="under load")],
               loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Before, during and after image processing", x=0.008, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_throughput(path: Path, scenarios: list[dict], required_rate: float):
    """Offered vs delivered, as a dumbbell rather than paired bars.

    The two rates span 0.67 → 30, which needs a log axis; but a log axis breaks
    bars, because a bar's LENGTH is its encoding and a log length is not
    proportional to its value. Dots encode by position, so they survive the log
    axis intact — and the connecting segment draws the shortfall directly,
    which is the thing the chart exists to show.
    """
    n = len(scenarios)
    ys = np.arange(n)[::-1]          # lowest rate at the top

    fig, ax = plt.subplots(figsize=(10, 1.05 * n + 2.6))
    for y, sc in zip(ys, scenarios):
        a, o = max(sc["achieved_rate"], 1e-3), sc["target_rate"]
        lo, hi = min(a, o), max(a, o)
        ax.plot([lo, hi], [y, y], lw=2.0, color=BASELINE,
                solid_capstyle="round", zorder=1)
        ax.scatter([o], [y], s=110, color=C_S1, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
        ax.scatter([a], [y], s=110, color=C_S2, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"{o:g}", xy=(o, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_2)
        ax.annotate(f"{a:.2f}", xy=(a, y), xytext=(0, -17), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_2)
        ax.annotate(f"{STATUS_ICON[sc['verdict']]} {sc['verdict'].title()}",
                    xy=(1.015, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9,
                    color=STATUS_COLOR[sc["verdict"]])

    ax.axvline(required_rate, color=ST_GOOD, lw=1.6, ls="--", zorder=0)
    ax.annotate(f"business requirement\n{required_rate:.2f} img/s",
                xy=(required_rate, -0.12), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8.5, color=ST_GOOD)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{sc['target_rate']:g} img/s offered" for sc in scenarios])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(max(0.2, min(s_["achieved_rate"] for s_ in scenarios) * 0.45),
                max(s_["target_rate"] for s_ in scenarios) * 1.9)
    ax.set_xlabel("images / second  (log scale)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _titles(ax, "Offered load vs. delivered throughput",
            "the grey segment between the two dots is the capacity shortfall")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=C_S1, label="offered"),
        plt.Line2D([], [], marker="o", ls="", markersize=9, color=C_S2, label="delivered"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_latency(path: Path, scenarios: list[dict], sla_p95: float):
    keys = ["p50", "p90", "p95", "p99"]
    x = np.arange(len(scenarios))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, k in enumerate(keys):
        vals = [s["latency_s"][k] for s in scenarios]
        off = (i - 1.5) * width
        bars = ax.bar(x + off, vals, width=width * 0.92, color=SEQ[i], label=k)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_2)

    ax.axhline(sla_p95, color=ST_CRIT, lw=1.6, ls="--")
    ax.annotate(f"p95 SLA {sla_p95:.0f}s", xy=(len(scenarios) - 0.5, sla_p95),
                xytext=(0, 4), textcoords="offset points", ha="right",
                fontsize=8.5, color=ST_CRIT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['target_rate']:g}/s" for s in scenarios])
    ax.set_xlabel("offered arrival rate")
    ax.set_ylabel("end-to-end latency (s)")
    _titles(ax, "Latency distribution by scenario",
            "measured at the client, through the proxy — includes queue wait")
    ax.legend(loc="upper left", ncol=4, fontsize=9, title=None)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_latency_cdf(path: Path, per_scenario_lat: dict[str, list[float]]):
    live = {k: v for k, v in per_scenario_lat.items() if v}
    if not live:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, vals) in enumerate(live.items()):
        a = np.sort(np.asarray(vals, dtype=float))
        y = np.arange(1, a.size + 1) / a.size * 100
        ax.plot(a, y, lw=2.0, color=SEQ[i % len(SEQ)], label=name, solid_capstyle="round")
        # Direct label at the curve's end — relief for the low-contrast steps.
        ax.annotate(name, xy=(a[-1], 100), xytext=(4, -6), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="top")
    ax.set_xlabel("end-to-end latency (s)")
    ax.set_ylabel("% of successful requests")
    ax.set_ylim(0, 104)
    _titles(ax, "Latency CDF — how much of the run met a given deadline",
            "read across from a % to find the deadline that share of work met")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_concurrency(path: Path, req_rows, phases, t_origin, max_concurrent: int):
    grid, counts = concurrency_series(req_rows)
    if not grid.size:
        return False
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(grid - t_origin, counts, lw=2.0, color=C_S1, solid_capstyle="round")
    ax.axhline(max_concurrent, color=ST_CRIT, lw=1.6, ls="--")
    ax.annotate(f"server max_concurrent = {max_concurrent}",
                xy=(grid[-1] - t_origin, max_concurrent), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=ST_CRIT)
    _band_phases(ax, phases, t_origin)
    ax.set_xlabel("seconds since run start")
    ax.set_ylabel("requests in flight")
    _titles(ax, "In-flight requests vs. the server's admission limit",
            "everything above the dashed line is queued, not being served")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_sizing(path: Path, sizing: list[dict]):
    labels = [f"{s['scenario_rate']:g}/s" for s in sizing]
    nodes  = [s["nodes_required"] for s in sizing]
    x = np.arange(len(sizing))
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    bars = ax.bar(x, nodes, width=0.55, color=C_S1)
    for b, s in zip(bars, sizing):
        ax.annotate(f"{s['nodes_required']}", xy=(b.get_x() + b.get_width() / 2, s["nodes_required"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=10, color=INK, fontweight="bold")
    ax.axhline(1, color=ST_GOOD, lw=1.6, ls="--")
    ax.annotate("current estate: 1 node (2× H200 NVL)", xy=(len(sizing) - 0.5, 1),
                xytext=(0, 5), textcoords="offset points", ha="right",
                fontsize=8.5, color=ST_GOOD)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("sustained rate to be supported")
    ax.set_ylabel("H200 nodes required")
    _titles(ax, "Nodes required to hold each rate",
            "derived from measured per-node sustained throughput")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════ report ═══════════════════════════════════════

def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(ctx: dict, out: Path) -> str:
    a          = ctx["args"]
    meta       = ctx["meta"]
    scenarios  = ctx["scenarios"]
    phase_gpu  = ctx["phase_gpu"]
    phase_host = ctx["phase_host"]
    sizing     = ctx["sizing"]
    econ       = ctx["economics"]
    charts     = ctx["charts"]
    idle_key   = ctx["idle_before_key"]

    req_rate = a.daily_images / (a.window_hours * 3600.0)
    L: list[str] = []
    w = L.append

    w("# Infra & Compute Sourcing Report")
    w("## Vision-language document extraction on 2× NVIDIA H200 NVL")
    w("")
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
      f"from a measured benchmark run of {ctx['total_ok']} successful inferences "
      f"across {len(scenarios)} load scenarios.*")
    w("")

    if ctx["synthetic"]:
        w("> ## ⚠ THESE NUMBERS ARE SYNTHETIC")
        w("> This run was executed against `mock_gpu_server.py`, which returns "
          "arithmetic placeholder latencies — **not** Qwen3-VL inference. The report "
          "proves the measurement pipeline works end to end. Every performance, "
          "sizing and cost figure below is a placeholder until the run is repeated "
          "against the real H200 server.")
        w("")

    # ── 1. Executive summary ────────────────────────────────────────────────
    sustained   = ctx["sustained_rate"]
    ceiling     = ctx["ceiling_rate"]
    nodes_daily = ctx["nodes_for_daily"]

    w("## 1. Executive summary")
    w("")
    w(md_table(
        ["Question", "Measured answer"],
        [
            ["Business volume", f"{a.daily_images:,} images/day over a {a.window_hours:g}-hour window"],
            ["Required sustained rate", f"**{req_rate:.2f} images/sec**"],
            ["Measured sustained rate, 1 node",
             (f"**{sustained:.2f} images/sec** (all requests served within the "
              f"{a.sla_p95:.0f}s p95 SLA)") if ctx["sustained_is_measured"] else
             (f"**not established** — no scenario met the SLA. "
              f"{sustained:.2f} images/sec is a *saturated ceiling* used as an "
              f"optimistic lower bound below")],
            ["Measured throughput ceiling, 1 node", f"{ceiling:.2f} images/sec "
                                                    f"(saturated — requests shed above this)"],
            ["Headroom on the daily requirement",
             ("n/a" if not req_rate else
              f"**{sustained / req_rate:.1f}×**" if ctx["sustained_is_measured"] else
              f"{sustained / req_rate:.1f}× *against the saturated ceiling* — "
              f"not a headroom figure to plan on")],
            ["Nodes needed for the daily volume", f"**{nodes_daily}**"],
            ["Daily capacity of the current node", f"**{ctx['daily_capacity']:,.0f} images** "
                                                   f"in a {a.window_hours:g}-hour window"],
            ["Energy per image", f"{econ['wh_per_image']:.2f} Wh (GPU board power)"],
            ["Compute cost per 1,000 images", f"{a.currency}{econ['cost_per_1k_images']:,.2f} "
                                              f"(at {a.currency}{a.cost_gpu_hour:g}/GPU-hour)"],
        ]))
    w("")
    w("### The headline finding")
    w("")
    if not ctx["sustained_is_measured"]:
        w(f"> **No tested scenario qualified as sustainable**, including the "
          f"{scenarios[0]['target_rate']:g}/s steady-state case. Every rate either "
          f"breached the {a.sla_p95:.0f}s p95 SLA, shed requests, or failed to keep "
          f"pace. The figures below therefore size against the **saturated ceiling** "
          f"of {sustained:.2f} images/sec, which is an optimistic bound: real "
          f"provisioning needs headroom above it, and the lowest offered rate should "
          f"be re-tested after the section 6 configuration changes before any "
          f"hardware is sourced.")
        w("")
    if ctx["sustained_is_measured"] and sustained >= req_rate:
        w(f"**One node covers the committed workload with {sustained / req_rate:.1f}× headroom.** "
          f"The {a.daily_images:,}-image daily volume needs {req_rate:.2f} images/sec averaged "
          f"across the {a.window_hours:g}-hour window, and a single node sustains "
          f"{sustained:.2f} images/sec. No additional compute is required for the "
          f"steady-state commitment.")
    elif ctx["sustained_is_measured"]:
        w(f"**One node does not cover the committed workload.** The daily volume needs "
          f"{req_rate:.2f} images/sec sustained and a single node delivers "
          f"{sustained:.2f} images/sec, so **{nodes_daily} nodes** are required.")
    else:
        w(f"Against that unqualified ceiling the daily volume would need "
          f"**{nodes_daily} node(s)** — a floor on the answer, not the answer.")
    w("")

    failed = [s for s in scenarios if s["verdict"] != "SUSTAINABLE" and s["target_rate"] >= 1]
    if failed:
        worst = failed[0]
        w(f"**The 10 / 20 / 30 images-per-second scenarios are a different question, "
          f"and the answer is no — not on this node, and not primarily because of the GPU.** "
          f"At {worst['target_rate']:g} images/sec the node delivered "
          f"{worst['achieved_rate']:.2f} images/sec "
          f"({worst['rate_ratio'] * 100:.0f}% of offered) and lost "
          f"{(1 - worst['success_rate']) * 100:.1f}% of requests. "
          f"Section 6 shows why: the binding constraint is the serving configuration, "
          f"not the silicon.")
        w("")

    # ── 2. What was measured ────────────────────────────────────────────────
    w("## 2. What was measured, and how")
    w("")
    w(md_table(["Parameter", "Value"], [
        ["Model", f"`{meta.get('model', 'qwen3-vl')}` — Qwen3-VL-30B-A3B-Instruct (MoE, ~3B active)"],
        ["Unit of work", "1 page image in → structured JSON extraction with OCR'd values out"],
        ["Output cap", f"{meta.get('max_new_tokens', '?')} tokens (`max_new_tokens`)"],
        ["Sampling temperature", f"{meta.get('temperature', '?')} (near-deterministic, as extraction requires)"],
        ["Corpus", f"{meta.get('corpus_images', '?')} distinct images, "
                   f"{meta.get('corpus_avg_kb', '?')} kB average, "
                   f"{meta.get('corpus_bytes', 0) / 1e6:.0f} MB total"],
        ["Image transport", f"`{meta.get('image_mode', '?')}` to the proxy"],
        ["Arrival process", f"{meta.get('arrival', '?')} — **open loop**, requests fire on schedule "
                            f"regardless of whether earlier ones returned"],
        ["Request path", f"load generator → `llm_proxy_v3` → GPU server `/infer`"],
        ["Observability", "1 Hz `nvidia-smi` sampling plus host RAM/disk and the vLLM server's own counters"],
        ["p95 latency SLA", f"{a.sla_p95:.0f}s (configurable — drives the SUSTAINABLE verdict)"],
    ]))
    w("")
    w("**Why open-loop matters.** A closed-loop test that holds *N* requests in flight "
      "cannot fail: offered load throttles itself to whatever the server can absorb, so "
      "it measures throughput but can never tell you whether 30 images/sec is reachable. "
      "This harness fires on a schedule and records what falls on the floor. The shortfall "
      "**is** the capacity answer.")
    w("")

    # ── 3. Scenario results ─────────────────────────────────────────────────
    w("## 3. Scenario results")
    w("")
    w(md_table(
        ["Scenario", "Offered", "Delivered", "% of offered", "Success",
         "p50", "p95", "p99", "Peak in-flight", "Verdict"],
        [[
            f"`{s['scenario']}`", f"{s['target_rate']:g}/s", f"{s['achieved_rate']:.2f}/s",
            f"{s['rate_ratio'] * 100:.0f}%", f"{s['success_rate'] * 100:.1f}%",
            f"{s['latency_s']['p50']:.1f}s", f"{s['latency_s']['p95']:.1f}s",
            f"{s['latency_s']['p99']:.1f}s", s["peak_concurrency"],
            f"{STATUS_ICON[s['verdict']]} **{s['verdict'].title()}**",
        ] for s in scenarios]))
    w("")
    w("Verdict key — ✔ Sustainable: kept pace, nothing shed, p95 within SLA · "
      "▲ Degraded: ≥80% of offered rate but breached SLA or lost requests · "
      "✖ Failed: could not keep pace.")
    w("")
    for s in scenarios:
        w(f"- **{s['target_rate']:g}/s** — {s['verdict_reason']}.")
    w("")

    w("### Request disposition")
    w("")
    w(md_table(
        ["Scenario", "Fired", "OK", "Saturated (503)", "Timeout", "Error", "Client-shed"],
        [[f"{s['target_rate']:g}/s", s["fired"], s["ok"], s["saturated"],
          s["timeout"], s["errors"], s["shed_client"]] for s in scenarios]))
    w("")
    w("`Saturated (503)` is the GPU server refusing admission after a request waited "
      "`QUEUE_TIMEOUT_S` for a semaphore slot — the capacity ceiling announcing itself. "
      "`Client-shed` means the load generator itself hit `--max-inflight`; those rows are "
      "excluded from success rates so client limits are never mistaken for server limits.")
    w("")

    w("### Token economics per image")
    w("")
    w(md_table(
        ["Scenario", "Prompt tokens (mean)", "Output tokens (mean)",
         "GPU time (mean)", "Queue wait (p95)", "Output tok/s (aggregate)"],
        [[f"{s['target_rate']:g}/s", f"{s['prompt_tokens']['mean']:,.0f}",
          f"{s['new_tokens']['mean']:,.0f}", f"{s['gpu_time_s']['mean']:.2f}s",
          f"{s['queue_wait_s']['p95']:.2f}s", f"{s['output_tokens_per_s']:,.1f}"]
         for s in scenarios]))
    w("")
    w("Prompt tokens are dominated by **vision tokens**, not the text instruction. "
      "That count is set by `mm_processor_kwargs.max_pixels` (currently `1280*28*28`) and "
      "is the single largest lever on both latency and VRAM — see section 6.")
    w("")

    # ── 4. GPU observability ────────────────────────────────────────────────
    w("## 4. GPU utilization, memory, and power")
    w("")
    w(f"![GPU utilization timeline]({charts['util']})")
    w("")
    w("**Read this chart carefully.** `utilization.gpu` from `nvidia-smi` is a "
      "*time-occupancy* metric: the fraction of the sampling window in which at least one "
      "kernel was resident. It is **not** the fraction of the SM array doing work. A single "
      "decoding sequence at batch size 1 can read near 100% while most of the H200 idles. "
      "Read it alongside delivered throughput in section 3, never on its own.")
    w("")
    w(f"![VRAM timeline]({charts['vram']})")
    w("")
    w(f"![Power timeline]({charts['power']})")
    w("")
    w(f"![Before and after]({charts['before_after']})")
    w("")

    w("### Idle baseline vs. load, per phase")
    w("")
    rows = []
    for key in [idle_key] + [s["scenario"] for s in scenarios] + [ctx["idle_after_key"]]:
        st = phase_gpu.get(key)
        if not st or not st.get("gpus"):
            continue
        gpus = st["gpus"]
        util = np.mean([g["util_pct"]["mean"] for g in gpus.values()])
        umax = np.max([g["util_pct"]["max"] for g in gpus.values()])
        vram = sum(g["mem_used_gb"]["mean"] for g in gpus.values())
        vmax = sum(g["mem_used_gb"]["max"] for g in gpus.values())
        temp = np.mean([g["temp_c"]["mean"] for g in gpus.values()])
        rows.append([
            f"`{key}`", f"{util:.1f}%", f"{umax:.0f}%",
            f"{vram:.1f} GB", f"{vmax:.1f} GB", f"{temp:.0f}°C",
            f"{st['total_power_w']['mean']:.0f} W", f"{st['total_power_w']['max']:.0f} W",
        ])
    w(md_table(["Phase", "GPU util (mean)", "GPU util (max)", "VRAM (mean, both GPUs)",
                "VRAM (max)", "Temp (mean)", "Power (mean)", "Power (max)"], rows))
    w("")

    if ctx["gpu_imbalance"]:
        w(f"> **Finding — the second GPU is nearly idle.** {ctx['gpu_imbalance']} "
          f"With `tensor_parallel_size: 1`, the Qwen3-VL engine lands entirely on GPU 0 and "
          f"GPU 1 contributes nothing to this workload. Half the purchased VRAM and half the "
          f"purchased FLOPs are unused. See section 6.")
        w("")

    w("### Host memory, storage and load")
    w("")
    w(f"![Host RAM]({charts['ram']})")
    w("")
    host_load = phase_host.get(scenarios[-1]["scenario"], {}) if scenarios else {}
    host_idle = phase_host.get(idle_key, {})
    if host_load:
        w(md_table(["Host metric", "Idle", "Under peak load"], [
            ["RAM in use", f"{host_idle.get('ram_used_gb', {}).get('mean', 0):.1f} GB",
             f"{host_load.get('ram_used_gb', {}).get('mean', 0):.1f} GB"],
            ["RAM utilization", f"{host_idle.get('ram_used_pct', {}).get('mean', 0):.1f}%",
             f"{host_load.get('ram_used_pct', {}).get('mean', 0):.1f}%"],
            ["Installed RAM", f"{host_load.get('ram_total_gb', 0):.0f} GB", "—"],
            ["Swap in use", f"{host_idle.get('swap_used_gb', {}).get('mean', 0):.2f} GB",
             f"{host_load.get('swap_used_gb', {}).get('mean', 0):.2f} GB"],
            ["1-minute load average", f"{host_idle.get('load1', {}).get('mean', 0):.2f}",
             f"{host_load.get('load1', {}).get('mean', 0):.2f}"],
            [f"Model store (`{host_load.get('disk_path', '?')}`)",
             f"{host_load.get('disk_used_gb', 0):,.0f} GB used of "
             f"{host_load.get('disk_total_gb', 0):,.0f} GB "
             f"({host_load.get('disk_used_pct', 0):.0f}%)",
             f"{host_load.get('disk_free_gb', 0):,.0f} GB free"],
        ]))
        w("")

    # ── 5. Capacity & sizing ────────────────────────────────────────────────
    w("## 5. Capacity and sizing")
    w("")
    w(f"![Throughput]({charts['throughput']})")
    w("")
    w(f"![Latency]({charts['latency']})")
    w("")
    w(f"![Latency CDF]({charts['cdf']})")
    w("")
    w(f"![Concurrency]({charts['concurrency']})")
    w("")
    w("### Sizing against each scenario")
    w("")
    w(md_table(
        ["Target rate", "Images / 10h day at that rate", "Per-node sustained",
         "Nodes required", "GPUs required", "Verdict on current estate"],
        [[f"{s['scenario_rate']:g}/s", f"{s['daily_images_at_rate']:,.0f}",
          f"{sustained:.2f}/s", f"**{s['nodes_required']}**", s["gpus_required"],
          "✔ covered" if s["nodes_required"] <= 1 else f"✖ short by {s['nodes_required'] - 1} node(s)"]
         for s in sizing]))
    w("")
    w(f"![Sizing]({charts['sizing']})")
    w("")
    if ctx["sustained_is_measured"]:
        w(f"Sizing is `ceil(target_rate ÷ {sustained:.3f})`, using the **sustained** rate "
          f"rather than the saturated ceiling. Sizing on the ceiling would mean "
          f"provisioning a fleet that is by definition running at the point where it "
          f"starts dropping work.")
    else:
        w(f"Sizing is `ceil(target_rate ÷ {sustained:.3f})`. **That divisor is a saturated "
          f"ceiling, not a sustainable rate** — no scenario met the SLA — so every node "
          f"count in this table is a lower bound. Treat them as 'at least this many', and "
          f"re-measure after section 6 before committing to a number.")
    w("")

    # ── 6. Why the high rates fail ──────────────────────────────────────────
    w("## 6. Why the high rates fail — and what to change first")
    w("")
    w("The ceiling measured here is **not** an H200 limit. It is a serving-configuration "
      "limit, and most of it is recoverable from the config file before any hardware is "
      "bought. In rough order of value:")
    w("")
    w(md_table(
        ["#", "Constraint", "Current", "Recommended", "Why"],
        [
            ["1", "`max_concurrent` (qwen3-vl)", "`2`", "`16`–`32`",
             "This is an `asyncio.Semaphore` **in front of** vLLM. vLLM's own continuous "
             "batching can hold far more sequences; a depth of 2 caps the batch at 2 and "
             "forfeits nearly all batching throughput. This is the single biggest lever."],
            ["2", "`tensor_parallel_size` (qwen3-vl)", "`1`", "`2`",
             "TP=1 puts the whole engine on GPU 0 and leaves GPU 1 idle — half the estate "
             "unused. The file's own header and VRAM plan both specify TP=2; the registry "
             "entry contradicts them."],
            ["3", "`max_num_seqs`", "`max_concurrent * 4` = `8`", "`64`–`128`",
             "Derived from `max_concurrent`, so it inherits the same cap. Raising item 1 "
             "without this just moves the bottleneck one layer down."],
            ["4", "`mm_processor_kwargs.max_pixels`", "`1280*28*28`", "tune per accuracy test",
             "Vision tokens dominate prompt length. Halving max_pixels roughly halves "
             "prefill cost and KV footprint per image. Trade against OCR accuracy on YOUR "
             "documents — this is an accuracy decision, not a performance one."],
            ["5", "`QUEUE_TIMEOUT_S`", "`60`", "keep, but alert on it",
             "60s of queueing before a 503 is a long time to hold a client. The 503 rate "
             "is the correct saturation alarm for capacity monitoring."],
            ["6", "Client batching", "1 image/request", "consider 2–4",
             "`limit_mm_per_prompt` allows 4. Batching amortizes per-request overhead, at "
             "the cost of coarser failure granularity."],
        ]))
    w("")
    w("**Re-run this harness after changing items 1–3.** The numbers in this report are "
      "the *current configuration's* capacity, not the hardware's. Sourcing a second node "
      "before exhausting the config changes would very likely be buying capacity that is "
      "already sitting idle on GPU 1.")
    w("")

    # ── 7. Economics ────────────────────────────────────────────────────────
    w("## 7. Compute economics")
    w("")
    w("> Cost inputs are **assumptions supplied at the command line**, not measured. "
      "Replace them with your own quotes and re-run; only the energy and throughput "
      "figures come from the benchmark.")
    w("")
    w(md_table(["Input", "Value", "Source"], [
        ["GPU rental rate", f"{a.currency}{a.cost_gpu_hour:g} / GPU-hour", "assumption (`--cost-gpu-hour`)"],
        ["Electricity", f"{a.currency}{a.power_cost_kwh:g} / kWh", "assumption (`--power-cost-kwh`)"],
        ["Datacentre PUE", f"{a.pue:g}", "assumption (`--pue`)"],
        ["Non-GPU node draw", f"{a.node_overhead_w:g} W", "assumption (`--node-overhead-w`)"],
        ["Mean GPU board power under load", f"{econ['load_power_w']:.0f} W", "**measured**"],
        ["Mean GPU board power at idle", f"{econ['idle_power_w']:.0f} W", "**measured**"],
        ["Sustained throughput", f"{sustained:.2f} images/sec", "**measured**"],
    ]))
    w("")
    w(md_table(["Metric", "Per image", "Per 1,000 images", f"Per day ({a.daily_images:,} images)",
                "Per year (250 working days)"], [
        ["GPU busy time", f"{econ['gpu_s_per_image']:.2f} GPU-s",
         f"{econ['gpu_s_per_image'] * 1000 / 3600:.2f} GPU-h",
         f"{econ['gpu_hours_per_day']:.1f} GPU-h", f"{econ['gpu_hours_per_day'] * 250:,.0f} GPU-h"],
        ["GPU energy", f"{econ['wh_per_image']:.2f} Wh",
         f"{econ['wh_per_image']:.2f} kWh",
         f"{econ['kwh_per_day']:.1f} kWh", f"{econ['kwh_per_day'] * 250:,.0f} kWh"],
        ["Node energy (incl. PUE + overhead)", f"{econ['node_wh_per_image']:.2f} Wh",
         f"{econ['node_wh_per_image']:.2f} kWh",
         f"{econ['node_kwh_per_day']:.1f} kWh", f"{econ['node_kwh_per_day'] * 250:,.0f} kWh"],
        ["Electricity cost", f"{a.currency}{econ['power_cost_per_image']:.4f}",
         f"{a.currency}{econ['power_cost_per_image'] * 1000:,.2f}",
         f"{a.currency}{econ['power_cost_per_day']:,.2f}",
         f"{a.currency}{econ['power_cost_per_day'] * 250:,.0f}"],
        ["Rental-equivalent compute cost", f"{a.currency}{econ['cost_per_image']:.4f}",
         f"{a.currency}{econ['cost_per_1k_images']:,.2f}",
         f"{a.currency}{econ['cost_per_day']:,.2f}",
         f"{a.currency}{econ['cost_per_day'] * 250:,.0f}"],
    ]))
    w("")
    w(f"Duty cycle: at {sustained:.2f} images/sec the committed {a.daily_images:,} images "
      f"occupy **{econ['busy_hours_per_day']:.1f} hours** of the "
      f"{a.window_hours:g}-hour window — a **{econ['duty_cycle'] * 100:.0f}% duty cycle**. "
      f"The remaining {a.window_hours - econ['busy_hours_per_day']:.1f} hours are idle "
      f"capacity available to the other models in the registry, which is the strongest "
      f"argument for consolidating rather than sourcing dedicated hardware.")
    w("")

    # ── 8. Storage ──────────────────────────────────────────────────────────
    st = ctx["storage"]
    w("## 8. Storage")
    w("")
    w(md_table(["Component", "Size", "Note"], [
        ["Image ingest, per day", f"{st['daily_gb']:.1f} GB",
         f"{a.daily_images:,} × {meta.get('corpus_avg_kb', 0):.0f} kB measured average"],
        [f"Image retention, {a.retention_days:g} days", f"**{st['retention_gb']:,.0f} GB**",
         "raw images only; excludes extracted JSON and any derived artefacts"],
        ["Extracted JSON, per day", f"{st['json_daily_mb']:.0f} MB",
         f"{st['json_kb_per_image']:.1f} kB/image estimated from measured output tokens"],
        ["Model weights on the GPU box", f"{st['models_measured_gb']:,.0f} GB used",
         f"measured on `{st['disk_path']}`; {st['disk_free_gb']:,.0f} GB free of "
         f"{st['disk_total_gb']:,.0f} GB"],
        ["VRAM, Qwen3-VL resident", f"{st['vram_load_gb']:.1f} GB",
         f"peak across both GPUs under load, of {st['vram_total_gb']:.0f} GB installed"],
    ]))
    w("")
    if st["disk_runway_days"] is not None:
        w(f"At the measured ingest rate, the free space on `{st['disk_path']}` is "
          f"**{st['disk_runway_days']:,.0f} days** of runway if raw images are stored there. "
          f"They should not be — the model store and the image store should be separate "
          f"volumes, so a corpus backlog can never prevent a model from loading.")
        w("")

    # ── 9. Method & caveats ─────────────────────────────────────────────────
    w("## 9. Method, reproduction, and caveats")
    w("")
    w("```bash")
    w("# on the GPU box")
    w("python3 gpu_monitor.py --out ./results --interval 1.0")
    w("")
    w("# on the RHEL VM")
    w("python3 loadgen.py --corpus ./corpus --out ./results \\")
    w(f"    --proxy {meta.get('proxy_url', 'http://127.0.0.1:8071/v1/infer').rsplit('/v1', 1)[0]} \\")
    w(f"    --rates {' '.join(str(s['target_rate']) for s in scenarios)}")
    w("")
    w("# anywhere")
    w("python3 analyze.py --results ./results --out ./report \\")
    w(f"    --daily-images {a.daily_images} --window-hours {a.window_hours:g}")
    w("```")
    w("")
    w("**Caveats a reviewer should hold this report to:**")
    w("")
    w(f"1. **Corpus realism.** Throughput scales with vision-token count, which scales "
      f"with image size and visual density. This run used "
      f"{meta.get('corpus_images', '?')} images averaging "
      f"{meta.get('corpus_avg_kb', 0):.0f} kB. If production documents are denser, "
      f"multi-page, or higher-resolution, throughput will be lower.")
    w(f"2. **Burst duration.** The high-rate scenarios ran for "
      f"{max((s['duration_s'] for s in scenarios if s['target_rate'] >= 1), default=0):.0f}s. "
      f"They establish the saturation point, not thermal steady state. A full-hour soak "
      f"is needed before committing to a sustained SLA, since H200 clocks throttle on "
      f"sustained thermal load.")
    w("3. **No accuracy measurement.** This report measures throughput only. Extraction "
      "accuracy on your documents is a separate exercise and is the input that decides "
      "whether `max_pixels` can be lowered — which is the largest available throughput win.")
    w("4. **Single-tenant assumption.** The run had the box to itself apart from the "
      "resident `mistral` engine. Concurrent Falcon traffic will reduce these figures.")
    w("5. **Clock alignment.** GPU samples and request records come from different hosts, "
      f"joined on wall-clock time with a {a.clock_offset_s:+.1f}s offset. Verify NTP sync "
      f"before trusting phase-aligned GPU statistics.")
    w("")
    w("---")
    w("")
    w(f"*Charts and the full numeric dump are in `{out.name}/`. Every figure in this "
      f"report is reproducible from `summary.json`.*")

    return "\n".join(L)


# ═══════════════════════════════ main ═════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out",     default="./report")
    ap.add_argument("--daily-images", type=int,   default=24000)
    ap.add_argument("--window-hours", type=float, default=10.0)
    ap.add_argument("--sla-p95",      type=float, default=30.0,
                    help="p95 latency objective in seconds")
    ap.add_argument("--clock-offset-s", type=float, default=0.0,
                    help="added to GPU-box timestamps to align with the load generator")
    ap.add_argument("--max-concurrent", type=int, default=2,
                    help="the GPU server's max_concurrent for this model, for the charts")
    ap.add_argument("--retention-days", type=float, default=90.0)
    ap.add_argument("--cost-gpu-hour",  type=float, default=3.50)
    ap.add_argument("--power-cost-kwh", type=float, default=0.12)
    ap.add_argument("--pue",            type=float, default=1.4)
    ap.add_argument("--node-overhead-w", type=float, default=700.0,
                    help="CPU, RAM, NICs, fans, PSU loss — everything but the GPU boards")
    ap.add_argument("--currency", default="$")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    req_rows  = read_csv(res / "requests.csv")
    phases    = read_csv(res / "phases.csv")
    gpu_rows  = read_csv(res / "gpu_samples.csv")
    host_rows = read_csv(res / "host_samples.csv")
    meta_path = res / "run_meta.json"
    meta      = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if not req_rows or not phases:
        raise SystemExit(f"No requests.csv/phases.csv under {res} — run loadgen.py first.")

    offset = args.clock_offset_s

    # ── clock sanity ─────────────────────────────────────────────────────────
    if gpu_rows:
        g0 = min(fnum(r, "ts") for r in gpu_rows) + offset
        g1 = max(fnum(r, "ts") for r in gpu_rows) + offset
        l0 = min(fnum(p, "start_ts") for p in phases)
        l1 = max(fnum(p, "end_ts")   for p in phases)
        if g1 < l0 or g0 > l1:
            print(f"!! GPU samples ({g0:.0f}–{g1:.0f}) do not overlap the load window "
                  f"({l0:.0f}–{l1:.0f}). Clocks are out of sync — pass --clock-offset-s "
                  f"≈ {l0 - g0:+.0f}. GPU statistics will be empty until you do.")
        elif g0 > l0 + 5 or g1 < l1 - 5:
            print("!! GPU sampling did not cover the whole load window; some phases will "
                  "have partial or missing GPU statistics.")
    else:
        print("!! No gpu_samples.csv — GPU sections will be omitted. Did gpu_monitor.py "
              "run on the GPU box?")

    # ── per-scenario analysis ────────────────────────────────────────────────
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in req_rows:
        by_scenario[r["scenario"]].append(r)

    load_phases = [p for p in phases if p["kind"] == "load"]
    scenarios = [analyse_scenario(p["phase"], p, by_scenario.get(p["phase"], []), args.sla_p95)
                 for p in load_phases]
    scenarios.sort(key=lambda s: s["target_rate"])

    phase_gpu  = {p["phase"]: gpu_window_stats(gpu_rows, fnum(p, "start_ts"),
                                               fnum(p, "end_ts"), offset) for p in phases}
    phase_host = {p["phase"]: host_window_stats(host_rows, fnum(p, "start_ts"),
                                                fnum(p, "end_ts"), offset) for p in phases}

    idle_before_key = next((p["phase"] for p in phases if p["phase"] == "idle_before"), "idle_before")
    idle_after_key  = next((p["phase"] for p in phases if p["phase"] == "idle_after"), "idle_after")

    # ── headline capacity ────────────────────────────────────────────────────
    ok_scen  = [s for s in scenarios if s["verdict"] == "SUSTAINABLE"]
    ceiling  = max((s["achieved_rate"] for s in scenarios), default=0.0)
    sustained_is_measured = bool(ok_scen)
    if ok_scen:
        sustained = max(s["achieved_rate"] for s in ok_scen)
    else:
        # Nothing cleared the bar. The best delivered rate is a SATURATED
        # ceiling, not a sustainable rate — sizing off it would provision a
        # fleet that runs permanently at the point where it drops work. It is
        # used as a lower bound and the report says so explicitly.
        sustained = ceiling
    sustained = max(sustained, 1e-9)

    req_rate       = args.daily_images / (args.window_hours * 3600.0)
    daily_capacity = sustained * args.window_hours * 3600.0
    nodes_daily    = max(1, math.ceil(args.daily_images / daily_capacity)) if daily_capacity else 0

    sizing = []
    for s in scenarios:
        rate = s["target_rate"]
        nodes = max(1, math.ceil(rate / sustained))
        sizing.append({
            "scenario_rate": rate,
            "daily_images_at_rate": rate * args.window_hours * 3600.0,
            "nodes_required": nodes,
            "gpus_required": nodes * 2,
        })

    # ── power & economics ────────────────────────────────────────────────────
    def phase_power(keys: list[str]) -> float:
        vals = [phase_gpu[k]["total_power_w"]["mean"] for k in keys
                if k in phase_gpu and phase_gpu[k].get("total_power_w", {}).get("n")]
        return float(np.mean(vals)) if vals else 0.0

    load_power = phase_power([s["scenario"] for s in scenarios])
    idle_power = phase_power([idle_before_key, idle_after_key])

    gpu_s_per_image = 1.0 / sustained
    wh_per_image    = load_power / sustained / 3600.0 if sustained else 0.0
    node_w          = load_power * args.pue + args.node_overhead_w
    node_wh_per_img = node_w / sustained / 3600.0 if sustained else 0.0

    busy_hours   = args.daily_images / sustained / 3600.0 if sustained else 0.0
    duty_cycle   = busy_hours / args.window_hours if args.window_hours else 0.0
    n_gpus       = max(1, len(phase_gpu.get(idle_before_key, {}).get("gpus", {}) or {1: 1}))
    gpu_hours_pd = busy_hours * n_gpus

    kwh_per_day      = args.daily_images * wh_per_image / 1000.0
    node_kwh_per_day = args.daily_images * node_wh_per_img / 1000.0
    cost_per_day     = gpu_hours_pd * args.cost_gpu_hour
    power_cost_day   = node_kwh_per_day * args.power_cost_kwh

    economics = {
        "load_power_w": load_power, "idle_power_w": idle_power,
        "gpu_s_per_image": gpu_s_per_image, "wh_per_image": wh_per_image,
        "node_wh_per_image": node_wh_per_img, "node_power_w": node_w,
        "busy_hours_per_day": busy_hours, "duty_cycle": duty_cycle,
        "gpu_hours_per_day": gpu_hours_pd,
        "kwh_per_day": kwh_per_day, "node_kwh_per_day": node_kwh_per_day,
        "cost_per_day": cost_per_day,
        "cost_per_image": cost_per_day / args.daily_images if args.daily_images else 0.0,
        "cost_per_1k_images": cost_per_day / args.daily_images * 1000 if args.daily_images else 0.0,
        "power_cost_per_day": power_cost_day,
        "power_cost_per_image": power_cost_day / args.daily_images if args.daily_images else 0.0,
    }

    # ── storage ──────────────────────────────────────────────────────────────
    avg_kb    = float(meta.get("corpus_avg_kb", 0) or 0)
    daily_gb  = args.daily_images * avg_kb / 1e6
    host_any  = next((h for h in phase_host.values() if h), {})
    mean_out  = float(np.mean([s["new_tokens"]["mean"] for s in scenarios])) if scenarios else 0.0
    json_kb   = mean_out * 4 / 1024.0          # ~4 bytes/token for JSON-ish output
    free_gb   = host_any.get("disk_free_gb", 0)

    vram_load = 0.0
    for s in scenarios:
        st = phase_gpu.get(s["scenario"], {})
        vram_load = max(vram_load, sum(g["mem_used_gb"]["max"] for g in st.get("gpus", {}).values()))
    vram_total = sum(g["mem_total_gb"] for g in
                     phase_gpu.get(idle_before_key, {}).get("gpus", {}).values())

    storage = {
        "daily_gb": daily_gb,
        "retention_gb": daily_gb * args.retention_days,
        "json_kb_per_image": json_kb,
        "json_daily_mb": args.daily_images * json_kb / 1024.0,
        "models_measured_gb": host_any.get("disk_used_gb", 0),
        "disk_free_gb": free_gb,
        "disk_total_gb": host_any.get("disk_total_gb", 0),
        "disk_path": host_any.get("disk_path", "n/a"),
        "vram_load_gb": vram_load, "vram_total_gb": vram_total,
        "disk_runway_days": (free_gb / daily_gb) if daily_gb > 0 else None,
    }

    # ── GPU imbalance finding ────────────────────────────────────────────────
    imbalance = ""
    busiest = phase_gpu.get(scenarios[-1]["scenario"], {}).get("gpus", {}) if scenarios else {}
    if len(busiest) >= 2:
        utils = {g: v["util_pct"]["mean"] for g, v in busiest.items()}
        hi = max(utils, key=utils.get)
        lo = min(utils, key=utils.get)
        if utils[hi] - utils[lo] > 25:
            imbalance = (f"At the highest offered rate GPU {hi} averaged "
                         f"{utils[hi]:.0f}% utilization while GPU {lo} averaged "
                         f"{utils[lo]:.0f}%.")

    # ── charts ───────────────────────────────────────────────────────────────
    t_origin = min(fnum(p, "start_ts") for p in phases)
    charts = {
        "util": "chart_gpu_util.png", "vram": "chart_vram.png",
        "power": "chart_power.png", "ram": "chart_host_ram.png",
        "before_after": "chart_before_after.png", "throughput": "chart_throughput.png",
        "latency": "chart_latency.png", "cdf": "chart_latency_cdf.png",
        "concurrency": "chart_concurrency.png", "sizing": "chart_sizing.png",
    }

    chart_timeline(out / charts["util"], gpu_rows, phases, offset, t_origin,
                   "util_gpu_pct", "utilization (%)",
                   "GPU utilization before, during and after processing",
                   "shaded bands are load phases, labelled with their offered rate",
                   ylim=(0, 105))
    chart_timeline(out / charts["vram"], gpu_rows, phases, offset, t_origin,
                   "mem_used_mb", "VRAM in use (GB)",
                   "VRAM occupancy across the run",
                   "weights stay resident between phases; the variable part is KV cache "
                   "and vision-encoder activations",
                   scale=1 / 1024)
    chart_timeline(out / charts["power"], gpu_rows, phases, offset, t_origin,
                   "power_w", "board power (W)",
                   "GPU board power — the basis for energy-per-image",
                   "idle draw between phases is the floor cost of keeping the model resident")
    chart_host_timeline(out / charts["ram"], host_rows, phases, offset, t_origin)
    chart_before_after(out / charts["before_after"], phase_gpu, scenarios,
                       idle_before_key, idle_after_key)
    chart_throughput(out / charts["throughput"], scenarios, req_rate)
    chart_latency(out / charts["latency"], scenarios, args.sla_p95)
    chart_latency_cdf(out / charts["cdf"],
                      {f"{s['target_rate']:g}/s":
                       [fnum(r, "latency_s") for r in by_scenario[s["scenario"]] if r["status"] == "ok"]
                       for s in scenarios})
    chart_concurrency(out / charts["concurrency"], req_rows, phases, t_origin,
                      args.max_concurrent)
    chart_sizing(out / charts["sizing"], sizing)

    ctx = {
        "args": args, "meta": meta, "scenarios": scenarios,
        "phase_gpu": phase_gpu, "phase_host": phase_host,
        "sizing": sizing, "economics": economics, "storage": storage,
        "charts": charts, "sustained_rate": sustained, "ceiling_rate": ceiling,
        "nodes_for_daily": nodes_daily, "daily_capacity": daily_capacity,
        "idle_before_key": idle_before_key, "idle_after_key": idle_after_key,
        "gpu_imbalance": imbalance,
        "sustained_is_measured": sustained_is_measured,
        "total_ok": sum(s["ok"] for s in scenarios),
        "synthetic": bool(meta.get("mock_detected")),
    }

    (out / "REPORT.md").write_text(build_report(ctx, out))
    (out / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": ctx["synthetic"],
        "run_meta": meta, "scenarios": scenarios, "sizing": sizing,
        "economics": economics, "storage": storage,
        "phase_gpu": phase_gpu, "phase_host": phase_host,
        "sustained_rate_img_s": sustained, "ceiling_rate_img_s": ceiling,
        "sustained_is_measured": sustained_is_measured,
        "required_rate_img_s": req_rate, "nodes_for_daily_volume": nodes_daily,
        "daily_capacity_images": daily_capacity,
    }, indent=2, default=str))

    print(f"\nReport written to {out / 'REPORT.md'}")
    print(f"  sustained  : {sustained:.3f} img/s   ceiling: {ceiling:.3f} img/s")
    print(f"  required   : {req_rate:.3f} img/s    nodes for daily volume: {nodes_daily}")
    for s in scenarios:
        print(f"  {s['target_rate']:>5g}/s → {s['achieved_rate']:>6.2f}/s "
              f"({s['success_rate'] * 100:5.1f}% ok)  {s['verdict']}")
    if ctx["synthetic"]:
        print("\n  *** SYNTHETIC RUN — numbers are placeholders, not measurements ***")


if __name__ == "__main__":
    main()
