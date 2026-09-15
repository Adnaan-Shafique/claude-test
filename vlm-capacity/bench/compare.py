#!/usr/bin/env python3
"""
compare.py — A/B (or A/B/C) comparison across serving configurations.

Answers the question a sourcing report actually has to answer: *what did the
second GPU buy, over and above what a config change on one GPU would have
bought anyway?*

That distinction is the whole point. Run the second GPU and a raised
`max_concurrent` as a single change and you cannot tell them apart — the
combined number gets attributed to the hardware, which is the error that gets a
sourcing report sent back. So the recommended sequence is three runs:

  A  baseline   TP=1, gpu_memory_utilization 0.60, max_concurrent 2   (as deployed)
  B  semaphore  TP=1, gpu_memory_utilization 0.60, max_concurrent 32  (one GPU still)
  C  two GPUs   TP=2, gpu_memory_utilization 0.35, max_concurrent 32

B − A is what the config change was worth on one GPU.
C − B is what the second GPU was worth. Only that second number is a
hardware argument.

Usage:
    python3 compare.py --out ./report-compare \\
        --run "A baseline TP=1 c=2:./results-a" \\
        --run "B semaphore TP=1 c=32:./results-b" \\
        --run "C two GPUs TP=2 c=32:./results-c"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from analyze import (                                  # noqa: E402
    C_S1, C_S2, C_S3, SURFACE, INK, INK_2, MUTED, BASELINE,
    ST_GOOD, ST_WARN, ST_CRIT, STATUS_ICON,
    analyse_scenario, gpu_window_stats, host_window_stats,
    read_csv, fnum, md_table, _titles,
)

SERIES = [C_S1, C_S2, C_S3]

# Below this relative change, a difference between two runs is not a result.
# Back-to-back runs of an identical configuration vary by a few percent from
# scheduling and thermal state alone; calling a 3% delta a win is how a
# benchmark talks itself into a conclusion.
NOISE_FLOOR = 0.05


def load_run(label: str, path: Path, sla_p95: float) -> dict:
    req    = read_csv(path / "requests.csv")
    phases = read_csv(path / "phases.csv")
    gpu    = read_csv(path / "gpu_samples.csv")
    host   = read_csv(path / "host_samples.csv")
    mpath  = path / "run_meta.json"
    meta   = json.loads(mpath.read_text()) if mpath.exists() else {}

    if not req or not phases:
        raise SystemExit(f"'{label}': no requests.csv/phases.csv under {path}")

    by_scenario = defaultdict(list)
    for r in req:
        by_scenario[r["scenario"]].append(r)

    scenarios = [analyse_scenario(p["phase"], p, by_scenario.get(p["phase"], []), sla_p95)
                 for p in phases if p["kind"] == "load"]
    scenarios.sort(key=lambda s: s["target_rate"])

    load_marks = [p for p in phases if p["kind"] == "load"]
    gpu_load, host_load = {}, {}
    if load_marks and gpu:
        t0 = min(fnum(p, "start_ts") for p in load_marks)
        t1 = max(fnum(p, "end_ts")   for p in load_marks)
        gpu_load  = gpu_window_stats(gpu, t0, t1, 0.0)
        host_load = host_window_stats(host, t0, t1, 0.0)

    entry = (meta.get("server_config") or {}).get("model_entry") or {}

    return {
        "label": label, "path": str(path), "meta": meta, "scenarios": scenarios,
        "gpu": gpu_load, "host": host_load, "entry": entry,
        "by_scenario": by_scenario,
        "synthetic": bool(meta.get("mock_detected")),
        "best_rate": max((s["achieved_rate"] for s in scenarios), default=0.0),
        "sustained": max((s["achieved_rate"] for s in scenarios
                          if s["verdict"] == "SUSTAINABLE"), default=0.0),
    }


def fmt_delta(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    rel = (new - old) / old
    if abs(rel) < NOISE_FLOOR:
        return f"{rel * 100:+.1f}% *(within noise)*"
    return f"**{rel * 100:+.1f}%**"


# ═══════════════════════════════ charts ═══════════════════════════════════════

def chart_rate_compare(path: Path, runs: list[dict]):
    rates = sorted({s["target_rate"] for r in runs for s in r["scenarios"]})
    x = np.arange(len(rates))
    width = min(0.26, 0.78 / len(runs))

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, run in enumerate(runs):
        vals = []
        for rate in rates:
            sc = next((s for s in run["scenarios"] if s["target_rate"] == rate), None)
            vals.append(sc["achieved_rate"] if sc else 0.0)
        off = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width=width * 0.9, color=SERIES[i % len(SERIES)],
                      label=run["label"])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}/s offered" for r in rates])
    ax.set_ylabel("delivered images / second")
    _titles(ax, "Delivered throughput by serving configuration",
            "same corpus, same prompt, same arrival schedule — only the server config differs")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(runs), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_attribution(path: Path, runs: list[dict], metric: str, ylabel: str, title: str):
    """Waterfall: what each successive change was actually worth."""
    vals = [r["best_rate"] if metric == "best_rate" else r["sustained"] for r in runs]
    labels = [r["label"] for r in runs]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    bottoms, heights, colors, texts = [], [], [], []
    for i, v in enumerate(vals):
        if i == 0:
            bottoms.append(0.0); heights.append(v)
            colors.append(BASELINE); texts.append(f"{v:.2f}")
        else:
            prev = vals[i - 1]
            bottoms.append(min(prev, v)); heights.append(abs(v - prev))
            colors.append(ST_GOOD if v >= prev else ST_CRIT)
            texts.append(f"{v - prev:+.2f}")

    x = np.arange(len(vals))
    bars = ax.bar(x, heights, bottom=bottoms, width=0.55, color=colors,
                  linewidth=2, edgecolor=SURFACE)
    for i, (b, t, v) in enumerate(zip(bars, texts, vals)):
        ax.annotate(t, xy=(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9.5, color=INK, fontweight="bold")
        ax.annotate(f"total {v:.2f}", xy=(i, -0.075), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8.5, color=MUTED)
        if i:
            ax.plot([i - 1 + 0.275, i - 0.275], [vals[i - 1]] * 2,
                    color=BASELINE, lw=1.2, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) else 1)
    _titles(ax, title, "each bar is the increment that change was worth, not its total")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_vram_compare(path: Path, runs: list[dict]):
    labels, used, total = [], [], []
    for r in runs:
        gpus = r["gpu"].get("gpus", {})
        if not gpus:
            continue
        labels.append(r["label"])
        used.append(sum(g["mem_used_gb"]["max"] for g in gpus.values()))
        total.append(sum(g["mem_total_gb"] for g in gpus.values()))
    if not labels:
        return False

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x, total, width=0.55, color=BASELINE, label="installed")
    ax.bar(x, used,  width=0.55, color=C_S1, label="peak in use")
    for xi, (u, t) in enumerate(zip(used, total)):
        ax.annotate(f"{u:.0f} GB used", xy=(xi, u), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK, fontweight="bold")
        ax.annotate(f"{t - u:.0f} GB idle", xy=(xi, t), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("VRAM (GB)")
    ax.set_ylim(0, max(total) * 1.18)
    _titles(ax, "VRAM engaged by each configuration",
            "the grey headroom above each bar is purchased memory the config never touches")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


# ═══════════════════════════════ report ═══════════════════════════════════════

def build(runs: list[dict], args, out: Path) -> str:
    L: list[str] = []
    w = L.append
    base, last = runs[0], runs[-1]
    req_rate = args.daily_images / (args.window_hours * 3600.0)

    w("# Serving-configuration comparison")
    w("## What the second GPU is actually worth")
    w("")
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
      f"from {len(runs)} runs of an identical workload.*")
    w("")
    if any(r["synthetic"] for r in runs):
        w("> ## ⚠ ONE OR MORE RUNS ARE SYNTHETIC")
        w("> At least one directory was produced against `mock_gpu_server.py`. "
          "Every figure below is a placeholder until all runs are repeated against "
          "the real H200 server.")
        w("")

    # ── configurations ──────────────────────────────────────────────────────
    w("## 1. What differed between the runs")
    w("")
    rows = []
    for r in runs:
        e = r["entry"] or {}
        gpus = len(r["gpu"].get("gpus", {})) or "?"
        rows.append([
            f"**{r['label']}**",
            e.get("tensor_parallel_size", "?"),
            e.get("gpu_memory_utilization", "?"),
            e.get("max_concurrent", "?"),
            e.get("max_model_len", "?"),
            gpus,
        ])
    w(md_table(["Run", "tensor_parallel_size", "gpu_memory_utilization",
                "max_concurrent", "max_model_len", "GPUs sampled"], rows))
    w("")
    if any(not r["entry"] for r in runs):
        w("> Some runs could not read the server's live registry, so their columns show "
          "`?`. Those runs are **not safely comparable** — a configuration you cannot "
          "prove was deployed is a configuration you are assuming. Re-run with the "
          "proxy's `/v1/models` reachable.")
        w("")

    # ── throughput ──────────────────────────────────────────────────────────
    w("## 2. Delivered throughput")
    w("")
    w(f"![Throughput by config]({args.chart_rate})")
    w("")
    rates = sorted({s["target_rate"] for r in runs for s in r["scenarios"]})
    header = ["Offered"] + [r["label"] for r in runs] + [f"Δ vs {base['label']}"]
    rows = []
    for rate in rates:
        cells, first, lastv = [], None, None
        for r in runs:
            sc = next((s for s in r["scenarios"] if s["target_rate"] == rate), None)
            if sc is None:
                cells.append("—"); continue
            cells.append(f"{sc['achieved_rate']:.2f}/s {STATUS_ICON[sc['verdict']]}")
            if first is None:
                first = sc["achieved_rate"]
            lastv = sc["achieved_rate"]
        rows.append([f"{rate:g}/s"] + cells
                    + [fmt_delta(lastv, first) if first is not None and lastv is not None else "n/a"])
    w(md_table(header, rows))
    w("")
    w("### Latency at p95")
    w("")
    rows = []
    for rate in rates:
        cells = []
        for r in runs:
            sc = next((s for s in r["scenarios"] if s["target_rate"] == rate), None)
            cells.append(f"{sc['latency_s']['p95']:.1f}s" if sc else "—")
        rows.append([f"{rate:g}/s"] + cells)
    w(md_table(["Offered"] + [r["label"] for r in runs], rows))
    w("")

    # ── attribution ─────────────────────────────────────────────────────────
    w("## 3. Attribution — which change earned what")
    w("")
    w(f"![Attribution]({args.chart_attr})")
    w("")
    rows = []
    for i, r in enumerate(runs):
        if i == 0:
            rows.append([f"**{r['label']}**", f"{r['best_rate']:.2f}/s", "—", "baseline"])
        else:
            prev = runs[i - 1]
            gain = r["best_rate"] - prev["best_rate"]
            rows.append([
                f"**{r['label']}**", f"{r['best_rate']:.2f}/s", f"{gain:+.2f}/s",
                fmt_delta(r["best_rate"], prev["best_rate"]),
            ])
    w(md_table(["Run", "Best delivered rate", "Increment", "Relative to previous"], rows))
    w("")

    # A run that delivered essentially all of the highest rate offered never
    # found its ceiling — its "best rate" is the test's limit, not the
    # configuration's. Comparing two such runs measures the load generator.
    max_offered = max((s["target_rate"] for r in runs for s in r["scenarios"]), default=0.0)
    capped = [r for r in runs
              if max_offered and r["best_rate"] >= 0.95 * max_offered]
    if len(capped) >= 2:
        w(f"> **⚠ The offered load did not reach these configurations' ceiling.** "
          f"{', '.join(repr(r['label']) for r in capped)} each delivered ≥95% of the "
          f"highest rate offered ({max_offered:g} img/s), which means the test ran out "
          f"of work before they ran out of capacity. Any difference between them is "
          f"bounded by the load generator, not by the hardware — **these runs cannot "
          f"show what the second GPU is worth.** Re-run with `--rates` extended well "
          f"past {max_offered:g} until every configuration visibly saturates.")
        w("")

    total_gain = last["best_rate"] - base["best_rate"]
    if len(runs) >= 3 and total_gain > 0 and len(capped) < 2:
        config_gain = runs[-2]["best_rate"] - base["best_rate"]
        hw_gain     = last["best_rate"] - runs[-2]["best_rate"]
        w(f"Of the **{total_gain:+.2f} img/s** total improvement from "
          f"`{base['label']}` to `{last['label']}`:")
        w("")
        w(f"- **{config_gain:+.2f} img/s ({config_gain / total_gain * 100:.0f}%)** came from "
          f"the configuration change on the *same* GPU count.")
        w(f"- **{hw_gain:+.2f} img/s ({hw_gain / total_gain * 100:.0f}%)** came from adding "
          f"the second GPU on top of it.")
        w("")
        w("**Only the second line is a hardware argument.** Reporting the combined figure "
          "as the value of the second GPU would overstate it by "
          f"{config_gain / max(hw_gain, 1e-9):.1f}×.")
        w("")
    elif len(runs) >= 3 and len(capped) >= 2:
        pass                     # the cap warning above already says what is wrong
    elif len(runs) == 2:
        w("> **Two runs cannot separate the config change from the hardware change.** If "
          "these two runs differ in both `max_concurrent` and `tensor_parallel_size`, the "
          "delta above is the *combined* effect and cannot be attributed to the second GPU. "
          "Add the intermediate run (raise `max_concurrent` on one GPU, change nothing "
          "else) before using this to justify hardware.")
        w("")

    # ── VRAM ────────────────────────────────────────────────────────────────
    section = 4
    if args.chart_vram_ok:
        w(f"## {section}. VRAM engaged")
        w("")
        w(f"![VRAM by config]({args.chart_vram})")
        w("")
        rows = []
        for r in runs:
            gpus = r["gpu"].get("gpus", {})
            if not gpus:
                continue
            used  = sum(g["mem_used_gb"]["max"] for g in gpus.values())
            total = sum(g["mem_total_gb"] for g in gpus.values())
            utils = {g: v["util_pct"]["mean"] for g, v in gpus.items()}
            rows.append([
                f"**{r['label']}**", f"{used:.1f} GB", f"{total:.0f} GB",
                f"{used / total * 100:.0f}%" if total else "—",
                " · ".join(f"GPU{g} {u:.0f}%" for g, u in sorted(utils.items())),
            ])
        w(md_table(["Run", "Peak VRAM in use", "Installed", "Engaged", "Mean utilization"], rows))
        w("")
        section += 1

    # ── verdict ─────────────────────────────────────────────────────────────
    w(f"## {section}. Verdict")
    w("")
    w(f"The committed workload needs **{req_rate:.2f} images/sec** sustained "
      f"({args.daily_images:,} images ÷ {args.window_hours:g} h).")
    w("")
    rows = []
    for r in runs:
        meets = r["sustained"] >= req_rate
        rows.append([
            f"**{r['label']}**",
            f"{r['sustained']:.2f}/s" if r["sustained"] else "none qualified",
            f"{r['sustained'] / req_rate:.1f}×" if r["sustained"] and req_rate else "—",
            ("✔ meets the requirement" if meets else "✖ does not meet it"),
        ])
    w(md_table(["Run", "Sustained rate (SLA-clean)", "Headroom", "Against the requirement"], rows))
    w("")

    single_gpu_runs = [r for r in runs
                       if str((r["entry"] or {}).get("tensor_parallel_size")) == "1"]
    best_single = max((r["sustained"] for r in single_gpu_runs), default=0.0)

    if best_single >= req_rate > 0:
        w(f"> **One GPU meets the committed workload.** The best single-GPU configuration "
          f"sustained {best_single:.2f} images/sec against a requirement of "
          f"{req_rate:.2f} — {best_single / req_rate:.1f}× headroom. The honest case for "
          f"the second GPU is therefore **burst capacity, model-swap headroom and "
          f"failover**, not steady-state capacity. State it that way: a reviewer who "
          f"checks the arithmetic will reach this conclusion whether or not the report "
          f"does, and it is a stronger position than an overstated one.")
    elif best_single > 0:
        w(f"> **One GPU does not meet the committed workload.** The best single-GPU "
          f"configuration sustained {best_single:.2f} images/sec against a requirement of "
          f"{req_rate:.2f}. That is a direct, measured capacity argument for the second "
          f"GPU — provided the single-GPU run had `max_concurrent` raised, so the "
          f"shortfall is genuinely the hardware and not the semaphore.")
    else:
        w("> **No single-GPU configuration produced an SLA-clean sustained rate**, so the "
          "comparison cannot yet support a hardware conclusion in either direction. The "
          "most likely cause is that `max_concurrent` was left at its deployed value, "
          "which caps the batch regardless of how many GPUs are present.")
    w("")
    w("### A note on what this comparison can and cannot show")
    w("")
    w("This measures **throughput**. It does not measure accuracy, and the two trade "
      "against each other through `mm_processor_kwargs.max_pixels`: fewer vision tokens "
      "per image is faster and cheaper, and at some point it stops reading your documents "
      "correctly. A configuration that wins here and loses on extraction accuracy is not "
      "the better configuration. Settle accuracy on your own documents first, then use "
      "this to choose among the options that pass.")

    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="'label:path' — repeat, in the order the changes were made")
    ap.add_argument("--out", default="./report-compare")
    ap.add_argument("--sla-p95",      type=float, default=30.0)
    ap.add_argument("--daily-images", type=int,   default=24000)
    ap.add_argument("--window-hours", type=float, default=10.0)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if ":" not in spec:
            raise SystemExit(f"--run needs 'label:path', got '{spec}'")
        label, _, path = spec.rpartition(":")
        runs.append(load_run(label.strip(), Path(path.strip()), args.sla_p95))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    args.chart_rate = "chart_rate_compare.png"
    args.chart_attr = "chart_attribution.png"
    args.chart_vram = "chart_vram_compare.png"

    chart_rate_compare(out / args.chart_rate, runs)
    chart_attribution(out / args.chart_attr, runs, "best_rate",
                      "best delivered images / second",
                      "Where the throughput gain actually came from")
    args.chart_vram_ok = chart_vram_compare(out / args.chart_vram, runs)

    (out / "COMPARISON.md").write_text(build(runs, args, out))
    (out / "comparison.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [{k: v for k, v in r.items() if k != "by_scenario"} for r in runs],
    }, indent=2, default=str))

    print(f"\nComparison written to {out / 'COMPARISON.md'}")
    for r in runs:
        e = r["entry"] or {}
        print(f"  {r['label']:<32} TP={e.get('tensor_parallel_size', '?')} "
              f"c={e.get('max_concurrent', '?'):<4} "
              f"best={r['best_rate']:.2f}/s sustained={r['sustained']:.2f}/s")


if __name__ == "__main__":
    main()
