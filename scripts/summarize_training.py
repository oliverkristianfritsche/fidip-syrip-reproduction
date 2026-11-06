#!/usr/bin/env python3
"""Tabulate training runs: one row per run, then mean and spread per config.

    python scripts/summarize_training.py                       # reads results/training/*/results/summary.json
    python scripts/summarize_training.py --out results/training/summary.csv

Each run folder under results/training/ holds the files pulled from the cluster with
scripts/pull_results.sh: results/summary.json, results/epochs.csv, results/final/metrics.csv, and
results/best/metrics.csv. The table reports the final-epoch AP as the primary number and the
test-selected model_best AP beside it. Standard library only.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "results" / "training"), help="folder with one subfolder per run")
    ap.add_argument("--out", default="", help="summary csv; default <runs>/summary.csv")
    a = ap.parse_args()
    runs = Path(a.runs)
    out = Path(a.out) if a.out else runs / "summary.csv"
    rows = []
    for summary in sorted(runs.glob("*/results/summary.json")) + sorted(runs.glob("*/summary.json")):
        s = json.load(open(summary, encoding="utf-8"))
        run = summary.parent.parent.name if summary.parent.name == "results" else summary.parent.name
        # The group is the run name without its repeat suffix, so a run on another data root with the
        # same config file forms its own group.
        group = re.sub(r"_rep\d+$", "", run)
        cfg = Path(s.get("cfg", "")).stem
        rows.append({
            "run": run, "config": group, "config_file": cfg, "base_path": s.get("base_path", ""),
            "final_AP": s.get("final_AP"), "final_AR": s.get("final_AR"),
            "best_AP": s.get("best_AP"), "best_epoch": s.get("best_epoch_by_validation"),
            "epochs": s.get("epochs_trained"), "train_s": s.get("train_s"), "commit": s.get("commit"),
            "upstream_commit": (s.get("upstream_commit") or "")[:12], "smoke": bool(s.get("max_samples")),
        })
    if not rows:
        sys.exit(f"no summary.json under {runs}")
    rows = [r for r in rows if not r["smoke"]]
    if not rows:
        sys.exit(f"only smoke runs under {runs}; nothing to tabulate")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'run':<48} {'final AP':>8} {'best AP':>8} {'best ep':>7} {'epochs':>6}")
    for r in rows:
        print(f"{r['run']:<48} {r['final_AP']!s:>8} {r['best_AP']!s:>8} {r['best_epoch']!s:>7} {r['epochs']!s:>6}")
    print()
    by_cfg = {}
    for r in rows:
        if r["final_AP"] is not None:
            by_cfg.setdefault(r["config"], []).append(float(r["final_AP"]))
    agg = []
    for cfg, vals in sorted(by_cfg.items()):
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        agg.append({"config": cfg, "n": len(vals), "final_AP_mean": round(mean, 2),
                    "final_AP_sd": round(sd, 2) if sd == sd else "", "final_AP_min": min(vals), "final_AP_max": max(vals)})
        print(f"{cfg:<44} n={len(vals)} final AP mean {mean:.2f} sd {sd:.2f} range {min(vals)} to {max(vals)}")
    agg_path = out.with_name(out.stem + "_by_config.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"wrote {out} and {agg_path}")


if __name__ == "__main__":
    main()
