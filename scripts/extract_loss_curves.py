#!/usr/bin/env python3
"""Per-epoch training and Test100 losses of the reproduction runs, from the upstream training logs.

    python scripts/extract_loss_curves.py                  # the nine runs of the three training configs

Inputs, per run folder <root>/raw/train_<config>_rep<n>/train/ (on the cluster, or a local copy of these files):
  - losses_P.npy: the pose loss of each training iteration, as the upstream trainer logs it;
  - losses_D.npy: the domain-classifier loss of each iteration, warm-up iterations first;
  - upstream_log/.../events.out.tfevents.*: the Test100 loss after each epoch (TensorBoard tag valid_loss). Some runs
    also write one subfolder per AP metric; the script reads the one file that holds valid_loss.
Output: results/training/<config>_rep<n>/results/losses.csv with one row per epoch.

The logged pose loss is the heatmap loss minus LAMBDA times the classifier loss. The script adds LAMBDA times the
classifier loss of the same iteration back, so train_heatmap_loss compares with the Test100 loss. Upstream logs the
classifier loss of the classifier step; the pose step sees the classifier after that step. On the FiDIP config the two differ
by about 2 percent of the heatmap loss on average. Upstream evaluates Test100 after each epoch; these numbers
describe training and never choose an epoch.
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from smalldata.paths import Paths  # noqa: E402

CONFIGS = ["hrnet_w48_384x288_infant_baseline", "hrnet_w48_384x288_infant_fidip",
           "hrnet_w48_384x288_infant_fidip_paper"]


def curves(run_dir, lam):
    train = run_dir / "train"
    loss_p = np.load(train / "losses_P.npy")
    loss_d = np.load(train / "losses_D.npy")[-len(loss_p):]  # drop the warm-up iterations
    events = []
    for path in glob.glob(str(train / "upstream_log" / "**" / "events.out.tfevents.*"), recursive=True):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        if "valid_loss" in acc.Tags()["scalars"]:
            events.append(acc)
    if len(events) != 1:
        sys.exit(f"expected one TensorBoard file with valid_loss under {train}, found {len(events)}")
    acc = events[0]
    test_loss = [e.value for e in sorted(acc.Scalars("valid_loss"), key=lambda e: e.step)]
    epochs = len(test_loss)
    if len(loss_p) % epochs:
        sys.exit(f"{run_dir.name}: {len(loss_p)} iterations do not split into {epochs} epochs")
    per = len(loss_p) // epochs
    rows = []
    for e in range(epochs):
        sl = slice(e * per, (e + 1) * per)
        logged = float(loss_p[sl].mean())
        rows.append({"epoch": e + 1, "train_loss_logged": round(logged, 7),
                     "train_heatmap_loss": round(logged + lam * float(loss_d[sl].mean()), 7),
                     "test100_loss": round(float(test_loss[e]), 7)})
    return rows, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="folder that holds the train_<config>_rep<n> run folders; default <root>/raw")
    a = ap.parse_args()
    runs = Path(a.runs) if a.runs else Paths.load().root / "raw"
    for config in CONFIGS:
        with open(REPO / "configs" / f"{config}.yaml", encoding="utf-8") as f:
            lam = float(yaml.safe_load(f)["TRAIN"].get("LAMBDA", 0.0))
        for n in (1, 2, 3):
            name = f"{config}_rep{n}"
            rows, per = curves(runs / f"train_{name}", lam)
            out = REPO / "results" / "training" / name / "results" / "losses.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"{name}: LAMBDA {lam}, {per} iterations per epoch, train heatmap loss "
                  f"{rows[0]['train_heatmap_loss']:.6f} -> {rows[-1]['train_heatmap_loss']:.6f}, Test100 loss "
                  f"{rows[0]['test100_loss']:.6f} -> {rows[-1]['test100_loss']:.6f}, min at epoch "
                  f"{min(rows, key=lambda r: r['test100_loss'])['epoch']}; wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
