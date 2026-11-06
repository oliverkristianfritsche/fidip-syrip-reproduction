#!/usr/bin/env python3
"""Paired per-image comparison of two groups of training runs on the same test images.

    python scripts/compare_runs.py --treatment hrnet_w48_384x288_infant_fidip_clean --control hrnet_w48_384x288_infant_fidip
    python scripts/compare_runs.py --treatment <group> --control <group> --checkpoint best
    python scripts/compare_runs.py --treatment <group> --control <group> --checkpoint best --control-checkpoint final

Stages:
  1. For each run in each group, score the predictions of the chosen checkpoint (final or best)
     against the test annotations and keep one OKS per image.
  2. Average the per-image OKS over the runs of each group, pair the two groups by image, and
     report the mean difference, a bootstrap confidence interval (seed 0), the counts of images
     that improved or got worse by more than 0.01, and an exact two-sided sign test.
  3. Write <out>/paired_<treatment>_vs_<control>_<checkpoint>.csv with one row per image.

Groups are run-folder names under results/training/ without the repeat suffix. Scoring needs
pycocotools; everything else is the standard library.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.cocoeval import per_image_oks  # noqa: E402
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def group_runs(runs_dir, group, checkpoint):
    paths = sorted(runs_dir.glob(f"{group}_rep*/results/{checkpoint}/predictions_validate_infant.json"))
    if not paths:
        sys.exit(f"no runs for group {group} with checkpoint {checkpoint} under {runs_dir}")
    return paths


def sign_test_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--checkpoint", default="final", choices=["final", "best"], help="checkpoint of the treatment group")
    ap.add_argument("--control-checkpoint", default="", choices=["", "final", "best"],
                    help="checkpoint of the control group; default the same. With the same group on both sides, "
                         "this pairs the selected checkpoint against the final epoch of the same runs")
    ap.add_argument("--runs", default=str(REPO / "results" / "training"))
    ap.add_argument("--out", default=str(REPO / "results" / "training"))
    ap.add_argument("--data", default="validate_infant")
    a = ap.parse_args()
    p = Paths.load()
    gt = p.abs(f"data.annotations.{a.data}")
    if not gt.is_file():
        sys.exit(f"missing annotation file: {gt}")
    runs_dir = Path(a.runs)
    control_ck = a.control_checkpoint or a.checkpoint
    groups = {}
    for key, name, ck in (("t", a.treatment, a.checkpoint), ("c", a.control, control_ck)):
        per_run = [per_image_oks(gt, q) for q in group_runs(runs_dir, name, ck)]
        images = sorted(per_run[0])
        groups[key] = {i: statistics.mean(r[i] for r in per_run) for i in images}
        print(f"{name} ({ck}): {len(per_run)} runs, mean OKS over images {statistics.mean(groups[key].values()):.4f}")
    t, c = groups["t"], groups["c"]
    images = sorted(t)
    diffs = [t[i] - c[i] for i in images]
    wins = sum(d > 0.01 for d in diffs)
    losses = sum(d < -0.01 for d in diffs)
    ties = len(diffs) - wins - losses
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    p_sign = sign_test_p(wins, losses)
    print(f"paired over {len(images)} images, checkpoint {a.checkpoint}: mean OKS difference {statistics.mean(diffs):+.4f} "
          f"(95% bootstrap {lo:+.4f} to {hi:+.4f}), median {statistics.median(diffs):+.4f}")
    print(f"images improved by more than 0.01: {wins}, worse: {losses}, within 0.01: {ties}; exact sign test p = {p_sign:.3f}")
    suffix = a.checkpoint if control_ck == a.checkpoint else f"{a.checkpoint}_vs_{control_ck}"
    out = Path(a.out) / f"paired_{a.treatment}_vs_{a.control}_{suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", f"oks_{a.treatment}_{a.checkpoint}", f"oks_{a.control}_{control_ck}", "difference"])
        for i in images:
            w.writerow([i, round(t[i], 4), round(c[i], 4), round(t[i] - c[i], 4)])
        w.writerow(["mean", round(statistics.mean(t.values()), 4), round(statistics.mean(c.values()), 4),
                    round(statistics.mean(diffs), 4)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
