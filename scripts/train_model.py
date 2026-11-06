#!/usr/bin/env python3
"""Train FiDIP with the upstream trainer from the COCO DarkPose weights, then evaluate the result.

    python scripts/train_model.py --cfg configs/hrnet_w48_384x288_infant_fidip.yaml      # the FiDIP config
    python scripts/train_model.py --cfg configs/hrnet_w48_384x288_infant_baseline.yaml   # the guide config, plain fine-tuning
    python scripts/train_model.py --cfg configs/hrnet_w48_384x288_infant_fidip.yaml \\
        --epochs 1 --pre-epochs 1 --max-samples 16 --out results/_smoke_train           # smoke test

Stages:
  1. Resolve paths. Stop on a missing config, start checkpoint, annotation file, or images folder.
  2. With --max-samples, build a reduced root with that many images in each of train_pre_infant,
     train_infant, and validate_infant.
  3. Run upstream tools/train_adaptive_model_hrnet.py as a subprocess from the work folder, with
     MODEL.INIT_WEIGHTS False, TRAIN.CHECKPOINT <start weights>, DEBUG.DEBUG False, and absolute
     output folders. The upstream trainer resumes from checkpoint.pth in its output folder when
     AUTO_RESUME is on, so a rerun with the same --work continues. A rerun whose --out already
     holds summary.json stops at once, so a chained job after a finished run does nothing.
  4. Parse the per-epoch validation table from the upstream training log into epochs.csv.
  5. Evaluate final_state.pth (primary, the last epoch) and model_best.pth (secondary, selected by
     upstream on the test split) on --data, into <out>/final/ and <out>/best/, and write summary.json.

Flags: --cfg --model --data --base-path --out --work --seed --max-samples --epochs --pre-epochs --batch-size --workers --gpus.
--model is the start checkpoint: a weights key from configs/paths.yaml (default coco_w48_384x288) or a path.
--seed sets PYTHONHASHSEED only. The upstream trainer sets no seed and runs cudnn.benchmark, so
repeats differ. Checkpoints stay under --work, which defaults to <root>/raw/train_<out name>/ and
never sits in the repository. Smoke runs must use a results/_smoke_* folder, which git ignores.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.cocoeval import STAT_NAMES  # noqa: E402
from smalldata.evaluate import evaluate_checkpoint, git_head, md5, run_tool  # noqa: E402
from smalldata.paths import Paths  # noqa: E402
from smalldata.subset import make_subset_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "hrnet_w48_384x288_infant_fidip.yaml"
TRAIN_SPLITS = ["train_pre_infant", "train_infant"]
# The upstream logger prefixes each line with a timestamp, so the table row is searched, not anchored.
ROW = re.compile(r"\|\s*\S+\s*\|((?:\s*-?\d+(?:\.\d+)?\s*\|){10})\s*$")


def resolve_model(p, arg):
    if arg in p.get("weights"):
        return p.abs(f"weights.{arg}")
    return Path(arg).resolve()


def parse_epoch_table(log_paths):
    """Rows of the upstream validation table, one per epoch, as percentages."""
    rows = []
    for log in sorted(log_paths):
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ROW.search(line.rstrip())
                if m:
                    values = [round(float(v) * 100, 2) for v in m.group(1).replace("|", " ").split()]
                    rows.append(dict(zip(STAT_NAMES, values)))
    return [{"epoch": i + 1, **r} for i, r in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--model", default="coco_w48_384x288", help="start checkpoint: weights key or .pth path")
    ap.add_argument("--data", default="validate_infant", help="split for the final evaluation")
    ap.add_argument("--base-path", default="", help="project root; default from configs/paths.yaml")
    ap.add_argument("--out", default="", help="result folder; default results/training/<cfg stem>_<model>")
    ap.add_argument("--work", default="", help="run folder with checkpoints; default <root>/raw/train_<out name>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help="images per split for a smoke test")
    ap.add_argument("--epochs", type=int, default=0, help="override TRAIN.END_EPOCH")
    ap.add_argument("--pre-epochs", type=int, default=-1, help="override TRAIN.PRE_EPOCH")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="override both batch sizes, for a smoke test on a small GPU; never for a real run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", default="(0,)", help="GPUS value for the upstream config")
    a = ap.parse_args()

    # ----- stage 1: paths
    p = Paths.load()
    if a.base_path:
        p.root = Path(a.base_path)
    cfg = Path(a.cfg).resolve()
    start = resolve_model(p, a.model)
    trainer = p.upstream / "tools" / "train_adaptive_model_hrnet.py"
    checks = [("config", cfg), ("start checkpoint", start), ("upstream trainer", trainer)]
    for split in TRAIN_SPLITS + [a.data]:
        checks += [(f"{split} annotations", p.abs(f"data.annotations.{split}")),
                   (f"{split} images", p.abs(f"data.images.{split}"))]
    for label, path in checks:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")
    model_stem = Path(a.model).stem if a.model.endswith(".pth") else a.model
    out = Path(a.out) if a.out else REPO / "results" / "training" / f"{cfg.stem}_{model_stem}"
    if a.max_samples and "_smoke" not in str(out):
        sys.exit("a --max-samples run must write to a results/_smoke_* folder")
    work = Path(a.work) if a.work else p.root / "raw" / f"train_{out.name}"
    if (out / "summary.json").is_file():
        print(f"already done: {out / 'summary.json'} exists. Remove it to run again.")
        return
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"root {p.root}\nupstream {p.upstream}\nstart {start}\ncfg {cfg}\nout {out}\nwork {work}")

    # ----- stage 2: optional subset
    data_dir, gt_ann = p.root, p.abs(f"data.annotations.{a.data}")
    if a.max_samples:
        counts = {s: a.max_samples for s in TRAIN_SPLITS + [a.data]}
        data_dir, anns = make_subset_root(p, counts, work)
        gt_ann = anns[a.data]

    # ----- stage 3: upstream training
    opts = ["MODEL.INIT_WEIGHTS", "False", "TRAIN.CHECKPOINT", str(start), "DEBUG.DEBUG", "False",
            "WORKERS", str(a.workers), "GPUS", a.gpus]
    if a.epochs:
        opts += ["TRAIN.END_EPOCH", str(a.epochs)]
    if a.pre_epochs >= 0:
        opts += ["TRAIN.PRE_EPOCH", str(a.pre_epochs)]
    if a.batch_size:
        if not a.max_samples:
            sys.exit("--batch-size is for smoke tests only; pass --max-samples with it")
        opts += ["TRAIN.BATCH_SIZE_PER_GPU", str(a.batch_size), "TRAIN.PRE_BATCH_SIZE_PER_GPU", str(a.batch_size),
                 "TEST.BATCH_SIZE_PER_GPU", str(a.batch_size)]
    cmd = [sys.executable, trainer, "--cfg", cfg, "--dataDir", data_dir,
           "--modelDir", work / "upstream_output", "--logDir", work / "upstream_log"] + opts
    t0 = time.time()
    rc = run_tool(cmd, cwd=work, log_path=work / "upstream_stdout.log",
                  env=dict(os.environ, PYTHONHASHSEED=str(a.seed)))
    if rc != 0:
        sys.exit(f"upstream trainer failed with exit code {rc}. See {work / 'upstream_stdout.log'}")
    train_s = round(time.time() - t0, 1)

    # ----- stage 4: the per-epoch table
    final_dirs = sorted(q.parent for q in (work / "upstream_output").rglob("final_state.pth"))
    if not final_dirs:
        sys.exit(f"no final_state.pth under {work / 'upstream_output'}")
    final_dir = final_dirs[-1]
    epochs = parse_epoch_table(final_dir.glob("*_train.log"))
    with open(out / "epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + STAT_NAMES)
        w.writeheader()
        w.writerows(epochs)
    print(f"{len(epochs)} validation rows -> {out / 'epochs.csv'}")

    # ----- stage 5: evaluation of the last and the selected epoch
    summary = {
        "cfg": str(cfg.relative_to(REPO)) if cfg.is_relative_to(REPO) else str(cfg),
        "base_path": str(p.root),
        "start_model": a.model, "start_path": str(start), "start_md5": md5(start),
        "epochs_trained": len(epochs), "epochs_flag": a.epochs, "pre_epochs_flag": a.pre_epochs,
        "batch_size_flag": a.batch_size, "max_samples": a.max_samples, "seed": a.seed, "train_s": train_s,
        "commit": git_head(),
        "upstream_commit": p.upstream_commit(), "work": str(work), "final_dir": str(final_dir),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for tag, name in [("final", "final_state.pth"), ("best", "model_best.pth")]:
        ckpt = final_dir / name
        if not ckpt.is_file():
            print(f"no {name}, skipping the {tag} evaluation")
            continue
        rec = evaluate_checkpoint(p, cfg, ckpt, a.data, data_dir, gt_ann, work / f"eval_{tag}", out / tag,
                                  workers=a.workers, gpus=a.gpus, seed=a.seed,
                                  extra={"checkpoint": tag, "max_samples": a.max_samples})
        summary[f"{tag}_AP"] = rec["AP"]
        summary[f"{tag}_AR"] = rec["AR"]
    if epochs:
        best = max(epochs, key=lambda r: r["AP"])
        summary["best_epoch_by_validation"] = best["epoch"]
        summary["best_epoch_AP"] = best["AP"]
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfinal_state AP {summary.get('final_AP')}  model_best AP {summary.get('best_AP')}  "
          f"wrote {out / 'summary.json'}")
    if a.max_samples:
        print("smoke test: the numbers above are not a result")


if __name__ == "__main__":
    main()
