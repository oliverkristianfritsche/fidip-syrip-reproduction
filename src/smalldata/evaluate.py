#!/usr/bin/env python3
"""Run the upstream test script on one checkpoint and score its predictions.

    from smalldata.evaluate import evaluate_checkpoint
    record = evaluate_checkpoint(p, cfg, model, split, data_dir, gt_ann, work, out)

evaluate_checkpoint runs tools/test_adaptive_model.py as a subprocess with absolute paths and
DEBUG.DEBUG off, rescores the results json with pycocotools (the same COCOeval call as upstream),
copies the predictions to <out>, writes <out>/metrics.json and <out>/metrics.csv, and returns the
record. The upstream output, with a feature dump of about 100 MB, stays under <work>.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from smalldata.cocoeval import STAT_NAMES, score_keypoints

REPO = Path(__file__).resolve().parents[2]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo=REPO):
    """Short commit of the repository, or the COMMIT file of a pushed copy, or empty."""
    marker = repo / "COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_tool(cmd, cwd, log_path, env=None):
    """Run a command, stream its output to the console and to log_path, return the exit code."""
    print("run:", " ".join(str(c) for c in cmd), flush=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered child output, so the job log shows progress line by line instead of in late chunks.
    env = dict(env if env is not None else os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def write_record(out, record):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record.keys()))
        w.writeheader()
        w.writerow(record)


def evaluate_checkpoint(p, cfg, model, split, data_dir, gt_ann, work, out, workers=4, gpus="(0,)",
                        seed=0, extra=None):
    cfg, model, work, out = Path(cfg), Path(model), Path(work), Path(out)
    tool = p.upstream / "tools" / "test_adaptive_model.py"
    for label, path in [("config", cfg), ("model", model), ("annotation file", Path(gt_ann)),
                        ("upstream test script", tool)]:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, tool, "--cfg", cfg, "--dataDir", data_dir,
        "--modelDir", work / "upstream_output", "--logDir", work / "upstream_log",
        "TEST.MODEL_FILE", model, "TEST.USE_GT_BBOX", "True", "DATASET.TEST_SET", split,
        "DEBUG.DEBUG", "False", "WORKERS", str(workers), "GPUS", gpus,
    ]
    t0 = time.time()
    rc = run_tool(cmd, cwd=p.upstream, log_path=work / "upstream_stdout.log",
                  env=dict(os.environ, PYTHONHASHSEED=str(seed)))
    if rc != 0:
        sys.exit(f"upstream test script failed with exit code {rc}. See {work / 'upstream_stdout.log'}")
    hits = sorted((work / "upstream_output").rglob(f"keypoints_{split}_results_*.json"))
    if not hits:
        sys.exit(f"no results json under {work / 'upstream_output'}")
    dt_json = hits[-1]
    stats = score_keypoints(gt_ann, dt_json)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dt_json, out / f"predictions_{split}.json")
    with open(gt_ann, encoding="utf-8") as f:
        n_images = len(json.load(f)["images"])
    cfg_label = str(cfg.relative_to(REPO)) if cfg.is_relative_to(REPO) else str(cfg)
    record = {
        "model_path": str(model), "model_md5": md5(model), "split": split, "n_images": n_images,
        "cfg": cfg_label, "use_gt_bbox": True, "seed": seed, "commit": git_head(),
        "upstream_commit": p.upstream_commit(), "work": str(work), "elapsed_s": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    record.update(stats)
    record.update(extra or {})
    write_record(out, record)
    print("\n" + "  ".join(f"{k} {stats[k]:.2f}" for k in STAT_NAMES))
    print(f"wrote {out / 'metrics.csv'}")
    return record
