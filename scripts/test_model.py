#!/usr/bin/env python3
"""Evaluate one FiDIP checkpoint on a SyRIP split with the upstream test script, then score it.

    python scripts/test_model.py                                        # released hrnet_fidip.pth on validate_infant
    python scripts/test_model.py --model <file.pth> --data validate_infant --seed 0
    python scripts/test_model.py --max-samples 8 --out results/_smoke_eval   # smoke test

Stages:
  1. Resolve the root, the upstream clone, the config, the checkpoint, and the split. Stop on a missing path.
  2. With --max-samples, build a reduced root under --work/subset_root/ that links the images folder.
  3. Run upstream tools/test_adaptive_model.py and rescore its predictions (src/smalldata/evaluate.py).
     Results: metrics.csv, metrics.json, and a copy of the predictions in --out.

Flags: --cfg --model --data --base-path --out --work --seed --max-samples --workers --gpus.
--model takes a key from configs/paths.yaml (hrnet_fidip, mobile_fidip, coco_w48_384x288) or a path.
--out holds the small result files. --work holds the upstream output, including a feature dump of
about 100 MB, so it defaults to <root>/raw/eval_<out name>/ and never sits in the repository.
Smoke runs must use a results/_smoke_* folder, which git ignores.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.evaluate import evaluate_checkpoint  # noqa: E402
from smalldata.paths import Paths  # noqa: E402
from smalldata.subset import make_subset_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "hrnet_w48_384x288_infant_baseline.yaml"


def resolve_model(p, arg):
    if arg in p.get("weights"):
        return p.abs(f"weights.{arg}")
    return Path(arg).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--model", default="hrnet_fidip", help="weights key from configs/paths.yaml, or a .pth path")
    ap.add_argument("--data", default="validate_infant", help="SyRIP split")
    ap.add_argument("--base-path", default="", help="project root; default from configs/paths.yaml")
    ap.add_argument("--out", default="", help="result folder; default results/testing/<model>_<split>_gtbox")
    ap.add_argument("--work", default="", help="upstream output folder; default <root>/raw/eval_<out name>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help="evaluate the first N images only (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", default="(0,)", help="GPUS value for the upstream config")
    a = ap.parse_args()

    p = Paths.load()
    if a.base_path:
        p.root = Path(a.base_path)
    cfg = Path(a.cfg).resolve()
    model = resolve_model(p, a.model)
    gt_ann = p.abs(f"data.annotations.{a.data}")
    images = p.abs(f"data.images.{a.data}")
    for label, path in [("config", cfg), ("model", model), ("annotation file", gt_ann), ("images folder", images)]:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")
    model_stem = Path(a.model).stem if a.model.endswith(".pth") else a.model
    out = Path(a.out) if a.out else REPO / "results" / "testing" / f"{model_stem}_{a.data}_gtbox"
    if a.max_samples and "_smoke" not in str(out):
        sys.exit("a --max-samples run must write to a results/_smoke_* folder")
    work = Path(a.work) if a.work else p.root / "raw" / f"eval_{out.name}"
    print(f"root {p.root}\nupstream {p.upstream}\nmodel {model}\nout {out}\nwork {work}")

    data_dir = p.root
    if a.max_samples:
        data_dir, anns = make_subset_root(p, {a.data: a.max_samples}, work)
        gt_ann = anns[a.data]

    evaluate_checkpoint(p, cfg, model, a.data, data_dir, gt_ann, work, out, workers=a.workers, gpus=a.gpus,
                        seed=a.seed, extra={"model": a.model, "max_samples": a.max_samples})
    if a.max_samples:
        print("smoke test: the numbers above are not a result")


if __name__ == "__main__":
    main()
