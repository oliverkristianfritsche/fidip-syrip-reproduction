#!/usr/bin/env python3
"""Check the environment for smalldata-project: torch, GPU, the upstream FiDIP code, and paths.

    python scripts/check_env.py                  # full check, needs a GPU
    python scripts/check_env.py --no-gpu         # skip the GPU stages, for a login node
    python scripts/check_env.py --out <folder>   # also write env_check.json to <folder>

Stages:
  1. Report python, torch, torchvision, CUDA, and the visible GPU.
  2. Resolve the upstream clone and compare its commit with configs/paths.yaml.
  3. Import the upstream modules that training and evaluation need, with the prebuilt nms extension.
  4. Build the adaptive HRNet-W48 model from the baseline config and run one forward pass on the GPU.
  5. Report which data files and weights exist under the project root. A missing file is a
     warning in this part and an error in Part 2.

The script stops with exit code 1 at the first failed stage. The Part 1 check passes when
stages 1 to 4 pass on a cluster GPU node.
"""
import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.paths import Paths  # noqa: E402

BASELINE_CFG = Path(__file__).resolve().parents[1] / "configs" / "hrnet_w48_384x288_infant_baseline.yaml"
# Only packages that upstream lib/ and tools/ import, plus the project's own yaml and tensorboard.
PACKAGES = ["numpy", "cv2", "yacs", "json_tricks", "pycocotools", "yaml", "tensorboard", "tensorboardX", "dsntnn"]
UPSTREAM_MODULES = [
    "config", "models", "models.adaptive_pose_hrnet", "dataset", "dataset.syrip",
    "core.function", "core.loss", "core.evaluate", "utils.utils", "utils.transforms", "nms.nms",
]
PATH_KEYS = [
    "data.annotations.train_pre_infant", "data.annotations.train_infant", "data.annotations.validate_infant",
    "data.images.train_pre_infant", "data.images.train_infant", "data.images.validate_infant",
    "weights.hrnet_fidip", "weights.mobile_fidip", "weights.coco_w48_384x288",
]


def stage(name):
    print(f"\n----- {name}")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="skip the GPU stages")
    ap.add_argument("--out", default="", help="folder for env_check.json")
    a = ap.parse_args()
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "node": platform.node()}

    stage("1. versions")
    import torch
    import torchvision

    report["python"] = platform.python_version()
    report["torch"] = torch.__version__
    report["torchvision"] = torchvision.__version__
    report["cuda_build"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    print(f"python {report['python']}  torch {report['torch']}  torchvision {report['torchvision']}  "
          f"cuda build {report['cuda_build']}")
    if a.no_gpu:
        print("GPU stages skipped")
    elif not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    else:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        print(f"gpu {report['gpu']}  {report['gpu_mem_gb']} GB")

    stage("2. upstream clone")
    p = Paths.load()
    report["host_kind"] = p.host
    report["root"] = str(p.root)
    report["upstream"] = str(p.upstream)
    print(f"host {p.host}  root {p.root}")
    print(f"upstream {p.upstream}")
    if not (p.upstream / "lib").is_dir():
        fail("upstream lib/ is missing. Clone the pinned commit locally, or run scripts/push_upstream.sh.")
    found, pinned = p.upstream_commit(), p.pinned_commit()
    report["upstream_commit"] = found
    if found != pinned:
        fail(f"upstream commit {found or 'unknown'} differs from the pin {pinned}")
    print(f"commit {found} matches the pin")

    stage("3. imports")
    p.add_upstream_lib()
    for name in PACKAGES + UPSTREAM_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # name the module and the reason, then stop
            fail(f"import {name}: {type(e).__name__}: {e}")
        print(f"ok  {name} {getattr(mod, '__version__', '')}".rstrip())
    report["imports"] = "ok"

    stage("4. model forward pass")
    from config import cfg, update_config
    import models

    if not BASELINE_CFG.is_file():
        fail(f"missing config {BASELINE_CFG}")
    args = SimpleNamespace(cfg=str(BASELINE_CFG), opts=[], modelDir="", logDir="", dataDir=str(p.root))
    update_config(cfg, args)
    model_p, model_d = models.adaptive_pose_hrnet.get_adaptive_pose_net(cfg, is_train=False)
    report["pose_params_m"] = round(sum(t.numel() for t in model_p.parameters()) / 1e6, 2)
    report["domain_params_m"] = round(sum(t.numel() for t in model_d.parameters()) / 1e6, 2)
    print(f"adaptive_pose_hrnet built: pose branch {report['pose_params_m']} M parameters, "
          f"domain branch {report['domain_params_m']} M parameters")
    if a.no_gpu:
        print("forward pass skipped")
    else:
        width, height = cfg.MODEL.IMAGE_SIZE
        x = torch.randn(1, 3, height, width, device="cuda")
        model_p = model_p.cuda().eval()
        with torch.no_grad():
            out = model_p(x)
        outs = out if isinstance(out, (tuple, list)) else [out]
        shapes = [tuple(t.shape) for t in outs if hasattr(t, "shape")]
        report["forward_shapes"] = shapes
        print(f"input {tuple(x.shape)} -> outputs {shapes}")
        want = (cfg.MODEL.NUM_JOINTS, cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0])
        if not any(len(s) == 4 and tuple(s[1:]) == want for s in shapes):
            fail(f"no output has the heatmap shape (1, {want[0]}, {want[1]}, {want[2]})")
        print(f"one output has the heatmap shape (1, {want[0]}, {want[1]}, {want[2]})")

    stage("5. data and weights under the root")
    missing = []
    for key in PATH_KEYS:
        path = p.abs(key)
        present = path.exists()
        print(f"{'ok     ' if present else 'missing'} {key}: {path}")
        if not present:
            missing.append(key)
    report["missing_paths"] = missing
    if missing:
        print(f"warning: {len(missing)} paths are missing. Part 2 stages them.")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "env_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out / 'env_check.json'}")
    print("\nPASS: environment check")


if __name__ == "__main__":
    main()
