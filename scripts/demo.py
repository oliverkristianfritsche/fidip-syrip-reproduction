#!/usr/bin/env python3
"""Run one FiDIP checkpoint on one image and draw the pose.

    python scripts/demo.py --image <file.jpg>                                  # released hrnet_fidip.pth, whole image as the box
    python scripts/demo.py --image <file.jpg> --box 120 40 300 400 --model <file.pth>
    python scripts/demo.py --image <test image> --annotation validate_infant   # box from the SyRIP annotation of that image

Stages:
  1. Resolve the config, the checkpoint, and the image. Stop on a missing path.
  2. Box to center and scale the upstream way: the aspect ratio of MODEL.IMAGE_SIZE, pixel_std 200,
     enlargement 1.25. Affine crop to MODEL.IMAGE_SIZE, ImageNet normalization.
  3. Forward pass, the flip test when TEST.FLIP_TEST is on, DARK post-processing
     (upstream core.inference.get_final_preds).
  4. Write <out>.jpg with the skeleton and <out>.json with 17 keypoints [x, y, score], the box, and
     the provenance (model md5, config, commits).

The preprocessing and the post-processing equal the evaluation path of the upstream test script, so
with --annotation the keypoints of a test image equal the evaluation's predictions for that image.
Flags: --cfg --model --image --box --annotation --out --no-flip --device --threshold.
--model takes a key from configs/paths.yaml (hrnet_fidip, mobile_fidip, coco_w48_384x288) or a path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.evaluate import git_head, md5  # noqa: E402
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "hrnet_w48_384x288_infant_baseline.yaml"
PIXEL_STD = 200
ENLARGE = 1.25
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


def resolve_model(p, arg):
    if arg in p.get("weights"):
        return p.abs(f"weights.{arg}")
    return Path(arg).resolve()


def box_from_annotation(ann_file, image_name):
    """The first person box of the image in a COCO annotation file, as [x, y, w, h]."""
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)
    ids = [img["id"] for img in coco["images"] if Path(img["file_name"]).name == image_name]
    if not ids:
        sys.exit(f"{image_name} is not in {ann_file}")
    boxes = [a["bbox"] for a in coco["annotations"] if a["image_id"] == ids[0]]
    if not boxes:
        sys.exit(f"no annotation for image id {ids[0]} in {ann_file}")
    return [float(v) for v in boxes[0]], ids[0]


def box_to_center_scale(box, aspect_ratio):
    """Center and scale as lib/dataset/syrip.py computes them from a box."""
    x, y, w, h = box
    center = np.array([x + (w - 1) * 0.5, y + (h - 1) * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / PIXEL_STD, h / PIXEL_STD], dtype=np.float32) * ENLARGE
    return center, scale


def load_pose_net(cfg, model_path, device):
    import torch
    import models  # upstream

    model_p, _ = models.adaptive_pose_hrnet.get_adaptive_pose_net(cfg, is_train=False)
    state = torch.load(str(model_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    result = model_p.load_state_dict(state, strict=False)
    if result.missing_keys:
        sys.exit(f"{len(result.missing_keys)} model tensors are missing from {model_path}, "
                 f"first: {result.missing_keys[0]}")
    if result.unexpected_keys:
        print(f"ignored {len(result.unexpected_keys)} extra tensors in the checkpoint")
    return model_p.to(device).eval()


def draw(image_bgr, keypoints, threshold):
    import cv2

    out = image_bgr.copy()
    thick = max(2, int(round(min(out.shape[:2]) / 250)))
    for a, b in SKELETON:
        if keypoints[a, 2] >= threshold and keypoints[b, 2] >= threshold:
            pa = (int(round(keypoints[a, 0])), int(round(keypoints[a, 1])))
            pb = (int(round(keypoints[b, 0])), int(round(keypoints[b, 1])))
            cv2.line(out, pa, pb, (0, 200, 255), thick, cv2.LINE_AA)
    for j, (x, y, s) in enumerate(keypoints):
        if s < threshold:
            continue
        color = (255, 128, 0) if j in LEFT else (0, 0, 255)
        cv2.circle(out, (int(round(x)), int(round(y))), thick + 1, color, -1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--model", default="hrnet_fidip", help="weights key from configs/paths.yaml, or a .pth path")
    ap.add_argument("--image", required=True)
    ap.add_argument("--box", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="person box; default the whole image")
    ap.add_argument("--annotation", default="", help="SyRIP split whose annotation gives the box, for example validate_infant")
    ap.add_argument("--out", default="", help="output stem; default results/demo/<image stem>_<model stem>")
    ap.add_argument("--no-flip", action="store_true", help="skip the flip test even when the config turns it on")
    ap.add_argument("--device", default="", help="cuda or cpu; default cuda when available")
    ap.add_argument("--threshold", type=float, default=0.2, help="lowest score drawn")
    a = ap.parse_args()

    # ----- stage 1: paths
    p = Paths.load()
    cfg_path = Path(a.cfg).resolve()
    model_path = resolve_model(p, a.model)
    image_path = Path(a.image).resolve()
    checks = [("config", cfg_path), ("model", model_path), ("image", image_path)]
    if a.annotation:
        checks.append(("annotation file", p.abs(f"data.annotations.{a.annotation}")))
    for label, path in checks:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")
    model_stem = Path(a.model).stem if a.model.endswith(".pth") else a.model
    out = Path(a.out) if a.out else REPO / "results" / "demo" / f"{image_path.stem}_{model_stem}"
    out.parent.mkdir(parents=True, exist_ok=True)

    p.add_upstream_lib()
    import cv2
    import torch
    from config import cfg  # upstream
    from core.inference import get_final_preds  # upstream
    from utils.transforms import flip_back, get_affine_transform  # upstream

    cfg.defrost()
    cfg.merge_from_file(str(cfg_path))
    cfg.freeze()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    flip = bool(cfg.TEST.FLIP_TEST) and not a.no_flip
    print(f"model {model_path}\nimage {image_path}\ndevice {device}, flip test {flip}, blur kernel {cfg.TEST.BLUR_KERNEL}")

    # ----- stage 2: the crop
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image_bgr is None:
        sys.exit(f"cannot read {image_path}")
    height, width = image_bgr.shape[:2]
    image_id = None
    if a.annotation:
        box, image_id = box_from_annotation(p.abs(f"data.annotations.{a.annotation}"), image_path.name)
    elif a.box:
        box = list(a.box)
    else:
        box = [0.0, 0.0, float(width), float(height)]
    image_size = np.array(cfg.MODEL.IMAGE_SIZE, dtype=np.float32)  # width, height
    center, scale = box_to_center_scale(box, float(image_size[0] / image_size[1]))
    data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if cfg.DATASET.COLOR_RGB else image_bgr
    trans = get_affine_transform(center, scale, 0, image_size)
    crop = cv2.warpAffine(data, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    tensor = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(tensor.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)

    # ----- stage 3: forward, flip test, DARK
    model = load_pose_net(cfg, model_path, device)
    t0 = time.time()
    with torch.no_grad():
        _, output = model(x)
        if isinstance(output, list):
            output = output[-1]
        if flip:
            _, flipped = model(torch.flip(x, [3]))
            if isinstance(flipped, list):
                flipped = flipped[-1]
            flipped = flip_back(flipped.cpu().numpy(), FLIP_PAIRS)
            output = (output + torch.from_numpy(flipped.copy()).to(device)) * 0.5
    preds, maxvals = get_final_preds(cfg, output.cpu().numpy(), np.array([center]), np.array([scale]))
    forward_s = round(time.time() - t0, 3)
    keypoints = np.concatenate([preds[0], maxvals[0]], axis=1)  # 17 x [x, y, score]

    # ----- stage 4: outputs
    cv2.imwrite(str(out.with_suffix(".jpg")), draw(image_bgr, keypoints, a.threshold))
    record = {
        "image": str(image_path), "image_size": [width, height], "image_id": image_id,
        "box": box, "box_source": "annotation" if a.annotation else ("flag" if a.box else "whole image"),
        "center": center.tolist(), "scale": scale.tolist(),
        "keypoints": [round(float(v), 3) for v in keypoints.reshape(-1)],
        "named": [{"name": n, "x": round(float(k[0]), 2), "y": round(float(k[1]), 2), "score": round(float(k[2]), 4)}
                  for n, k in zip(NAMES, keypoints)],
        "mean_score": round(float(keypoints[:, 2].mean()), 4),
        "flip_test": flip, "blur_kernel": int(cfg.TEST.BLUR_KERNEL), "device": device, "forward_s": forward_s,
        "model": str(model_path), "model_md5": md5(model_path),
        "cfg": str(cfg_path.relative_to(REPO)) if cfg_path.is_relative_to(REPO) else str(cfg_path),
        "commit": git_head(), "upstream_commit": p.upstream_commit(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    for n, k in zip(NAMES, keypoints):
        print(f"{n:15s} {k[0]:8.1f} {k[1]:8.1f}  {k[2]:.3f}")
    print(f"mean score {record['mean_score']}  forward {forward_s} s  wrote {out.with_suffix('.jpg')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
