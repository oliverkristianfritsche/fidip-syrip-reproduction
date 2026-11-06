#!/usr/bin/env python3
"""Build a reduced data root for smoke tests: fewer images per split, same layout.

    from smalldata.subset import make_subset_root
    root, anns = make_subset_root(p, {"train_infant": 8, "validate_infant": 4}, work)

The root is <work>/subset_root/. It holds reduced annotation files under data/syrip/annotations/
and copies of the kept images under data/syrip/images/<split>/, so the upstream loaders read it
through --dataDir <root>. Copies instead of a link: a Windows mount under WSL may refuse symlinks,
and a subset holds few images. A split with an is_synthetic flag keeps half real and half
synthetic images, so the balanced sampler still sees both classes.
"""
import json
import shutil
from pathlib import Path


def _by_id(images):
    return sorted(images, key=lambda im: int(im["id"]))


def _pick(images, n):
    if not images or "is_synthetic" not in images[0]:
        return _by_id(images)[:n]
    real = _by_id(im for im in images if not im.get("is_synthetic"))
    syn = _by_id(im for im in images if im.get("is_synthetic"))
    half = n // 2
    keep = real[:half] + syn[:n - half]
    if len(keep) < n:  # one class is short, so fill from the other
        pool = real[half:] + syn[n - half:]
        keep += pool[:n - len(keep)]
    return keep


def make_subset_root(p, counts, work):
    root = Path(work) / "subset_root"
    syrip = root / "data" / "syrip"
    (syrip / "annotations").mkdir(parents=True, exist_ok=True)
    anns = {}
    for split, n in counts.items():
        src = p.abs(f"data.annotations.{split}")
        src_images = p.abs(f"data.images.{split}")
        with open(src, encoding="utf-8") as f:
            tree = json.load(f)
        keep = _pick(tree["images"], int(n))
        ids = {int(im["id"]) for im in keep}
        tree["images"] = keep
        tree["annotations"] = [an for an in tree["annotations"] if int(an["image_id"]) in ids]
        dst = syrip / "annotations" / src.name
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(tree, f)
        dst_images = syrip / "images" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        for im in keep:
            target = dst_images / Path(im["file_name"]).name
            if not target.is_file():
                shutil.copy2(src_images / Path(im["file_name"]).name, target)
        anns[split] = dst
        print(f"subset {split}: {len(keep)} images, {len(tree['annotations'])} annotations -> {dst}")
    return root, anns
