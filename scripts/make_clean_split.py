#!/usr/bin/env python3
"""Build the cleaned training variant: train_infant without the images whose labels sit in another frame.

    python scripts/make_clean_split.py                      # builds <root>/variants/clean/ for this host
    python scripts/make_clean_split.py --base-path <root>   # another root
    python scripts/make_clean_split.py --check              # list what would be dropped, write nothing

The variant root mirrors the data layout the upstream loader expects, so the trainer runs on it
through --base-path <root>/variants/clean with no change to upstream or to the driver:

    variants/clean/data/syrip/annotations/person_keypoints_train_infant.json   rewritten, dropped images removed
    variants/clean/data/syrip/annotations/<other annotation files>            links to the originals
    variants/clean/data/syrip/images/<split>                                   links to the original folders
    variants/clean/data/models                                                 link to the weights folder

An image is dropped when its stored pixel size differs from the width and height in the
annotation file. The dropped list goes to results/data_check/clean_split_dropped.csv. A link is a
symlink; when the file system refuses symlinks (a Windows mount), the folder is copied instead.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT = "train_infant"


def link_or_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        return "kept"
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return "linked"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied"


def main():
    from PIL import Image  # heavy import stays inside the function

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default="", help="project root; default from configs/paths.yaml")
    ap.add_argument("--out", default=str(REPO / "results" / "data_check" / "clean_split_dropped.csv"))
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    p = Paths.load()
    if a.base_path:
        p.root = Path(a.base_path)
    src_ann = p.abs(f"data.annotations.{SPLIT}")
    src_images = p.abs(f"data.images.{SPLIT}")
    for label, path in [("annotation file", src_ann), ("images folder", src_images)]:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")

    with open(src_ann, encoding="utf-8") as f:
        tree = json.load(f)
    dropped, keep = [], []
    for im in tree["images"]:
        with Image.open(src_images / Path(im["file_name"]).name) as img:
            stored = img.size
        if stored != (int(im["width"]), int(im["height"])):
            dropped.append({"file_name": im["file_name"], "image_id": im["id"], "is_synthetic": im.get("is_synthetic", ""),
                            "json_width": im["width"], "json_height": im["height"],
                            "stored_width": stored[0], "stored_height": stored[1]})
        else:
            keep.append(im)
    ids = {int(im["id"]) for im in keep}
    n_syn = sum(1 for im in keep if im.get("is_synthetic"))
    print(f"{SPLIT}: {len(tree['images'])} images, drop {len(dropped)}, keep {len(keep)} "
          f"({len(keep) - n_syn} real, {n_syn} synthetic)")
    if a.check:
        for d in dropped[:10]:
            print("  drop", d["file_name"], d["json_width"], "x", d["json_height"], "vs stored",
                  d["stored_width"], "x", d["stored_height"])
        return

    root = p.root / "variants" / "clean"
    syrip = root / "data" / "syrip"
    (syrip / "annotations").mkdir(parents=True, exist_ok=True)
    (syrip / "images").mkdir(parents=True, exist_ok=True)
    tree["images"] = keep
    tree["annotations"] = [an for an in tree["annotations"] if int(an["image_id"]) in ids]
    with open(syrip / "annotations" / src_ann.name, "w", encoding="utf-8") as f:
        json.dump(tree, f)
    for other in p.abs("data.syrip").joinpath("annotations").glob("*.json"):
        if other.name != src_ann.name:
            print(f"  annotations/{other.name}: {link_or_copy(other, syrip / 'annotations' / other.name)}")
    for split_dir in p.abs("data.syrip").joinpath("images").iterdir():
        print(f"  images/{split_dir.name}: {link_or_copy(split_dir, syrip / 'images' / split_dir.name)}")
    models = p.abs("weights.hrnet_fidip").parent
    print(f"  data/models: {link_or_copy(models, root / 'data' / 'models')}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dropped[0].keys()) if dropped else ["file_name"])
        w.writeheader()
        w.writerows(dropped)
    print(f"variant root {root}\nwrote {out}")


if __name__ == "__main__":
    main()
