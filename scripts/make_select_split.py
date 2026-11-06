#!/usr/bin/env python3
"""Build the selection split: the real Test500 images outside Test100, for checkpoint selection.

    python scripts/make_select_split.py                      # builds <root>/variants/select/ for this host
    python scripts/make_select_split.py --check              # report what is kept and dropped, write nothing
    python scripts/make_select_split.py --base-path <root>   # another root

Source: the original SyRIP release under data/syrip_original/, the validate500 annotation file and
its images (configs/paths.yaml, data.original). An image is dropped when its file name is in Test100,
when its bytes equal a Test100 image, when its stored size differs from the annotation, when its
file name does not follow test<id>.jpg (the upstream loader derives the name from the id), or when
its source video (image_source) also appears in Test100. --keep-shared-sources keeps the last group.

The variant mirrors the layout the upstream loader expects, under the split name validate_infant,
because the loader hard-codes the split names:

    variants/select/data/syrip/annotations/person_keypoints_validate_infant.json
    variants/select/data/syrip/images/validate_infant/test<id>.jpg              links to the original files

The per-image table goes to results/data_check/select_split.csv. A link is a symlink; when the file
system refuses symlinks (a Windows mount), the file is copied.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT = "validate_infant"
FIELDS = ["file_name", "image_id", "image_source", "json_width", "json_height", "stored_width", "stored_height",
          "status", "reason"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        return "kept"
    try:
        os.symlink(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main():
    from PIL import Image  # heavy import stays inside the function

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default="", help="project root; default from configs/paths.yaml")
    ap.add_argument("--out", default=str(REPO / "results" / "data_check" / "select_split.csv"))
    ap.add_argument("--keep-shared-sources", action="store_true",
                    help="keep images whose source video also appears in Test100")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    p = Paths.load()
    if a.base_path:
        p.root = Path(a.base_path)
    src_ann = p.abs("data.original.validate500_annotations")
    src_images = p.abs("data.original.validate_images")
    test_ann = p.abs(f"data.annotations.{SPLIT}")
    test_images = p.abs(f"data.images.{SPLIT}")
    for label, path in [("Test500 annotation file", src_ann), ("Test500 images", src_images),
                        ("Test100 annotation file", test_ann), ("Test100 images", test_images)]:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")

    # ----- stage 1: what Test100 holds
    with open(test_ann, encoding="utf-8") as f:
        test = json.load(f)
    test_names = {im["file_name"] for im in test["images"]}
    test_sources = {im.get("image_source", "") for im in test["images"]} - {""}
    test_md5 = {md5(test_images / im["file_name"]): im["file_name"] for im in test["images"]}
    print(f"Test100: {len(test_names)} images, {len(test_sources)} distinct image_source values")

    # ----- stage 2: judge every Test500 image
    with open(src_ann, encoding="utf-8") as f:
        tree = json.load(f)
    rows, keep = [], []
    for im in tree["images"]:
        name, source = im["file_name"], im.get("image_source", "")
        row = {"file_name": name, "image_id": im["id"], "image_source": source,
               "json_width": im["width"], "json_height": im["height"], "stored_width": "", "stored_height": ""}
        reason = ""
        path = src_images / name
        if name in test_names:
            reason = "in Test100"
        elif not path.is_file():
            reason = "missing file"
        elif name != f"test{im['id']}.jpg":
            reason = "file name does not match the image id"
        else:
            twin = test_md5.get(md5(path))
            with Image.open(path) as img:
                row["stored_width"], row["stored_height"] = img.size
            if twin:
                reason = f"bytes equal Test100 image {twin}"
            elif (row["stored_width"], row["stored_height"]) != (int(im["width"]), int(im["height"])):
                reason = "stored size differs from the annotation"
            elif source and source in test_sources and not a.keep_shared_sources:
                reason = "source video shared with Test100"
        row["status"] = "drop" if reason else "keep"
        row["reason"] = reason
        rows.append(row)
        if not reason:
            keep.append(im)
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    print(f"Test500: {len(tree['images'])} images, keep {len(keep)}, drop {len(rows) - len(keep)}")
    for reason, n in reasons.most_common():
        print(f"  {n:4d} {reason}")
    if a.check:
        return

    # ----- stage 3: the variant root
    root = p.root / "variants" / "select"
    syrip = root / "data" / "syrip"
    images = syrip / "images" / SPLIT
    (syrip / "annotations").mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    ids = {int(im["id"]) for im in keep}
    tree["images"] = keep
    tree["annotations"] = [an for an in tree["annotations"] if int(an["image_id"]) in ids]
    with open(syrip / "annotations" / f"person_keypoints_{SPLIT}.json", "w", encoding="utf-8") as f:
        json.dump(tree, f)
    how = Counter(link_or_copy(src_images / im["file_name"], images / im["file_name"]) for im in keep)
    print(f"images/{SPLIT}: " + ", ".join(f"{n} {k}" for k, n in how.items()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"variant root {root}: {len(keep)} images, {len(tree['annotations'])} annotations\nwrote {out}")


if __name__ == "__main__":
    main()
