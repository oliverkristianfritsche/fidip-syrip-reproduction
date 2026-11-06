#!/usr/bin/env python3
"""Check the SyRIP data and the weight files under the project root, and write the Part 2 tables.

    python scripts/prepare_data.py                        # check the root for this host
    python scripts/prepare_data.py --base-path <root>     # check another root
    python scripts/prepare_data.py --out results/data_check  # where the tables go (default)

Stages:
  1. For each split in configs/paths.yaml: parse the annotation file, count images and
     annotations, and confirm that every listed image file exists on disk.
  2. Compare the image counts with counts: in configs/paths.yaml.
  3. For each weight file: record size and md5.
  4. For each image: compare the stored pixel size with the width and height in the annotation
     file, and check that the labels fit the annotated frame. A mismatch means the labels belong
     to another version of the image. image_size_check.csv lists every mismatch.
  5. Write data_check.csv, weights.csv, and image_size_check.csv to --out.

The script writes the tables first, then stops with exit code 1 when a file is missing or a
count differs from the registry. A different count means a wrong download or a wrong registry.
Change the registry only with a source for the new number, and record the change in AGENT.md.
Size mismatches are reported, not fatal: the released training set has 46 of them, and the
reproduction trains on the data as released.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLITS = ["train_pre_infant", "train_infant", "validate_infant"]


def upstream_file_name(split, image_id):
    """The image name that upstream SYRIPDataset.image_path_from_index builds."""
    if split == "validate_infant":
        return f"test{image_id}.jpg"
    return f"train{image_id:05d}.jpg"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_split(p, split):
    ann = p.abs(f"data.annotations.{split}")
    images = p.abs(f"data.images.{split}")
    expected = int(p.get(f"counts.{split}"))
    row = {"split": split, "annotation_file": str(ann), "n_images": "", "n_annotations": "",
           "expected_images": expected, "n_files_on_disk": "", "n_missing_files": "", "status": ""}
    if not ann.is_file():
        row["status"] = "missing annotation file"
        return row
    if not images.is_dir():
        row["status"] = "missing images folder"
        return row
    with open(ann, encoding="utf-8") as f:
        tree = json.load(f)
    imgs = tree.get("images", [])
    row["n_images"] = len(imgs)
    row["n_annotations"] = len(tree.get("annotations", []))
    row["n_files_on_disk"] = sum(1 for q in images.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    missing = 0
    for im in imgs:
        name = im.get("file_name") or upstream_file_name(split, int(im["id"]))
        if not (images / Path(name).name).is_file():
            missing += 1
    row["n_missing_files"] = missing
    if missing:
        row["status"] = f"{missing} image files missing"
    elif len(imgs) != expected:
        row["status"] = f"count {len(imgs)} differs from registry {expected}"
    else:
        row["status"] = "ok"
    return row


def check_weight(p, key):
    path = p.abs(f"weights.{key}")
    row = {"key": key, "path": str(path), "bytes": "", "md5": "", "status": ""}
    if not path.is_file():
        row["status"] = "missing"
        return row
    row["bytes"] = path.stat().st_size
    row["md5"] = md5(path)
    row["status"] = "ok" if row["bytes"] > 0 else "empty"
    return row


def check_image_sizes(p, split):
    """Rows for images whose stored size differs from the annotation, with the frame the labels fit."""
    from PIL import Image  # heavy import stays inside the function

    ann = p.abs(f"data.annotations.{split}")
    images = p.abs(f"data.images.{split}")
    if not ann.is_file() or not images.is_dir():
        return []
    with open(ann, encoding="utf-8") as f:
        tree = json.load(f)
    labels = {}
    for an in tree.get("annotations", []):
        labels.setdefault(an["image_id"], []).append(an)
    rows = []
    for im in tree.get("images", []):
        path = images / Path(im.get("file_name") or upstream_file_name(split, int(im["id"]))).name
        if not path.is_file():
            continue
        with Image.open(path) as f:
            stored = f.size
        json_size = (int(im["width"]), int(im["height"]))
        if stored == json_size:
            continue
        xs, ys = [0.0], [0.0]
        for an in labels.get(im["id"], []):
            x, y, w, h = an["bbox"]
            xs.append(x + w)
            ys.append(y + h)
            kp = an.get("keypoints", [])
            xs += [kp[i] for i in range(0, len(kp), 3) if kp[i + 2] > 0]
            ys += [kp[i + 1] for i in range(0, len(kp), 3) if kp[i + 2] > 0]
        frame = "labels fit the annotated size" if max(xs) <= json_size[0] and max(ys) <= json_size[1] \
            else "labels exceed the annotated size"
        rows.append({"split": split, "file_name": path.name, "is_synthetic": im.get("is_synthetic", ""),
                     "json_width": json_size[0], "json_height": json_size[1],
                     "stored_width": stored[0], "stored_height": stored[1],
                     "max_label_x": round(max(xs)), "max_label_y": round(max(ys)), "labels_frame": frame})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default="", help="project root; default from configs/paths.yaml")
    ap.add_argument("--out", default=str(REPO / "results" / "data_check"), help="folder for the tables")
    a = ap.parse_args()
    p = Paths.load()
    if a.base_path:
        p.root = Path(a.base_path)
    print(f"root {p.root}")

    data_rows = [check_split(p, s) for s in SPLITS]
    weight_rows = [check_weight(p, k) for k in p.get("weights")]
    size_rows = [r for s in SPLITS for r in check_image_sizes(p, s)]
    out = Path(a.out)
    write_csv(out / "data_check.csv", data_rows)
    write_csv(out / "weights.csv", weight_rows)
    size_header = ["split", "file_name", "is_synthetic", "json_width", "json_height", "stored_width", "stored_height",
                   "max_label_x", "max_label_y", "labels_frame"]
    write_csv(out / "image_size_check.csv", size_rows or [dict.fromkeys(size_header, "")])

    for r in data_rows:
        print(f"{r['split']:<18} images={r['n_images']!s:<6} annotations={r['n_annotations']!s:<6} "
              f"on_disk={r['n_files_on_disk']!s:<6} {r['status']}")
    for r in weight_rows:
        print(f"{r['key']:<18} {r['bytes']!s:<11} {r['md5']:<34} {r['status']}")
    for s in SPLITS:
        n = sum(1 for r in size_rows if r["split"] == s)
        print(f"{s:<18} stored size differs from the annotation for {n} images"
              + (": see image_size_check.csv" if n else ""))
    print(f"wrote {out / 'data_check.csv'}, {out / 'weights.csv'}, and {out / 'image_size_check.csv'}")

    bad = [r for r in data_rows + weight_rows if r["status"] != "ok"]
    if bad:
        sys.exit(f"FAIL: {len(bad)} items are not ok")
    print("PASS: data and weights match the registry")


if __name__ == "__main__":
    main()
