#!/usr/bin/env python3
"""Find copies and near-copies of the test images in the other image folders (a leakage check).

    python scripts/find_near_duplicates.py
    python scripts/find_near_duplicates.py --sheet ../_data/smalldata-project/analysis/near_duplicates.jpg --sheet-max 30

Each image gets a 256-bit difference hash: grayscale, resized to 17 by 16, one bit per horizontal
gradient. The hash survives resizing and re-encoding. The distance of two images is the number of
differing bits, the lower of the plain and the mirrored comparison. Unrelated photos sit near 80;
the same photo sits at 0; frames a moment apart in one video sit near 5 to 30. Frames of the same
video with a different pose can sit higher, so a clean result bounds the leakage from below only.

For every test image and every other folder, the closest image is written to
results/analysis/near_duplicates_test100.csv with: the distance, whether the match is mirrored,
whether the match is listed in that folder's annotation file (a listed image is used), and whether
the bytes are identical. The folders: train_infant (all files on disk), train_pre_infant,
and the selection split when it exists. --sheet writes a side-by-side contact sheet of the pairs up to --sheet-max; it
holds dataset images, so keep it beside the data, not in the repository.

--embed adds a second signal that sees the scene, not the frame: ImageNet ResNet-50 features
(global average pool, the mean of the plain and the mirrored image, unit length) and their cosine.
Frames of one video share the room, the light, and the clothes, so they stay close when the pose
changes. It compares the test images with the real images the project uses (the listed real
train_infant images, and the selection split when it exists) and, for information, the unlisted train_infant files.
It writes results/analysis/near_duplicates_test100_embed.csv (the --top closest per test image and
folder, with the hash distance) and, under --review-dir, sheets of every test image beside its
closest used image, 25 pairs a sheet, for a review by eye. --expand test3.jpg,... draws the --top
closest used images of the named test images, to find every frame of a confirmed video.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.paths import Paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUFFIXES = {".jpg", ".jpeg", ".png"}


def dhash(img):
    g = np.asarray(img.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hash_folder(folder):
    names, plain, mirror = [], [], []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUFFIXES:
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            plain.append(dhash(im))
            mirror.append(dhash(im.transpose(Image.FLIP_LEFT_RIGHT)))
        names.append(path.name)
    return names, np.array(plain), np.array(mirror)


def listed_names(annotation_file):
    with open(annotation_file, encoding="utf-8") as f:
        return {im["file_name"] for im in json.load(f)["images"]}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_sheet(rows, folders, test_dir, out, max_distance):
    tiles = []
    for r in rows:
        if r["distance"] > max_distance:
            continue
        pair = []
        for path in (test_dir / r["test"], folders[r["folder"]][0] / r["match"]):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((220, 220))
                pair.append(im)
        tiles.append((f"{r['test']} | {r['folder']}/{r['match']} d{r['distance']}", pair))
    if not tiles:
        return 0
    w, h = 460, 250
    sheet = Image.new("RGB", (2 * w, h * ((len(tiles) + 1) // 2)), "white")
    for i, (label, (a, b)) in enumerate(tiles):
        x, y = (i % 2) * w, (i // 2) * h
        sheet.paste(a, (x, y + 20))
        sheet.paste(b, (x + 230, y + 20))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return len(tiles)


def embed_folder(folder, names, model, device):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for name in names:
            with Image.open(folder / name) as im:
                x = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
            x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
            x = (torch.cat([x, x.flip(-1)]) - mean) / std
            f = model(x).mean(0)
            feats.append((f / f.norm()).cpu().numpy())
    return np.array(feats)


def tile_sheet(tiles, out):
    w, h = 460, 250
    sheet = Image.new("RGB", (2 * w, h * ((len(tiles) + 1) // 2)), "white")
    for i, (label, paths) in enumerate(tiles):
        x, y = (i % 2) * w, (i // 2) * h
        for k, path in enumerate(paths):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((220, 220))
                sheet.paste(im, (x + 230 * k, y + 20))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def embed_scan(a, p, select_root, test_dir):
    import torch
    import torchvision

    train_ann = json.load(open(p.abs("data.annotations.train_infant"), encoding="utf-8"))
    real_listed = sorted(im["file_name"] for im in train_ann["images"] if not im.get("is_synthetic"))
    source = {im["file_name"]: im.get("image_source", "") for im in train_ann["images"]}
    listed = {im["file_name"] for im in train_ann["images"]}
    train_dir = p.abs("data.images.train_infant")
    unlisted = sorted(q.name for q in train_dir.iterdir() if q.suffix.lower() in SUFFIXES and q.name not in listed)
    select_dir = select_root / "data/syrip/images/validate_infant"
    folders = {  # name: (image folder, file names, used by the project)
        "train_infant_real": (train_dir, real_listed, True),
        "train_infant_unlisted": (train_dir, unlisted, False),
    }
    if select_dir.is_dir():  # the selection split is optional
        folders["select"] = (select_dir, sorted(listed_names(
            select_root / "data/syrip/annotations/person_keypoints_validate_infant.json")), True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    t_names = sorted(q.name for q in test_dir.iterdir() if q.suffix.lower() in SUFFIXES)
    t_feat = embed_folder(test_dir, t_names, model, device)
    t_hash = {n: hash_folder_files(test_dir, [n])[0] for n in t_names}

    rows, best_used = [], {}
    for folder, (images, names, used) in folders.items():
        feat = embed_folder(images, names, model, device)
        cos = t_feat @ feat.T
        hashes = hash_folder_files(images, names)
        for i, test in enumerate(t_names):
            order = np.argsort(-cos[i])[: a.top]
            for rank, j in enumerate(order, 1):
                plain, mirror = t_hash[test]
                dist = int(min((plain != hashes[j][0]).sum(), (mirror != hashes[j][0]).sum()))
                row = {"test": test, "folder": folder, "used": int(used), "rank": rank, "match": names[j],
                       "cosine": round(float(cos[i, j]), 4), "hash_distance": dist,
                       "image_source": source.get(names[j], "") if folder.startswith("train") else ""}
                rows.append(row)
                if used and rank == 1 and (test not in best_used or row["cosine"] > best_used[test][0]["cosine"]):
                    best_used[test] = (row, images)
        top1 = cos.max(1)
        print(f"{folder}: {len(names)} images; closest cosine per test image: median {np.median(top1):.3f}, "
              f"above 0.9: {(top1 > 0.9).sum()}, above 0.8: {(top1 > 0.8).sum()}, above 0.7: {(top1 > 0.7).sum()}")

    out = Path(a.out).with_name(Path(a.out).stem + "_embed.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    if a.review_dir:
        review = Path(a.review_dir)
        ordered = sorted(best_used.values(), key=lambda v: -v[0]["cosine"])
        for s in range(0, len(ordered), 25):
            tiles = [(f"{r['test']} | {r['folder']}/{r['match']} c{r['cosine']:.3f} h{r['hash_distance']}",
                      (test_dir / r["test"], images / r["match"])) for r, images in ordered[s:s + 25]]
            tile_sheet(tiles, review / f"review_{s // 25 + 1}.jpg")
        print(f"review sheets in {review}: {len(ordered)} pairs, sorted by cosine")
        if a.expand:
            tiles = []
            for test in a.expand.split(","):
                for r in rows:
                    if r["test"] == test and r["used"]:
                        images = folders[r["folder"]][0]
                        tiles.append((f"{test} | {r['folder']}/{r['match']} #{r['rank']} c{r['cosine']:.3f}",
                                      (test_dir / test, images / r["match"])))
            tile_sheet(tiles, review / "expand.jpg")
            print(f"expand sheet: {len(tiles)} pairs")


def hash_folder_files(folder, names):
    out = []
    for name in names:
        with Image.open(folder / name) as im:
            im = im.convert("RGB")
            out.append((dhash(im), dhash(im.transpose(Image.FLIP_LEFT_RIGHT))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="validate_infant", help="the test split")
    ap.add_argument("--select-root", default="", help="selection split root, optional; default <root>/variants/select")
    ap.add_argument("--out", default=str(REPO / "results/analysis/near_duplicates_test100.csv"))
    ap.add_argument("--sheet", default="", help="contact sheet path, kept outside the repository")
    ap.add_argument("--sheet-max", type=int, default=30)
    ap.add_argument("--embed", action="store_true", help="add the ResNet-50 scene signal")
    ap.add_argument("--top", type=int, default=5, help="closest images kept per test image and folder")
    ap.add_argument("--review-dir", default="", help="review sheets, kept outside the repository")
    ap.add_argument("--expand", default="", help="comma-separated test images to draw with their --top matches")
    a = ap.parse_args()
    p = Paths.load()
    select_root = Path(a.select_root) if a.select_root else Path(p.root) / "variants/select"
    if a.embed:
        embed_scan(a, p, select_root, p.abs(f"data.images.{a.data}"))
        return
    folders = {
        "train_infant": (p.abs("data.images.train_infant"), p.abs("data.annotations.train_infant")),
        "train_pre_infant": (p.abs("data.images.train_pre_infant"), p.abs("data.annotations.train_pre_infant")),
    }
    if (select_root / "data/syrip/images/validate_infant").is_dir():  # the selection split is optional
        folders["select"] = (select_root / "data/syrip/images/validate_infant",
                             select_root / "data/syrip/annotations/person_keypoints_validate_infant.json")
    test_dir = p.abs(f"data.images.{a.data}")
    t_names, t_plain, t_mirror = hash_folder(test_dir)
    print(f"{a.data}: {len(t_names)} images")

    rows = []
    for folder, (images, annotations) in folders.items():
        names, plain, _ = hash_folder(images)
        listed = listed_names(annotations)
        d_plain = (t_plain[:, None, :] != plain[None, :, :]).sum(-1)
        d_mirror = (t_mirror[:, None, :] != plain[None, :, :]).sum(-1)
        dist = np.minimum(d_plain, d_mirror)
        best, arg = dist.min(1), dist.argmin(1)
        for i, test in enumerate(t_names):
            j = arg[i]
            rows.append({
                "test": test, "folder": folder, "match": names[j], "distance": int(best[i]),
                "mirrored": int(d_mirror[i, j] < d_plain[i, j]),
                "listed": int(names[j] in listed),
                "identical_bytes": int(best[i] == 0 and md5(test_dir / test) == md5(images / names[j])),
            })
        close = best <= a.sheet_max
        print(f"{folder}: {len(names)} files, {len(listed)} listed; closest distance per test image: "
              f"median {np.median(best):.0f}; at most 10: {(best <= 10).sum()}, "
              f"at most {a.sheet_max}: {close.sum()}, of those listed: "
              f"{sum(names[arg[i]] in listed for i in np.flatnonzero(close))}")

    rows.sort(key=lambda r: (r["distance"], r["test"], r["folder"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        if r["distance"] <= a.sheet_max:
            print(r)
    if a.sheet:
        n = write_sheet(rows, folders, test_dir, Path(a.sheet), a.sheet_max)
        print(f"sheet {a.sheet}: {n} pairs")


if __name__ == "__main__":
    main()
