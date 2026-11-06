#!/usr/bin/env python3
"""Figures and a per-image table for the course report. No GPU, no weights.

    python scripts/make_report_figures.py                 # writes results/figures/*.pdf and *.png

Inputs:
  - results/training/<run>/results/: the per-epoch AP (epochs.csv) and losses (losses.csv) of the nine reproduction
    runs (three configs, three runs each), and the Test100 predictions of their final epochs;
  - results/baseline/results/hrnet_fidip/: the Test100 predictions of the released checkpoint (cluster baseline job);
  - the Test100 annotations and images through configs/paths.yaml.
"Our best run" is the reproduction run with the highest final-epoch Test100 AP. The figures show it next to the
released checkpoint; it defines no reported number.
Outputs in results/figures/: one PDF (for LaTeX) and one PNG per figure, and per_image_oks.csv.
The script also prints the numbers that the report cites, with the left/right swap counts:
  - a mirrored image: the OKS rises by more than 0.1 when every left/right pair is exchanged;
  - a swapped pair: exchanging the two joints raises the sum of their OKS terms by more than 0.2.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.image import imread  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smalldata.oks import JOINTS, PAIRS, joint_terms, oks, swapped  # noqa: E402
from smalldata.paths import Paths  # noqa: E402

# Validated categorical slots 1 and 2 (the pair passes the CVD and normal-vision floors on white).
BLUE, ORANGE = "#2a78d6", "#eb6834"
# The training configs. The paper-settings config holds our best run, so it takes orange. Aqua, violet, orange in
# legend order pass the CVD and normal-vision floors on white; aqua is below 3:1 contrast, so the legend and the
# report table carry the values.
AQUA, VIOLET = "#1baf7a", "#6250d6"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
RELEASED_C, OURS_C = BLUE, ORANGE  # the same model keeps its color in every figure

TRAIN = ROOT / "results" / "training"
RELEASED = ROOT / "results" / "baseline" / "results" / "hrnet_fidip" / "predictions_validate_infant.json"
RELEASED_2024 = ROOT / "results" / "baseline" / "results" / "hrnet_fidip_images2024" / "predictions_validate_infant.json"
TRAINING_CONFIGS = [  # label, config, color, published AP: the two upstream configs and the paper's settings
    ("Guide config", "hrnet_w48_384x288_infant_baseline", AQUA, 92.7),
    ("FiDIP config", "hrnet_w48_384x288_infant_fidip", VIOLET, 93.6),
    ("Paper-settings config", "hrnet_w48_384x288_infant_fidip_paper", ORANGE, 93.6),
]
JOINT_TYPES = ["nose", "eyes", "ears", "shoulders", "elbows", "wrists", "hips", "knees", "ankles"]
POSTURES = ["supine", "prone", "sitting", "standing"]
WRONG = 0.5  # a joint with an OKS term below this is a missed joint in the example figure
TEMPLATE_C = "#c3c2b7"  # the T-pose of the example figure: a recessive neutral under the model colors
TPOSE = np.array([  # front-facing T-pose in COCO joint order; the infant's left has positive x
    (0.0, 1.62), (0.07, 1.70), (-0.07, 1.70), (0.15, 1.66), (-0.15, 1.66),  # nose, eyes, ears
    (0.25, 1.40), (-0.25, 1.40), (0.62, 1.40), (-0.62, 1.40), (0.98, 1.40), (-0.98, 1.40),  # shoulders, elbows, wrists
    (0.17, 0.85), (-0.17, 0.85), (0.19, 0.45), (-0.19, 0.45), (0.20, 0.05), (-0.20, 0.05),  # hips, knees, ankles
])
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "legend.fontsize": 7.5,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 200,
    })


def style(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, color=AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def epoch_values(config, file, column):
    """One list per run of a per-epoch column: epochs.csv (AP) or losses.csv (scripts/extract_loss_curves.py)."""
    curves = []
    for rep in (1, 2, 3):
        with open(TRAIN / f"{config}_rep{rep}" / "results" / file, encoding="utf-8") as f:
            curves.append([float(r[column]) for r in csv.DictReader(f)])
    return curves


def fig_training_curves(out):
    """Per epoch of the three training configs: training heatmap loss, Test100 loss, and Test100 AP. The epoch axis is
    logarithmic, so the 20-epoch upstream configs and the 100-epoch paper-settings config both stay readable."""
    panels = [("Training loss (heatmap MSE)", "losses.csv", "train_heatmap_loss", 1e4),
              ("Test100 loss (heatmap MSE)", "losses.csv", "test100_loss", 1e4),
              ("Test100 AP", "epochs.csv", "AP", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, (title, file, column, scale) in zip(axes, panels):
        style(ax)
        for label, config, color, published in TRAINING_CONFIGS:
            curves = [[v * scale for v in c] for c in epoch_values(config, file, column)]
            x = range(1, len(curves[0]) + 1)
            for c in curves:
                ax.plot(x, c, color=color, linewidth=0.7, alpha=0.35)
            ax.plot(x, [statistics.mean(v) for v in zip(*curves)], color=color, linewidth=1.8, label=label)
            if column == "AP":
                print(f"  {label}: final epochs {[c[-1] for c in curves]}, best epochs {[max(c) for c in curves]}")
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 110)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlabel("epoch (log scale)")
        ax.axvline(40, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)  # LR drop, paper settings
    for value, name in ((93.6, "93.6 paper, FiDIP"), (92.7, "92.7 paper, fine-tuning")):
        axes[2].axhline(value, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        axes[2].text(1.0, value + 0.12, name, va="bottom", ha="left", fontsize=6.5, color=INK2)  # no curve there
    axes[0].set_ylabel("loss (x 1e-4)")
    axes[1].set_ylim(3, 7)
    axes[2].set_ylim(84, 95)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
    save(fig, out, "training_curves")


def best_run():
    """The reproduction run with the highest final-epoch Test100 AP, for display next to the released checkpoint."""
    with open(TRAIN / "summary.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["config"] in {c for _, c, _, _ in TRAINING_CONFIGS}]
    best = max(rows, key=lambda r: float(r["final_AP"]))
    config = next(label for label, c, _, _ in TRAINING_CONFIGS if c == best["config"]).lower().replace("fidip", "FiDIP")
    label = f"our best reproduction run ({config}, run {best['run'][-1]})"
    print(f"  best run {best['run']}: final-epoch AP {best['final_AP']}")
    return TRAIN / best["run"] / "results" / "final" / "predictions_validate_infant.json", label


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {d["image_id"]: d["keypoints"] for d in json.load(f)}


def load_test(paths):
    with open(paths.abs("data.annotations.validate_infant"), encoding="utf-8") as f:
        coco = json.load(f)
    return {im["id"]: im for im in coco["images"]}, {an["image_id"]: an for an in coco["annotations"]}


def per_image_table(images, gts, released, ours, out):
    rows = []
    for i, g in sorted(gts.items()):
        rows.append({"image_id": i, "file": images[i]["file_name"], "posture": images[i]["posture"],
                     "oks_released": round(oks(released[i], g["keypoints"], g["area"]), 4),
                     "oks_ours": round(oks(ours[i], g["keypoints"], g["area"]), 4)})
    with open(out / "per_image_oks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for key in ("oks_released", "oks_ours"):
        values = [r[key] for r in rows]
        print(f"  {key}: mean {statistics.mean(values):.4f}; images above 0.9: {sum(v > 0.9 for v in values)}; "
              f"below 0.8: {sum(v < 0.8 for v in values)}")
    return rows


def swap_counts(name, gts, preds):
    """Mirrored images and swapped left/right pairs; the rules are in the module docstring."""
    mirrored, swaps, pairs = 0, 0, 0
    for img, gt in gts.items():
        pr, area = preds[img], gt["area"]
        mirrored += oks(swapped(pr), gt["keypoints"], area) - oks(pr, gt["keypoints"], area) > 0.1
        terms, terms_sw = joint_terms(pr, gt["keypoints"], area), joint_terms(swapped(pr), gt["keypoints"], area)
        for left, right in PAIRS:
            if terms[left] is None or terms[right] is None:
                continue
            pairs += 1
            swaps += (terms_sw[left] + terms_sw[right]) - (terms[left] + terms[right]) > 0.2
    print(f"  {name}: mirrored images {mirrored}; swapped pairs {swaps} of {pairs} labeled pairs")


def joint_type_means(gts, preds):
    names = {"eye": "eyes", "ear": "ears", "shoulder": "shoulders", "elbow": "elbows", "wrist": "wrists",
             "hip": "hips", "knee": "knees", "ankle": "ankles"}
    acc = {t: [] for t in JOINT_TYPES}
    for img, gt in gts.items():
        for j, term in enumerate(joint_terms(preds[img], gt["keypoints"], gt["area"])):
            if term is not None:
                acc[names.get(JOINTS[j].split("_")[-1], "nose")].append(term)
    return {t: statistics.mean(v) for t, v in acc.items()}


def dot_rows(ax, labels, released, ours, ours_label):
    """Two dots on each row: the released checkpoint and our best run. A white edge keeps close dots apart."""
    for y, key in enumerate(labels):
        ax.scatter([released[key]], [y], s=34, color=RELEASED_C, edgecolor="white", linewidth=0.6, zorder=3,
                   label="released checkpoint, our evaluation" if y == 0 else None)
        ax.scatter([ours[key]], [y], s=34, color=OURS_C, edgecolor="white", linewidth=0.6, zorder=4,
                   label=ours_label if y == 0 else None)


def fig_joint_posture(rows, gts, released, ours, ours_label, out):
    """Left: mean OKS term per joint type. Right: mean OKS per posture. Both models on Test100."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1.1, 1]})
    style(a, "x")
    style(b, "x")
    jr, jo = joint_type_means(gts, released), joint_type_means(gts, ours)
    order = sorted(JOINT_TYPES, key=lambda t: jr[t], reverse=True)
    dot_rows(a, order, jr, jo, ours_label)
    a.set_yticks(range(len(order)))
    a.set_yticklabels(order)
    a.set_xlabel("mean OKS term")
    a.set_title("Per joint type, left and right pooled", loc="left")
    for t in order:
        print(f"  joint {t:<10} released {jr[t]:.4f}  ours {jo[t]:.4f}  diff {jo[t] - jr[t]:+.4f}")
    pr = {p: statistics.mean(r["oks_released"] for r in rows if r["posture"] == p) for p in POSTURES}
    po = {p: statistics.mean(r["oks_ours"] for r in rows if r["posture"] == p) for p in POSTURES}
    counts = {p: sum(r["posture"] == p for r in rows) for p in POSTURES}
    order = sorted(POSTURES, key=lambda p: pr[p], reverse=True)
    dot_rows(b, order, pr, po, ours_label)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([f"{p} (n = {counts[p]})" for p in order])
    b.set_xlabel("mean OKS")
    b.set_title("Per posture", loc="left")
    for p in order:
        print(f"  posture {p:<9} n {counts[p]:>3}  released {pr[p]:.4f}  ours {po[p]:.4f}")
    lo = min(list(jr.values()) + list(jo.values()) + list(pr.values()) + list(po.values())) - 0.01
    for ax in (a, b):
        ax.set_xlim(lo, 1.0)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
    save(fig, out, "joint_posture")


def missed_joints(kps, gt):
    """The joints whose OKS term is below WRONG."""
    return {j for j, t in enumerate(joint_terms(kps, gt["keypoints"], gt["area"])) if t is not None and t < WRONG}


def draw_overlay(ax, kps, color):
    """A predicted skeleton on the photo, in the model's color."""
    xy = [(kps[3 * j], kps[3 * j + 1]) for j in range(17)]
    for a, b in SKELETON:
        ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]], color=color, linewidth=1.3, alpha=0.95)
    ax.scatter([v[0] for v in xy], [v[1] for v in xy], s=9, color=color, edgecolor="white", linewidth=0.4, zorder=4)


def miss_marker(models):
    """Marker style for a joint that the given models miss: the model's color, or half blue and half orange."""
    if models == {"released", "ours"}:
        return dict(markerfacecolor=RELEASED_C, markerfacecoloralt=OURS_C, fillstyle="left")
    return dict(markerfacecolor=RELEASED_C if models == {"released"} else OURS_C)


def draw_tpose(ax, missed):
    """The missed joints on a front-facing T-pose: the infant's right is on the viewer's left. missed maps each joint
    to the set of models that miss it."""
    neck = [(0, 1.40), tuple(TPOSE[0])]  # COCO has no neck bone; this one joins the head to the shoulders
    ax.plot(*zip(*neck), color=TEMPLATE_C, linewidth=2.2, solid_capstyle="round", zorder=1)
    for a, b in SKELETON:
        ax.plot(TPOSE[[a, b], 0], TPOSE[[a, b], 1], color=TEMPLATE_C, linewidth=2.2 if a > 4 else 1.4,
                solid_capstyle="round", zorder=1)
    ax.scatter(TPOSE[:, 0], TPOSE[:, 1], s=7, color=TEMPLATE_C, zorder=2)
    for j, models in missed.items():
        ax.plot(*TPOSE[j], marker="o", markersize=7, markeredgecolor="white", markeredgewidth=0.8, linestyle="",
                zorder=3, **miss_marker(models))
    for x, text, ha in ((-1.08, "right", "left"), (1.08, "left", "right")):
        ax.text(x, 1.94, text, ha=ha, va="top", fontsize=5.8, color=INK2)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.12, 1.98)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


def square_crop(ax, pic, gt):
    """Show the image on a square crop around the person box, without ticks."""
    x, y, w, h = gt["bbox"]
    side = 1.12 * max(w, h)
    cx, cy = x + w / 2, y + h / 2
    ax.imshow(pic)
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy + side / 2, cy - side / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(AXIS)


def fig_examples(rows, images, gts, released, ours, image_dir, out):
    """The two best and the two worst Test100 images of our best run, ranked by its OKS, with both models on each
    photo. Next to each worst image: a T-pose that marks the joints each model misses."""
    ranked = sorted(rows, key=lambda r: r["oks_ours"])
    picks = [(ranked[-1], False), (ranked[-2], False), (ranked[1], True), (ranked[0], True)]  # best two, worst two
    fig = plt.figure(figsize=(6.6, 1.95))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 0.12, 1, 1, 1, 1])  # column 2 separates the two groups
    col, groups, present = 0, [], set()
    for r, worst in picks:
        img = r["image_id"]
        gt = gts[img]
        name = r["file"].replace(".jpg", "")
        ax = fig.add_subplot(gs[0, col])
        square_crop(ax, imread(image_dir / images[img]["file_name"]), gt)
        draw_overlay(ax, released[img], RELEASED_C)
        draw_overlay(ax, ours[img], OURS_C)
        ax.set_title(f"{name}, {r['posture']}", fontsize=7)
        ax.set_xlabel(f"released: OKS {r['oks_released']:.2f}\nours: OKS {r['oks_ours']:.2f}", fontsize=5.9, color=INK,
                      labelpad=3)
        groups.append(ax)
        col += 2 if col == 1 else 1
        miss = {"released": missed_joints(released[img], gt), "ours": missed_joints(ours[img], gt)}
        print(f"  example {name:<7} {r['posture']:<9} released OKS {r['oks_released']:.4f}, missed "
              f"{sorted(JOINTS[j] for j in miss['released'])}; ours OKS {r['oks_ours']:.4f}, missed "
              f"{sorted(JOINTS[j] for j in miss['ours'])}")
        if worst:
            tp = fig.add_subplot(gs[0, col])
            by_joint = {j: {m for m in miss if j in miss[m]} for j in miss["released"] | miss["ours"]}
            present |= {frozenset(v) for v in by_joint.values()}
            draw_tpose(tp, by_joint)
            tp.set_title("missed joints", fontsize=7)
            tp.set_xlabel(f"released: {len(miss['released'])} missed\nours: {len(miss['ours'])} missed", fontsize=5.9,
                          color=INK, labelpad=3)
            groups.append(tp)
            col += 1
    handles = [plt.Line2D([], [], marker="o", markersize=3, color=RELEASED_C, label="released checkpoint"),
               plt.Line2D([], [], marker="o", markersize=3, color=OURS_C, label="our best reproduction run")]
    for models, text in (({"released", "ours"}, "missed by both"), ({"released"}, "missed by the released checkpoint only"),
                         ({"ours"}, "missed by our run only")):
        if frozenset(models) in present:
            handles.append(plt.Line2D([], [], marker="o", markersize=7, markeredgecolor="white", linestyle="",
                                      label=text, **miss_marker(models)))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.92), w_pad=0.6)
    for first, last, text in ((0, 1, "Two best images of our run"), (2, 5, "Two worst images of our run")):
        a, b = groups[first].get_position(), groups[last].get_position()
        fig.text((a.x0 + b.x1) / 2, 0.985, text, ha="center", va="top", fontsize=7.5, color=INK)
    save(fig, out, "examples")


def archive_test11(paths):
    """test11.jpg from the 2024 archive (syrip_for_train.zip), read without extracting the archive."""
    import io
    import zipfile

    from PIL import Image
    for zp in (paths.root / "data" / "_zips" / "syrip_for_train.zip", paths.root / "_zips" / "syrip_for_train.zip"):
        if zp.is_file():
            with zipfile.ZipFile(zp) as z:
                return np.asarray(Image.open(io.BytesIO(z.read("syrip/images/validate_infant/test11.jpg"))).convert("RGB"))
    sys.exit("syrip_for_train.zip not found under <root>/data/_zips or <root>/_zips")


def fig_test11(paths, images, gts, out):
    """test11.jpg in the 2024 and the 2023 archive, with the annotated box and the released checkpoint's prediction."""
    img = next(i for i, im in images.items() if im["file_name"] == "test11.jpg")
    gt = gts[img]
    halo = [patheffects.withStroke(linewidth=2.8, foreground=INK)]  # keeps the white dashes visible on any background
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.5))  # printed at this size, so the text stays readable
    files = [("2024 archive", archive_test11(paths), RELEASED_2024),
             ("2023 archive", imread(Path(paths.abs("data.images.validate_infant")) / "test11.jpg"), RELEASED)]
    for ax, (name, pic, pred_file) in zip(axes, files):
        kps = load_preds(pred_file)[img]
        ax.imshow(pic)
        x, y, w, h = gt["bbox"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=1.4, linestyle=(0, (4, 2)),
                                   path_effects=halo))
        xy = [(kps[3 * j], kps[3 * j + 1]) for j in range(17)]
        for p, q in SKELETON:
            ax.plot([xy[p][0], xy[q][0]], [xy[p][1], xy[q][1]], color=RELEASED_C, linewidth=1.3)
        ax.scatter([v[0] for v in xy], [v[1] for v in xy], s=8, color=RELEASED_C, edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_xlim(0, pic.shape[1])
        ax.set_ylim(pic.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        score = oks(kps, gt["keypoints"], gt["area"])
        ax.set_title(f"{name}: {pic.shape[1]} x {pic.shape[0]}\nOKS {score:.2f}", fontsize=7)
        print(f"  test11 {name}: {pic.shape[1]} x {pic.shape[0]}, OKS {score:.4f}")
    handles = [plt.Line2D([], [], color="white", linewidth=1.4, linestyle=(0, (4, 2)), path_effects=halo,
                          label="person box of the annotation file"),
               plt.Line2D([], [], color=RELEASED_C, marker="o", markersize=3.5, label="pose predicted by the released checkpoint")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    save(fig, out, "test11")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    paths = Paths.load()
    fig_training_curves(out)
    images, gts = load_test(paths)
    ours_path, ours_label = best_run()
    released, ours = load_preds(RELEASED), load_preds(ours_path)
    rows = per_image_table(images, gts, released, ours, out)
    swap_counts("released", gts, released)
    swap_counts("ours", gts, ours)
    fig_joint_posture(rows, gts, released, ours, ours_label, out)
    fig_examples(rows, images, gts, released, ours, Path(paths.abs("data.images.validate_infant")), out)
    fig_test11(paths, images, gts, out)


if __name__ == "__main__":
    main()
