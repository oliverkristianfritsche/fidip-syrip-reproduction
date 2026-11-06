#!/usr/bin/env python3
"""Train FiDIP with checkpoint selection on a held-out real split, with a seed and a chosen freeze depth.

    python scripts/train_with_validation.py --cfg configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml --seed 0
    python scripts/train_with_validation.py --cfg <cfg> --base-path <root>/variants/clean --freeze-tensors 0 --seed 1
    python scripts/train_with_validation.py --cfg <cfg> --epochs 1 --pre-epochs 1 --max-samples 16 --batch-size 4 \\
        --out results/_smoke_heldout                                                  # smoke test

What differs from the upstream trainer (tools/train_adaptive_model_hrnet.py, which scripts/train_model.py drives):
  1. The per-epoch validation runs on the selection split, <root>/variants/select by default (the real
     Test500 images outside Test100, built by scripts/make_select_split.py), never on the test split.
     model_best.pth is the epoch with the highest selection AP.
  2. The test split is evaluated once per checkpoint at the end, with the upstream test script:
     final_state.pth into <out>/final/ and model_best.pth into <out>/best/.
  3. --seed seeds Python, NumPy, torch, and the loader workers. --deterministic turns off
     cudnn.benchmark and asks torch for deterministic kernels.
  4. --freeze-tensors N freezes the first N parameter tensors of the pose net. Upstream freezes 109
     (the stem and stage 2); its comment names 484 for stage 3 as well; 0 trains everything.
  5. The resume checkpoint holds both networks, both optimizers, the epoch, and the epoch table, so
     a resumed run continues instead of repeating the warm-up.
Everything else follows upstream: the warm-up of the domain classifier on train_pre_infant, the
balanced sampler, Step I and Step II per batch (core.function.train_adaptive), Adam at TRAIN.LR, and
the flip test with DARK in validation (core.function.validate_adaptive). The learning rates stay
constant: the MultiStepLR milestones 40 and 200 lie beyond the 20 epochs.

Outputs in --out: epochs.csv (selection AP and mean OKS per epoch), final/ and best/ (test metrics),
summary.json. Checkpoints and every epoch's selection predictions (select_eval/epoch_<n>.json) stay
under --work, default <root>/raw/train_<out name>/, never in the repository.
Flags: --cfg --model --data --base-path --select-root --out --work --seed --deterministic
       --freeze-tensors --max-samples --epochs --pre-epochs --batch-size --workers.
"""
import argparse
import copy
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smalldata.cocoeval import STAT_NAMES, per_image_oks, score_keypoints  # noqa: E402
from smalldata.evaluate import evaluate_checkpoint, git_head, md5  # noqa: E402
from smalldata.paths import Paths  # noqa: E402
from smalldata.subset import make_subset_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "hrnet_w48_384x288_infant_fidip.yaml"
TRAIN_SPLITS = ["train_pre_infant", "train_infant"]
# The selection variant names its split validate_infant: the upstream loader hard-codes the split names.
SELECT_SPLIT = "validate_infant"
UPSTREAM_FREEZE = 109


def resolve_model(p, arg):
    if arg in p.get("weights"):
        return p.abs(f"weights.{arg}")
    return Path(arg).resolve()


def seed_everything(seed, deterministic):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init(_worker_id):
    """Seed NumPy and random in a loader worker from the torch seed torch derives per worker and epoch."""
    import torch

    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def freeze_first(model, n):
    frozen = 0
    for i, param in enumerate(model.parameters()):
        if i < n:
            param.requires_grad = False
            frozen += 1
    return frozen


def warm_up(cfg, loader, model_p, model_d, criterion_d, optimizer_d, epochs, log):
    """The upstream warm-up: the classifier learns on the pose features; the pose net gets no update.

    The pose net stays in train mode, as upstream, so its BatchNorm statistics drift on the adult data.
    """
    import torch

    for epoch in range(epochs):
        model_d.train()
        model_p.train()
        hits = total = 0
        loss_sum = 0.0
        for i, (inp, _, _, meta) in enumerate(loader):
            with torch.no_grad():
                features, _ = model_p(inp)
            logits = model_d(features)
            label = (meta["synthetic"].unsqueeze(-1) * 1.0).cuda(non_blocking=True)
            loss = criterion_d(logits, label)
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()
            hits += int(((logits > 0).float() == label).sum().item())
            total += label.numel()
            loss_sum += loss.item() * label.numel()
            if i % cfg.PRINT_FREQ == 0:
                log.info(f"warm-up epoch {epoch} [{i}/{len(loader)}] loss_d {loss.item():.4f} acc_d {hits / total:.3f}")
        log.info(f"warm-up epoch {epoch} done: loss_d {loss_sum / max(total, 1):.4f} acc_d {hits / max(total, 1):.3f}")


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + STAT_NAMES + ["mean_OKS"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--model", default="coco_w48_384x288", help="start checkpoint: weights key or .pth path")
    ap.add_argument("--data", default="validate_infant", help="test split, evaluated once per checkpoint at the end")
    ap.add_argument("--base-path", default="", help="data root for training; default from configs/paths.yaml")
    ap.add_argument("--select-root", default="", help="root of the selection split; default <project root>/variants/select")
    ap.add_argument("--out", default="", help="result folder; default results/training/<cfg stem>_heldout_<model>")
    ap.add_argument("--work", default="", help="run folder with checkpoints; default <root>/raw/train_<out name>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true", help="cudnn.benchmark off, deterministic kernels")
    ap.add_argument("--freeze-tensors", type=int, default=UPSTREAM_FREEZE,
                    help=f"freeze the first N parameter tensors of the pose net; upstream {UPSTREAM_FREEZE}")
    ap.add_argument("--max-samples", type=int, default=0, help="images per split for a smoke test")
    ap.add_argument("--epochs", type=int, default=0, help="override TRAIN.END_EPOCH")
    ap.add_argument("--pre-epochs", type=int, default=-1, help="override TRAIN.PRE_EPOCH")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="override both batch sizes, for a smoke test on a small GPU; never for a real run")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    # ----- stage 1: paths
    p = Paths.load()
    project_root = p.root
    if a.base_path:
        p.root = Path(a.base_path)
    # Absolute folders throughout: the upstream test script runs from the upstream folder as a subprocess.
    select_root = Path(a.select_root).resolve() if a.select_root else project_root / "variants" / "select"
    cfg_path = Path(a.cfg).resolve()
    start = resolve_model(p, a.model)
    select_ann = select_root / "data" / "syrip" / "annotations" / f"person_keypoints_{SELECT_SPLIT}.json"
    select_images = select_root / "data" / "syrip" / "images" / SELECT_SPLIT
    checks = [("config", cfg_path), ("start checkpoint", start),
              ("selection annotations", select_ann), ("selection images", select_images)]
    for split in TRAIN_SPLITS + [a.data]:
        checks += [(f"{split} annotations", p.abs(f"data.annotations.{split}")),
                   (f"{split} images", p.abs(f"data.images.{split}"))]
    for label, path in checks:
        if not path.exists():
            sys.exit(f"missing {label}: {path}")
    model_stem = Path(a.model).stem if a.model.endswith(".pth") else a.model
    out = Path(a.out).resolve() if a.out else REPO / "results" / "training" / f"{cfg_path.stem}_heldout_{model_stem}"
    if a.max_samples and "_smoke" not in str(out):
        sys.exit("a --max-samples run must write to a results/_smoke_* folder")
    if a.batch_size and not a.max_samples:
        sys.exit("--batch-size is for smoke tests only; pass --max-samples with it")
    work = Path(a.work).resolve() if a.work else p.root / "raw" / f"train_{out.name}"
    if (out / "summary.json").is_file():
        print(f"already done: {out / 'summary.json'} exists. Remove it to run again.")
        return
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"root {p.root}\nselection {select_root}\nupstream {p.upstream}\nstart {start}\ncfg {cfg_path}\nout {out}\nwork {work}")

    # ----- stage 2: optional subsets for a smoke test
    data_dir, gt_ann = p.root, p.abs(f"data.annotations.{a.data}")
    select_dir = select_root
    if a.max_samples:
        counts = {s: a.max_samples for s in TRAIN_SPLITS + [a.data]}
        data_dir, anns = make_subset_root(p, counts, work)
        gt_ann = anns[a.data]
        q = copy.copy(p)
        q.root = select_root
        select_dir, sel_anns = make_subset_root(q, {SELECT_SPLIT: a.max_samples}, work / "select_subset")
        select_ann = sel_anns[SELECT_SPLIT]

    # ----- stage 3: upstream imports, config, seed, logging
    p.add_upstream_lib()
    import torch
    import torchvision.transforms as transforms
    from torch.utils.tensorboard import SummaryWriter
    from config import cfg  # upstream
    from core.function import AverageMeter, train_adaptive, validate_adaptive  # upstream
    from core.loss import JointsMSELoss  # upstream
    from utils.sampler import BalancedBatchSampler  # upstream
    from utils.utils import get_optimizer  # upstream
    import dataset  # upstream
    import models  # upstream

    if not torch.cuda.is_available():
        sys.exit("this trainer needs a CUDA device: the upstream functions call .cuda()")
    cfg.defrost()
    cfg.merge_from_file(str(cfg_path))
    opts = ["MODEL.INIT_WEIGHTS", False, "DEBUG.DEBUG", False, "WORKERS", a.workers, "GPUS", (0,)]
    if a.epochs:
        opts += ["TRAIN.END_EPOCH", a.epochs]
    if a.pre_epochs >= 0:
        opts += ["TRAIN.PRE_EPOCH", a.pre_epochs]
    if a.batch_size:
        opts += ["TRAIN.BATCH_SIZE_PER_GPU", a.batch_size, "TRAIN.PRE_BATCH_SIZE_PER_GPU", a.batch_size,
                 "TEST.BATCH_SIZE_PER_GPU", a.batch_size]
    cfg.merge_from_list(opts)
    cfg.freeze()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(work / "train.log")])
    log = logging.getLogger("train_with_validation")
    seed_everything(a.seed, a.deterministic)
    log.info(f"seed {a.seed}, deterministic {a.deterministic}, LR {cfg.TRAIN.LR}, LAMBDA {cfg.TRAIN.LAMBDA}, "
             f"warm-up epochs {cfg.TRAIN.PRE_EPOCH}, epochs {cfg.TRAIN.END_EPOCH}, batch {cfg.TRAIN.BATCH_SIZE_PER_GPU}")

    # ----- stage 4: networks, losses, optimizers, freezing (the upstream order: optimizers first)
    model_p, model_d = getattr(models, cfg.MODEL.NAME).get_adaptive_pose_net(cfg, is_train=True)
    model_p.load_state_dict(torch.load(str(start), map_location="cpu"))
    model_p = torch.nn.DataParallel(model_p, device_ids=list(cfg.GPUS)).cuda()
    model_d = torch.nn.DataParallel(model_d, device_ids=list(cfg.GPUS)).cuda()
    criterion_p = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    criterion_d = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer_p = get_optimizer(cfg, model_p)
    optimizer_d = get_optimizer(cfg, model_d)
    n_tensors = sum(1 for _ in model_p.parameters())
    n_frozen = freeze_first(model_p, a.freeze_tensors)
    log.info(f"pose net: {n_tensors} parameter tensors, {n_frozen} frozen")

    # ----- stage 5: data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.ToTensor(), normalize])
    syrip_root = str(Path(data_dir) / "data" / "syrip")
    train_pre_set = dataset.syrip(cfg, syrip_root, cfg.DATASET.TRAIN_PRE_SET, True, tf)
    train_set = dataset.syrip(cfg, syrip_root, cfg.DATASET.TRAIN_SET, True, tf)
    select_set = dataset.syrip(cfg, str(Path(select_dir) / "data" / "syrip"), SELECT_SPLIT, False, tf)
    generator = torch.Generator()
    generator.manual_seed(a.seed)
    pre_loader = torch.utils.data.DataLoader(
        train_pre_set, batch_size=cfg.TRAIN.PRE_BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, worker_init_fn=worker_init, generator=generator)
    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=BalancedBatchSampler(train_set, train_set._load_syrip_syn_annotations()),
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS), num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY, worker_init_fn=worker_init, generator=generator)
    select_loader = torch.utils.data.DataLoader(
        select_set, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=False,
        num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    log.info(f"samples: warm-up {len(train_pre_set)}, train {len(train_set)}, selection {len(select_set)}")

    # ----- stage 6: resume or warm-up
    ckpt_file = work / "checkpoint.pth"
    begin_epoch, best_perf, best_epoch, rows = 0, -1.0, 0, []
    t0 = time.time()
    if ckpt_file.is_file():
        ck = torch.load(ckpt_file, map_location="cpu")
        model_p.module.load_state_dict(ck["model_p"])
        model_d.module.load_state_dict(ck["model_d"])
        optimizer_p.load_state_dict(ck["optimizer_p"])
        optimizer_d.load_state_dict(ck["optimizer_d"])
        begin_epoch, best_perf, best_epoch, rows = ck["epoch"], ck["best_perf"], ck["best_epoch"], ck["rows"]
        log.info(f"resumed {ckpt_file}: epoch {begin_epoch}, best selection AP {best_perf} at epoch {best_epoch}")
    else:
        warm_up(cfg, pre_loader, model_p, model_d, criterion_d, optimizer_d, cfg.TRAIN.PRE_EPOCH, log)
    scheduler_p = torch.optim.lr_scheduler.MultiStepLR(optimizer_p, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR)
    for _ in range(begin_epoch):
        scheduler_p.step()

    # ----- stage 7: epochs with selection on the held-out split
    writer = SummaryWriter(log_dir=str(work / "tb"))
    writer_dict = {"writer": writer, "pre_train_global_steps": 0, "train_global_steps": 0, "valid_global_steps": 0}
    losses_p, acc_p, losses_d = AverageMeter(), AverageMeter(), AverageMeter()
    lists = ([], [], [], [])
    select_eval = work / "select_eval"
    (select_eval / "results").mkdir(parents=True, exist_ok=True)
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        scheduler_p.step()
        lists = train_adaptive(cfg, train_loader, model_p, model_d, criterion_p, criterion_d, optimizer_p, optimizer_d,
                               epoch, str(work), str(work / "tb"), writer_dict, *lists, 0, 0, losses_p, acc_p, losses_d)
        perf = validate_adaptive(cfg, select_loader, select_set, model_p, criterion_p, str(select_eval),
                                 str(work / "tb"), writer_dict)
        dt_json = select_eval / "results" / f"keypoints_{SELECT_SPLIT}_results_0.json"
        stats = score_keypoints(select_ann, dt_json)
        if abs(stats["AP"] - 100 * perf) > 0.05:
            log.warning(f"rescored selection AP {stats['AP']:.2f} differs from upstream's {100 * perf:.2f}")
        # Keep every epoch's selection predictions and their mean OKS, so another selection rule can be
        # computed later without retraining. The selection rule of the run stays AP.
        oks = per_image_oks(select_ann, dt_json)
        shutil.copy2(dt_json, select_eval / f"epoch_{epoch + 1}.json")
        rows.append({"epoch": epoch + 1, **stats, "mean_OKS": round(sum(oks.values()) / max(len(oks), 1), 4)})
        write_rows(out / "epochs.csv", rows)
        if stats["AP"] > best_perf:
            best_perf, best_epoch = stats["AP"], epoch + 1
            torch.save(model_p.module.state_dict(), work / "model_best.pth")
        torch.save({"epoch": epoch + 1, "model_p": model_p.module.state_dict(), "model_d": model_d.module.state_dict(),
                    "optimizer_p": optimizer_p.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                    "best_perf": best_perf, "best_epoch": best_epoch, "rows": rows}, ckpt_file)
        log.info(f"epoch {epoch + 1}/{cfg.TRAIN.END_EPOCH}: selection AP {stats['AP']:.2f}, "
                 f"best {best_perf:.2f} at epoch {best_epoch}, {time.time() - t0:.0f} s so far")
    torch.save(model_p.module.state_dict(), work / "final_state.pth")
    writer.close()
    train_s = round(time.time() - t0, 1)
    del model_p, model_d, optimizer_p, optimizer_d, train_loader, pre_loader, select_loader
    torch.cuda.empty_cache()

    # ----- stage 8: one test evaluation per checkpoint, then the summary
    summary = {
        "cfg": str(cfg_path.relative_to(REPO)) if cfg_path.is_relative_to(REPO) else str(cfg_path),
        "trainer": "scripts/train_with_validation.py", "base_path": str(p.root), "selection_root": str(select_root),
        "selection_images": len(select_set), "epochs_metric": "AP on the selection split",
        "start_model": a.model, "start_path": str(start), "start_md5": md5(start),
        "seed": a.seed, "deterministic": a.deterministic, "freeze_tensors": a.freeze_tensors, "frozen": n_frozen,
        "n_tensors": n_tensors, "epochs_trained": len(rows), "epochs_flag": a.epochs, "pre_epochs_flag": a.pre_epochs,
        "batch_size_flag": a.batch_size, "max_samples": a.max_samples, "train_s": train_s,
        "best_epoch_by_validation": best_epoch, "best_select_AP": best_perf,
        "final_select_AP": rows[-1]["AP"] if rows else None,
        "commit": git_head(), "upstream_commit": p.upstream_commit(), "work": str(work),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for tag, name in [("final", "final_state.pth"), ("best", "model_best.pth")]:
        ckpt = work / name
        if not ckpt.is_file():
            print(f"no {name}, skipping the {tag} evaluation")
            continue
        rec = evaluate_checkpoint(p, cfg_path, ckpt, a.data, data_dir, gt_ann, work / f"eval_{tag}", out / tag,
                                  workers=a.workers, gpus="(0,)", seed=a.seed,
                                  extra={"checkpoint": tag, "max_samples": a.max_samples,
                                         "epoch": best_epoch if tag == "best" else len(rows)})
        summary[f"{tag}_AP"] = rec["AP"]
        summary[f"{tag}_AR"] = rec["AR"]
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nselection: best epoch {best_epoch} at {best_perf} AP; test: final_state AP {summary.get('final_AP')}  "
          f"model_best AP {summary.get('best_AP')}  wrote {out / 'summary.json'}")
    if a.max_samples:
        print("smoke test: the numbers above are not a result")


if __name__ == "__main__":
    main()
