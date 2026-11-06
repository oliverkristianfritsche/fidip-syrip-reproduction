#!/usr/bin/env python3
"""Score a COCO-format keypoint results file against a SyRIP annotation file.

    from smalldata.cocoeval import score_keypoints
    stats = score_keypoints(gt_json, dt_json)
    stats["AP"]   # percent, two decimals

The function mirrors SYRIPDataset._do_python_keypoint_eval in the upstream code: COCOeval with
iouType "keypoints" and the default OKS sigmas. Values are percentages, the unit of the upstream
README tables.
"""
import contextlib
import io

STAT_NAMES = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]


def per_image_oks(gt_json, dt_json):
    """One OKS per ground-truth image: the best match of its detection, 0 when none."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        dt = gt.loadRes(str(dt_json))
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.useSegm = None
        ev.evaluate()
    out = {}
    for img in ev.params.imgIds:
        ious = ev.ious.get((img, 1))
        out[int(img)] = float(np.max(ious)) if ious is not None and len(ious) else 0.0
    return out


def score_keypoints(gt_json, dt_json, quiet=True):
    from pycocotools.coco import COCO  # heavy import stays inside the function
    from pycocotools.cocoeval import COCOeval

    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        gt = COCO(str(gt_json))
        dt = gt.loadRes(str(dt_json))
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.useSegm = None
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {name: round(float(value) * 100, 2) for name, value in zip(STAT_NAMES, ev.stats)}
