#!/usr/bin/env python3
"""COCO keypoint similarity in the standard library, joint by joint.

    from smalldata.oks import JOINTS, PAIRS, joint_terms, oks

    terms = joint_terms(pred, gt, area)   # one exp(-d^2 / (2 s^2 k^2)) per joint, None where the label is absent
    value = oks(pred, gt, area)           # the mean over labeled joints, as COCOeval computes it

pred and gt are flat COCO lists [x, y, v] * 17; area is the annotation's area (s^2); k = 2 sigma with
the COCO sigmas. The formula equals pycocotools' computeOks up to its 1e-16 spacing term.
"""
import math

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
SIGMAS = [.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089]


def joint_terms(pred, gt, area):
    out = []
    for j in range(17):
        if gt[3 * j + 2] <= 0:
            out.append(None)
            continue
        d2 = (pred[3 * j] - gt[3 * j]) ** 2 + (pred[3 * j + 1] - gt[3 * j + 1]) ** 2
        out.append(math.exp(-d2 / (2 * area * (2 * SIGMAS[j]) ** 2)))
    return out


def oks(pred, gt, area):
    terms = [t for t in joint_terms(pred, gt, area) if t is not None]
    return sum(terms) / len(terms) if terms else 0.0


def swapped(pred, pairs=PAIRS):
    """The prediction with every left/right pair exchanged."""
    out = list(pred)
    for a, b in pairs:
        out[3 * a:3 * a + 3], out[3 * b:3 * b + 3] = pred[3 * b:3 * b + 3], pred[3 * a:3 * a + 3]
    return out
