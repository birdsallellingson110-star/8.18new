#!/usr/bin/env python3
"""Summarize CMU V2..V5 camera-combination metrics from preds_gt_*_dict.pkl.

Matches the RUMPL project protocol: all combinations of cameras
(3, 6, 12, 13, 23), report per-combination and unweighted mean across
combinations.  Supports absolute MPJPE (GBT / RUMPL CMU baseline style)
and pelvis-relative (mid-hip, COCO indices 11/12).
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

CAMERAS = (3, 6, 12, 13, 23)
KP_STAR = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-pkl", required=True)
    parser.add_argument("--n-views", type=int, required=True, choices=range(2, 6))
    parser.add_argument("--output-json")
    return parser.parse_args()


def combination_label(name, n_views):
    tail = name.rsplit("_", n_views)[-n_views:]
    return "_".join("{:02d}".format(int(part)) for part in tail)


def metrics_mm(error_mm):
    return {
        "all17_mm": float(error_mm.mean()),
        "kp_star_mm": float(error_mm[:, KP_STAR].mean()),
    }


def pelvis_relative_m(pred_m, gt_m):
    pred = pred_m.copy()
    gt = gt_m.copy()
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0
    pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
    return pred - pv_p, gt - pv_g


def main():
    args = parse_args()
    with open(args.dict_pkl, "rb") as stream:
        data = pickle.load(stream)

    pred = np.asarray(data["pred"], dtype=np.float64)
    gt = np.asarray(data["gt"], dtype=np.float64)
    fnames = list(data["fnames"])
    if pred.ndim == 2:
        pred = pred.reshape(-1, 17, 3)
        gt = gt.reshape(-1, 17, 3)

    labels = [combination_label(name, args.n_views) for name in fnames]
    expected = [
        "_".join("{:02d}".format(c) for c in combo)
        for combo in itertools.combinations(CAMERAS, args.n_views)
    ]

    abs_err = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    rel_pred, rel_gt = pelvis_relative_m(pred, gt)
    rel_err = np.linalg.norm(rel_pred - rel_gt, axis=-1) * 1000.0

    per_combo = {}
    for combo in expected:
        idx = np.asarray(labels) == combo
        if not np.any(idx):
            raise ValueError(f"missing combination {combo} in predictions")
        per_combo[combo] = {
            "samples": int(idx.sum()),
            "absolute": metrics_mm(abs_err[idx]),
            "pelvis_relative": metrics_mm(rel_err[idx]),
        }

    unknown = sorted(set(labels) - set(expected))
    if unknown:
        raise ValueError(f"unexpected camera combinations: {unknown[:5]} ...")

    def mean_over_combos(field, metric):
        vals = [per_combo[c][field][metric] for c in expected]
        return float(np.mean(vals))

    result = {
        "dict_pkl": str(Path(args.dict_pkl).resolve()),
        "n_views": args.n_views,
        "n_combinations": len(expected),
        "kp_star_indices": KP_STAR,
        "cameras": list(CAMERAS),
        "per_combination": per_combo,
        "mean_over_combinations": {
            "absolute": {
                "all17_mm": mean_over_combos("absolute", "all17_mm"),
                "kp_star_mm": mean_over_combos("absolute", "kp_star_mm"),
            },
            "pelvis_relative": {
                "all17_mm": mean_over_combos("pelvis_relative", "all17_mm"),
                "kp_star_mm": mean_over_combos("pelvis_relative", "kp_star_mm"),
            },
        },
        "overall_pooled": {
            "samples": int(len(fnames)),
            "absolute": metrics_mm(abs_err),
            "pelvis_relative": metrics_mm(rel_err),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
