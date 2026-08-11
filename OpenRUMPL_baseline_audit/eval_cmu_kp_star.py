#!/usr/bin/env python3
"""CMU official-style metrics on RUMPL preds_gt_*_dict.pkl (COCO-17 order).

Reports pelvis-relative MPJPE in centimeters for All-17 and KP* (matched body
joints only: shoulders, elbows, wrists, hips, knees, ankles).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

# COCO-17 indices shared with CMU / H36M cross-dataset body evaluation.
KP_STAR = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_STAR]

JOINT_NAMES = {
    5: "lsho",
    6: "rsho",
    7: "lelb",
    8: "relb",
    9: "lwri",
    10: "rwri",
    13: "lkne",
    14: "rkne",
    15: "lank",
    16: "rank",
}


def pelvis_relative_coco(pred_m, gt_m):
    pred = pred_m * 100.0
    gt = gt_m * 100.0
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0
    pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
    return pred - pv_p, gt - pv_g


def summarize(pred_m, gt_m):
    pred, gt = pelvis_relative_coco(pred_m, gt_m)
    err = np.linalg.norm(pred - gt, axis=-1)
    per_joint_cm = err.mean(axis=0)
    return {
        "n_samples": int(len(pred)),
        "all17_cm": float(err.mean()),
        "all17_mm": float(err.mean() * 10.0),
        "kp_star_cm": float(err[:, KP_STAR].mean()),
        "kp_star_mm": float(err[:, KP_STAR].mean() * 10.0),
        "kp_star_indices": KP_STAR,
        "per_joint_kp_star_cm": {
            JOINT_NAMES[i]: float(per_joint_cm[i]) for i in KP_STAR
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-pkl", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with open(args.dict_pkl, "rb") as stream:
        data = pickle.load(stream)

    pred = np.asarray(data["pred"], dtype=np.float64)
    gt = np.asarray(data["gt"], dtype=np.float64)
    if pred.ndim == 3 and pred.shape[-1] == 3:
        pass
    elif pred.ndim == 2 and pred.shape[-1] == 17 * 3:
        pred = pred.reshape(-1, 17, 3)
        gt = gt.reshape(-1, 17, 3)
    else:
        raise ValueError(f"unexpected pred shape {pred.shape}")

    if gt.shape != pred.shape:
        gt = gt.reshape(pred.shape)

    out = summarize(pred, gt)
    out["dict_pkl"] = str(Path(args.dict_pkl).resolve())
    text = (
        f"n={out['n_samples']}  "
        f"KP*={out['kp_star_cm']:.3f} cm ({out['kp_star_mm']:.2f} mm)  "
        f"All-17={out['all17_cm']:.3f} cm (reference)"
    )
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
