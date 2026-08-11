#!/usr/bin/env python3
"""GBT-style CMU metrics: absolute All-17 MPJPE (mm), optional KP*."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

KP_STAR = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-pkl", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output-json")
    args = ap.parse_args()

    data = pickle.load(open(args.dict_pkl, "rb"))
    pred = np.asarray(data["pred"], dtype=np.float64)
    gt = np.asarray(data["gt"], dtype=np.float64)
    if pred.ndim == 2:
        pred = pred.reshape(-1, 17, 3)
        gt = gt.reshape(-1, 17, 3)

    err_mm = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    per_joint = err_mm.mean(axis=0)
    out = {
        "tag": args.tag,
        "dict_pkl": str(Path(args.dict_pkl).resolve()),
        "n_samples": int(len(pred)),
        "metric": "absolute_MPJPE_mm_no_alignment",
        "all17_mm": float(err_mm.mean()),
        "kp_star_mm": float(err_mm[:, KP_STAR].mean()),
        "median_sample_all17_mm": float(np.median(err_mm.mean(axis=1))),
        "p90_sample_all17_mm": float(np.percentile(err_mm.mean(axis=1), 90)),
        "per_joint_mm": {str(i): float(per_joint[i]) for i in range(17)},
    }
    text = (
        f"[{args.tag}] n={out['n_samples']}  "
        f"All-17={out['all17_mm']:.2f} mm  KP*={out['kp_star_mm']:.2f} mm  "
        f"median={out['median_sample_all17_mm']:.2f}  p90={out['p90_sample_all17_mm']:.2f}"
    )
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
