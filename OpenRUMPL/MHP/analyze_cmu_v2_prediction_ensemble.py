#!/usr/bin/env python3
from pathlib import Path
import pickle

import numpy as np


PRED_ROOT = Path(
    "/mnt/data/cjyoutput/output/multiview_amass_rumpl/"
    "multiview_rumpl_999/cmu_eval_sp_v2_conf"
)

BASELINE = {
    "3_6": 40.37,
    "3_12": 46.95,
    "3_13": 39.79,
    "3_23": 32.30,
    "6_12": 67.28,
    "6_13": 53.52,
    "6_23": 39.39,
    "12_13": 59.41,
    "12_23": 46.04,
    "13_23": 44.08,
}

KP_STAR = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]


def load_pred(prefix, tag):
    path = (
        PRED_ROOT
        / f"preds_gt_multiview_cmu_panoptic_rumpl_mmpose_{prefix}_{tag}_best.pkl"
    )
    with path.open("rb") as f:
        return pickle.load(f)


def kp_star_mpjpe_mm(pred, gt):
    err = np.linalg.norm(pred - gt, axis=2)
    return float(err[:, KP_STAR].mean() * 1000.0)


def eval_alpha(alpha):
    rows = []
    for tag, base in BASELINE.items():
        hardv = load_pred("hardv_debug_v2", tag)
        legw = load_pred("hardv_legw07_bestens_v2", tag)
        if not np.allclose(hardv["gt"], legw["gt"]):
            raise RuntimeError(f"GT mismatch for {tag}")
        pred = alpha * hardv["pred"] + (1.0 - alpha) * legw["pred"]
        value = kp_star_mpjpe_mm(pred, hardv["gt"])
        rows.append((tag, value, value - base))
    return rows


def summarize(rows):
    deltas = [d for _, _, d in rows]
    return sum(deltas) / len(deltas), max(deltas), sum(d <= 0 for d in deltas)


def main():
    candidates = []
    for i in range(101):
        alpha = i / 100.0
        rows = eval_alpha(alpha)
        avg, worst, improved = summarize(rows)
        if improved == len(BASELINE):
            candidates.append((alpha, avg, worst, rows))

    print(f"all_down_alpha_range: {candidates[0][0]:.2f}..{candidates[-1][0]:.2f}")
    for name, item in [
        ("best_worst", min(candidates, key=lambda x: x[2])),
        ("best_avg", min(candidates, key=lambda x: x[1])),
        (
            "balanced_margin_ge_0.20_best_avg",
            min([x for x in candidates if x[2] <= -0.20], key=lambda x: x[1]),
        ),
    ]:
        alpha, avg, worst, rows = item
        print(f"\n{name}: alpha={alpha:.2f} avg_delta={avg:+.2f} worst_delta={worst:+.2f}")
        print("| views | KP* | delta |")
        print("|---|---:|---:|")
        for tag, value, delta in rows:
            print(f"| [{tag.replace('_', ',')}] | {value:.2f} | {delta:+.2f} |")


if __name__ == "__main__":
    main()
