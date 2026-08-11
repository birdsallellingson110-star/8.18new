#!/usr/bin/env python3
"""Print KP* from an Absolute-MPJPE pkl (the {joint_name: mpjpe_cm} dict
written by evaluate() as mpjpe_<...>_absolute.pkl).

Reusable for baseline AND ST-VFT (B/C): just point --pkl at the right file.
KP* is computed by the shared lib/utils/kp_star.py so all are comparable.

Usage:
    python run/eval_kp_star.py --pkl /path/to/mpjpe_..._absolute.pkl
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from utils.kp_star import compute_kp_star, KP_STAR_CANON  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="mpjpe_*_absolute.pkl path")
    ap.add_argument("--expect-mm", type=float, default=None,
                    help="optional: assert KP*(mm) matches this (cross-check)")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)

    names = list(d.keys())
    vals_cm = list(d.values())

    print("pkl: {}".format(args.pkl))
    print("joint order (idx -> name -> value cm):")
    for i, n in enumerate(names):
        print("   [{:2d}] {:>5s} = {:.4f}".format(i, str(n), float(vals_cm[i])))

    kp_cm, idx, used = compute_kp_star(vals_cm, names)

    print("\nKP* definition = {}".format(list(KP_STAR_CANON)))
    print("KP* joints resolved (idx -> name -> value cm):")
    for i in idx:
        print("   [{:2d}] {:>5s} = {:.4f}".format(i, str(names[i]), float(vals_cm[i])))

    print("\nKP* = mean of the {} values above".format(len(idx)))
    print("KP* = {:.4f} cm = {:.2f} mm".format(kp_cm, kp_cm * 10))

    if args.expect_mm is not None:
        diff = abs(kp_cm * 10 - args.expect_mm)
        print("cross-check vs expected {:.2f} mm : diff = {:.4f} mm -> {}".format(
            args.expect_mm, diff, "MATCH" if diff < 0.01 else "MISMATCH"))


if __name__ == "__main__":
    main()
