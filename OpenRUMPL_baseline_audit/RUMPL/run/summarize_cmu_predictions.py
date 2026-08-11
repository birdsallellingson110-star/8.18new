#!/usr/bin/env python3
"""Summarize CMU Absolute MPJPE from a RUMPL prediction dictionary."""

import argparse
import itertools
import json
import os
import pickle

import numpy as np


CAMERAS = (3, 6, 12, 13, 23)
KP_STAR = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file")
    parser.add_argument("--output-json")
    parser.add_argument("--n-views", type=int, default=2, choices=range(2, 6))
    parser.add_argument("--expected-frames-per-pair", type=int, default=642)
    return parser.parse_args()


def metrics(error_mm):
    return {
        "all17_mm": float(error_mm.mean()),
        "kpstar_mm": float(error_mm[:, KP_STAR].mean()),
    }


def main():
    args = parse_args()
    with open(args.prediction_file, "rb") as handle:
        data = pickle.load(handle)

    pred = np.asarray(data["pred"])
    gt = np.asarray(data["gt"])
    fnames = data["fnames"]
    if pred.shape != gt.shape or pred.shape != (len(fnames), 17, 3):
        raise ValueError(
            f"unexpected shapes: pred={pred.shape}, gt={gt.shape}, fnames={len(fnames)}"
        )

    error_mm = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    combination_labels = [
        "_".join("{:02d}".format(int(camera)) for camera in name.rsplit("_", args.n_views)[-args.n_views:])
        for name in fnames
    ]
    expected_combinations = [
        "_".join("{:02d}".format(camera) for camera in combination)
        for combination in itertools.combinations(CAMERAS, args.n_views)
    ]

    per_combination = {}
    for combination in expected_combinations:
        indexes = np.flatnonzero(np.asarray(combination_labels) == combination)
        if len(indexes) != args.expected_frames_per_pair:
            raise ValueError(
                f"combination {combination}: expected "
                f"{args.expected_frames_per_pair} frames, got {len(indexes)}"
            )
        per_combination[combination] = {
            "samples": int(len(indexes)),
            **metrics(error_mm[indexes]),
        }

    unknown = sorted(set(combination_labels) - set(expected_combinations))
    if unknown:
        raise ValueError(f"unexpected camera combinations: {unknown}")

    result = {
        "prediction_file": os.path.abspath(args.prediction_file),
        "samples": int(len(fnames)),
        "units": "mm",
        "metric": "absolute MPJPE",
        "n_views": args.n_views,
        "kpstar_indices": list(KP_STAR),
        "overall": metrics(error_mm),
        "per_combination": per_combination,
    }
    result["best_combination_all17"] = min(
        per_combination, key=lambda item: per_combination[item]["all17_mm"]
    )
    result["worst_combination_all17"] = max(
        per_combination, key=lambda item: per_combination[item]["all17_mm"]
    )
    result["best_combination_kpstar"] = min(
        per_combination, key=lambda item: per_combination[item]["kpstar_mm"]
    )
    result["worst_combination_kpstar"] = max(
        per_combination, key=lambda item: per_combination[item]["kpstar_mm"]
    )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w", encoding="ascii") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
