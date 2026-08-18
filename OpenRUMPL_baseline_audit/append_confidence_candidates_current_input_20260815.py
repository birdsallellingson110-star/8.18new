#!/usr/bin/env python3
"""Append confidence-weighted ray-triangulation candidates to an 11-candidate cache.

The cache contains the frozen H76 prediction for every 2/3/4-view subset. This
script adds the same 11 subsets solved by confidence-weighted closest-point
triangulation. It never reads targets while generating candidates; targets are
only copied into the output for the supervised E2 training/evaluation code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_h76_candidate_pool_20260812 import ray_solver
from train_h76_hypothesis_utility_20260811 import COMBINATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--irls-iters", type=int, default=3)
    parser.add_argument("--huber-threshold-m", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.input, allow_pickle=False)
    required = {"group_indices", "actions", "subjects", "predictions", "targets", "rays"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"missing cache fields: {sorted(missing)}")
    predictions = source["predictions"]
    rays = source["rays"]
    if predictions.shape[1:] != (len(COMBINATIONS), 17, 3):
        raise ValueError(
            f"expected {len(COMBINATIONS)} original candidates, got {predictions.shape}"
        )
    if rays.shape[1:] != (17, 4, 7):
        raise ValueError(f"unexpected ray shape {rays.shape}")

    confidence = np.stack(
        [ray_solver(rays, combo, "confidence", args.irls_iters, args.huber_threshold_m)
         for combo in COMBINATIONS],
        axis=1,
    ).astype(np.float32, copy=False)
    expanded = np.concatenate((predictions, confidence), axis=1)
    if expanded.shape[1:] != (2 * len(COMBINATIONS), 17, 3):
        raise RuntimeError(f"unexpected expanded shape {expanded.shape}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    payload = {key: source[key] for key in required}
    payload["predictions"] = expanded
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    manifest = {
        "method": "current GBT-style H76 cache + confidence-weighted triangulation",
        "input": str(Path(args.input).resolve()),
        "output": str(output),
        "original_candidate_count": len(COMBINATIONS),
        "added_candidate_count": len(COMBINATIONS),
        "candidate_count": int(expanded.shape[1]),
        "candidate_combinations": [list(c) for c in COMBINATIONS] * 2,
        "added_solver": "confidence",
        "irls_iters": args.irls_iters,
        "huber_threshold_m": args.huber_threshold_m,
        "groups": int(len(expanded)),
        "generation_note": "targets were not read for candidate generation",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
