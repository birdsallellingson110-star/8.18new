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
    parser.add_argument(
        "--solver-mode", choices=("uniform", "confidence", "irls"),
        default="confidence",
        help="Ray solver used for the added candidate branch.",
    )
    parser.add_argument("--irls-iters", type=int, default=3)
    parser.add_argument("--huber-threshold-m", type=float, default=0.03)
    parser.add_argument(
        "--blend-alpha", type=float, default=1.0,
        help="Blend added geometry with the matching frozen H76 candidate.",
    )
    parser.add_argument(
        "--max-delta-m", type=float, default=0.0,
        help=(
            "Clip the geometry-H76 displacement before blending. Zero keeps "
            "the historical unbounded candidate.")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("blend-alpha must be in [0, 1]")
    if args.max_delta_m < 0.0:
        raise ValueError("max-delta-m must be non-negative")
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

    geometric = np.stack(
        [ray_solver(
            rays, combo, args.solver_mode,
            args.irls_iters, args.huber_threshold_m
        ) for combo in COMBINATIONS],
        axis=1,
    ).astype(np.float32, copy=False)
    if args.max_delta_m > 0.0 or args.blend_alpha < 1.0:
        displacement = geometric - predictions
        if args.max_delta_m > 0.0:
            norm = np.linalg.norm(displacement, axis=-1, keepdims=True)
            displacement = displacement * np.minimum(
                1.0, args.max_delta_m / np.maximum(norm, 1e-8)
            )
        geometric = predictions + args.blend_alpha * displacement
    expanded = np.concatenate((predictions, geometric), axis=1)
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
        "method": (
            "current GBT-style H76 cache + optional bounded geometric "
            "residual candidate"
        ),
        "input": str(Path(args.input).resolve()),
        "output": str(output),
        "original_candidate_count": len(COMBINATIONS),
        "added_candidate_count": len(COMBINATIONS),
        "candidate_count": int(expanded.shape[1]),
        "candidate_combinations": [list(c) for c in COMBINATIONS] * 2,
        "added_solver": args.solver_mode,
        "irls_iters": args.irls_iters,
        "huber_threshold_m": args.huber_threshold_m,
        "blend_alpha": args.blend_alpha,
        "max_delta_m": args.max_delta_m,
        "groups": int(len(expanded)),
        "generation_note": (
            "targets were not read for candidate generation; when enabled, "
            "the added branch is clipped and anchored to matching H76"
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
