#!/usr/bin/env python3
"""Build the paper-backed candidate pool used by the V2/V3/V4 E2 scorer.

The input cache already contains the frozen H76/RIGR candidates, the original
uniform pairwise hypotheses, and (for V3/V4) the learned-triangulation
hypotheses.  This script appends two deterministic, inference-only candidates
for every 2/3/4-view subset:

* confidence-weighted closest-point triangulation;
* robust IRLS closest-point triangulation.

No target or training label is read.  The candidate order is part of the
output manifest and must be kept identical by the E2 trainer.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from diagnose_h76_candidate_pool_20260812 import ray_solver
from train_h76_hypothesis_utility_20260811 import COMBINATIONS


PAIRWISE_COMBINATIONS = tuple(itertools.combinations(range(4), 2))
LEARNED_COMBINATIONS = tuple(combo for combo in COMBINATIONS if len(combo) >= 3)

# Existing cache layout produced by train_h76_learned_candidate_e2_20260814.py.
EXISTING_COMBINATIONS = COMBINATIONS + PAIRWISE_COMBINATIONS + LEARNED_COMBINATIONS
NEW_COMBINATIONS = COMBINATIONS + COMBINATIONS
ALL_CANDIDATE_COMBINATIONS = EXISTING_COMBINATIONS + NEW_COMBINATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="e2_v234")
    parser.add_argument("--irls-iters", type=int, default=3)
    parser.add_argument("--huber-threshold-m", type=float, default=0.03)
    parser.add_argument("--shards", type=int, default=2)
    return parser.parse_args()


def load_files(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {
        key: np.concatenate([source[key] for source in loaded], axis=0)
        for key in keys
    }
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate group_indices after concatenation")
    if arrays["predictions"].shape[1:] != (22, 17, 3):
        raise ValueError(
            "expected the existing 22-candidate RIGR/E2 cache; got "
            f"{arrays['predictions'].shape}"
        )
    if arrays["rays"].shape[1:] != (17, 4, 7):
        raise ValueError(f"unexpected ray shape {arrays['rays'].shape}")
    return arrays


def append_candidates(arrays: dict[str, np.ndarray], args: argparse.Namespace):
    rays = arrays["rays"]
    generated_confidence = []
    generated_irls = []
    for combo in COMBINATIONS:
        generated_confidence.append(
            ray_solver(rays, combo, "confidence", args.irls_iters, args.huber_threshold_m)
        )
    for combo in COMBINATIONS:
        generated_irls.append(
            ray_solver(rays, combo, "irls", args.irls_iters, args.huber_threshold_m)
        )
    generated = np.concatenate(
        [np.stack(generated_confidence, axis=1), np.stack(generated_irls, axis=1)],
        axis=1,
    )
    predictions = np.concatenate([arrays["predictions"], generated], axis=1)
    expected = (len(ALL_CANDIDATE_COMBINATIONS), 17, 3)
    if predictions.shape[1:] != expected:
        raise RuntimeError(f"bad expanded predictions shape {predictions.shape}; expected {expected}")
    result = dict(arrays)
    result["predictions"] = predictions.astype(np.float32, copy=False)
    return result


def write_shards(arrays: dict[str, np.ndarray], output_dir: Path, prefix: str, count: int):
    if count < 1:
        raise ValueError("--shards must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.array_split(np.arange(len(arrays["targets"])), count)
    paths = []
    for shard_id, rows in enumerate(indices):
        path = output_dir / f"{prefix}_shard{shard_id}of{count}.npz"
        payload = {
            key: value[rows]
            for key, value in arrays.items()
            if key != "candidate_combinations"
        }
        np.savez_compressed(path, **payload)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    arrays = append_candidates(load_files(args.input_files), args)
    output_dir = Path(args.output_dir).resolve()
    paths = write_shards(arrays, output_dir, args.prefix, args.shards)
    manifest = {
        "method": "RIGR/E2 cache plus confidence-weighted and IRLS DLT candidates",
        "input_files": [str(Path(item).resolve()) for item in args.input_files],
        "output_shards": paths,
        "candidate_count": len(ALL_CANDIDATE_COMBINATIONS),
        "candidate_combinations": [list(item) for item in ALL_CANDIDATE_COMBINATIONS],
        "existing_candidate_count": len(EXISTING_COMBINATIONS),
        "new_candidate_count": len(NEW_COMBINATIONS),
        "irls_iters": args.irls_iters,
        "huber_threshold_m": args.huber_threshold_m,
        "groups": int(len(arrays["targets"])),
        "note": "GT is retained only as the existing cache target for later supervised training/evaluation; generation reads rays only.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
