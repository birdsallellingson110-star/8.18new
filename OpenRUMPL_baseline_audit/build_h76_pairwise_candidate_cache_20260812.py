#!/usr/bin/env python3
"""Append deterministic pairwise ray-intersection hypotheses to H76 caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import itertools

import numpy as np

from diagnose_h76_candidate_pool_20260812 import ray_solver


PAIRWISE_COMBINATIONS = tuple(itertools.combinations(range(4), 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.input)
    predictions = source["predictions"]
    rays = source["rays"]
    pairwise = np.stack([
        ray_solver(rays, pair, "uniform", 0, 0.03)
        for pair in PAIRWISE_COMBINATIONS
    ], axis=1)
    expanded = np.concatenate([predictions, pairwise], axis=1).astype(np.float32)
    if expanded.shape[1] != 17:
        raise RuntimeError(f"unexpected expanded candidate count: {expanded.shape}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: source[key] for key in source.files}
    arrays["predictions"] = expanded
    arrays["pairwise_combinations"] = np.asarray(PAIRWISE_COMBINATIONS, dtype=np.int16)
    np.savez_compressed(output, **arrays)
    print({
        "input": str(Path(args.input).resolve()),
        "output": str(output),
        "frames": int(len(expanded)),
        "predictions_shape": list(expanded.shape),
        "pairwise_combinations": PAIRWISE_COMBINATIONS,
    })


if __name__ == "__main__":
    main()

