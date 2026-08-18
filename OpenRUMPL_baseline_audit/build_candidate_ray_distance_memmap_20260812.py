#!/usr/bin/env python3
"""Build per-candidate mean ray-to-pose distance for the I3 geometry ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_h76_hypothesis_utility_20260811 import COMBINATIONS, load_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--chunk", type=int, default=2048)
    args = p.parse_args()

    arrays = load_arrays(args.shards)
    predictions = arrays["predictions"]
    rays = arrays["rays"]
    output_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "candidate_ray_distance.npy"
    shape = (len(predictions), len(COMBINATIONS), predictions.shape[2])
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=shape
    )
    direction = rays[..., :3].astype(np.float32, copy=False)
    direction = direction / np.linalg.norm(
        direction, axis=-1, keepdims=True
    ).clip(min=1e-8)
    point = rays[..., 3:6].astype(np.float32, copy=False)
    for start in range(0, len(predictions), args.chunk):
        stop = min(start + args.chunk, len(predictions))
        pose = predictions[start:stop]
        # N,C,J,V,3 -> N,C,J,V.  The candidate index is aligned with the
        # corresponding H76 view combination.
        residual = np.linalg.norm(
            np.cross(
                pose[:, :, :, None, :] - point[start:stop, None, :, :, :],
                direction[start:stop, None, :, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        for candidate_index, combination in enumerate(COMBINATIONS):
            output[start:stop, candidate_index] = residual[
                :, candidate_index, :, :
            ][:, :, np.asarray(combination, dtype=np.int64)].mean(axis=-1)
    output.flush()
    manifest = {
        "shards": args.shards,
        "output": str(destination),
        "shape": list(shape),
        "dtype": "float32",
        "definition": "mean point-to-ray distance over views in each H76 candidate",
    }
    (output_dir / "candidate_ray_distance_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
