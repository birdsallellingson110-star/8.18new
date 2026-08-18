#!/usr/bin/env python3
"""Prepare geometry-conditioned view-quality features for the RIGR probe.

The features are computed only from the frozen H76 query and calibrated rays;
ground truth is never read.  For every subset-specific query and view we store:

  1. ray confidence;
  2. point-to-ray distance of the current 3-D query (metres, clipped);
  3. mean sine of the ray angle to the other selected views;
  4. absolute query depth along the ray (metres, clipped);
  5. maximum sine of the ray angle to the other selected views.

The output follows the same [G,11,V,J,D] layout as the detector auxiliaries,
so the feature refiner can be compared under identical subset/masking rules.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


COMBINATIONS = tuple(
    combo for n in (2, 3, 4) for combo in itertools.combinations(range(4), n)
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h76-cache", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group-indices-file", default="")
    p.add_argument("--limit-groups", type=int, default=0)
    return p.parse_args()


def load_cache(paths: list[str]) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path) as source:
            parts.append({key: source[key].copy() for key in source.files})
    keys = parts[0].keys()
    cache = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    order = np.argsort(cache["group_indices"])
    return {key: value[order] for key, value in cache.items()}


def selected_rows(cache: dict[str, np.ndarray], path: str, limit: int) -> np.ndarray:
    if path:
        group_path = Path(path)
        if group_path.suffix == ".npy":
            ids = np.asarray(np.load(group_path), dtype=np.int64).reshape(-1)
        else:
            ids = np.asarray(
                [int(line.strip()) for line in group_path.read_text().splitlines() if line.strip()],
                dtype=np.int64,
            )
        rows = {int(group_id): row for row, group_id in enumerate(cache["group_indices"])}
        if len(ids) == 0 or any(int(x) not in rows for x in ids):
            raise ValueError(f"invalid group-indices-file: {path}")
        return np.asarray([rows[int(x)] for x in ids], dtype=np.int64)
    count = min(limit, len(cache["group_indices"])) if limit else len(cache["group_indices"])
    return np.arange(count, dtype=np.int64)


def main() -> None:
    args = parse_args()
    cache = load_cache(args.h76_cache)
    rows = selected_rows(cache, args.group_indices_file, args.limit_groups)
    predictions = cache["predictions"][rows].astype(np.float32, copy=False)
    rays = cache["rays"][rows].astype(np.float32, copy=False)
    # rays: [G,J,V,7], predictions: [G,11,J,3]
    if rays.ndim != 4 or rays.shape[2] != 4 or rays.shape[-1] < 7:
        raise ValueError(f"unexpected rays shape {rays.shape}")
    if predictions.ndim != 4 or predictions.shape[1] != len(COMBINATIONS):
        raise ValueError(f"unexpected predictions shape {predictions.shape}")

    groups = len(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path.with_suffix(output_path.suffix + ".tmp"), mode="w+",
        dtype=np.float16, shape=(groups, len(COMBINATIONS), 4, rays.shape[1], 5),
    )
    directions = rays[..., :3]
    origins = rays[..., 3:6]
    confidence = np.clip(rays[..., 6], 0.0, 1.0)
    for group in range(groups):
        for combo_id, combo in enumerate(COMBINATIONS):
            views = np.asarray(combo, dtype=np.int64)
            query = predictions[group, combo_id]
            # Slice the group first; mixing a basic slice and an advanced
            # view index in one expression moves the view axis in NumPy.
            d = directions[group][:, views]
            o = origins[group][:, views]
            rel = query[:, None, :] - o
            point_distance = np.linalg.norm(np.cross(rel, d), axis=-1)
            depth = np.abs(np.sum(rel * d, axis=-1))
            pair_sin = np.zeros((rays.shape[1], len(views)), dtype=np.float32)
            for local in range(len(views)):
                other = np.delete(np.arange(len(views)), local)
                cross = np.cross(d[:, local, None, :], d[:, other, :])
                pair_sin[:, local] = np.linalg.norm(cross, axis=-1).mean(axis=-1)
            max_sin = np.zeros_like(pair_sin)
            for local in range(len(views)):
                other = np.delete(np.arange(len(views)), local)
                cross = np.cross(d[:, local, None, :], d[:, other, :])
                max_sin[:, local] = np.linalg.norm(cross, axis=-1).max(axis=-1)
            values = np.stack(
                (
                    confidence[group][:, views].T,
                    np.clip(point_distance / 2.0, 0.0, 1.0).T,
                    pair_sin.T,
                    np.clip(depth / 5.0, 0.0, 1.0).T,
                    max_sin.T,
                ),
                axis=-1,
            )
            output[group, combo_id, views] = values.astype(np.float16)
        if (group + 1) % 1000 == 0 or group + 1 == groups:
            print(f"groups {group + 1}/{groups}", flush=True)
    output.flush()
    Path(output.filename).replace(output_path)
    metadata = {
        "groups": groups, "shape": list(output.shape), "dtype": "float16",
        "aux_dim": 5, "subset_specific": True,
        "definition": ["confidence", "point_to_ray_distance_over_2m",
                        "mean_pair_ray_sine", "query_depth_over_5m", "max_pair_ray_sine"],
        "combinations": [list(c) for c in COMBINATIONS],
    }
    Path(str(output_path) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
