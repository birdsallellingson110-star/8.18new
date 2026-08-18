#!/usr/bin/env python3
"""Export confidence-ray and robust IRLS candidates for manifest subsets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from export_h36m_ray_geometry_features_20260813 import camera_rays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--huber-threshold-mm", type=float, default=20.0)
    return parser.parse_args()


def intersect(
    centers: np.ndarray, directions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    # Inputs are (J,V,3), (J,V,3), (J,V).
    identity = np.eye(3, dtype=np.float64)
    projection = identity - directions[..., :, None] * directions[..., None, :]
    weighted = weights[..., None, None] * projection
    lhs = weighted.sum(axis=1) + 1e-8 * identity
    rhs = np.einsum("jvab,jvb->ja", weighted, centers)
    return np.linalg.solve(lhs, rhs[..., None])[..., 0]


def main() -> None:
    args = parse_args()
    with open(args.input_pkl, "rb") as stream:
        database = pickle.load(stream)
    with open(args.selection_manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    confidence_candidates, irls_candidates = [], []
    for position, group in enumerate(manifest["groups"]):
        records = [database[int(index)] for index in group["record_indices"]]
        if [record["image"] for record in records] != group["images"]:
            raise ValueError(f"alignment failed at group {position}")
        directions, centers, confidence = zip(*(camera_rays(record) for record in records))
        directions = np.stack(directions, axis=1)
        centers = np.stack(centers, axis=1)
        confidence = np.stack(confidence, axis=1)
        base_weights = np.clip(confidence, 1e-4, None)
        estimate = intersect(centers, directions, base_weights)
        confidence_candidates.append(estimate.astype(np.float32))
        for _ in range(args.iterations):
            residual = np.linalg.norm(
                np.cross(estimate[:, None] - centers, directions), axis=-1
            )
            robust = np.minimum(
                1.0, args.huber_threshold_mm / np.maximum(residual, 1e-8)
            )
            estimate = intersect(centers, directions, base_weights * robust)
        irls_candidates.append(estimate.astype(np.float32))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            confidence_ray=np.stack(confidence_candidates),
            robust_irls=np.stack(irls_candidates),
            uses_ground_truth=np.asarray(False),
        )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output), "records": len(confidence_candidates),
        "uses_ground_truth": False,
    }))


if __name__ == "__main__":
    main()
