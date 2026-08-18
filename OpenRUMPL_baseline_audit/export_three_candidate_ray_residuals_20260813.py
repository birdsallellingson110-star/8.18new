#!/usr/bin/env python3
"""Export GT-free ray residuals for H76/Volumetric/Algebraic candidates."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from export_h36m_ray_geometry_features_20260813 import camera_rays
from train_adaptive_joint_branch_gate_20260813 import load_pair
from train_unified_multiview_joint_gate_20260813 import combinations


BRANCHES = ("h76", "vol", "alg")
STATISTICS = (
    "log1p_weighted_mean_distance_mm", "log1p_max_distance_mm",
    "weighted_mean_angle_rad", "max_angle_rad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--h76-pkl", required=True)
    parser.add_argument("--vol-npz", required=True)
    parser.add_argument("--alg-npz", required=True)
    parser.add_argument("--views", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = combinations(args.views)
    data = load_pair(args.h76_pkl, args.vol_npz, args.views, combos)
    alg_payload = np.load(args.alg_npz)
    algebraic = np.stack([
        np.asarray(alg_payload[f"prediction_V{args.views}_{combo.replace('-', '_')}"])
        for combo in combos
    ], axis=1).astype(np.float32)
    if algebraic.shape != data["h76"].shape:
        raise ValueError("Algebraic/H76 shape mismatch")
    candidates = np.stack((data["h76"], data["vol"], algebraic), axis=3)
    # Flatten synchronized-frame major, camera-combination minor, matching the manifest.
    candidates = candidates.reshape(-1, 17, 3, 3)

    with open(args.input_pkl, "rb") as stream:
        records = pickle.load(stream)
    with open(args.selection_manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if len(manifest["groups"]) != len(candidates):
        raise ValueError("manifest/candidate count mismatch")

    output_features = []
    for position, (group, candidate) in enumerate(zip(manifest["groups"], candidates)):
        selected_records = [records[int(index)] for index in group["record_indices"]]
        if [record["image"] for record in selected_records] != group["images"]:
            raise ValueError(f"record alignment failed at {position}")
        directions, centers, confidence = zip(*(camera_rays(x) for x in selected_records))
        # (J,V,3), (J,V,3), (J,V)
        directions = np.stack(directions, axis=1)
        centers = np.stack(centers, axis=1)
        confidence = np.stack(confidence, axis=1)
        weights = np.clip(confidence, 0.0, None)
        weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

        # candidate: (J,B,3); displacement: (J,B,V,3)
        displacement = candidate[:, :, None, :] - centers[:, None, :, :]
        distance = np.linalg.norm(
            np.cross(displacement, directions[:, None, :, :]), axis=-1
        )
        displacement_unit = displacement / np.linalg.norm(
            displacement, axis=-1, keepdims=True
        ).clip(1e-12)
        cross_norm = np.linalg.norm(
            np.cross(displacement_unit, directions[:, None, :, :]), axis=-1
        )
        dot = np.sum(displacement_unit * directions[:, None, :, :], axis=-1)
        angle = np.arctan2(cross_norm, np.clip(dot, -1.0, 1.0))
        blocks = []
        for branch in range(3):
            blocks.extend((
                np.log1p(np.sum(weights * distance[:, branch], axis=1)),
                np.log1p(distance[:, branch].max(axis=1)),
                np.sum(weights * angle[:, branch], axis=1),
                angle[:, branch].max(axis=1),
            ))
        output_features.append(np.stack(blocks, axis=-1).astype(np.float32))

    feature_names = tuple(
        f"{branch}_{statistic}" for branch in BRANCHES for statistic in STATISTICS
    )
    features = np.stack(output_features)
    if not np.isfinite(features).all():
        raise ValueError("non-finite candidate residual features")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream, features=features,
            feature_names=np.asarray(feature_names),
            uses_ground_truth=np.asarray(False),
        )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output), "shape": list(features.shape),
        "feature_names": feature_names, "uses_ground_truth": False,
    }, indent=2))


if __name__ == "__main__":
    main()
