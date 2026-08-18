#!/usr/bin/env python3
"""Fail-fast audit for official-LT observations exported into RUMPL PKLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(record: dict) -> tuple[int, ...]:
    return tuple(
        int(record[name])
        for name in ("subject", "action", "subaction", "image_id", "camera_id")
    )


def project_undistorted(record: dict) -> np.ndarray:
    xyz = np.asarray(record["joints_3d_camera"], dtype=np.float64)
    intrinsic = np.asarray(record["camera"]["K"], dtype=np.float64)
    homogeneous = xyz @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-identity-projection-error", type=float, default=15.0)
    args = parser.parse_args()

    source_path, export_path = Path(args.source), Path(args.export)
    with source_path.open("rb") as handle:
        source = pickle.load(handle)
    with export_path.open("rb") as handle:
        exported = pickle.load(handle)

    if len(source) != len(exported):
        raise RuntimeError(f"record count differs: source={len(source)}, export={len(exported)}")
    source_keys = [record_key(record) for record in source]
    export_keys = [record_key(record) for record in exported]
    if source_keys != export_keys:
        raise RuntimeError("export record identity/order differs from source")
    if len(set(export_keys)) != len(export_keys):
        raise RuntimeError("duplicate record keys in export")

    observations = np.stack([
        np.asarray(record["joints_2d"], dtype=np.float64) for record in exported
    ])
    confidences = np.stack([
        np.asarray(record["joints_2d_conf"], dtype=np.float64).reshape(17)
        for record in exported
    ])
    if observations.shape != (len(exported), 17, 2):
        raise RuntimeError(f"bad observation shape {observations.shape}")
    if confidences.shape != (len(exported), 17):
        raise RuntimeError(f"bad confidence shape {confidences.shape}")
    if not np.isfinite(observations).all() or not np.isfinite(confidences).all():
        raise RuntimeError("non-finite observation or confidence")
    if confidences.min() < 0.0 or confidences.max() > 1.0:
        raise RuntimeError(
            f"confidence out of range: min={confidences.min()}, max={confidences.max()}"
        )
    sources = Counter(record.get("joints_2d_source", "") for record in exported)
    expected_source = "official_lt_alg_undistorted_annotation_box"
    if sources != Counter({expected_source: len(exported)}):
        raise RuntimeError(f"unexpected joints_2d_source values: {sources}")

    # The export must remain in the same H36M/RUMPL joint order as 3D targets.
    # A left/right lower-body swap is a historically plausible failure mode.
    stride = max(1, len(exported) // 4096)
    sample = exported[::stride]
    projections = np.stack([project_undistorted(record) for record in sample])
    sample_observations = np.stack([
        np.asarray(record["joints_2d"], dtype=np.float64) for record in sample
    ])
    identity_errors = np.linalg.norm(sample_observations - projections, axis=-1)
    swap = np.arange(17)
    swap[1:4], swap[4:7] = np.arange(4, 7), np.arange(1, 4)
    swapped_errors = np.linalg.norm(sample_observations - projections[:, swap], axis=-1)
    identity_mean = float(identity_errors.mean())
    swapped_leg_mean = float(swapped_errors[:, 1:7].mean())
    identity_leg_mean = float(identity_errors[:, 1:7].mean())
    if identity_mean > args.max_identity_projection_error:
        raise RuntimeError(f"2D/3D semantics or observations suspect: mean={identity_mean:.3f}px")
    if swapped_leg_mean <= identity_leg_mean:
        raise RuntimeError("left/right swap unexpectedly fits better than identity")

    group_counts = Counter(key[:-1] for key in export_keys)
    if set(group_counts.values()) != {4}:
        raise RuntimeError(f"not every synchronized group has four cameras: {Counter(group_counts.values())}")

    result = {
        "source": str(source_path.resolve()),
        "export": str(export_path.resolve()),
        "records": len(exported),
        "groups": len(group_counts),
        "sha256": file_sha256(export_path),
        "observation_shape": list(observations.shape),
        "confidence": {
            "min": float(confidences.min()),
            "mean": float(confidences.mean()),
            "max": float(confidences.max()),
        },
        "sampled_2d_vs_undistorted_3d_projection_px": {
            "identity_mean": identity_mean,
            "identity_median": float(np.median(identity_errors)),
            "identity_lower_body_mean": identity_leg_mean,
            "swapped_lower_body_mean": swapped_leg_mean,
        },
        "flip_lower_body_kp_test_required": False,
        "status": "PASS",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
