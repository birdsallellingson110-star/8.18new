#!/usr/bin/env python3
"""Geometry/provenance gates for the GBT-aligned HRNet coordinate cache.

This read-only audit is run after shard merging and before any 3D training. It
does not use predicted 3D poses.  Instead it verifies that the exported 2D
stream is in the advertised full-resolution undistorted coordinate system and
that the H36M 3D annotation projects into the same camera convention.

The most useful gate is ``pred_vs_gt3d_pinhole_px``: the merged HRNet points
are compared with the known 3D annotation projected with K and no distortion.
The raw source points are separately compared against the distorted and
undistorted projections.  A large raw-to-undistorted shift is expected for
H36M; a large merged-vs-undistorted 3D projection error indicates an exporter,
detector, or MMPose coordinate-map problem rather than a 3D-network problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-pkl", required=True)
    parser.add_argument("--source-pkl", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-records", type=int, default=4096)
    parser.add_argument(
        "--expected-coordinate-system",
        choices=("undistorted_K_equals_K", "original_distorted"),
        default="undistorted_K_equals_K",
        help="coordinate/camera convention expected in the merged cache",
    )
    parser.add_argument(
        "--roundtrip-tolerance-px",
        type=float,
        default=1e-3,
        help="maximum accepted MMPose crop inverse-affine round-trip error",
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_parameters(
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = record["camera"]
    K = np.asarray(camera.get("K"), dtype=np.float64).reshape(3, 3)
    R = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera["T"], dtype=np.float64).reshape(3)
    radial = np.asarray(
        record.get("camera_original_distortion_k", camera.get("k", np.zeros(3))),
        dtype=np.float64,
    ).reshape(-1)
    tangential = np.asarray(
        record.get("camera_original_distortion_p", camera.get("p", np.zeros(2))),
        dtype=np.float64,
    ).reshape(-1)
    distortion = np.array(
        [radial[0], radial[1], tangential[0], tangential[1], radial[2]],
        dtype=np.float64,
    )
    return K, R, T, distortion


def project_world(
    joints_3d_camera: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray | None,
) -> np.ndarray:
    points = np.asarray(joints_3d_camera, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(
        points.reshape(-1, 3), rvec, tvec, K,
        None if distortion is None else distortion,
    )
    return projected.reshape(-1, 2)


def values_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("subject"),
        record.get("action"),
        record.get("subaction"),
        record.get("video_id"),
        record.get("image_id"),
    )


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_pkl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, list) or not cache:
        raise ValueError("cache must be a non-empty record list")
    source = None
    source_path = None
    if args.source_pkl:
        source_path = Path(args.source_pkl).resolve()
        with source_path.open("rb") as handle:
            source = pickle.load(handle)
        if not isinstance(source, list) or len(source) != len(cache):
            raise ValueError("source pkl must have exactly the same record count/order")

    n = min(int(args.sample_records), len(cache))
    sample_indices = np.linspace(0, len(cache) - 1, n, dtype=np.int64)
    finite_failures = []
    camera_zero_failures = []
    coordinate_system_failures = []
    shape_failures = []
    raw_to_undistorted = []
    raw_to_distorted_projection = []
    raw_to_undistorted_projection = []
    predicted_to_undistorted_projection = []
    predicted_to_raw_projection = []
    detector_scores = []
    image_shapes = Counter()
    camera_ids = Counter()
    protocol_counts = Counter()
    coordinate_system_counts = Counter()
    inverse_affine_roundtrips = []
    inverse_affine_failures = []

    for index in sample_indices.tolist():
        record = cache[int(index)]
        raw_record = source[int(index)] if source is not None else None
        try:
            points = np.asarray(record["joints_2d"], dtype=np.float64)
            confidence = np.asarray(record["joints_2d_conf"], dtype=np.float64)
            if points.shape != (17, 2) or confidence.shape not in {(17,), (17, 1)}:
                shape_failures.append(int(index))
            if not np.isfinite(points).all() or not np.isfinite(confidence).all():
                finite_failures.append(int(index))
            camera = record["camera"]
            k = np.asarray(camera.get("k", []), dtype=np.float64)
            p = np.asarray(camera.get("p", []), dtype=np.float64)
            if not np.allclose(k, 0.0) or not np.allclose(p, 0.0):
                camera_zero_failures.append(int(index))
            protocol = record.get("source_2d_protocol", "missing")
            protocol_counts[str(protocol)] += 1
            coordinate_system = record.get(
                "source_2d_coordinate_system",
                "undistorted_K_equals_K"
                if record.get("source_2d_undistorted_full_image") is True
                else "original_distorted",
            )
            coordinate_system_counts[str(coordinate_system)] += 1
            if coordinate_system != args.expected_coordinate_system:
                coordinate_system_failures.append(int(index))
            roundtrip = record.get("source_2d_mmpose_inverse_affine_roundtrip_max_px")
            if roundtrip is not None:
                roundtrip = float(roundtrip)
                inverse_affine_roundtrips.append(roundtrip)
                if not np.isfinite(roundtrip) or roundtrip > args.roundtrip_tolerance_px:
                    inverse_affine_failures.append(int(index))
            camera_ids[int(record.get("camera_id", -1))] += 1
            detector_score = record.get("source_2d_detector_score")
            if detector_score is not None:
                detector_scores.append(float(detector_score))
            shape = record.get("source_2d_image_shape")
            if shape is not None:
                image_shapes[tuple(shape)] += 1

            K, R, T, distortion = camera_parameters(record)
            gt_camera = np.asarray(record["joints_3d_camera"], dtype=np.float64)
            projected_distorted = project_world(gt_camera, K, distortion)
            projected_undistorted = project_world(gt_camera, K, None)
            predicted_to_undistorted_projection.extend(
                np.linalg.norm(points - projected_undistorted, axis=1).tolist()
            )
            predicted_to_raw_projection.extend(
                np.linalg.norm(points - projected_distorted, axis=1).tolist()
            )
            if raw_record is not None:
                raw_points = np.asarray(raw_record["joints_2d"], dtype=np.float64)
                raw_to_distorted_projection.extend(
                    np.linalg.norm(raw_points - projected_distorted, axis=1).tolist()
                )
                raw_to_undistorted_projection.extend(
                    np.linalg.norm(raw_points - projected_undistorted, axis=1).tolist()
                )
                undistorted_raw = cv2.undistortPoints(
                    raw_points[:, None, :], K, distortion, P=K
                ).reshape(-1, 2)
                raw_to_undistorted.extend(
                    np.linalg.norm(raw_points - undistorted_raw, axis=1).tolist()
                )
        except Exception:
            finite_failures.append(int(index))

    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(cache):
        groups[group_key(record)].append(index)
    complete_groups = 0
    group_sizes = Counter()
    for indices in groups.values():
        group_sizes[len(indices)] += 1
        if len(indices) == 4 and {int(cache[i]["camera_id"]) for i in indices} == {0, 1, 2, 3}:
            complete_groups += 1

    report = {
        "cache_pkl": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "source_pkl": None if source_path is None else str(source_path),
        "source_sha256": None if source_path is None else sha256(source_path),
        "record_count": len(cache),
        "sample_count": int(n),
        "sample_indices_mode": "linspace",
        "protocol_counts": dict(protocol_counts),
        "coordinate_system_counts": dict(coordinate_system_counts),
        "camera_id_counts_sample": dict(camera_ids),
        "image_shapes_sample": {str(k): v for k, v in image_shapes.items()},
        "detector_score_sample": values_summary(detector_scores),
        "mmpose_inverse_affine_roundtrip_px": values_summary(inverse_affine_roundtrips),
        "camera_zero_distortion_failures": camera_zero_failures[:20],
        "coordinate_system_failures": coordinate_system_failures[:20],
        "inverse_affine_failures": inverse_affine_failures[:20],
        "shape_failures": shape_failures[:20],
        "finite_failures": finite_failures[:20],
        "raw_to_undistorted_shift_px": values_summary(raw_to_undistorted),
        "raw_vs_distorted_3d_projection_px": values_summary(raw_to_distorted_projection),
        "raw_vs_undistorted_3d_projection_px": values_summary(raw_to_undistorted_projection),
        "predicted_vs_undistorted_3d_projection_px": values_summary(
            predicted_to_undistorted_projection
        ),
        "predicted_vs_raw_3d_projection_px": values_summary(predicted_to_raw_projection),
        "group_count": len(groups),
        "group_size_counts": dict(group_sizes),
        "complete_four_camera_groups": complete_groups,
        "gates": {
            "record_count_matches_source": source is None or len(source) == len(cache),
            "all_sample_shapes_valid": not shape_failures,
            "all_sample_finite": not finite_failures,
            "all_sample_cameras_zero_distortion": not camera_zero_failures,
            "all_sample_coordinate_system_matches": not coordinate_system_failures,
            "all_sample_inverse_affine_roundtrip": not inverse_affine_failures,
            "sample_complete_four_camera_groups": complete_groups > 0,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GBT-aligned HRNet cache geometry gate (2026-08-14)",
        "",
        f"- records/sample: `{len(cache)}/{n}`",
        f"- protocol counts: `{dict(protocol_counts)}`",
        f"- coordinate systems: `{dict(coordinate_system_counts)}` (expected `{args.expected_coordinate_system}`)",
        f"- camera IDs in sample: `{dict(camera_ids)}`",
        f"- complete 4-camera groups: `{complete_groups}/{len(groups)}`",
        f"- raw → undistorted shift: `{report['raw_to_undistorted_shift_px']}` px",
        f"- raw vs distorted 3D projection: `{report['raw_vs_distorted_3d_projection_px']}` px",
        f"- raw vs undistorted 3D projection: `{report['raw_vs_undistorted_3d_projection_px']}` px",
        f"- exported HRNet vs undistorted 3D projection: `{report['predicted_vs_undistorted_3d_projection_px']}` px",
        f"- MMPose inverse-affine round-trip: `{report['mmpose_inverse_affine_roundtrip_px']}` px",
        f"- camera zero-distortion failures: `{len(camera_zero_failures)}`",
        f"- coordinate-system failures: `{len(coordinate_system_failures)}`",
        "",
        "The raw-vs-undistorted gap is expected for H36M. The exported HRNet-vs-undistorted projection is the detector/pose 2D gate; it is not a 3D MPJPE result.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    hard_fail = any(
        [
            shape_failures,
            finite_failures,
            camera_zero_failures,
            coordinate_system_failures,
            inverse_affine_failures,
        ]
    )
    if args.fail_on_gate and hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
