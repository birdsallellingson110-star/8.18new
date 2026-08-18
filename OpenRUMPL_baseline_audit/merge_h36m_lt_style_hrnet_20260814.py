#!/usr/bin/env python3
"""Merge LT-style HRNet coordinate shards into a crop-coordinate RUMPL PKL.

The resulting records contain only HRNet COCO-17 coordinates/confidences as
the learned 2-D input.  Coordinates are in the 384x384 LT crop frame and the
record camera K is updated with LT's crop/resize transform.  This is
mathematically equivalent to full-image undistorted coordinates with the
original K, but preserves the exact LT coordinate/camera contract for a fair
frontend ablation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from merge_h36m_gbt_aligned_hrnet_20260814 import (  # noqa: E402
    MMPOSE2H36M,
    convert,
    sha256_file,
    write_json_atomic,
    write_pickle_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--no-swap-lower-body", action="store_true")
    return parser.parse_args()


def entry_fields(entry: Any) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any]]:
    if not isinstance(entry, dict):
        raise TypeError(f"LT-style entry must be a dict, got {type(entry).__name__}")
    index = entry.get("record_index", entry.get("index"))
    points = entry.get("keypoints_coco", entry.get("keypoints"))
    scores = entry.get("keypoint_scores_coco", entry.get("scores"))
    if index is None or points is None or scores is None:
        raise KeyError("LT-style prediction requires record_index/keypoints/scores")
    metadata = {
        str(key): value
        for key, value in entry.items()
        if str(key).startswith("lt_") or str(key).startswith("mmpose_")
    }
    return (
        int(index),
        np.asarray(points, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        metadata,
    )


def matrix_metadata(metadata: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"prediction has no required metadata {key}")
    value = np.asarray(metadata[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key} has invalid shape/value {value.shape}")
    return value


def vector_metadata(metadata: dict[str, Any], key: str, size: int) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"prediction has no required metadata {key}")
    value = np.asarray(metadata[key]).reshape(-1)
    if value.size != size:
        raise ValueError(f"{key} has invalid size {value.size}, expected {size}")
    return value


def update_camera_for_lt(record: dict[str, Any], K_lt: np.ndarray) -> None:
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise KeyError("record has no camera dictionary")
    original_K = np.asarray(camera.get("K"), dtype=np.float64).copy()
    original_k = np.asarray(camera.get("k", np.zeros(3)), dtype=np.float64).copy()
    original_p = np.asarray(camera.get("p", np.zeros(2)), dtype=np.float64).copy()
    record["camera_original_K"] = original_K
    record["camera_original_distortion_k"] = original_k
    record["camera_original_distortion_p"] = original_p
    camera["K"] = np.asarray(K_lt, dtype=np.float64).copy()
    camera["fx"] = float(K_lt[0, 0])
    camera["fy"] = float(K_lt[1, 1])
    camera["cx"] = float(K_lt[0, 2])
    camera["cy"] = float(K_lt[1, 2])
    # The image presented to HRNet was already undistorted.  Keep an explicit
    # zero-distortion camera instead of mixing crop coordinates with original
    # distortion coefficients.
    camera["k"] = np.zeros_like(original_k)
    camera["p"] = np.zeros_like(original_p)
    if "distCoef" in camera:
        camera["distCoef"] = np.zeros_like(np.asarray(camera["distCoef"]))
    record["camera_2d_coordinate_system"] = "lt_crop_384x384_undistorted_K_updated"
    record["camera_lt_K_after_crop_resize"] = np.asarray(K_lt, dtype=np.float64).copy()


def main() -> None:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    output = Path(args.output).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    with input_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"expected list source records, got {type(records).__name__}")

    found: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    source_manifests: list[dict[str, Any]] = []
    for shard_name in args.shards:
        shard = Path(shard_name).resolve()
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        predictions = payload.get("predictions") if isinstance(payload, dict) and "predictions" in payload else payload
        if not isinstance(predictions, list):
            raise TypeError(f"{shard}: expected list predictions")
        for raw_entry in predictions:
            index, points, scores, metadata = entry_fields(raw_entry)
            if not 0 <= index < len(records):
                raise RuntimeError(f"{shard}: out-of-range record index {index}")
            if index in found:
                raise RuntimeError(f"duplicate LT-style prediction for record {index}")
            if points.shape != (17, 2) or scores.shape != (17,):
                raise RuntimeError(f"{shard}: record {index} has {points.shape}/{scores.shape}")
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise RuntimeError(f"{shard}: non-finite LT-style prediction at record {index}")
            matrix_metadata(metadata, "lt_camera_K_after_crop_resize", (3, 3))
            bbox = vector_metadata(metadata, "lt_bbox_xyxy_int", 4)
            crop_shape = vector_metadata(metadata, "lt_crop_shape_before_resize", 2)
            resize_shape = vector_metadata(metadata, "lt_resize_shape", 2)
            if np.any(resize_shape != resize_shape[0]) or resize_shape[0] <= 0:
                raise RuntimeError(f"{shard}: invalid LT resize shape at record {index}")
            if np.any(crop_shape <= 0) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise RuntimeError(f"{shard}: invalid LT bbox/crop metadata at record {index}")
            found[index] = (points, scores, metadata)
        sidecar = shard.with_suffix(shard.suffix + ".manifest.json")
        if sidecar.is_file():
            try:
                source_manifests.append(json.loads(sidecar.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                source_manifests.append({"path": str(sidecar), "parse_error": True})

    if len(found) != len(records):
        missing = sorted(set(range(len(records))) - set(found))
        raise RuntimeError(f"missing {len(missing)} LT-style predictions; first indices: {missing[:20]}")

    swap_lower_body = not args.no_swap_lower_body
    output_records: list[dict[str, Any]] = []
    for index, source_record in enumerate(records):
        points, scores, metadata = found[index]
        updated = copy.deepcopy(source_record)
        # Preserve the source annotation for a later pixel-level audit, then
        # replace only the 2-D input consumed by RUMPL.
        updated["source_2d_original_joints_2d"] = np.asarray(
            source_record["joints_2d"], dtype=np.float32
        ).copy()
        updated["source_2d_original_camera_K"] = np.asarray(
            source_record["camera"]["K"], dtype=np.float64
        ).copy()
        joints, confidence = convert(points, scores, swap_lower_body)
        updated["joints_2d"] = joints
        updated["joints_2d_conf"] = confidence
        K_lt = matrix_metadata(metadata, "lt_camera_K_after_crop_resize", (3, 3))
        update_camera_for_lt(updated, K_lt)
        updated["source_2d_protocol"] = "LT-style-HRNet-coordinate-only-v1"
        updated["source_2d_keypoint_model"] = "MMPose-HRNet-W32-COCO"
        updated["source_2d_detector_model"] = "none_GT_annotation_bbox"
        updated["source_2d_bbox_source"] = "H36M_annotation_box_rounded_like_LT"
        updated["source_2d_undistortion"] = "cv2.undistort_K_equals_K_before_crop"
        updated["source_2d_coordinate_system"] = "lt_crop_384x384_undistorted_K_updated"
        updated["source_2d_lower_body_swap"] = swap_lower_body
        updated["source_2d_lt_bbox_xyxy_int"] = np.asarray(
            metadata["lt_bbox_xyxy_int"], dtype=np.int32
        ).copy()
        updated["source_2d_lt_crop_shape_before_resize"] = tuple(
            int(x) for x in np.asarray(metadata["lt_crop_shape_before_resize"]).reshape(-1)
        )
        updated["source_2d_lt_resize_shape"] = tuple(
            int(x) for x in np.asarray(metadata["lt_resize_shape"]).reshape(-1)
        )
        for key, value in metadata.items():
            if key.startswith("mmpose_"):
                updated[f"source_2d_{key}"] = copy.deepcopy(value)
        output_records.append(updated)

    write_pickle_atomic(output, output_records)
    merged_manifest = {
        "protocol": "LT-style-HRNet-coordinate-only-v1",
        "created_unix": time.time(),
        "input_pkl": str(input_pkl),
        "input_sha256": sha256_file(input_pkl),
        "shards": [str(Path(x).resolve()) for x in args.shards],
        "shard_sha256": {str(Path(x).resolve()): sha256_file(x) for x in args.shards},
        "record_count": len(records),
        "prediction_count": len(found),
        "swap_lower_body": swap_lower_body,
        "bbox_source": "H36M_annotation_box_rounded_like_LT",
        "lt_preprocessing": {
            "undistort": "full_image_cv2_undistort_new_K_original_K",
            "crop": "official_LT_PIL_crop_image",
            "resize": "official_LT_cv2_INTER_AREA_384x384",
            "camera": "official_LT_update_after_crop_then_update_after_resize",
        },
        "coordinate_contract": "HRNet COCO-17 crop pixels + confidences, converted to RUMPL H36M-17",
        "camera_contract": "R/K/t with crop-resized K and zero distortion",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_manifests": source_manifests,
    }
    write_json_atomic(manifest_path, merged_manifest)
    print(f"wrote {len(output_records)} LT-style records to {output}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
