#!/usr/bin/env python3
"""Merge official HRNet prediction shards into a RUMPL H36M coordinate cache.

The merge is intentionally strict: every source record must have exactly one
prediction, shapes must be COCO-17, and all values must be finite.  COCO-17 is
converted using the same mapping used by the existing RUMPL/MMPose exporter;
the optional lower-body swap is enabled by default to preserve the established
``*_legswap`` protocol.

Because the exporter runs on an undistorted full image, the merged records use
the same K/R/T but an explicit zero-distortion camera for the 2D coordinate
stream.  Original camera ``k``/``p`` values are copied into audit fields before
being zeroed.  This avoids feeding undistorted points to a downstream routine
that assumes distorted image coordinates, while retaining the original
calibration for later diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


MMPOSE2H36M = {
    1: 12,
    2: 14,
    3: 16,
    4: 11,
    5: 13,
    6: 15,
    9: 0,
    11: 5,
    12: 7,
    13: 9,
    14: 6,
    15: 8,
    16: 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--no-swap-lower-body",
        action="store_true",
        help="disable the established RUMPL *_legswap conversion",
    )
    parser.add_argument(
        "--keep-camera-distortion",
        action="store_true",
        help="diagnostic only; do not use for final undistorted coordinate cache",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_fields(
    entry: Any,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    float | None,
    tuple[int, ...] | None,
    dict[str, Any],
]:
    metadata: dict[str, Any] = {}
    if isinstance(entry, dict):
        index = entry.get("record_index", entry.get("index"))
        points = entry.get("keypoints_coco", entry.get("keypoints"))
        scores = entry.get("keypoint_scores_coco", entry.get("scores"))
        bbox = entry.get("detector_bbox_xyxy", entry.get("bbox"))
        detector_score = entry.get("detector_score")
        image_shape = entry.get("image_shape")
        metadata = {
            str(key): value
            for key, value in entry.items()
            if str(key).startswith("mmpose_") or str(key) == "detector_fallback"
        }
    elif isinstance(entry, (tuple, list)) and len(entry) >= 3:
        index, points, scores = entry[:3]
        bbox = entry[3] if len(entry) >= 4 else None
        detector_score = entry[4] if len(entry) >= 5 else None
        image_shape = entry[5] if len(entry) >= 6 else None
    else:
        raise TypeError(f"unsupported prediction entry {type(entry).__name__}")
    if index is None or points is None or scores is None:
        raise KeyError("prediction entry must contain index/keypoints/scores")
    return (
        int(index),
        np.asarray(points, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        None if bbox is None else np.asarray(bbox, dtype=np.float32),
        None if detector_score is None else float(detector_score),
        None if image_shape is None else tuple(int(x) for x in image_shape),
        metadata,
    )


def convert(
    keypoints: np.ndarray, scores: np.ndarray, swap_lower_body: bool
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(17)
    joints = np.zeros((17, 2), dtype=np.float32)
    confidence = np.zeros((17, 1), dtype=np.float32)
    for dst, src in MMPOSE2H36M.items():
        joints[dst] = keypoints[src]
        confidence[dst, 0] = scores[src]
    # H36M virtual joints used by RUMPL: head, neck, pelvis/root, and belly.
    joints[10] = keypoints[0:5].mean(axis=0)
    confidence[10, 0] = scores[0:5].mean()
    joints[8] = keypoints[3:7].mean(axis=0)
    confidence[8, 0] = scores[3:7].mean()
    joints[0] = keypoints[11:13].mean(axis=0)
    confidence[0, 0] = scores[11:13].mean()
    joints[7] = (joints[8] + joints[0]) / 2.0
    confidence[7, 0] = (confidence[8, 0] + confidence[0, 0]) / 2.0
    if swap_lower_body:
        joints[1:4], joints[4:7] = joints[4:7].copy(), joints[1:4].copy()
        confidence[1:4], confidence[4:7] = (
            confidence[4:7].copy(),
            confidence[1:4].copy(),
        )
    return joints, confidence


def zero_camera_distortion(record: dict[str, Any], keep_original: bool) -> None:
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise KeyError("record has no camera dictionary")
    original_k = np.asarray(camera.get("k", np.zeros(3)), dtype=np.float64).copy()
    original_p = np.asarray(camera.get("p", np.zeros(2)), dtype=np.float64).copy()
    record["camera_original_distortion_k"] = original_k
    record["camera_original_distortion_p"] = original_p
    record["camera_2d_coordinate_system"] = (
        "original_distorted" if keep_original else "undistorted_K_equals_K"
    )
    if keep_original:
        return
    camera["k"] = np.zeros_like(original_k)
    camera["p"] = np.zeros_like(original_p)
    if "distCoef" in camera:
        camera["distCoef"] = np.zeros_like(np.asarray(camera["distCoef"]))


def write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else str(value),
        )
        handle.write("\n")
    temporary.replace(path)


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

    found: dict[
        int,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            float | None,
            tuple[int, ...] | None,
            dict[str, Any],
        ],
    ] = {}
    source_manifests: list[dict[str, Any]] = []
    for shard_name in args.shards:
        shard = Path(shard_name).resolve()
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "predictions" in payload:
            predictions = payload["predictions"]
        else:
            predictions = payload
        if not isinstance(predictions, list):
            raise TypeError(f"{shard}: expected list predictions")
        for raw_entry in predictions:
            (
                index,
                points,
                scores,
                bbox,
                detector_score,
                image_shape,
                metadata,
            ) = _entry_fields(raw_entry)
            if not 0 <= index < len(records):
                raise RuntimeError(f"{shard}: out-of-range record index {index}")
            if index in found:
                raise RuntimeError(f"duplicate prediction for record {index}")
            if points.shape != (17, 2) or scores.shape != (17,):
                raise RuntimeError(
                    f"{shard}: record {index} has {points.shape}/{scores.shape}, "
                    "expected (17,2)/(17,)"
                )
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise RuntimeError(f"{shard}: non-finite prediction at record {index}")
            if bbox is not None and (bbox.shape != (4,) or not np.isfinite(bbox).all()):
                raise RuntimeError(f"{shard}: invalid detector bbox at record {index}")
            found[index] = (
                points,
                scores,
                bbox,
                detector_score,
                image_shape,
                metadata,
            )
        sidecar = shard.with_suffix(shard.suffix + ".manifest.json")
        if sidecar.is_file():
            try:
                source_manifests.append(json.loads(sidecar.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                source_manifests.append({"path": str(sidecar), "parse_error": True})

    if len(found) != len(records):
        missing = sorted(set(range(len(records))) - set(found))
        raise RuntimeError(
            f"missing {len(missing)} predictions; first indices: {missing[:20]}"
        )

    swap_lower_body = not args.no_swap_lower_body
    output_records = []
    for index, source_record in enumerate(records):
        points, scores, bbox, detector_score, image_shape, metadata = found[index]
        updated = copy.deepcopy(source_record)
        joints, confidence = convert(points, scores, swap_lower_body)
        updated["joints_2d"] = joints
        updated["joints_2d_conf"] = confidence
        updated["source_2d_protocol"] = "GBT-aligned-HRNet-coordinate-only-v2"
        updated["source_2d_keypoint_model"] = "MMPose-HRNet-W32-COCO"
        updated["source_2d_detector_model"] = "MMDetection-person-detector"
        updated["source_2d_undistorted_full_image"] = not args.keep_camera_distortion
        updated["source_2d_coordinate_system"] = (
            "original_distorted" if args.keep_camera_distortion else "undistorted_K_equals_K"
        )
        updated["source_2d_lower_body_swap"] = swap_lower_body
        if bbox is not None:
            updated["source_2d_detector_bbox_xyxy"] = bbox
        if detector_score is not None:
            updated["source_2d_detector_score"] = detector_score
        if image_shape is not None:
            updated["source_2d_image_shape"] = image_shape
        for key, value in metadata.items():
            updated[f"source_2d_{key}"] = copy.deepcopy(value)
        zero_camera_distortion(updated, keep_original=args.keep_camera_distortion)
        output_records.append(updated)

    write_pickle_atomic(output, output_records)
    merged_manifest = {
        "protocol": "GBT-aligned-HRNet-coordinate-only-v2",
        "created_unix": time.time(),
        "input_pkl": str(input_pkl),
        "input_sha256": sha256_file(input_pkl),
        "shards": [str(Path(x).resolve()) for x in args.shards],
        "shard_sha256": {str(Path(x).resolve()): sha256_file(x) for x in args.shards},
        "record_count": len(records),
        "prediction_count": len(found),
        "swap_lower_body": swap_lower_body,
        "keep_camera_distortion": args.keep_camera_distortion,
        "camera_policy": (
            "retain original k/p for diagnostic distorted coordinates"
            if args.keep_camera_distortion
            else "zero k/p after preserving original values in audit fields"
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_manifests": source_manifests,
    }
    write_json_atomic(manifest_path, merged_manifest)
    print(f"wrote {len(output_records)} records to {output}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
