#!/usr/bin/env python3
"""Audit the dense GBT-aligned H36M validation cache used by H8.

The temporal experiment must not silently train/evaluate on a partial shard,
an unexpected camera set, detector fallback boxes, or a mixed distorted/
undistorted coordinate convention.  This check is intentionally independent
of the RUMPL dataset loader and writes a compact machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-records", type=int, default=105076)
    parser.add_argument("--expected-subjects", nargs="+", type=int, default=[9, 11])
    parser.add_argument("--expected-cameras", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--expected-frame-stride", type=int, default=5)
    parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=64,
        help="explicit upper bound for source H36M bbox fallback records",
    )
    return parser.parse_args()


def _as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise RuntimeError(f"{name} shape {array.shape} != {shape}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return array


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return payload


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_pkl).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output_json).resolve()

    with input_path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise RuntimeError(f"expected a list in {input_path}")
    if len(records) != args.expected_records:
        raise RuntimeError(
            f"record count {len(records)} != expected {args.expected_records}"
        )

    merged_manifest = _read_json(manifest_path)
    if int(merged_manifest.get("record_count", -1)) != len(records):
        raise RuntimeError("merged manifest record_count does not match payload")
    if merged_manifest.get("protocol") != "GBT-aligned-HRNet-coordinate-only-v2":
        raise RuntimeError("unexpected merged protocol")
    if bool(merged_manifest.get("keep_camera_distortion", True)):
        raise RuntimeError("validation cache keeps camera distortion; expected zero-distortion convention")

    source_manifests = merged_manifest.get("source_manifests", [])
    # The historical merge utility looked for ``shard.pkl.manifest.json``
    # while the exporter writes ``shard.manifest.json``.  Recover the latter
    # from the explicit shard list instead of treating an empty embedded list
    # as evidence that the run was unaudited.
    if not isinstance(source_manifests, list):
        raise RuntimeError("merged manifest source_manifests is not a list")
    if not source_manifests:
        for shard_name in merged_manifest.get("shards", []):
            shard_path = Path(str(shard_name))
            sidecar = shard_path.with_name(shard_path.stem + ".manifest.json")
            if sidecar.is_file():
                source_manifests.append(_read_json(sidecar))
    if not source_manifests:
        raise RuntimeError("could not locate shard manifests")
    shard_summary = []
    manifest_fallback_total = 0
    for shard in source_manifests:
        if int(shard.get("error_count", -1)) != 0:
            raise RuntimeError(f"shard has errors: {shard.get('error_count')}")
        fallback_count = int(shard.get("fallback_count", -1))
        if fallback_count < 0:
            raise RuntimeError("shard manifest has no fallback_count")
        manifest_fallback_total += fallback_count
        shard_summary.append(
            {
                "shard_id": int(shard.get("shard_id", -1)),
                "prediction_count": int(shard.get("prediction_count", -1)),
                "error_count": int(shard.get("error_count", -1)),
                "fallback_count": fallback_count,
            }
        )

    expected_cameras = tuple(sorted(args.expected_cameras))
    subjects = Counter()
    frame_cameras: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    sequence_frames: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    protocol_counts = Counter()
    detector_scores = []
    fallback_record_count = 0
    max_camera_distortion = 0.0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"record {index} is not a dict")
        subject = int(record["subject"])
        action = int(record["action"])
        subaction = int(record["subaction"])
        image_id = int(record["image_id"])
        camera_id = int(record["camera_id"])
        subjects[subject] += 1
        frame_key = (subject, action, subaction, image_id)
        sequence_key = (subject, action, subaction, camera_id)
        frame_cameras[frame_key].append(camera_id)
        sequence_frames[sequence_key].append(image_id)

        protocol_counts[str(record.get("source_2d_protocol"))] += 1
        if record.get("source_2d_coordinate_system") != "undistorted_K_equals_K":
            raise RuntimeError(f"record {index} has unexpected 2D coordinate system")
        if not bool(record.get("source_2d_undistorted_full_image", False)):
            raise RuntimeError(f"record {index} was not full-image undistorted")
        _as_array(record.get("joints_2d"), (17, 2), f"record {index} joints_2d")
        _as_array(record.get("joints_2d_conf"), (17, 1), f"record {index} joints_2d_conf")
        camera = record.get("camera")
        if not isinstance(camera, dict):
            raise RuntimeError(f"record {index} has no camera dictionary")
        for key in ("k", "p"):
            values = np.asarray(camera.get(key))
            if not np.isfinite(values).all():
                raise RuntimeError(f"record {index} camera.{key} is non-finite")
            max_camera_distortion = max(max_camera_distortion, float(np.max(np.abs(values))))
        fallback_marker = (
            "detector_fallback" in record or "source_2d_detector_fallback" in record
        )
        if fallback_marker:
            fallback_record_count += 1
        detector_score = float(record.get("source_2d_detector_score", -1.0))
        if not np.isfinite(detector_score) or (
            detector_score < 0.01 and not (fallback_marker and detector_score == 0.0)
        ):
            raise RuntimeError(f"record {index} has invalid detector score {detector_score}")
        detector_scores.append(detector_score)

    if set(subjects) != set(args.expected_subjects):
        raise RuntimeError(f"subjects {sorted(subjects)} != expected {sorted(args.expected_subjects)}")
    bad_camera_frames = {
        key: sorted(values)
        for key, values in frame_cameras.items()
        if tuple(sorted(values)) != expected_cameras or len(values) != len(set(values))
    }
    if bad_camera_frames:
        first = next(iter(bad_camera_frames.items()))
        raise RuntimeError(f"camera coverage failure ({len(bad_camera_frames)} frames), first={first}")

    stride_counts = Counter()
    for key, frame_ids in sequence_frames.items():
        unique = sorted(set(frame_ids))
        if len(unique) != len(frame_ids):
            raise RuntimeError(f"duplicate image_id in sequence {key}")
        stride_counts.update(np.diff(unique).tolist())
    if stride_counts and set(stride_counts) != {args.expected_frame_stride}:
        raise RuntimeError(f"unexpected frame strides: {stride_counts.most_common(10)}")
    if fallback_record_count != manifest_fallback_total:
        raise RuntimeError(
            "fallback count mismatch: records="
            f"{fallback_record_count}, shard manifests={manifest_fallback_total}"
        )
    if fallback_record_count > args.max_fallbacks:
        raise RuntimeError(
            f"fallback count {fallback_record_count} exceeds max {args.max_fallbacks}"
        )
    if max_camera_distortion > 1e-8:
        raise RuntimeError(
            "merged camera distortion is not zero under the undistorted_K_equals_K protocol: "
            f"max_abs={max_camera_distortion}"
        )

    report = {
        "status": "PASS",
        "input_pkl": str(input_path),
        "manifest": str(manifest_path),
        "record_count": len(records),
        "subjects": dict(sorted(subjects.items())),
        "num_synchronized_frames": len(frame_cameras),
        "num_camera_sequences": len(sequence_frames),
        "camera_set": list(expected_cameras),
        "frame_stride_counts": {str(k): int(v) for k, v in sorted(stride_counts.items())},
        "protocol_counts": dict(protocol_counts),
        "detector_score_min": float(min(detector_scores)),
        "detector_score_mean": float(np.mean(detector_scores)),
        "fallback_record_count": fallback_record_count,
        "camera_distortion_max_abs": max_camera_distortion,
        "shards": shard_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
