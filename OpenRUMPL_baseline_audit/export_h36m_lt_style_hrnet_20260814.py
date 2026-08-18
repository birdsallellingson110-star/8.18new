#!/usr/bin/env python3
"""Export an HRNet coordinate cache with the official LT image protocol.

This is deliberately a *coordinate-only* adapter.  The source of the 2-D
observations is still MMPose HRNet-W32 (COCO-17 coordinates and peak scores).
The only frontend change relative to the full-image HRNet line is the
Learnable-Triangulation (LT, ICCV'19) preprocessing contract:

  1. read the H36M annotation bbox (LT's released H36M setup uses GT boxes);
  2. undistort the full image with the original K as the new K;
  3. crop using LT's PIL crop semantics and integer bbox labels;
  4. resize the crop to 384x384;
  5. run HRNet on that crop and keep only coordinates/confidences;
  6. record the crop-space camera update so a downstream RUMPL cache can use
     the coordinates and the matching K.

No LT heatmaps, ResNet features, learned triangulator outputs, or image
features are written.  A separate merge script converts COCO-17 to RUMPL's
H36M-17 order and updates each record's camera K after crop/resize.

The GT-box version is intentionally labelled as an LT preprocessing upper
bound.  It must not be confused with a detector-box comparison line; a
detector-box variant can be added later with the same crop/camera code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mmpose.apis import inference_topdown, init_model

# Reuse the audited MMPose protocol helpers without touching the running
# full-image exporter.  The imported module is local to this repository.
from export_h36m_gbt_aligned_hrnet_20260814 import (  # noqa: E402
    _metadata_array,
    _to_numpy,
    camera_parameters,
    configure_pose_test_protocol,
    extract_pose_transform_metadata,
    json_value,
    package_version,
    resolve_image,
    selected_indices,
    sha256_file,
    undistort_full_image,
)

# These two functions are copied from the official LT repository's
# ``mvn.utils.img`` implementation via a small import, so PIL crop edge
# behavior and INTER_AREA resizing stay identical to LT.
LT_ROOT = Path("/home/lixiaob/cjy/reference/learnable-triangulation-official")
sys.path.insert(0, str(LT_ROOT))
from mvn.utils.img import crop_image, resize_image  # noqa: E402
from mvn.utils.multiview import Camera as LTCamera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LT-style crop/undistort frontend with HRNet coordinates"
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=1.0,
        help="HRNet's internal bbox padding; 1.0 avoids adding LT-style context",
    )
    parser.add_argument("--no-flip-test", action="store_true")
    parser.add_argument("--no-shift-heatmap", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write an error entry instead of aborting; never use for final training",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="diagnostic only; final LT-style protocol leaves this disabled",
    )
    return parser.parse_args()


def integer_lt_bbox(record: dict[str, Any]) -> tuple[int, int, int, int]:
    """Mirror LT's int16 label and PIL crop convention."""
    if "box" not in record:
        raise KeyError("record has no LT/H36M annotation box")
    values = tuple(int(round(float(value))) for value in np.asarray(record["box"]).reshape(4))
    left, upper, right, lower = values
    if right <= left or lower <= upper:
        raise ValueError(f"invalid LT bbox {values}")
    return values


def update_lt_camera(record: dict[str, Any], bbox: tuple[int, int, int, int],
                     crop_shape: tuple[int, int], crop_size: int) -> np.ndarray:
    """Return the official LT camera K after crop and resize.

    LT keeps R and t fixed and updates only K.  The source image was
    undistorted with original K, so the output distortion is zero.
    """
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise KeyError("record has no camera dictionary")
    K, distortion = camera_parameters(record)
    R = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    if "t" in camera:
        t = np.asarray(camera["t"], dtype=np.float64).reshape(3, 1)
    else:
        T = np.asarray(camera["T"], dtype=np.float64).reshape(3, 1)
        t = -R @ T
    lt_camera = LTCamera(R, t, K, distortion)
    lt_camera.update_after_crop(bbox)
    lt_camera.update_after_resize(crop_shape, (crop_size, crop_size))
    return np.asarray(lt_camera.K, dtype=np.float64)


def infer_pose_on_lt_crop(
    pose_model: Any, image_crop: np.ndarray, crop_size: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run official MMPose HRNet on the already cropped LT image.

    The synthetic full-image bbox covers the complete 384x384 LT crop.  The
    HRNet model still uses its released 384x288 codec; no image/feature tensor
    is exposed to RUMPL, only the decoded points and scores.
    """
    bbox = np.asarray([[0.0, 0.0, float(crop_size), float(crop_size)]], dtype=np.float32)
    samples = inference_topdown(
        pose_model,
        image_crop,
        bboxes=bbox,
        bbox_format="xyxy",
    )
    if len(samples) != 1:
        raise RuntimeError(f"expected one HRNet pose sample, got {len(samples)}")
    instances = samples[0].pred_instances
    keypoints = _to_numpy(instances.keypoints)
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 2)
    if keypoints.shape[0] != 1:
        raise RuntimeError(f"expected one HRNet pose, got {keypoints.shape}")
    if hasattr(instances, "keypoint_scores"):
        scores = _to_numpy(instances.keypoint_scores)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, 17)
        if scores.shape[0] != 1:
            raise RuntimeError(f"unexpected HRNet score shape {scores.shape}")
        confidence = scores[0]
    else:
        confidence = np.ones((17,), dtype=np.float32)
    points = keypoints[0]
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise RuntimeError("HRNet returned non-finite keypoints/confidence")
    metadata = extract_pose_transform_metadata(samples[0], points)
    metadata["lt_hrnet_point_space"] = "lt_crop_pixels_384x384"
    metadata["lt_hrnet_crop_bbox_input_xyxy"] = np.asarray(
        [0.0, 0.0, float(crop_size), float(crop_size)], dtype=np.float32
    )
    if hasattr(instances, "keypoints_visible"):
        visibility = _to_numpy(instances.keypoints_visible)
        visibility = np.asarray(visibility, dtype=np.float32).reshape(-1, 17)
        if visibility.shape[0] == 1 and np.isfinite(visibility[0]).all():
            metadata["mmpose_keypoint_visibility_coco"] = visibility[0]
    return points, confidence, metadata


def prediction_entry(
    index: int,
    points: np.ndarray,
    confidence: np.ndarray,
    bbox: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
    crop_size: int,
    camera_K: np.ndarray,
    image_shape: tuple[int, ...],
    transform_metadata: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "record_index": int(index),
        "keypoints_coco": np.asarray(points, dtype=np.float32),
        "keypoint_scores_coco": np.asarray(confidence, dtype=np.float32),
        "lt_bbox_xyxy_int": np.asarray(bbox, dtype=np.int32),
        "lt_crop_shape_before_resize": tuple(int(x) for x in crop_shape),
        "lt_resize_shape": (int(crop_size), int(crop_size)),
        "lt_camera_K_after_crop_resize": np.asarray(camera_K, dtype=np.float64),
        "image_shape_original": tuple(int(x) for x in image_shape),
        "frontend_bbox_source": "H36M_annotation_box_rounded_like_LT",
        "frontend_image_protocol": "LT_undistort_crop_resize_384",
    }
    entry.update(transform_metadata)
    return entry


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
        json.dump(json_value(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if args.no_undistort:
        print("WARNING: --no-undistort is diagnostic only", file=sys.stderr)

    input_pkl = Path(args.input_pkl).resolve()
    images_root = Path(args.images_root).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else output.with_suffix(output.suffix + ".manifest.json")
    with input_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"expected list input records, got {type(records).__name__}")
    indices = selected_indices(len(records), args.shard_id, args.num_shards, args.max_records)

    print(
        f"loading HRNet on {args.device}; LT-style records={len(records)} "
        f"selected={len(indices)} shard={args.shard_id}/{args.num_shards}",
        flush=True,
    )
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    pose_protocol = configure_pose_test_protocol(pose_model, args)
    print(f"HRNet test protocol: {json_value(pose_protocol)}", flush=True)

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()
    for done, index in enumerate(indices, start=1):
        record = records[index]
        try:
            image_path = resolve_image(images_root, record)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2.imread failed for {image_path}")
            K, distortion = camera_parameters(record)
            processed = image if args.no_undistort else undistort_full_image(image, K, distortion)
            bbox = integer_lt_bbox(record)
            crop = crop_image(processed, bbox)
            crop_shape = tuple(int(x) for x in crop.shape[:2])
            if min(crop_shape) <= 0:
                raise RuntimeError(f"empty LT crop {bbox} for {image_path}")
            resized = resize_image(crop, (args.crop_size, args.crop_size))
            K_lt = update_lt_camera(record, bbox, crop_shape, args.crop_size)
            points, confidence, metadata = infer_pose_on_lt_crop(
                pose_model, resized, args.crop_size
            )
            metadata["lt_bbox_xyxy_int"] = np.asarray(bbox, dtype=np.int32)
            metadata["lt_crop_shape_before_resize"] = np.asarray(crop_shape, dtype=np.int32)
            metadata["lt_resize_shape"] = np.asarray([args.crop_size, args.crop_size], dtype=np.int32)
            metadata["lt_camera_K_after_crop_resize"] = K_lt
            predictions.append(
                prediction_entry(
                    index, points, confidence, bbox, crop_shape, args.crop_size,
                    K_lt, tuple(image.shape), metadata,
                )
            )
        except Exception as exc:
            error = {
                "record_index": int(index),
                "image": record.get("image"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)
            if not args.allow_missing:
                raise RuntimeError(json.dumps(json_value(error), ensure_ascii=False)) from exc
        if done == 1 or done % 100 == 0 or done == len(indices):
            rate = done / max(time.time() - started, 1e-6)
            print(
                f"shard={args.shard_id} {done}/{len(indices)} "
                f"predictions={len(predictions)} errors={len(errors)} rate={rate:.2f}/s",
                flush=True,
            )

    write_pickle_atomic(output, predictions)
    payload = {
        "protocol": "LT-style-HRNet-coordinate-only-v1",
        "created_unix": time.time(),
        "command": " ".join(str(x) for x in sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "mmpose": package_version("mmpose"),
            "mmengine": package_version("mmengine"),
            "mmcv": package_version("mmcv"),
        },
        "input_pkl": str(input_pkl),
        "input_sha256": sha256_file(input_pkl),
        "images_root": str(images_root),
        "pose_config": str(Path(args.pose_config).resolve()),
        "pose_config_sha256": sha256_file(args.pose_config),
        "pose_checkpoint": str(Path(args.pose_checkpoint).resolve()),
        "pose_checkpoint_sha256": sha256_file(args.pose_checkpoint),
        "device": args.device,
        "record_count_total": len(records),
        "record_count_selected": len(indices),
        "prediction_count": len(predictions),
        "error_count": len(errors),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "bbox_source": "H36M_annotation_box_rounded_like_LT",
        "bbox_scale": 1.0,
        "crop_semantics": "official_LT_mvn.utils.img.crop_image_PIL",
        "resize_semantics": "official_LT_mvn.utils.img.resize_image_cv2_INTER_AREA",
        "crop_size": [args.crop_size, args.crop_size],
        "undistort_full_image": not args.no_undistort,
        "camera_update": "LT_Camera.update_after_crop_then_update_after_resize",
        "camera_coordinate_system": "lt_crop_384x384_undistorted_K_updated",
        "pose_api": "mmpose.apis.inference_topdown_on_external_LT_crop",
        "saved_payload": "COCO-17 HRNet crop coordinates and scores plus LT audit metadata",
        "errors": errors,
    }
    write_json_atomic(manifest, payload)
    print(f"wrote {len(predictions)} predictions to {output}")
    print(f"wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
