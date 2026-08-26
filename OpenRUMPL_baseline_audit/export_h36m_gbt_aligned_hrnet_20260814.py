#!/usr/bin/env python3
"""Export a coordinate-only, GBT-aligned H36M HRNet input cache.

This exporter deliberately uses the public MMDetection and MMPose APIs rather
than the high-level MMPose inferencer.  That keeps the two stages explicit:

1. read the original H36M image and calibration;
2. undistort the *full image* with ``K_new=K``;
3. run an official MMDetection person detector on that same image;
4. run official MMPose HRNet-W32 top-down inference using the detector box;
5. save only COCO-17 coordinates/confidences (plus auditable metadata).

The output is a prediction shard.  ``merge_h36m_gbt_aligned_hrnet_20260814.py``
converts COCO-17 to the RUMPL H36M-17 layout and makes the corresponding
zero-distortion camera convention explicit.  No heatmaps, image features, A1D
features, or 3D predictions are written to the 3D-network input.

The detector and pose model are intentionally command-line arguments.  This
prevents silently changing the external-comparison protocol when a checkpoint
is replaced.  For a local smoke test, the existing RTMDet-M checkpoint can be
passed; the selected final engineering line uses the official MMDetection
YOLOX-X checkpoint.  The manifest must still record that GBT only says YOLOX
and does not specify its exact variant.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmengine.registry import init_default_scope

from h36m_occlusion_protocol_20260822 import (
    apply_white_joint_squares,
    protocol_manifest as occlusion_protocol_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official MMDetection + MMPose coordinate-only H36M exporter"
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument(
        "--detector-test-score-thr",
        type=float,
        default=None,
        help=(
            "override MMDetection model test_cfg.score_thr; needed when the "
            "external threshold is below the config threshold"
        ),
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=None,
        help=(
            "override MMPose GetBBoxCenterScale padding; None keeps the official "
            "checkpoint config (1.25)"
        ),
    )
    parser.add_argument(
        "--no-flip-test",
        action="store_true",
        help="disable the official HRNet horizontal flip test (diagnostic only)",
    )
    parser.add_argument(
        "--no-shift-heatmap",
        action="store_true",
        help="disable the official HRNet heatmap shift (diagnostic only)",
    )
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="person class id in the detector label space (COCO=0)",
    )
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
        "--fallback-record-box",
        action="store_true",
        help=(
            "if the detector has no person proposal, use the source H36M box "
            "for that record and mark it in the manifest; this is an explicit "
            "engineering fallback and is not strict detector-only GBT"
        ),
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="diagnostic only; final GBT-aligned protocol must leave this disabled",
    )
    parser.add_argument(
        "--occlusion-prob",
        type=float,
        default=0.0,
        help="GBT H36M-Occl per-joint white-square probability (paper: 0.1)",
    )
    parser.add_argument(
        "--occlusion-square-fraction",
        type=float,
        default=0.15,
        help="square side / longer annotation-box side; absent from GBT paper",
    )
    parser.add_argument(
        "--occlusion-seed",
        type=int,
        default=20260822,
        help="deterministic mask seed; absent from GBT paper",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def json_value(value: Any) -> Any:
    """Convert numpy/scalar values to manifest-safe JSON values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def camera_parameters(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise KeyError("record has no camera dictionary")
    if "K" in camera:
        K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    else:
        K = np.array(
            [
                [float(camera["fx"]), 0.0, float(camera["cx"])],
                [0.0, float(camera["fy"]), float(camera["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    radial = np.asarray(camera.get("k", np.zeros(3)), dtype=np.float64).reshape(-1)
    tangential = np.asarray(camera.get("p", np.zeros(2)), dtype=np.float64).reshape(-1)
    if radial.size < 3 or tangential.size < 2:
        raise ValueError("H36M camera must contain k[3] and p[2] distortion values")
    # H36M/RUMPL stores k=[k1,k2,k3], p=[p1,p2], whereas OpenCV expects
    # [k1,k2,p1,p2,k3].
    distortion = np.array(
        [radial[0], radial[1], tangential[0], tangential[1], radial[2]],
        dtype=np.float64,
    )
    if not np.isfinite(K).all() or not np.isfinite(distortion).all():
        raise ValueError("non-finite camera calibration")
    return K, distortion


def resolve_image(images_root: Path, record: dict[str, Any]) -> Path:
    candidates = [record.get("image"), record.get("source_image")]
    for relative in candidates:
        if not relative:
            continue
        candidate = images_root / str(relative)
        if candidate.is_file():
            return candidate
    tried = [str(images_root / str(x)) for x in candidates if x]
    raise FileNotFoundError(f"record has no image file; tried: {tried}")


def undistort_full_image(image: np.ndarray, K: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Return same-resolution image with K as the new camera matrix."""
    return cv2.undistort(image, K, distortion, None, K)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def configure_pose_test_protocol(pose_model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Apply auditable, inference-only HRNet protocol overrides.

    MMPose's ``inference_topdown`` rebuilds the test pipeline from
    ``model.cfg.test_dataloader.dataset.pipeline``.  Changing only the model
    object would therefore leave the old crop transform in place.  We update
    both the config and model test_cfg so bbox padding and flip-test settings
    are effective and are recorded in the shard manifest.
    """

    pipeline = pose_model.cfg.test_dataloader.dataset.pipeline
    padding_hits = 0
    if args.bbox_padding is not None:
        if not np.isfinite(float(args.bbox_padding)) or float(args.bbox_padding) <= 0:
            raise ValueError("--bbox-padding must be a finite positive number")
        for transform in pipeline:
            if isinstance(transform, dict) and transform.get("type") == "GetBBoxCenterScale":
                transform["padding"] = float(args.bbox_padding)
                padding_hits += 1
        if padding_hits == 0:
            raise RuntimeError("pose test pipeline has no GetBBoxCenterScale transform")

    def set_test_cfg(key: str, value: Any) -> None:
        if hasattr(pose_model, "test_cfg") and pose_model.test_cfg is not None:
            pose_model.test_cfg[key] = value
        model_cfg = pose_model.cfg.get("model")
        if model_cfg is not None and model_cfg.get("test_cfg") is not None:
            model_cfg["test_cfg"][key] = value

    if args.no_flip_test:
        set_test_cfg("flip_test", False)
    if args.no_shift_heatmap:
        set_test_cfg("shift_heatmap", False)

    return {
        "bbox_padding_override": (
            None if args.bbox_padding is None else float(args.bbox_padding)
        ),
        "bbox_padding_pipeline_hits": int(padding_hits),
        "flip_test": bool(pose_model.test_cfg.get("flip_test", False)),
        "shift_heatmap": bool(pose_model.test_cfg.get("shift_heatmap", False)),
    }


def configure_detector_test_protocol(
    detector: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Optionally expose low-score detector candidates for a controlled test.

    MMDetection normally removes detections below ``model.test_cfg.score_thr``
    before the exporter can apply its own threshold.  The default leaves the
    official config untouched.  A low-score experiment must explicitly set
    both values and is kept separate from the main 0.01 protocol.
    """

    value = args.detector_test_score_thr
    if value is None:
        model_cfg = detector.cfg.get("model")
        test_cfg = model_cfg.get("test_cfg") if model_cfg is not None else None
        current = None if test_cfg is None else test_cfg.get("score_thr")
        return {
            "override": None,
            "effective_test_score_thr": None if current is None else float(current),
        }
    if not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError("--detector-test-score-thr must be finite and non-negative")
    if hasattr(detector, "test_cfg") and detector.test_cfg is not None:
        detector.test_cfg["score_thr"] = float(value)
    model_cfg = detector.cfg.get("model")
    if model_cfg is not None and model_cfg.get("test_cfg") is not None:
        model_cfg["test_cfg"]["score_thr"] = float(value)
    return {
        "override": float(value),
        "effective_test_score_thr": float(value),
    }


def _metadata_array(metainfo: dict[str, Any], key: str) -> np.ndarray | None:
    value = metainfo.get(key)
    if value is None:
        return None
    value = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    return value.copy()


def extract_pose_transform_metadata(
    sample: Any, points: np.ndarray
) -> dict[str, Any]:
    """Save the crop transform that maps HRNet input pixels to image pixels.

    MMPose decodes heatmap coordinates in the 288x384 input space and then
    applies ``p / input_size * input_scale + input_center - input_scale/2``.
    Saving the values makes the GHT-style inverse-crop contract auditable
    instead of relying on an undocumented assumption about coordinate space.
    """

    metainfo = dict(getattr(sample, "metainfo", {}) or {})
    center = _metadata_array(metainfo, "input_center")
    scale = _metadata_array(metainfo, "input_scale")
    input_size = _metadata_array(metainfo, "input_size")
    result: dict[str, Any] = {
        "mmpose_input_center": center,
        "mmpose_input_scale": scale,
        "mmpose_input_size": input_size,
        "mmpose_coordinate_space": "image_pixels_after_inverse_affine",
    }
    if center is not None and scale is not None and input_size is not None:
        if center.size != 2 or scale.size != 2 or input_size.size != 2:
            raise RuntimeError(
                "unexpected MMPose transform metadata shapes: "
                f"center={center.shape}, scale={scale.shape}, input_size={input_size.shape}"
            )
        crop_points = (
            (np.asarray(points, dtype=np.float32) - center[None, :]
             + 0.5 * scale[None, :])
            / scale[None, :]
            * input_size[None, :]
        )
        restored = (
            crop_points / input_size[None, :] * scale[None, :]
            + center[None, :] - 0.5 * scale[None, :]
        )
        result["mmpose_inverse_affine_roundtrip_max_px"] = float(
            np.max(np.linalg.norm(restored - points, axis=-1))
        )
    else:
        result["mmpose_inverse_affine_roundtrip_max_px"] = None
    return result


def select_person_detection(
    result: Any, score_thr: float, category_id: int
) -> tuple[np.ndarray, float, int]:
    """Extract the highest-scoring person from an MMDetection 3 sample."""
    sample = result[0] if isinstance(result, (list, tuple)) else result
    instances = getattr(sample, "pred_instances", None)
    if instances is None:
        raise RuntimeError("MMDetection result has no pred_instances")
    bboxes = _to_numpy(getattr(instances, "bboxes", np.empty((0, 4))))
    scores = _to_numpy(getattr(instances, "scores", np.empty((0,))))
    labels = _to_numpy(getattr(instances, "labels", np.empty((len(scores),))))
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (len(bboxes) == len(scores) == len(labels)):
        raise RuntimeError(
            f"detector output length mismatch: {len(bboxes)}/{len(scores)}/{len(labels)}"
        )
    keep = (labels == int(category_id)) & (scores >= float(score_thr))
    indices = np.flatnonzero(keep)
    if len(indices) == 0:
        raise RuntimeError(
            f"detector found no person above score threshold {score_thr:.3f}"
        )
    best = int(indices[np.argmax(scores[indices])])
    bbox = bboxes[best].copy()
    if not np.isfinite(bbox).all() or not np.isfinite(scores[best]):
        raise RuntimeError("detector returned non-finite person proposal")
    return bbox, float(scores[best]), best


def infer_pose(
    pose_model: Any,
    image: np.ndarray,
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # inference_topdown is the official MMPose API.  Passing the ndarray keeps
    # detector and pose in exactly the same (undistorted, full-image) pixels.
    samples = inference_topdown(
        pose_model,
        image,
        bboxes=np.asarray(bbox, dtype=np.float32).reshape(1, 4),
        bbox_format="xyxy",
    )
    if len(samples) != 1:
        raise RuntimeError(f"expected one top-down pose sample, got {len(samples)}")
    instances = samples[0].pred_instances
    keypoints = _to_numpy(instances.keypoints)
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 2)
    if keypoints.shape[0] != 1:
        raise RuntimeError(f"expected one top-down pose, got {keypoints.shape}")
    if hasattr(instances, "keypoint_scores"):
        scores = _to_numpy(instances.keypoint_scores)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, 17)
        if scores.shape[0] != 1:
            raise RuntimeError(f"unexpected keypoint score shape {scores.shape}")
        confidence = scores[0]
    else:
        # Some custom MMPose versions expose only keypoints.  This fallback is
        # explicit in the manifest and does not fabricate per-joint confidence.
        confidence = np.ones((17,), dtype=np.float32)
    points = keypoints[0]
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise RuntimeError("MMPose returned non-finite keypoints/confidence")
    transform_metadata = extract_pose_transform_metadata(samples[0], points)
    if hasattr(instances, "keypoints_visible"):
        visibility = _to_numpy(instances.keypoints_visible)
        visibility = np.asarray(visibility, dtype=np.float32).reshape(-1, 17)
        if visibility.shape[0] == 1 and np.isfinite(visibility[0]).all():
            transform_metadata["mmpose_keypoint_visibility_coco"] = visibility[0]
    return points, confidence, transform_metadata


def infer_detector(detector: Any, image: np.ndarray) -> Any:
    """Run MMDetection while its own registry scope is active.

    MMPose and MMDetection both register transforms in the shared MMEngine
    registry.  Initialising the pose model after the detector changes the
    default scope to ``mmpose``; without switching back, MMDetection's
    ``PackDetInputs`` is looked up in the wrong registry.  Explicit scope
    switching keeps the two official APIs interoperable and is important for
    reproducibility across MMPose/MMDetection versions.
    """
    init_default_scope("mmdet")
    try:
        return inference_detector(detector, image)
    finally:
        init_default_scope("mmpose")


def prediction_entry(
    index: int,
    points: np.ndarray,
    confidence: np.ndarray,
    bbox: np.ndarray,
    detector_score: float,
    image_shape: tuple[int, ...],
    transform_metadata: dict[str, Any],
    detector_fallback: str | None = None,
) -> dict[str, Any]:
    entry = {
        "record_index": int(index),
        "keypoints_coco": np.asarray(points, dtype=np.float32),
        "keypoint_scores_coco": np.asarray(confidence, dtype=np.float32),
        "detector_bbox_xyxy": np.asarray(bbox, dtype=np.float32),
        "detector_score": float(detector_score),
        "image_shape": tuple(int(x) for x in image_shape),
    }
    if detector_fallback is not None:
        entry["detector_fallback"] = str(detector_fallback)
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


def selected_indices(total: int, shard_id: int, num_shards: int, max_records: int | None) -> list[int]:
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError(f"invalid shard {shard_id}/{num_shards}")
    indices = list(range(int(shard_id), int(total), int(num_shards)))
    if max_records is not None:
        if max_records < 0:
            raise ValueError("--max-records must be non-negative")
        indices = indices[: int(max_records)]
    return indices


def main() -> None:
    args = parse_args()
    if args.occlusion_prob > 0 and args.no_undistort:
        raise ValueError("H36M-Occl requires full-image undistortion before masking")
    if args.no_undistort:
        print(
            "WARNING: --no-undistort is a diagnostic protocol; final aligned data must omit it",
            file=sys.stderr,
        )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    input_pkl = Path(args.input_pkl).resolve()
    images_root = Path(args.images_root).resolve()
    output = Path(args.output).resolve()
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    with input_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"expected list input records, got {type(records).__name__}")
    indices = selected_indices(len(records), args.shard_id, args.num_shards, args.max_records)

    print(
        f"loading official detector/pose models on {args.device}; "
        f"records={len(records)} selected={len(indices)} shard={args.shard_id}/{args.num_shards}",
        flush=True,
    )
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector_protocol = configure_detector_test_protocol(detector, args)
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    pose_protocol = configure_pose_test_protocol(pose_model, args)
    print(
        f"detector test protocol: {json_value(detector_protocol)}; "
        f"HRNet test protocol: {json_value(pose_protocol)}",
        flush=True,
    )

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    started = time.time()
    for done, index in enumerate(indices, start=1):
        record = records[index]
        try:
            image_path = resolve_image(images_root, record)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2.imread failed for {image_path}")
            K, distortion = camera_parameters(record)
            processed = (
                image
                if args.no_undistort
                else undistort_full_image(image, K, distortion)
            )
            occlusion = None
            if args.occlusion_prob > 0:
                occlusion = apply_white_joint_squares(
                    processed,
                    record,
                    probability=args.occlusion_prob,
                    square_fraction=args.occlusion_square_fraction,
                    seed=args.occlusion_seed,
                )
                processed = occlusion.image
            detection = infer_detector(detector, processed)
            detector_fallback = None
            try:
                bbox, detector_score, _ = select_person_detection(
                    detection, args.score_thr, args.det_cat_id
                )
            except RuntimeError:
                if not args.fallback_record_box:
                    raise
                source_box = np.asarray(record.get("box"), dtype=np.float32).reshape(-1)
                if source_box.size != 4 or not np.isfinite(source_box).all():
                    raise RuntimeError(
                        "detector failed and source record has no finite 4-value box"
                    )
                x1, y1, x2, y2 = [float(x) for x in source_box]
                if not (x2 > x1 and y2 > y1):
                    raise RuntimeError(
                        f"detector failed and source record box is invalid: {source_box.tolist()}"
                    )
                bbox = source_box.copy()
                detector_score = 0.0
                detector_fallback = "source_h36m_record_box"
                fallbacks.append(
                    {
                        "record_index": int(index),
                        "image": record.get("image"),
                        "reason": "no_person_proposal_above_threshold",
                        "score_threshold": float(args.score_thr),
                    }
                )
            points, confidence, transform_metadata = infer_pose(
                pose_model, processed, bbox
            )
            entry = prediction_entry(
                index,
                points,
                confidence,
                bbox,
                detector_score,
                tuple(processed.shape),
                transform_metadata,
                detector_fallback,
            )
            if occlusion is not None:
                entry["occlusion_masked_joints_rumpl"] = occlusion.masked_joints
                entry["occlusion_centers_full_undistorted_xy"] = occlusion.centers_xy
                entry["occlusion_square_side_px"] = int(occlusion.square_side_px)
            predictions.append(entry)
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
                f"predictions={len(predictions)} errors={len(errors)} "
                f"rate={rate:.2f}/s",
                flush=True,
            )

    write_pickle_atomic(output, predictions)
    manifest_payload = {
        "protocol": "GBT-aligned-HRNet-coordinate-only-v2",
        "created_unix": time.time(),
        "command": " ".join([str(x) for x in sys.argv]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "mmdet": package_version("mmdet"),
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
        "det_config": str(Path(args.det_config).resolve()),
        "det_config_sha256": sha256_file(args.det_config),
        "det_checkpoint": str(Path(args.det_checkpoint).resolve()),
        "det_checkpoint_sha256": sha256_file(args.det_checkpoint),
        "device": args.device,
        "record_count_total": len(records),
        "record_count_selected": len(indices),
        "prediction_count": len(predictions),
        "error_count": len(errors),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "score_threshold": args.score_thr,
        "detector_test_score_threshold": args.detector_test_score_thr,
        "fallback_record_box": bool(args.fallback_record_box),
        "fallback_count": len(fallbacks),
        "fallbacks": fallbacks,
        "detector_test_protocol": detector_protocol,
        "hrnet_test_protocol": pose_protocol,
        "detector_category_id": args.det_cat_id,
        "undistort_full_image": not args.no_undistort,
        "undistortion": {
            "method": "cv2.undistort",
            "new_camera_matrix": "original_K",
            "output_resolution": "original_resolution",
            "distortion_order": "[k1,k2,p1,p2,k3] from camera.k/camera.p",
        },
        "occlusion": (
            None
            if args.occlusion_prob <= 0
            else occlusion_protocol_manifest(
                args.occlusion_prob,
                args.occlusion_square_fraction,
                args.occlusion_seed,
            )
        ),
        "pose_api": "mmpose.apis.inference_topdown",
        "detector_api": "mmdet.apis.inference_detector",
        "bbox_format": "xyxy",
        "saved_payload": "COCO-17 keypoints and scores, detector bbox/score only",
        "errors": errors,
    }
    write_json_atomic(manifest, manifest_payload)
    print(f"wrote {len(predictions)} predictions to {output}")
    print(f"wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
