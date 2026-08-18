#!/usr/bin/env python3
"""Audit LT-style HRNet coordinates against the full-image HRNet line.

The two caches use different pixel frames, so comparing their stored
``joints_2d`` values directly would be invalid.  This report transforms H36M
GT points into each prediction cache's own frame (full undistorted pixels or
384x384 LT crop pixels), then evaluates 2-D error and public ray/DLT controls
with the matching camera K.  It never trains or uses a learned 3-D model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval_h36m_controlled_triangulation_20260813 import (
    ACTION_NAMES,
    DIRECT_COCO_H36M,
    JOINT_NAMES,
    algebraic_dlt,
    camera_matrices,
    ray_intersection,
    solve_ray,
    summarize,
    undistort_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pkl", required=True)
    parser.add_argument(
        "--input", action="append", required=True,
        help="LABEL=PKL[:full|lt]; repeat for each coordinate cache",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--irls-iters", type=int, default=5)
    parser.add_argument("--huber-threshold-mm", type=float, default=20.0)
    return parser.parse_args()


def group_records(records: list[dict]) -> OrderedDict[tuple, list[dict]]:
    grouped: OrderedDict[tuple, dict[int, dict]] = OrderedDict()
    for record in records:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        grouped.setdefault(key, {})[int(record["camera_id"])] = record
    return OrderedDict(
        (key, [by_camera[index] for index in range(4)])
        for key, by_camera in grouped.items()
        if set(by_camera) == {0, 1, 2, 3}
    )


def cache_frame(record: dict, mode: str) -> str:
    if mode in {"full", "lt"}:
        return mode
    protocol = str(record.get("source_2d_coordinate_system", ""))
    if protocol.startswith("lt_crop"):
        return "lt"
    return "full"


def gt_in_prediction_frame(gt_record: dict, pred_record: dict, mode: str) -> np.ndarray:
    pixels = np.asarray(gt_record["joints_2d"], dtype=np.float64)
    K, _, _, distortion = camera_matrices(gt_record)
    pixels = undistort_pixels(pixels, K, distortion)
    if mode == "full":
        return pixels
    if mode != "lt":
        raise ValueError(f"unknown coordinate mode {mode}")
    bbox = np.asarray(pred_record["source_2d_lt_bbox_xyxy_int"], dtype=np.float64).reshape(4)
    crop_shape = np.asarray(
        pred_record["source_2d_lt_crop_shape_before_resize"], dtype=np.float64
    ).reshape(2)
    resize_shape = np.asarray(
        pred_record.get("source_2d_lt_resize_shape", (384, 384)), dtype=np.float64
    ).reshape(2)
    h, w = crop_shape
    out_h, out_w = resize_shape
    if min(h, w, out_h, out_w) <= 0:
        raise ValueError("invalid LT crop/resize shape")
    return np.column_stack([
        (pixels[:, 0] - bbox[0]) * out_w / w,
        (pixels[:, 1] - bbox[1]) * out_h / h,
    ])


def evaluate_method(
    gt_groups: OrderedDict,
    pred_groups: OrderedDict,
    mode: str,
    irls_iters: int,
    huber_threshold_mm: float,
) -> dict:
    common_keys = [key for key in gt_groups if key in pred_groups]
    if not common_keys:
        raise RuntimeError("no complete synchronized groups")
    two_d = []
    accum: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for key in common_keys:
        gt_records = gt_groups[key]
        pred_records = pred_groups[key]
        if [x["image"] for x in gt_records] != [x["image"] for x in pred_records]:
            raise RuntimeError(f"image alignment mismatch at {key}")
        gt_pixels = np.stack([
            gt_in_prediction_frame(gt, pred, mode)
            for gt, pred in zip(gt_records, pred_records)
        ])
        pred_pixels = np.asarray([x["joints_2d"] for x in pred_records], dtype=np.float64)
        confidence = np.asarray([
            x.get("joints_2d_conf", np.ones((17, 1))) for x in pred_records
        ], dtype=np.float64).reshape(4, 17)
        target = np.asarray(gt_records[0]["joints_3d"], dtype=np.float64)
        two_d.append(np.linalg.norm(pred_pixels - gt_pixels, axis=-1))

        for n_views in (2, 3, 4):
            for subset in itertools.combinations(range(4), n_views):
                sub_pred = pred_pixels[list(subset)]
                sub_gt = gt_pixels[list(subset)]
                sub_conf = confidence[list(subset)]
                sub_records = [pred_records[index] for index in subset]
                centers, directions, projections, _ = _geometry(sub_records, sub_pred)
                stage = f"V{n_views}"
                estimates = {
                    "pred_ray_uniform": solve_ray(
                        centers, directions, sub_conf, "uniform", irls_iters,
                        huber_threshold_mm,
                    ),
                    "pred_ray_confidence": solve_ray(
                        centers, directions, sub_conf, "confidence", irls_iters,
                        huber_threshold_mm,
                    ),
                    "pred_ray_irls": solve_ray(
                        centers, directions, sub_conf, "irls", irls_iters,
                        huber_threshold_mm,
                    ),
                    "pred_algebraic_confidence": algebraic_dlt(
                        projections, sub_pred, sub_conf,
                    ),
                    "gt_ray_confidence": solve_ray(
                        centers, directions, np.ones_like(sub_conf), "confidence",
                        irls_iters, huber_threshold_mm,
                    ),
                }
                for name, estimate in estimates.items():
                    accum[stage][name].append(np.linalg.norm(estimate - target, axis=-1))
    two_d_values = np.asarray(two_d)
    report = {
        "frame": mode,
        "complete_four_view_groups": len(common_keys),
        "pred2d_vs_gt2d_px": {
            "mean": float(two_d_values.mean()),
            "median": float(np.median(two_d_values)),
            "p90": float(np.percentile(two_d_values, 90)),
            "per_joint_mean": {
                name: float(two_d_values[..., index].mean())
                for index, name in enumerate(JOINT_NAMES)
            },
            "direct_coco13_mean": float(two_d_values[..., DIRECT_COCO_H36M].mean()),
        },
        "results": {},
    }
    for stage, methods in accum.items():
        report["results"][stage] = {}
        # Each common group contributes one row per camera combination.
        repeats = len(next(iter(methods.values()))) // len(common_keys)
        stage_actions = np.asarray(
            [key[1] for key in common_keys for _ in range(repeats)], dtype=np.int64
        )
        for name, values in methods.items():
            report["results"][stage][name] = summarize(
                np.asarray(values), stage_actions
            )
    return report


def _geometry(records: list[dict], pixels: np.ndarray):
    centers, directions, projections = [], [], []
    for record, points in zip(records, pixels):
        K, rotation, center, distortion = camera_matrices(record)
        homogeneous = np.concatenate(
            [points.astype(np.float64), np.ones((len(points), 1))], axis=1
        )
        camera_rays = homogeneous @ np.linalg.inv(K).T
        world_rays = camera_rays @ rotation
        world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True).clip(1e-12)
        translation = -rotation @ center.reshape(3, 1)
        projections.append(K @ np.concatenate([rotation, translation], axis=1))
        centers.append(center)
        directions.append(world_rays)
    return np.asarray(centers), np.asarray(directions), np.asarray(projections), None


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_pkl).resolve()
    gt_groups = group_records(pickle.loads(gt_path.read_bytes()))
    methods = {}
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"expected LABEL=PKL[:full|lt], got {spec}")
        label, value = spec.split("=", 1)
        if ":" in value:
            path_text, mode = value.rsplit(":", 1)
        else:
            path_text, mode = value, "auto"
        path = Path(path_text).resolve()
        records = pickle.loads(path.read_bytes())
        pred_groups = group_records(records)
        if mode == "auto":
            mode = cache_frame(records[0], "auto")
        methods[label] = {
            "path": str(path),
            "mode": mode,
            "report": evaluate_method(
                gt_groups, pred_groups, mode,
                args.irls_iters, args.huber_threshold_mm,
            ),
        }
    output = {
        "protocol": {
            "gt_pkl": str(gt_path),
            "metric": "absolute MPJPE, all 17 joints, no alignment",
            "camera_combinations": "all 6/4/1 combinations for V2/V3/V4",
            "gt_transform": "undistort with original K then map to each cache frame",
            "lt_transform": "integer bbox + official PIL crop shape + 384 resize",
        },
        "methods": methods,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
