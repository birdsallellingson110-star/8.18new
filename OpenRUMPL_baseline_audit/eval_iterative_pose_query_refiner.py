#!/usr/bin/env python3
"""Evaluate H21 after each pose-query/triangulation refinement iteration."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_h36m_dense_epipolar_heatmaps import DenseHeatmapStore
from eval_h36m_sparse_epipolar_topk import (
    ACTION_NAMES,
    COCO_TO_H36M_DIRECT,
    KP_STAR,
    build_four_view_groups,
    camera_parameters,
    coco_to_h36m,
    pixels_to_rays,
    robust_intersection,
    target_world_metres,
)
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import (
    heatmap_to_image,
    image_to_heatmap,
    project_world_points,
    triangulate_points,
)


DIRECT_COCO = np.asarray(
    list(COCO_TO_H36M_DIRECT.keys()), dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(
    errors: dict[str, dict[str, list[float]]],
    kp_errors: dict[str, list[float]],
) -> dict:
    output = {}
    for method, action_values in errors.items():
        per_action = {
            action: float(np.mean(values))
            for action, values in sorted(action_values.items())
        }
        flat = [
            value
            for action_group in action_values.values()
            for value in action_group
        ]
        output[method] = {
            "frame_weighted_all17_mm": float(np.mean(flat)),
            "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
            "frame_weighted_kp_star_mm": float(np.mean(kp_errors[method])),
            "per_action_all17_mm": per_action,
        }
    return output


def evaluate_pose(
    group_records: list[dict],
    coco_xy: np.ndarray,
    coco_confidence: np.ndarray,
    irls_iterations: int,
) -> np.ndarray:
    h36m_xy = []
    h36m_confidence = []
    for view in range(len(group_records)):
        joints, confidence = coco_to_h36m(
            coco_xy[view], coco_confidence[view]
        )
        h36m_xy.append(joints)
        h36m_confidence.append(confidence)
    h36m_xy = np.stack(h36m_xy)
    h36m_confidence = np.stack(h36m_confidence)
    camera_data = [camera_parameters(record) for record in group_records]
    intrinsics = [item[0] for item in camera_data]
    rotations = [item[1] for item in camera_data]
    centers = np.stack([item[2] for item in camera_data])
    directions = np.stack(
        [
            pixels_to_rays(h36m_xy[view], intrinsics[view], rotations[view])
            for view in range(len(group_records))
        ]
    )
    return np.stack(
        [
            robust_intersection(
                centers,
                directions[:, joint],
                h36m_confidence[:, joint],
                irls_iterations,
            )
            for joint in range(17)
        ]
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    store = DenseHeatmapStore(args.dense_shards)
    groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    payload = torch.load(args.checkpoint, map_location=device)
    model = IterativePoseQueryRefiner().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    result = {
        "input_pkl": args.input_pkl,
        "checkpoint": args.checkpoint,
        "iterations": args.iterations,
        "complete_four_view_groups": len(groups),
        "results": {},
    }

    for n_views in args.views:
        work = [
            (group, combination)
            for group in groups
            for combination in itertools.combinations(range(4), n_views)
        ]
        if args.limit_groups:
            work = work[: args.limit_groups]
        errors = defaultdict(lambda: defaultdict(list))
        kp_errors = defaultdict(list)
        for position, (group, combination) in enumerate(work, start=1):
            indices = [group[index] for index in combination]
            group_records = [records[index] for index in indices]
            data = store.get(indices)
            heatmaps_np = data["heatmaps"][:, DIRECT_COCO].astype(
                np.float32
            )
            _, _, height, width = heatmaps_np.shape
            raw_coco = data["decoded_keypoints"].astype(np.float64)
            coco_confidence = data["decoded_scores"].astype(np.float64)
            direct_confidence = coco_confidence[:, DIRECT_COCO]
            detector_hm_np = image_to_heatmap(
                raw_coco[:, DIRECT_COCO],
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            heatmaps = torch.as_tensor(
                heatmaps_np, dtype=torch.float32, device=device
            )
            detector_hm = torch.as_tensor(
                detector_hm_np, dtype=torch.float32, device=device
            )
            confidence = torch.as_tensor(
                direct_confidence, dtype=torch.float32, device=device
            )
            current_coco = raw_coco.copy()
            target = target_world_metres(group_records[0])
            action = ACTION_NAMES[int(group_records[0]["action"])]

            for iteration in range(args.iterations + 1):
                prediction = evaluate_pose(
                    group_records,
                    current_coco,
                    coco_confidence,
                    args.irls_iterations,
                )
                joint_error = np.linalg.norm(
                    prediction - target, axis=-1
                ) * 1000.0
                method = f"iteration_{iteration}"
                errors[method][action].append(float(joint_error.mean()))
                kp_errors[method].append(
                    float(joint_error[list(KP_STAR)].mean())
                )
                if iteration == args.iterations:
                    break
                anchor = triangulate_points(
                    group_records,
                    current_coco[:, DIRECT_COCO],
                    direct_confidence,
                    args.irls_iterations,
                )
                query_image = project_world_points(group_records, anchor)
                query_hm_np = image_to_heatmap(
                    query_image,
                    data["input_center"],
                    data["input_scale"],
                    width,
                    height,
                )
                query_hm = torch.as_tensor(
                    query_hm_np, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    refined_hm, _ = model(
                        heatmaps,
                        query_hm,
                        detector_hm,
                        confidence,
                        joint_ids,
                    )
                current_coco[:, DIRECT_COCO] = heatmap_to_image(
                    refined_hm.cpu().numpy(),
                    data["input_center"],
                    data["input_scale"],
                    width,
                    height,
                )
            if position % 250 == 0 or position == len(work):
                print(f"V{n_views}: {position}/{len(work)}", flush=True)
        summary = summarize(errors, kp_errors)
        result["results"][f"V{n_views}"] = {
            "views": n_views,
            "groups": len(work),
            "methods": summary,
        }
        for method, metrics in summary.items():
            print(
                f"V{n_views} {method}: "
                f"All={metrics['action_equal_all17_mm']:.3f} "
                f"KP*={metrics['frame_weighted_kp_star_mm']:.3f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
