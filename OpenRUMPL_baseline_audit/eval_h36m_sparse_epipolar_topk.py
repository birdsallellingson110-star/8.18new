#!/usr/bin/env python3
"""Evaluate training-free sparse epipolar heatmap correction on H36M.

For every COCO joint, top-K HRNet heatmap modes are converted to calibrated
world rays.  Candidate pairs produce 3D hypotheses; every view scores each
hypothesis by heatmap confidence and point-to-ray consistency.  The selected
2D modes are converted to the RUMPL H36M-17 convention and triangulated.

This is a low-cost test of the spatial part shared by AdaFuse, TransFusion and
DenseWarper.  It deliberately contains no temporal path and no learned camera
identity, so a gain cannot come from memorizing the four H36M cameras.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sparse_epipolar_candidate_transformer import (
    SparseEpipolarCandidateTransformer,
)


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
# Direct COCO channel -> H36M joint after the verified lower-body swap.
COCO_TO_H36M_DIRECT = {
    11: 1,
    13: 2,
    15: 3,
    12: 4,
    14: 5,
    16: 6,
    0: 9,
    5: 11,
    7: 12,
    9: 13,
    6: 14,
    8: 15,
    10: 16,
}
DIRECT_COCO_JOINTS = np.asarray(list(COCO_TO_H36M_DIRECT), dtype=np.int64)
DIRECT_H36M_JOINTS = np.asarray(
    list(COCO_TO_H36M_DIRECT.values()), dtype=np.int64
)
KP_STAR = (11, 14, 12, 15, 13, 16, 5, 2, 6, 3)
ACTION_NAMES = {
    2: "Direction",
    3: "Discuss",
    4: "Eating",
    5: "Greet",
    6: "Phone",
    7: "Photo",
    8: "Pose",
    9: "Purchase",
    10: "Sitting",
    11: "SittingDown",
    12: "Smoke",
    13: "Wait",
    14: "WalkDog",
    15: "Walk",
    16: "WalkTwo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--topk-shards", nargs="+", required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--sigma-m",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
        help="Point-to-ray consistency scale in metres.",
    )
    parser.add_argument(
        "--unary-weight",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Multiplier on log heatmap confidence.",
    )
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument(
        "--skip-epipolar",
        action="store_true",
        help="Evaluate only top-1 and the GT-2D top-K diagnostic ceiling.",
    )
    parser.add_argument("--candidate-transformer-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=0,
        help="Optional smoke-test limit per view cardinality.",
    )
    return parser.parse_args()


def camera_parameters(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = record["camera"]
    rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    center = np.asarray(camera["T"], dtype=np.float64).reshape(3) / 1000.0
    intrinsic = np.asarray(
        [
            [float(np.asarray(camera["fx"]).reshape(-1)[0]), 0.0,
             float(np.asarray(camera["cx"]).reshape(-1)[0])],
            [0.0, float(np.asarray(camera["fy"]).reshape(-1)[0]),
             float(np.asarray(camera["cy"]).reshape(-1)[0])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsic, rotation, center


def pixels_to_rays(
    pixels: np.ndarray, intrinsic: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    homogeneous = np.concatenate(
        (pixels.astype(np.float64), np.ones((*pixels.shape[:-1], 1))), axis=-1
    )
    camera_rays = homogeneous @ np.linalg.inv(intrinsic).T
    world_rays = camera_rays @ rotation
    return world_rays / np.linalg.norm(world_rays, axis=-1, keepdims=True)


def solve_ray_intersection(
    centers: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    reg: float = 1e-8,
) -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)
    projection = identity - directions[..., :, None] * directions[..., None, :]
    weighted = projection * weights[..., None, None]
    lhs = weighted.sum(axis=-3) + reg * identity
    rhs = np.einsum("...vij,...vj->...i", weighted, centers)
    return np.linalg.solve(lhs, rhs)


def robust_intersection(
    centers: np.ndarray,
    directions: np.ndarray,
    confidence: np.ndarray,
    iterations: int,
) -> np.ndarray:
    weights = np.maximum(confidence.astype(np.float64), 1e-6)
    estimate = solve_ray_intersection(centers, directions, weights)
    for _ in range(iterations):
        offset = estimate[None] - centers
        residual = np.linalg.norm(np.cross(offset, directions), axis=-1)
        scale = max(float(np.median(residual)), 0.005)
        robust = 1.0 / (1.0 + np.square(residual / (2.3849 * scale)))
        estimate = solve_ray_intersection(
            centers, directions, weights * robust
        )
    return estimate


def select_epipolar_candidates(
    centers: np.ndarray,
    directions: np.ndarray,
    scores: np.ndarray,
    sigma_m: float,
    unary_weight: float,
) -> np.ndarray:
    """RANSAC-like top-K consensus; inputs are VxJxKx3 and VxJxK."""
    n_views, n_joints, n_candidates = scores.shape
    hypotheses = []
    for first, second in itertools.combinations(range(n_views), 2):
        dirs_first = np.repeat(
            directions[first], n_candidates, axis=1
        )
        dirs_second = np.tile(directions[second], (1, n_candidates, 1))
        pair_directions = np.stack((dirs_first, dirs_second), axis=2)
        pair_centers = np.broadcast_to(
            centers[[first, second]][None, None],
            (n_joints, n_candidates * n_candidates, 2, 3),
        )
        hypotheses.append(
            solve_ray_intersection(
                pair_centers,
                pair_directions,
                np.ones((n_joints, n_candidates * n_candidates, 2)),
            )
        )
    hypotheses = np.concatenate(hypotheses, axis=1)
    hypothesis_scores = np.zeros(hypotheses.shape[:2], dtype=np.float64)
    selected_per_view = []
    for view in range(n_views):
        offset = hypotheses[:, :, None, :] - centers[view]
        distance = np.linalg.norm(
            np.cross(offset, directions[view][:, None]), axis=-1
        )
        candidate_score = (
            unary_weight
            * np.log(np.maximum(scores[view].astype(np.float64), 1e-6))[:, None]
            - 0.5 * np.square(distance / sigma_m)
        )
        hypothesis_scores += candidate_score.max(axis=2)
        selected_per_view.append(candidate_score.argmax(axis=2))
    best = hypothesis_scores.argmax(axis=1)
    joints = np.arange(n_joints)
    return np.stack(
        [indices[joints, best] for indices in selected_per_view], axis=1
    )


def coco_to_h36m(
    keypoints: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((17, 2), dtype=np.float64)
    confidence = np.zeros(17, dtype=np.float64)
    for dst, src in MMPOSE2H36M.items():
        joints[dst] = keypoints[src]
        confidence[dst] = scores[src]
    joints[10] = keypoints[0:5].mean(axis=0)
    confidence[10] = scores[0:5].mean()
    joints[8] = keypoints[3:7].mean(axis=0)
    confidence[8] = scores[3:7].mean()
    joints[0] = keypoints[11:13].mean(axis=0)
    confidence[0] = scores[11:13].mean()
    joints[7] = (joints[8] + joints[0]) / 2.0
    confidence[7] = (confidence[8] + confidence[0]) / 2.0
    # Match the verified H36M lower-body semantic correction used by H12.
    joints[1:4], joints[4:7] = joints[4:7].copy(), joints[1:4].copy()
    confidence[1:4], confidence[4:7] = (
        confidence[4:7].copy(),
        confidence[1:4].copy(),
    )
    return joints, confidence


def target_world_metres(record: dict) -> np.ndarray:
    camera = record["camera"]
    rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    translation = camera.get("t")
    if translation is None:
        center_mm = np.asarray(camera["T"], dtype=np.float64).reshape(3)
        translation = -rotation @ center_mm
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera_pose = np.asarray(
        record["joints_3d_camera"], dtype=np.float64
    )
    return ((camera_pose - translation) @ rotation) / 1000.0


def damaged(record: dict) -> bool:
    return (
        int(record["subject"]) == 9
        and (
            (int(record["action"]), int(record["subaction"]))
            in ((5, 2), (10, 2), (13, 1))
        )
    )


def frame_key(record: dict) -> tuple[int, int, int, int]:
    return (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        int(record["image_id"]),
    )


def load_candidates(shards: list[str], n_records: int) -> dict[str, np.ndarray]:
    merged_xy = None
    merged_scores = None
    seen = np.zeros(n_records, dtype=bool)
    for shard_path in shards:
        with np.load(shard_path) as shard:
            indices = shard["record_indices"].astype(np.int64)
            shard_xy = shard["candidate_xy"].astype(np.float64)
            shard_scores = shard["candidate_scores"].astype(np.float64)
        if merged_xy is None:
            merged_xy = np.empty((n_records, *shard_xy.shape[1:]), np.float64)
            merged_scores = np.empty(
                (n_records, *shard_scores.shape[1:]), np.float64
            )
        if seen[indices].any():
            duplicate = indices[seen[indices]][0]
            raise RuntimeError(f"duplicate record {duplicate}")
        merged_xy[indices] = shard_xy
        merged_scores[indices] = shard_scores
        seen[indices] = True
    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise RuntimeError(
            f"top-K export has {int(seen.sum())}/{n_records} records; "
            f"first missing: {missing[:10].tolist()}"
        )
    return {"xy": merged_xy, "scores": merged_scores}


def build_four_view_groups(records: list[dict]) -> list[list[int]]:
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        if damaged(record):
            continue
        key = frame_key(record)
        grouped.setdefault(key, [-1, -1, -1, -1])
        grouped[key][int(record["camera_id"])] = index
    return [group for group in grouped.values() if min(group) >= 0]


def summarize(errors: dict[str, dict[str, list[float]]]) -> dict:
    output = {}
    for method, action_values in errors.items():
        action_means = {
            action: float(np.mean(values))
            for action, values in sorted(action_values.items())
        }
        output[method] = {
            "frame_weighted_all17_mm": float(
                np.mean([value for values in action_values.values() for value in values])
            ),
            "action_equal_all17_mm": float(np.mean(list(action_means.values()))),
            "per_action_all17_mm": action_means,
        }
    return output


def evaluate_cardinality(
    records: list[dict],
    candidates: dict[str, np.ndarray],
    four_view_groups: list[list[int]],
    n_views: int,
    sigmas: list[float],
    unary_weights: list[float],
    irls_iterations: int,
    limit_groups: int,
    skip_epipolar: bool,
    candidate_transformer: SparseEpipolarCandidateTransformer | None,
    transformer_device: torch.device,
) -> dict:
    groups = [
        [group[index] for index in combination]
        for group in four_view_groups
        for combination in itertools.combinations(range(4), n_views)
    ]
    if limit_groups:
        groups = groups[:limit_groups]
    errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kp_errors: dict[str, list[float]] = defaultdict(list)

    for group_number, group in enumerate(groups, start=1):
        group_records = [records[index] for index in group]
        camera_data = [camera_parameters(record) for record in group_records]
        intrinsics = [item[0] for item in camera_data]
        rotations = [item[1] for item in camera_data]
        centers = np.stack([item[2] for item in camera_data])
        target = target_world_metres(group_records[0])
        action = ACTION_NAMES[int(group_records[0]["action"])]

        raw_xy = candidates["xy"][group]  # V,Coco17,K,2
        raw_scores = candidates["scores"][group]  # V,Coco17,K
        raw_directions = np.stack(
            [
                pixels_to_rays(raw_xy[view], intrinsics[view], rotations[view])
                for view in range(n_views)
            ]
        )

        settings = [("top1", None, None), ("oracle_gt2d_topk", None, None)]
        learned_chosen = None
        if candidate_transformer is not None:
            direct_scores = torch.as_tensor(
                np.transpose(
                    raw_scores[:, DIRECT_COCO_JOINTS], (1, 0, 2)
                ),
                dtype=torch.float32,
                device=transformer_device,
            )
            direct_directions = torch.as_tensor(
                np.transpose(
                    raw_directions[:, DIRECT_COCO_JOINTS], (1, 0, 2, 3)
                ),
                dtype=torch.float32,
                device=transformer_device,
            )
            direct_centers = torch.as_tensor(
                centers[None],
                dtype=torch.float32,
                device=transformer_device,
            ).expand(len(DIRECT_COCO_JOINTS), -1, -1)
            joint_ids = torch.as_tensor(
                DIRECT_H36M_JOINTS,
                dtype=torch.long,
                device=transformer_device,
            )
            with torch.no_grad():
                learned_logits = candidate_transformer(
                    direct_scores,
                    direct_centers,
                    direct_directions,
                    joint_ids,
                )
            direct_chosen = learned_logits.argmax(dim=-1).cpu().numpy()
            learned_chosen = np.zeros((17, n_views), dtype=np.int64)
            learned_chosen[DIRECT_COCO_JOINTS] = direct_chosen
            settings.append(("learned_sparse_transformer", None, None))
        if not skip_epipolar:
            settings.extend(
                (
                    f"epi_sigma{sigma:g}_unary{unary:g}",
                    sigma,
                    unary,
                )
                for sigma in sigmas
                for unary in unary_weights
            )
        for method, sigma, unary in settings:
            if method == "top1":
                chosen = np.zeros((17, n_views), dtype=np.int64)
            elif method == "oracle_gt2d_topk":
                chosen = np.zeros((17, n_views), dtype=np.int64)
                for view, record in enumerate(group_records):
                    gt_2d = np.asarray(record["joints_2d"], dtype=np.float64)
                    for coco_joint, h36m_joint in COCO_TO_H36M_DIRECT.items():
                        distance = np.linalg.norm(
                            raw_xy[view, coco_joint] - gt_2d[h36m_joint],
                            axis=-1,
                        )
                        chosen[coco_joint, view] = int(distance.argmin())
            elif method == "learned_sparse_transformer":
                chosen = learned_chosen
            else:
                chosen = select_epipolar_candidates(
                    centers,
                    raw_directions,
                    raw_scores,
                    float(sigma),
                    float(unary),
                )

            h36m_xy = []
            h36m_confidence = []
            for view in range(n_views):
                selected_xy = raw_xy[
                    view, np.arange(17), chosen[:, view]
                ]
                selected_scores = raw_scores[
                    view, np.arange(17), chosen[:, view]
                ]
                joints, confidence = coco_to_h36m(
                    selected_xy, selected_scores
                )
                h36m_xy.append(joints)
                h36m_confidence.append(confidence)
            h36m_xy = np.stack(h36m_xy)
            h36m_confidence = np.stack(h36m_confidence)
            h36m_directions = np.stack(
                [
                    pixels_to_rays(
                        h36m_xy[view], intrinsics[view], rotations[view]
                    )
                    for view in range(n_views)
                ]
            )

            pred = np.stack(
                [
                    robust_intersection(
                        centers,
                        h36m_directions[:, joint],
                        h36m_confidence[:, joint],
                        irls_iterations,
                    )
                    for joint in range(17)
                ]
            )
            joint_error = np.linalg.norm(pred - target, axis=-1) * 1000.0
            errors[method][action].append(float(joint_error.mean()))
            kp_errors[method].append(float(joint_error[list(KP_STAR)].mean()))

        if group_number % 250 == 0:
            print(
                f"V{n_views}: {group_number}/{len(groups)} groups",
                flush=True,
            )

    result = {
        "views": n_views,
        "groups": len(groups),
        "methods": summarize(errors),
    }
    for method, values in kp_errors.items():
        result["methods"][method]["frame_weighted_kp_star_mm"] = float(
            np.mean(values)
        )
    return result


def main() -> None:
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    candidates = load_candidates(args.topk_shards, len(records))
    groups = build_four_view_groups(records)
    candidate_transformer = None
    transformer_device = torch.device(args.device)
    if args.candidate_transformer_checkpoint:
        checkpoint = torch.load(
            args.candidate_transformer_checkpoint,
            map_location=transformer_device,
        )
        train_args = checkpoint["args"]
        candidate_transformer = SparseEpipolarCandidateTransformer(
            dim=int(train_args["dim"]),
            depth=int(train_args["depth"]),
            num_heads=int(train_args["heads"]),
        ).to(transformer_device)
        candidate_transformer.load_state_dict(checkpoint["model"])
        candidate_transformer.eval()
        print(
            "loaded candidate transformer: "
            f"{args.candidate_transformer_checkpoint}",
            flush=True,
        )
    print(
        f"loaded {len(records)} records, {len(groups)} complete frames",
        flush=True,
    )
    results = []
    for n_views in args.views:
        result = evaluate_cardinality(
            records,
            candidates,
            groups,
            n_views,
            args.sigma_m,
            args.unary_weight,
            args.irls_iterations,
            args.limit_groups,
            args.skip_epipolar,
            candidate_transformer,
            transformer_device,
        )
        results.append(result)
        print(f"V{n_views}", flush=True)
        for method, values in result["methods"].items():
            print(
                f"  {method}: "
                f"All={values['action_equal_all17_mm']:.3f}, "
                f"KP*={values['frame_weighted_kp_star_mm']:.3f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_pkl": str(Path(args.input_pkl).resolve()),
        "topk_shards": [str(Path(path).resolve()) for path in args.topk_shards],
        "sigma_m": args.sigma_m,
        "unary_weight": args.unary_weight,
        "candidate_transformer_checkpoint": (
            str(Path(args.candidate_transformer_checkpoint).resolve())
            if args.candidate_transformer_checkpoint
            else None
        ),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
