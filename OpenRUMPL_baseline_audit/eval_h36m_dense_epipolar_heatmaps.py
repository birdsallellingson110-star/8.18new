#!/usr/bin/env python3
"""Evaluate full-heatmap, camera-generalizable fusion on real Human3.6M.

This is the dense counterpart of ``eval_h36m_sparse_epipolar_topk.py``.
Instead of choosing among pre-existing local maxima, every target heatmap
pixel is projected along its calibrated 3D ray into the other views.  The
maximum source heatmap response along that ray becomes a geometry-aligned
cross-view support map.  The implementation borrows the spatial principle
from AdaFuse and DenseWarper, but:

* uses the exact frozen MMPose HRNet heatmaps used by the RUMPL baseline;
* accepts variable view counts and never embeds camera identity;
* contains no temporal path;
* preserves the verified H36M camera/metric evaluation code.

The script first evaluates training-free fusion rules.  They are diagnostic:
they establish whether full heatmaps contain useful modes that sparse top-K
selection missed, before a learned residual fusion module is trained.
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
import torch.nn.functional as functional

from dense_geometry_residual_fusion import DenseGeometryResidualFusion
from eval_h36m_sparse_epipolar_topk import (
    ACTION_NAMES,
    DIRECT_COCO_JOINTS,
    KP_STAR,
    build_four_view_groups,
    camera_parameters,
    coco_to_h36m,
    pixels_to_rays,
    robust_intersection,
    target_world_metres,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument(
        "--dense-shards",
        nargs="+",
        required=True,
        help="Top-K .npz shards; matching *.heatmaps.npy files are inferred.",
    )
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument("--depth-min-m", type=float, default=1.0)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--depth-samples", type=int, default=64)
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


class DenseHeatmapStore:
    """Memory-map dense heatmaps and index them by validation PKL record."""

    def __init__(self, shard_paths: list[str]) -> None:
        self.shards: list[dict[str, np.ndarray]] = []
        self.locations: dict[int, tuple[int, int]] = {}
        for shard_id, name in enumerate(shard_paths):
            path = Path(name)
            dense_path = path.with_name(
                path.name.removesuffix(".npz") + ".heatmaps.npy"
            )
            if not dense_path.is_file():
                raise FileNotFoundError(dense_path)
            with np.load(path) as source:
                required = (
                    "record_indices",
                    "decoded_keypoints",
                    "decoded_scores",
                    "input_center",
                    "input_scale",
                    "input_size",
                )
                missing = [key for key in required if key not in source]
                if missing:
                    raise RuntimeError(f"{path}: missing arrays {missing}")
                arrays = {key: source[key].copy() for key in required}
            arrays["heatmaps"] = np.load(dense_path, mmap_mode="r")
            if len(arrays["record_indices"]) != len(arrays["heatmaps"]):
                raise RuntimeError(f"{path}: dense/metadata length mismatch")
            self.shards.append(arrays)
            for row, record_index in enumerate(arrays["record_indices"]):
                record_index = int(record_index)
                if record_index in self.locations:
                    raise RuntimeError(f"duplicate record {record_index}")
                self.locations[record_index] = (shard_id, row)

    def __contains__(self, record_index: int) -> bool:
        return int(record_index) in self.locations

    def get(self, record_indices: list[int]) -> dict[str, np.ndarray]:
        keys = (
            "heatmaps",
            "decoded_keypoints",
            "decoded_scores",
            "input_center",
            "input_scale",
            "input_size",
        )
        output: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for record_index in record_indices:
            shard_id, row = self.locations[int(record_index)]
            shard = self.shards[shard_id]
            for key in keys:
                output[key].append(np.asarray(shard[key][row]))
        return {key: np.stack(values) for key, values in output.items()}


def heatmap_to_image(
    heatmap_xy: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    heatmap_width: int,
    heatmap_height: int,
) -> np.ndarray:
    factor = np.asarray([heatmap_width, heatmap_height], dtype=np.float64)
    return (
        heatmap_xy.astype(np.float64) / factor * scale
        + center
        - 0.5 * scale
    )


def image_to_heatmap_torch(
    image_xy: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    heatmap_width: int,
    heatmap_height: int,
) -> torch.Tensor:
    size = image_xy.new_tensor([heatmap_width, heatmap_height])
    return (image_xy - center + 0.5 * scale) / scale * size


def projection_grid(
    target_intrinsic: np.ndarray,
    target_rotation: np.ndarray,
    target_center: np.ndarray,
    target_crop_center: np.ndarray,
    target_crop_scale: np.ndarray,
    source_intrinsic: np.ndarray,
    source_rotation: np.ndarray,
    source_center: np.ndarray,
    source_crop_center: np.ndarray,
    source_crop_scale: np.ndarray,
    heatmap_width: int,
    heatmap_height: int,
    depths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return a D x (H*W) grid into a source-view HRNet heatmap."""
    dtype = torch.float32
    ys, xs = torch.meshgrid(
        torch.arange(heatmap_height, dtype=dtype, device=device),
        torch.arange(heatmap_width, dtype=dtype, device=device),
        indexing="ij",
    )
    target_hm = torch.stack((xs.flatten(), ys.flatten()), dim=-1)
    size = target_hm.new_tensor([heatmap_width, heatmap_height])
    target_image_xy = (
        target_hm
        / size
        * torch.as_tensor(target_crop_scale, dtype=dtype, device=device)
        + torch.as_tensor(target_crop_center, dtype=dtype, device=device)
        - 0.5
        * torch.as_tensor(target_crop_scale, dtype=dtype, device=device)
    )
    homogeneous = torch.cat(
        (target_image_xy, torch.ones_like(target_image_xy[:, :1])), dim=-1
    )
    target_k_inv = torch.linalg.inv(
        torch.as_tensor(target_intrinsic, dtype=dtype, device=device)
    )
    target_r = torch.as_tensor(target_rotation, dtype=dtype, device=device)
    world_ray = homogeneous @ target_k_inv.T @ target_r
    target_c = torch.as_tensor(target_center, dtype=dtype, device=device)
    world = target_c[None, None] + depths[:, None, None] * world_ray[None]

    source_r = torch.as_tensor(source_rotation, dtype=dtype, device=device)
    source_c = torch.as_tensor(source_center, dtype=dtype, device=device)
    source_k = torch.as_tensor(source_intrinsic, dtype=dtype, device=device)
    camera = (world - source_c) @ source_r.T
    projected = camera @ source_k.T
    image_xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    source_hm = image_to_heatmap_torch(
        image_xy,
        torch.as_tensor(source_crop_center, dtype=dtype, device=device),
        torch.as_tensor(source_crop_scale, dtype=dtype, device=device),
        heatmap_width,
        heatmap_height,
    )
    normalizer = source_hm.new_tensor(
        [max(heatmap_width - 1, 1), max(heatmap_height - 1, 1)]
    )
    grid = source_hm / normalizer * 2.0 - 1.0
    # Points behind a source camera must not acquire border responses.
    behind = camera[..., 2] <= 0
    grid[behind] = 2.0
    return grid


def epipolar_support(
    normalized_heatmaps: torch.Tensor,
    intrinsics: list[np.ndarray],
    rotations: list[np.ndarray],
    centers: np.ndarray,
    crop_centers: np.ndarray,
    crop_scales: np.ndarray,
    depths: torch.Tensor,
) -> torch.Tensor:
    """V x V x J x H x W support; diagonal entries are zero."""
    n_views, n_joints, height, width = normalized_heatmaps.shape
    supports = normalized_heatmaps.new_zeros(
        (n_views, n_views, n_joints, height, width)
    )
    for target in range(n_views):
        for source in range(n_views):
            if source == target:
                continue
            grid = projection_grid(
                intrinsics[target],
                rotations[target],
                centers[target],
                crop_centers[target],
                crop_scales[target],
                intrinsics[source],
                rotations[source],
                centers[source],
                crop_centers[source],
                crop_scales[source],
                width,
                height,
                depths,
                normalized_heatmaps.device,
            )
            sampled = functional.grid_sample(
                normalized_heatmaps[source : source + 1],
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0]
            supports[target, source] = sampled.amax(dim=1).reshape(
                n_joints, height, width
            )
    return supports


def decode_quarter_pixel(heatmaps: torch.Tensor) -> np.ndarray:
    n_views, n_joints, height, width = heatmaps.shape
    flat_indices = heatmaps.flatten(-2).argmax(dim=-1)
    xs = flat_indices % width
    ys = torch.div(flat_indices, width, rounding_mode="floor")
    view_ids = torch.arange(n_views, device=heatmaps.device)[:, None]
    joint_ids = torch.arange(n_joints, device=heatmaps.device)[None, :]
    safe_x = xs.clamp(1, width - 2)
    safe_y = ys.clamp(1, height - 2)
    dx = (
        heatmaps[view_ids, joint_ids, safe_y, safe_x + 1]
        - heatmaps[view_ids, joint_ids, safe_y, safe_x - 1]
    )
    dy = (
        heatmaps[view_ids, joint_ids, safe_y + 1, safe_x]
        - heatmaps[view_ids, joint_ids, safe_y - 1, safe_x]
    )
    valid_x = (xs > 1) & (xs < width - 1) & (ys > 0) & (ys < height)
    valid_y = (ys > 1) & (ys < height - 1) & (xs > 0) & (xs < width)
    x = xs.float() + 0.25 * torch.sign(dx) * valid_x
    y = ys.float() + 0.25 * torch.sign(dy) * valid_y
    return torch.stack((x, y), dim=-1).cpu().numpy()


def a1d_corrected_coco(
    fusion_model,
    depths: torch.Tensor,
    joint_ids: torch.Tensor,
    heatmaps_full: np.ndarray,
    decoded_keypoints: np.ndarray,
    input_center: np.ndarray,
    input_scale: np.ndarray,
    intrinsics: list[np.ndarray],
    rotations: list[np.ndarray],
    centers: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return A1D dense-fusion corrected COCO 2D for one synchronized group.

    Mirrors the ``learned_dense_residual`` variant of the standalone dense
    evaluator so that trainer and evaluator produce identical corrections.
    Peaks whose argmax cell is unchanged keep the exact official decoded
    coordinate, matching the float16 quantization guard used at validation.
    """
    n_views = heatmaps_full.shape[0]
    full_hm = torch.as_tensor(
        heatmaps_full, dtype=torch.float32, device=device
    ).clamp_min(0.0)
    maximum = full_hm.flatten(-2).amax(dim=-1, keepdim=True)
    normalized = full_hm / maximum.clamp_min(1e-6)[..., None]
    support = epipolar_support(
        normalized,
        intrinsics,
        rotations,
        centers,
        input_center,
        input_scale,
        depths,
    )
    with torch.no_grad():
        learned_logits, _ = fusion_model(
            normalized[:, DIRECT_COCO_JOINTS],
            support[:, :, DIRECT_COCO_JOINTS],
            joint_ids=joint_ids,
        )
    learned_full = torch.log(normalized + 1e-4)
    learned_full[:, DIRECT_COCO_JOINTS] = learned_logits
    _, _, height, width = full_hm.shape
    heatmap_xy = decode_quarter_pixel(learned_full)
    coco = np.stack(
        [
            heatmap_to_image(
                heatmap_xy[view],
                input_center[view],
                input_scale[view],
                width,
                height,
            )
            for view in range(n_views)
        ]
    )
    original_peak = normalized.flatten(-2).argmax(dim=-1).cpu().numpy()
    fused_peak = learned_full.flatten(-2).argmax(dim=-1).cpu().numpy()
    unchanged = fused_peak == original_peak
    coco[unchanged] = np.asarray(decoded_keypoints)[unchanged]
    return coco


def fusion_variants(
    normalized: torch.Tensor,
    support: torch.Tensor,
    alphas: list[float],
) -> dict[str, torch.Tensor]:
    n_views = normalized.shape[0]
    cross = support.sum(dim=1) / max(n_views - 1, 1)
    output: dict[str, torch.Tensor] = {}
    epsilon = 1e-4
    for alpha in alphas:
        tag = f"{alpha:g}"
        # AdaFuse-style arithmetic exchange.  The denominator is irrelevant
        # to argmax and is omitted.
        output[f"dense_add_a{tag}"] = normalized + alpha * cross
        # A conservative residual gate: geometry can reorder existing
        # appearance evidence but cannot create an unsupported line peak.
        output[f"dense_residual_a{tag}"] = normalized * (
            1.0 + alpha * cross
        )
        # Product-of-experts form, useful when 3+ views agree strongly.
        output[f"dense_poe_a{tag}"] = torch.log(
            normalized + epsilon
        ) + alpha * torch.log(cross + epsilon)
    return output


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
        values = [
            value for action_group in action_values.values()
            for value in action_group
        ]
        output[method] = {
            "frame_weighted_all17_mm": float(np.mean(values)),
            "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
            "frame_weighted_kp_star_mm": float(np.mean(kp_errors[method])),
            "per_action_all17_mm": per_action,
        }
    return output


def evaluate_cardinality(
    records: list[dict],
    store: DenseHeatmapStore,
    four_view_groups: list[list[int]],
    n_views: int,
    alphas: list[float],
    depths: torch.Tensor,
    irls_iterations: int,
    limit_groups: int,
    device: torch.device,
    fusion_model: DenseGeometryResidualFusion | None,
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
        data = store.get(group)
        raw_heatmaps = torch.as_tensor(
            data["heatmaps"], dtype=torch.float32, device=device
        ).clamp_min_(0.0)
        maximum = raw_heatmaps.flatten(-2).amax(dim=-1, keepdim=True)
        normalized = raw_heatmaps / maximum.clamp_min(1e-6)[..., None]

        camera_data = [camera_parameters(record) for record in group_records]
        intrinsics = [item[0] for item in camera_data]
        rotations = [item[1] for item in camera_data]
        centers = np.stack([item[2] for item in camera_data])
        support = epipolar_support(
            normalized,
            intrinsics,
            rotations,
            centers,
            data["input_center"],
            data["input_scale"],
            depths,
        )
        variants = fusion_variants(normalized, support, alphas)
        if fusion_model is not None:
            joint_ids = torch.as_tensor(
                DIRECT_COCO_JOINTS, dtype=torch.long, device=device
            )
            with torch.no_grad():
                learned_logits, _ = fusion_model(
                    normalized[:, DIRECT_COCO_JOINTS],
                    support[:, :, DIRECT_COCO_JOINTS],
                    joint_ids=joint_ids,
                )
            learned_full = torch.log(normalized + 1e-4)
            learned_full[:, DIRECT_COCO_JOINTS] = learned_logits
            variants["learned_dense_residual"] = learned_full
        methods: dict[str, np.ndarray] = {
            "top1": data["decoded_keypoints"].astype(np.float64)
        }
        _, _, height, width = raw_heatmaps.shape
        original_peak = normalized.flatten(-2).argmax(dim=-1).cpu().numpy()
        for name, fused in variants.items():
            heatmap_xy = decode_quarter_pixel(fused)
            image_xy = np.stack(
                [
                    heatmap_to_image(
                        heatmap_xy[view],
                        data["input_center"][view],
                        data["input_scale"][view],
                        width,
                        height,
                    )
                    for view in range(n_views)
                ]
            )
            # Dense maps are stored as float16 to keep the full train export
            # practical.  Quantization can flip the sign of MMPose's tiny
            # quarter-pixel gradient even when the argmax cell is unchanged.
            # Preserve the exact official decoded coordinate in that case;
            # only genuinely changed peak cells use the fused-map decoder.
            fused_peak = fused.flatten(-2).argmax(dim=-1).cpu().numpy()
            unchanged = fused_peak == original_peak
            image_xy[unchanged] = data["decoded_keypoints"][unchanged]
            methods[name] = image_xy

        target = target_world_metres(group_records[0])
        action = ACTION_NAMES[int(group_records[0]["action"])]
        raw_confidence = data["decoded_scores"].astype(np.float64)
        for method, coco_xy in methods.items():
            h36m_xy = []
            h36m_confidence = []
            for view in range(n_views):
                joints, confidence = coco_to_h36m(
                    coco_xy[view], raw_confidence[view]
                )
                h36m_xy.append(joints)
                h36m_confidence.append(confidence)
            h36m_xy = np.stack(h36m_xy)
            h36m_confidence = np.stack(h36m_confidence)
            directions = np.stack(
                [
                    pixels_to_rays(
                        h36m_xy[view], intrinsics[view], rotations[view]
                    )
                    for view in range(n_views)
                ]
            )
            prediction = np.stack(
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
            joint_error = (
                np.linalg.norm(prediction - target, axis=-1) * 1000.0
            )
            errors[method][action].append(float(joint_error.mean()))
            kp_errors[method].append(
                float(joint_error[list(KP_STAR)].mean())
            )

        if group_number % 25 == 0 or group_number == len(groups):
            print(
                f"V{n_views}: {group_number}/{len(groups)} groups",
                flush=True,
            )

    return {
        "views": n_views,
        "groups": len(groups),
        "methods": summarize(errors, kp_errors),
    }


def main() -> None:
    args = parse_args()
    if args.depth_samples < 2:
        raise ValueError("--depth-samples must be at least two")
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    store = DenseHeatmapStore(args.dense_shards)
    four_view_groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    if not four_view_groups:
        raise RuntimeError("dense shards contain no complete four-view groups")
    device = torch.device(args.device)
    fusion_model = None
    if args.fusion_checkpoint:
        payload = torch.load(args.fusion_checkpoint, map_location=device)
        fusion_model = DenseGeometryResidualFusion().to(device)
        fusion_model.load_state_dict(payload["model"])
        fusion_model.eval()
    depths = torch.linspace(
        args.depth_min_m,
        args.depth_max_m,
        args.depth_samples,
        device=device,
    )
    results = {
        "input_pkl": args.input_pkl,
        "dense_shards": args.dense_shards,
        "complete_four_view_groups": len(four_view_groups),
        "depth_range_m": [args.depth_min_m, args.depth_max_m],
        "depth_samples": args.depth_samples,
        "alphas": args.alphas,
        "fusion_checkpoint": args.fusion_checkpoint,
        "results": {},
    }
    for n_views in args.views:
        cardinality = evaluate_cardinality(
            records,
            store,
            four_view_groups,
            n_views,
            args.alphas,
            depths,
            args.irls_iterations,
            args.limit_groups,
            device,
            fusion_model,
        )
        results["results"][f"V{n_views}"] = cardinality
        print(f"V{n_views}", flush=True)
        for method, metrics in cardinality["methods"].items():
            print(
                f"  {method}: All="
                f"{metrics['frame_weighted_all17_mm']:.3f}, KP*="
                f"{metrics['frame_weighted_kp_star_mm']:.3f}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
