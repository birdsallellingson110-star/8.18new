#!/usr/bin/env python3
"""Export camera-ID-free epipolar correspondence features for frozen RUMPL.

This is a query-level adaptation of Epipolar Transformer (CVPR 2020).  For
each frozen RUMPL joint query and each target view, the target feature is
matched against samples on the corresponding epipolar line in every other
available view.  Source views are reduced symmetrically and no world voxel,
camera identity, target 3-D label, or test-time oracle is used.

The output is compact and can be consumed by
``train_epipolar_query_rumpl_20260813.py`` without re-running HRNet.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_rigr_heatmap_oracle_20260812 import (
    build_four_view_groups,
    camera_parameters,
    project_world,
)


COMBINATIONS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--h76-cache", nargs="+", required=True)
    parser.add_argument("--feature-shards", nargs="+", required=True)
    parser.add_argument("--group-indices-file", default="")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--depth-samples", type=int, default=64)
    parser.add_argument("--ray-window-m", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.1767766953)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def load_cache(paths: list[str]) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path) as source:
            parts.append({key: source[key].copy() for key in source.files})
    merged = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }
    order = np.argsort(merged["group_indices"])
    return {key: value[order] for key, value in merged.items()}


class FeatureStore:
    """Map original PKL record indices to exported frozen HRNet features."""

    def __init__(self, shard_paths: list[str]) -> None:
        self.locations: dict[int, tuple[int, int]] = {}
        self.features: list[np.ndarray] = []
        self.metadata: list[dict[str, np.ndarray]] = []
        for shard_id, name in enumerate(shard_paths):
            path = Path(name)
            feature_path = path.with_name(
                path.name.replace(".npz", ".features.npy")
            )
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            with np.load(path) as source:
                indices = source["record_indices"].astype(np.int64)
                metadata = {
                    key: source[key].copy()
                    for key in ("input_center", "input_scale", "input_size")
                }
            features = np.load(feature_path, mmap_mode="r")
            if len(features) != len(indices):
                raise ValueError(f"feature/metadata mismatch: {path}")
            self.features.append(features)
            self.metadata.append(metadata)
            for row, record_index in enumerate(indices):
                record_index = int(record_index)
                if record_index in self.locations:
                    raise ValueError(f"duplicate feature record {record_index}")
                self.locations[record_index] = (shard_id, row)

    def group(self, record_indices: list[int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        maps, centers, scales, sizes = [], [], [], []
        for record_index in record_indices:
            shard, row = self.locations[int(record_index)]
            maps.append(np.asarray(self.features[shard][row]))
            centers.append(self.metadata[shard]["input_center"][row])
            scales.append(self.metadata[shard]["input_scale"][row])
            sizes.append(self.metadata[shard]["input_size"][row])
        return np.stack(maps), {
            "center": np.stack(centers).astype(np.float32),
            "scale": np.stack(scales).astype(np.float32),
            "size": np.stack(sizes).astype(np.float32),
        }


def read_group_ids(path: str, total: int, limit: int) -> np.ndarray:
    if path:
        source = Path(path)
        if source.suffix == ".npy":
            ids = np.asarray(np.load(source), dtype=np.int64).reshape(-1)
        else:
            ids = np.asarray(
                [int(line) for line in source.read_text().splitlines() if line.strip()],
                dtype=np.int64,
            )
    else:
        ids = np.arange(total, dtype=np.int64)
    if limit:
        ids = ids[:limit]
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= total:
        raise ValueError(f"invalid group IDs for total={total}: {ids[:10]}")
    return ids


def image_to_feature(
    image_xy: torch.Tensor,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
) -> torch.Tensor:
    center_t = image_xy.new_tensor(center)
    scale_t = image_xy.new_tensor(scale)
    size = image_xy.new_tensor([width, height])
    return (image_xy - center_t + 0.5 * scale_t) / scale_t * size


def normalize_grid(xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    denominator = xy.new_tensor([max(width - 1, 1), max(height - 1, 1)])
    return xy / denominator * 2.0 - 1.0


def sample_points(feature: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Sample CxHxW at arbitrary ``...x2`` feature coordinates."""
    height, width = feature.shape[-2:]
    shape = xy.shape[:-1]
    grid = normalize_grid(xy, width, height).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature[None], grid, mode="bilinear", padding_mode="zeros",
        align_corners=True,
    )[0, :, :, 0].transpose(0, 1)
    return sampled.reshape(*shape, feature.shape[0])


def descriptor_for_combo(
    maps: torch.Tensor,
    metadata: dict[str, np.ndarray],
    records: list[dict],
    prediction: np.ndarray,
    combo: tuple[int, ...],
    offsets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return padded VxJxD descriptors for one camera subset."""
    device = maps.device
    channels = maps.shape[1]
    joints = prediction.shape[0]
    descriptor_dim = 2 * channels + 6
    output = maps.new_zeros((4, joints, descriptor_dim))
    prediction_t = maps.new_tensor(prediction)

    for target in combo:
        target_xy_np = project_world(prediction, records[target]).astype(np.float32)
        target_xy = maps.new_tensor(target_xy_np)
        target_feature_xy = image_to_feature(
            target_xy, metadata["center"][target], metadata["scale"][target],
            maps.shape[-1], maps.shape[-2],
        )
        query = sample_points(maps[target], target_feature_xy)
        query_norm = F.normalize(query, dim=-1, eps=1e-6)

        intrinsic, rotation, camera_center = camera_parameters(records[target])
        homogeneous = np.concatenate(
            (target_xy_np, np.ones((joints, 1), dtype=np.float32)), axis=-1
        )
        direction = homogeneous @ np.linalg.inv(intrinsic).T @ rotation
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        direction_t = maps.new_tensor(direction)
        center_t = maps.new_tensor(camera_center)
        central_depth = ((prediction_t - center_t) * direction_t).sum(-1)
        depths = (central_depth[:, None] + offsets[None]).clamp_min(0.1)
        world = center_t[None, None] + depths[..., None] * direction_t[:, None]

        matches, similarities, entropies, expected_offsets, valid_fractions = [], [], [], [], []
        for source in combo:
            if source == target:
                continue
            source_k, source_r, source_center = camera_parameters(records[source])
            camera = (world - maps.new_tensor(source_center)) @ maps.new_tensor(source_r).T
            projected = camera @ maps.new_tensor(source_k).T
            image_xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
            feature_xy = image_to_feature(
                image_xy, metadata["center"][source], metadata["scale"][source],
                maps.shape[-1], maps.shape[-2],
            )
            valid = (
                (camera[..., 2] > 0)
                & (feature_xy[..., 0] >= 0)
                & (feature_xy[..., 0] <= maps.shape[-1] - 1)
                & (feature_xy[..., 1] >= 0)
                & (feature_xy[..., 1] <= maps.shape[-2] - 1)
            )
            sampled = sample_points(maps[source], feature_xy)
            similarity = (query_norm[:, None] * F.normalize(sampled, dim=-1, eps=1e-6)).sum(-1)
            similarity = similarity.masked_fill(~valid, -1e4)
            no_valid = ~valid.any(dim=-1)
            if no_valid.any():
                similarity[no_valid, offsets.numel() // 2] = 0.0
            attention = torch.softmax(similarity / temperature, dim=-1)
            match = (attention[..., None] * sampled).sum(dim=-2)
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
            entropy = entropy / np.log(float(offsets.numel()))
            matches.append(match)
            similarities.append((attention * similarity.clamp_min(-1.0)).sum(-1))
            entropies.append(entropy)
            expected_offsets.append((attention * offsets[None]).sum(-1))
            valid_fractions.append(valid.float().mean(-1))

        match_stack = torch.stack(matches, dim=0)
        mean_match = match_stack.mean(dim=0)
        similarity_stack = torch.stack(similarities, dim=0)
        entropy_stack = torch.stack(entropies, dim=0)
        offset_stack = torch.stack(expected_offsets, dim=0)
        valid_stack = torch.stack(valid_fractions, dim=0)
        scalar = torch.stack(
            (
                similarity_stack.mean(0), similarity_stack.amax(0),
                entropy_stack.mean(0), offset_stack.mean(0),
                offset_stack.std(0, unbiased=False), valid_stack.mean(0),
            ),
            dim=-1,
        )
        output[target] = torch.cat(
            (query, mean_match, scalar),
            dim=-1,
        )
    return output


def main() -> None:
    args = parse_args()
    if args.depth_samples < 2 or args.ray_window_m <= 0 or args.temperature <= 0:
        raise ValueError("invalid epipolar sampling parameters")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_four_view_groups(records)
    cache = load_cache(args.h76_cache)
    if len(groups) != len(cache["predictions"]):
        raise ValueError(f"group/cache mismatch {len(groups)} != {len(cache['predictions'])}")
    group_ids = read_group_ids(
        args.group_indices_file, len(groups), args.limit_groups
    )
    shard_start = len(group_ids) * args.shard_index // args.num_shards
    shard_stop = len(group_ids) * (args.shard_index + 1) // args.num_shards
    group_ids = group_ids[shard_start:shard_stop]
    cache_rows = {int(group_id): row for row, group_id in enumerate(cache["group_indices"])}
    rows = np.asarray([cache_rows[int(group_id)] for group_id in group_ids])
    predictions = cache["predictions"][rows]
    store = FeatureStore(args.feature_shards)
    missing = [
        record_index for group_id in group_ids for record_index in groups[int(group_id)]
        if int(record_index) not in store.locations
    ]
    if missing:
        raise ValueError(f"feature store misses {len(missing)} selected records; first={missing[:5]}")

    first_maps, _ = store.group(groups[int(group_ids[0])])
    channels = int(first_maps.shape[1])
    descriptor_dim = 2 * channels + 6
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16,
        shape=(len(group_ids), len(COMBINATIONS), 4, 17, descriptor_dim),
    )
    offsets = torch.linspace(
        -args.ray_window_m, args.ray_window_m, args.depth_samples,
        device=device,
    )
    with torch.no_grad():
        for position, group_id in enumerate(group_ids):
            record_indices = groups[int(group_id)]
            maps_np, metadata = store.group(record_indices)
            maps = torch.as_tensor(maps_np, dtype=torch.float32, device=device)
            group_records = [records[index] for index in record_indices]
            for combo_id, combo in enumerate(COMBINATIONS):
                output[position, combo_id] = descriptor_for_combo(
                    maps, metadata, group_records, predictions[position, combo_id],
                    combo, offsets, args.temperature,
                ).cpu().numpy().astype(np.float16)
            if (position + 1) % args.log_every == 0 or position + 1 == len(group_ids):
                print(f"groups={position + 1}/{len(group_ids)}", flush=True)
    output.flush()
    Path(output.filename).replace(output_path)
    group_path = output_path.with_suffix(output_path.suffix + ".group_indices.npy")
    np.save(group_path, group_ids)
    metadata = {
        "method": "query-level Epipolar Transformer feature correspondence for frozen RUMPL",
        "paper_basis": "Epipolar Transformer, CVPR 2020",
        "camera_identity": False,
        "world_voxel": False,
        "ground_truth_used": False,
        "groups": len(group_ids),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_id_slice": [shard_start, shard_stop],
        "shape": list(output.shape),
        "dtype": "float16",
        "channels": channels,
        "depth_samples": args.depth_samples,
        "ray_window_m": args.ray_window_m,
        "temperature": args.temperature,
        "group_indices": str(group_path.resolve()),
        "feature_shards": [str(Path(path).resolve()) for path in args.feature_shards],
        "combinations": [list(combo) for combo in COMBINATIONS],
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
