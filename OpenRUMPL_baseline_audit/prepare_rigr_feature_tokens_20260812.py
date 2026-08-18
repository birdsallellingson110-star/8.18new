#!/usr/bin/env python3
"""Prepare compact HRNet feature tokens at the H76 3-D query projections.

The detector is frozen.  For every complete four-camera group and H76 full
four-view prediction, this script projects each joint query into every crop
and samples a small feature patch from the exported HRNet high-resolution
feature map.  The resulting compact tensor is used by the feature-level
RIGR/MVGFormer probe, so the training dataloader does not repeatedly read a
4.2 GB feature memmap for the eleven camera subsets of the same frame.

The operation is inference-only and contains no ground-truth access.  GT is
only used later by the training/evaluation script for the supervised loss.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups, project_world


COMBINATIONS = tuple(
    combo for n in (2, 3, 4)
    for combo in __import__("itertools").combinations(range(4), n)
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--h76-cache", nargs="+", required=True)
    p.add_argument("--feature-shards", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--group-indices-file", default="",
                   help="Optional .npy/.txt complete-group IDs; preserves file order.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--all-combinations", action="store_true",
                   help="sample features separately at each of the 11 H76 subset predictions")
    return p.parse_args()


class FeatureStore:
    """Map original PKL record indices to float16 feature memmap rows."""

    def __init__(self, paths: list[str]) -> None:
        self.locations: dict[int, tuple[int, int]] = {}
        self.features = []
        self.metadata = []
        for shard_id, path_string in enumerate(paths):
            path = Path(path_string)
            feature_path = path.with_name(path.name.replace(".npz", ".features.npy"))
            if not feature_path.is_file():
                raise FileNotFoundError(f"missing feature memmap: {feature_path}")
            with np.load(path) as source:
                indices = source["record_indices"].copy()
                meta = {
                    "input_center": source["input_center"].copy(),
                    "input_scale": source["input_scale"].copy(),
                    "input_size": source["input_size"].copy(),
                }
            features = np.load(feature_path, mmap_mode="r")
            if len(indices) != len(features):
                raise ValueError(f"metadata/feature mismatch: {path}")
            self.features.append(features)
            self.metadata.append(meta)
            for row, record_index in enumerate(indices):
                record_index = int(record_index)
                if record_index in self.locations:
                    raise ValueError(f"duplicate feature record {record_index}")
                self.locations[record_index] = (shard_id, row)
        if not self.locations:
            raise ValueError("empty feature store")

def load_cache(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key].copy() for key in source.files}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_four_view_groups(records)
    cache_parts = [load_cache(path) for path in args.h76_cache]
    if len(cache_parts) == 1:
        cache = cache_parts[0]
    else:
        keys = cache_parts[0].keys()
        cache = {key: np.concatenate([part[key] for part in cache_parts], axis=0) for key in keys}
        order = np.argsort(cache["group_indices"])
        cache = {key: value[order] for key, value in cache.items()}
    if len(groups) != len(cache["predictions"]):
        raise ValueError(f"group/cache mismatch: {len(groups)} vs {len(cache['predictions'])}")
    if args.group_indices_file:
        path = Path(args.group_indices_file)
        if path.suffix == ".npy":
            selected_ids = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
        else:
            selected_ids = np.asarray(
                [int(line.strip()) for line in path.read_text().splitlines() if line.strip()],
                dtype=np.int64,
            )
        if len(selected_ids) == 0 or np.any(selected_ids < 0) or np.any(selected_ids >= len(groups)):
            raise ValueError(f"invalid group-indices-file: {path}")
        groups = [groups[int(i)] for i in selected_ids]
        cache_rows = {int(group_id): row for row, group_id in enumerate(cache["group_indices"])}
        rows = np.asarray([cache_rows[int(i)] for i in selected_ids], dtype=np.int64)
        cache = {key: value[rows] for key, value in cache.items()}
    else:
        limit = min(args.limit_groups, len(groups)) if args.limit_groups else len(groups)
        groups = groups[:limit]
        cache = {key: value[:limit] for key, value in cache.items()}
    limit = len(groups)
    predictions = cache["predictions"].astype(np.float32)
    store = FeatureStore(args.feature_shards)
    c, h, w = np.asarray(store.features[0][0]).shape
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Shape is known before sampling and makes interrupted runs detectable.
    if args.all_combinations:
        output_shape = (limit, len(COMBINATIONS), 4, 17, c, args.patch_size, args.patch_size)
    else:
        output_shape = (limit, 4, 17, c, args.patch_size, args.patch_size)
    output = np.lib.format.open_memmap(
        out_path.with_suffix(out_path.suffix + ".tmp"), mode="w+", dtype=np.float16,
        shape=output_shape,
    )
    combo_ids = range(len(COMBINATIONS)) if args.all_combinations else [10]
    for combo_id in combo_ids:
        combo = COMBINATIONS[combo_id]
        for start in range(0, limit, args.batch_size):
            end = min(start + args.batch_size, limit)
            sampled = _sample_group_batch(
                store, records, groups, predictions[:, combo_id], start, end,
                args.patch_size, device, combo,
            )
            if args.all_combinations:
                output[start:end, combo_id] = sampled
            else:
                output[start:end] = sampled
            if ((end % (args.batch_size * 10) == 0) or end == limit) and combo_id in (0, 10):
                print(f"combo {combo_id} groups {end}/{limit}", flush=True)
    output.flush()
    output.filename and Path(output.filename).replace(out_path)
    metadata = {"groups": limit, "shape": list(output.shape), "dtype": "float16",
                "patch_size": args.patch_size, "source": "H76 full-view query",
                "subset_specific": bool(args.all_combinations),
                "combinations": [list(c) for c in COMBINATIONS],
                "feature_shards": [str(Path(item).resolve()) for item in args.feature_shards]}
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def _sample_group_batch(store: FeatureStore, records: list[dict], groups: list[list[int]],
                        predictions: np.ndarray, start: int, end: int, patch_size: int,
                        device: torch.device, combo: tuple[int, ...] | None = None) -> np.ndarray:
    """Batch version of FeatureStore sampling with the correct group per row."""
    half = patch_size // 2
    b, v, j = end - start, 4, 17
    selected_views = tuple(range(v)) if combo is None else tuple(combo)
    maps, coords = [], np.zeros((b, v, j, 2), dtype=np.float32)
    valid = np.zeros((b, v), dtype=np.bool_)
    for local, gi in enumerate(range(start, end)):
        for camera in selected_views:
            valid[local, camera] = True
            record_index = int(groups[gi][camera])
            shard_id, row = store.locations[record_index]
            maps.append(store.features[shard_id][row])
            center = store.metadata[shard_id]["input_center"][row]
            scale = store.metadata[shard_id]["input_scale"][row]
            xy = project_world(predictions[gi], records[record_index])
            coords[local, camera] = (xy - center[None] + 0.5 * scale[None]) / scale[None] * np.asarray([72.0, 96.0], np.float32)
    fmap = torch.from_numpy(np.stack(maps)).to(device=device, dtype=torch.float32)
    c, h, w = fmap.shape[1:]
    base = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
    offsets = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    patch = torch.stack((ox, oy), dim=-1)
    query = base[:, :, :, None, None, :] + patch[None, None, None]
    grid = torch.stack((query[..., 0] / max(w - 1, 1) * 2.0 - 1.0,
                        query[..., 1] / max(h - 1, 1) * 2.0 - 1.0), dim=-1)
    # Only selected views were stacked above.  Keep zero-valued padded views
    # in the returned tensor so the regular collate path remains unchanged.
    nv = len(selected_views)
    grid = grid[:, selected_views].reshape(b * nv * j, patch_size, patch_size, 2)
    fmap_j = fmap[:, None].expand(-1, j, -1, -1, -1).reshape(b * nv * j, c, h, w)
    sampled = F.grid_sample(fmap_j, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    selected_sampled = sampled.reshape(b, nv, j, c, patch_size, patch_size)
    result = torch.zeros((b, v, j, c, patch_size, patch_size), device=device, dtype=selected_sampled.dtype)
    result[:, selected_views] = selected_sampled
    return result.cpu().numpy().astype(np.float16)


if __name__ == "__main__":
    main()
