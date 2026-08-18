#!/usr/bin/env python3
"""Prepare detector-geometry residual auxiliaries for RIGR feature tokens.

For each camera subset, the auxiliary vector at a joint/view is:
  [decoded_2d - projected_H76_3d] / bbox_scale (2),
  decoded detector confidence (1),
  decoded 2-D crop coordinate (2).

The H76 prediction determines the projection, and no ground truth is read.
The output is subset-specific so V2/V3/V4 never receives information from an
unused camera.  This makes the signal explicit for a selective 2-D correction
experiment instead of asking the feature encoder to infer pixel residuals
from appearance alone.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups, project_world


COMBINATIONS = tuple(
    combo for n in (2, 3, 4) for combo in itertools.combinations(range(4), n)
)
H36M_TO_COCO = {
    1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15,
    9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--h76-cache", nargs="+", required=True)
    p.add_argument("--detector-shards", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group-indices-file", default="")
    p.add_argument("--limit-groups", type=int, default=0)
    return p.parse_args()


def load_cache(paths: list[str]) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path) as source:
            parts.append({key: source[key].copy() for key in source.files})
    keys = parts[0].keys()
    cache = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    order = np.argsort(cache["group_indices"])
    return {key: value[order] for key, value in cache.items()}


class DetectorMetaStore:
    def __init__(self, paths: list[str]) -> None:
        self.locations: dict[int, tuple[int, int]] = {}
        self.shards = []
        for shard_id, path in enumerate(paths):
            with np.load(path) as source:
                required = ("record_indices", "decoded_keypoints", "decoded_scores",
                            "input_center", "input_scale")
                missing = [key for key in required if key not in source]
                if missing:
                    raise ValueError(f"{path}: missing {missing}")
                shard = {key: source[key].copy() for key in required}
            self.shards.append(shard)
            for row, record_index in enumerate(shard["record_indices"]):
                record_index = int(record_index)
                if record_index in self.locations:
                    raise ValueError(f"duplicate detector record {record_index}")
                self.locations[record_index] = (shard_id, row)

    @staticmethod
    def h36m_xy_score(coco_xy: np.ndarray, coco_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.zeros((17, 2), dtype=np.float32)
        score = np.zeros(17, dtype=np.float32)
        for joint, channel in H36M_TO_COCO.items():
            xy[joint] = coco_xy[channel]
            score[joint] = coco_score[channel]
        xy[0] = coco_xy[[11, 12]].mean(axis=0)
        score[0] = coco_score[[11, 12]].mean()
        xy[8] = coco_xy[[3, 4, 5, 6]].mean(axis=0)
        score[8] = coco_score[[3, 4, 5, 6]].mean()
        xy[10] = coco_xy[[0, 1, 2, 3, 4]].mean(axis=0)
        score[10] = coco_score[[0, 1, 2, 3, 4]].mean()
        xy[7] = 0.5 * (xy[0] + xy[8])
        score[7] = 0.5 * (score[0] + score[8])
        return xy, score

    def get(self, record_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shard_id, row = self.locations[int(record_index)]
        shard = self.shards[shard_id]
        xy, score = self.h36m_xy_score(shard["decoded_keypoints"][row], shard["decoded_scores"][row])
        return (xy, score, shard["input_center"][row], shard["input_scale"][row])


def main() -> None:
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_four_view_groups(records)
    cache = load_cache(args.h76_cache)
    if len(groups) != len(cache["predictions"]):
        raise ValueError(f"group/cache mismatch {len(groups)} vs {len(cache['predictions'])}")
    if args.group_indices_file:
        path = Path(args.group_indices_file)
        selected_ids = np.asarray(np.load(path), dtype=np.int64).reshape(-1) if path.suffix == ".npy" else np.asarray(
            [int(line.strip()) for line in path.read_text().splitlines() if line.strip()], dtype=np.int64)
        if len(selected_ids) == 0 or np.any(selected_ids < 0) or np.any(selected_ids >= len(groups)):
            raise ValueError(f"invalid group-indices-file {path}")
        groups = [groups[int(i)] for i in selected_ids]
        rows = {int(group_id): row for row, group_id in enumerate(cache["group_indices"])}
        cache = {key: value[np.asarray([rows[int(i)] for i in selected_ids])] for key, value in cache.items()}
    else:
        limit = min(args.limit_groups, len(groups)) if args.limit_groups else len(groups)
        groups = groups[:limit]
        cache = {key: value[:limit] for key, value in cache.items()}
    limit = len(groups)
    detector = DetectorMetaStore(args.detector_shards)
    output = np.lib.format.open_memmap(
        Path(args.output).with_suffix(Path(args.output).suffix + ".tmp"), mode="w+",
        dtype=np.float16, shape=(limit, len(COMBINATIONS), 4, 17, 5),
    )
    for group_id in range(limit):
        for combo_id, combo in enumerate(COMBINATIONS):
            prediction = cache["predictions"][group_id, combo_id].astype(np.float32)
            for camera in combo:
                record_index = groups[group_id][camera]
                decoded_xy, score, center, scale = detector.get(record_index)
                projected_xy = project_world(prediction, records[record_index]).astype(np.float32)
                residual = (decoded_xy - projected_xy) / np.maximum(scale[None], 1e-6)
                crop_xy = (decoded_xy - center[None] + 0.5 * scale[None]) / np.maximum(scale[None], 1e-6)
                output[group_id, combo_id, camera] = np.concatenate(
                    (residual, np.clip(score[:, None], 0.0, 1.0), crop_xy), axis=-1
                ).astype(np.float16)
        if (group_id + 1) % 1000 == 0 or group_id + 1 == limit:
            print(f"groups {group_id + 1}/{limit}", flush=True)
    output.flush()
    Path(output.filename).replace(args.output)
    metadata = {"groups": limit, "shape": list(output.shape), "dtype": "float16",
                "aux_dim": 5, "subset_specific": True,
                "combinations": [list(c) for c in COMBINATIONS]}
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
