#!/usr/bin/env python3
"""P0 diagnostic for the Ray--Image Geometry Refiner (RIGR).

This is deliberately training-free.  It projects the frozen H76 3-D
prediction into each calibrated view, retrieves local modes from the already
exported HRNet dense heatmap, and measures two ceilings:

* 2-D oracle: choose the local mode closest to the annotated 2-D joint in each
  view, then triangulate;
* 3-D candidate oracle: enumerate local mode combinations and choose the
  triangulated hypothesis closest to the annotated 3-D joint.

The diagnostic answers whether the image evidence contains recoverable signal
before a RIGR/Epipolar-Transformer module is trained.  GT is used only for
the reported oracle ceilings; it is never an inference path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_h36m_sparse_epipolar_topk import (
    ACTION_NAMES,
    camera_parameters,
    pixels_to_rays,
    solve_ray_intersection,
)


COMBINATIONS = tuple(
    combo
    for views in (2, 3, 4)
    for combo in itertools.combinations(range(4), views)
)
COMBINATION_INDEX = {combo: index for index, combo in enumerate(COMBINATIONS)}

# The prepared H36M PKL uses the original H36M lower-body ordering.  RUMPL's
# verified H21/H76 convention swaps the right and left 3-joint chains.
LOWER_SWAP = np.asarray([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# COCO channel -> H36M joint for directly observed joints.  The remaining
# joints are the same averages used by the existing RUMPL conversion.
# The exported heatmaps retain standard MMPose COCO channel order.  RUMPL's
# H36M convention (after its verified lower-body swap) follows this direct
# H36M-joint -> COCO-channel table from its own MMPose preprocessing code.
H36M_TO_COCO_DIRECT = {
    1: 12,  # rhip
    2: 14,  # rkne
    3: 16,  # rank
    4: 11,  # lhip
    5: 13,  # lkne
    6: 15,  # lank
    9: 0,   # nose
    11: 5,  # lsho
    12: 7,  # lelb
    13: 9,  # lwri
    14: 6,  # rsho
    15: 8,  # relb
    16: 10, # rwri
}
DERIVED_COCO = {
    0: (11, 12),       # pelvis/root
    7: ("root", "neck"),
    8: (3, 4, 5, 6),    # neck
    10: (0, 1, 2, 3, 4), # head
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--h76-cache", required=True)
    p.add_argument("--dense-shards", nargs="+", required=True)
    p.add_argument("--radii", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument(
        "--oracle-mode", choices=("2d", "both"), default="both",
        help="2d reports the useful local 2-D ceiling; both additionally "
        "enumerates all local-mode 3-D hypotheses (expensive for V4).",
    )
    p.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


class DenseStore:
    """Memory-map dense heatmaps and expose rows by original PKL index."""

    def __init__(self, shard_paths: list[str]) -> None:
        self.shards = []
        self.locations: dict[int, tuple[int, int]] = {}
        for shard_id, path_string in enumerate(shard_paths):
            path = Path(path_string)
            dense_path = path.with_name(path.name.removesuffix(".npz") + ".heatmaps.npy")
            with np.load(path) as source:
                required = ("record_indices", "input_center", "input_scale", "input_size")
                missing = [key for key in required if key not in source]
                if missing:
                    raise RuntimeError(f"{path}: missing {missing}")
                metadata = {key: source[key].copy() for key in required}
            heatmaps = np.load(dense_path, mmap_mode="r")
            if len(metadata["record_indices"]) != len(heatmaps):
                raise RuntimeError(f"metadata/heatmap length mismatch: {path}")
            metadata["heatmaps"] = heatmaps
            self.shards.append(metadata)
            for row, record_index in enumerate(metadata["record_indices"]):
                record_index = int(record_index)
                if record_index in self.locations:
                    raise RuntimeError(f"duplicate dense record {record_index}")
                self.locations[record_index] = (shard_id, row)

    def get(self, record_index: int) -> dict[str, np.ndarray]:
        shard_id, row = self.locations[int(record_index)]
        shard = self.shards[shard_id]
        heatmaps = np.asarray(shard["heatmaps"][row], dtype=np.float32)
        # Convert COCO channels to the 17 RUMPL/H36M maps once per record.
        # The scalar oracle loops over many camera subsets; recomputing the
        # four derived maps inside every subset/joint was the dominant CPU
        # cost and did not change numerical values.
        joint_maps = np.empty((17, *heatmaps.shape[-2:]), dtype=np.float32)
        for joint, channel in H36M_TO_COCO_DIRECT.items():
            joint_maps[joint] = heatmaps[channel]
        joint_maps[8] = heatmaps[[3, 4, 5, 6]].mean(axis=0)
        joint_maps[10] = heatmaps[[0, 1, 2, 3, 4]].mean(axis=0)
        joint_maps[0] = heatmaps[[11, 12]].mean(axis=0)
        joint_maps[7] = 0.5 * (joint_maps[0] + joint_maps[8])
        return {
            "heatmaps": heatmaps,
            "joint_maps": joint_maps,
            "input_center": np.asarray(shard["input_center"][row], dtype=np.float64),
            "input_scale": np.asarray(shard["input_scale"][row], dtype=np.float64),
            "input_size": np.asarray(shard["input_size"][row], dtype=np.float64),
        }


def build_four_view_groups(records: list[dict]) -> list[list[int]]:
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        grouped.setdefault(key, [-1, -1, -1, -1])[int(record["camera_id"])] = index
    return [group for group in grouped.values() if min(group) >= 0]


def h36m_2d(record: dict) -> np.ndarray:
    points = np.asarray(record["joints_2d"], dtype=np.float64).copy()
    return points[LOWER_SWAP]


def project_world(point: np.ndarray, record: dict) -> np.ndarray:
    intrinsic, rotation, center = camera_parameters(record)
    camera = (np.asarray(point, dtype=np.float64) - center) @ rotation.T
    projected = camera @ intrinsic.T
    return projected[..., :2] / np.clip(projected[..., 2:], 1e-8, None)


def image_to_heatmap(
    image_xy: np.ndarray, center: np.ndarray, scale: np.ndarray,
    heatmap_shape: tuple[int, int], input_size: np.ndarray,
) -> np.ndarray:
    # MMPose's affine crop convention is retained by the exporter.  The
    # output heatmap is HxW while input_size is [width, height].
    height, width = heatmap_shape
    del input_size  # the exact crop scale is the transform used here
    size = np.asarray([width, height], dtype=np.float64)
    return (image_xy - center + 0.5 * scale) / scale * size


def heatmap_to_image(
    heatmap_xy: np.ndarray, center: np.ndarray, scale: np.ndarray,
    heatmap_shape: tuple[int, int], input_size: np.ndarray,
) -> np.ndarray:
    height, width = heatmap_shape
    size = np.asarray([width, height], dtype=np.float64)
    # Keep input_size in the expression for auditability.  It is [288,384]
    # for the 384x288 HRNet crop and cancels with the heatmap ratio.
    input_xy = heatmap_xy / size * np.asarray(input_size, dtype=np.float64)
    return input_xy / np.asarray(input_size, dtype=np.float64) * scale + center - 0.5 * scale


def joint_heatmap(heatmaps: np.ndarray, joint: int) -> np.ndarray:
    if joint in H36M_TO_COCO_DIRECT:
        return heatmaps[H36M_TO_COCO_DIRECT[joint]]
    if joint == 8:  # neck
        return heatmaps[[3, 4, 5, 6]].mean(axis=0)
    if joint == 10:  # head
        return heatmaps[[0, 1, 2, 3, 4]].mean(axis=0)
    if joint == 0:  # pelvis/root
        return heatmaps[[11, 12]].mean(axis=0)
    if joint == 7:
        return 0.5 * (joint_heatmap(heatmaps, 0) + joint_heatmap(heatmaps, 8))
    raise ValueError(f"unsupported H36M joint {joint}")


def extract_local_modes(
    heatmaps: np.ndarray, joint: int, projected_image_xy: np.ndarray,
    center: np.ndarray, scale: np.ndarray, input_size: np.ndarray,
    radius: float, topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    mode_map = heatmaps if heatmaps.ndim == 2 else joint_heatmap(heatmaps, joint)
    height, width = mode_map.shape
    hm_xy = image_to_heatmap(
        projected_image_xy, center, scale, (height, width), input_size
    )
    # A projection outside the crop cannot yield a meaningful local window;
    # use the global top pixels as an explicit diagnostic fallback.
    x, y = float(hm_xy[0]), float(hm_xy[1])
    if x < -radius or x >= width + radius or y < -radius or y >= height + radius:
        x0, x1, y0, y1 = 0, width, 0, height
    else:
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius)) + 1)
    window = mode_map[y0:y1, x0:x1]
    flat = window.reshape(-1)
    count = min(int(topk), len(flat))
    if count < 1:
        raise RuntimeError("empty heatmap window")
    # Stable descending order makes the diagnostic deterministic across runs.
    selected = np.argsort(-flat, kind="stable")[:count]
    local_y, local_x = np.divmod(selected, window.shape[1])
    hm_candidates = np.stack((local_x + x0, local_y + y0), axis=-1).astype(np.float64)
    xy = heatmap_to_image(hm_candidates, center, scale, (height, width), input_size)
    return xy, flat[selected].astype(np.float64)


def solve_selected(
    candidates: list[np.ndarray], records: list[dict], subset: tuple[int, ...],
    selected: np.ndarray,
) -> np.ndarray:
    directions = []
    centers = []
    for local_view, camera_index in enumerate(subset):
        intrinsic, rotation, center = camera_parameters(records[camera_index])
        directions.append(pixels_to_rays(selected[local_view], intrinsic, rotation))
        centers.append(center)
    directions = np.stack(directions, axis=0)
    centers = np.stack(centers, axis=0)
    return solve_ray_intersection(
        np.broadcast_to(centers, directions.shape[:-1] + (3,)),
        directions,
        np.ones(directions.shape[:-1], dtype=np.float64),
    )


def oracle_for_joint(
    candidate_xy: list[np.ndarray], records: list[dict], subset: tuple[int, ...],
    target: np.ndarray, gt2d: np.ndarray, compute_3d: bool = True,
) -> tuple[float, float | None, float]:
    """Return 2D-oracle mm, 3D-oracle mm, and 2D pixel error."""
    selected_2d = np.stack([
        candidate_xy[view][np.argmin(
            np.linalg.norm(candidate_xy[view] - gt2d[camera_index], axis=-1)
        )]
        for view, camera_index in enumerate(subset)
    ])
    pred_2d = solve_selected(candidate_xy, records, subset, selected_2d)
    error_2d = float(np.linalg.norm(pred_2d - target) * 1000.0)
    pixel_error = float(np.mean([
        np.min(np.linalg.norm(candidate_xy[view] - gt2d[camera_index], axis=-1))
        for view, camera_index in enumerate(subset)
    ]))

    if not compute_3d:
        return error_2d, None, pixel_error

    k = min(len(item) for item in candidate_xy)
    choices = np.asarray(list(itertools.product(range(k), repeat=len(subset))), dtype=np.int64)
    selected = np.stack([
        candidate_xy[view][choices[:, view]] for view in range(len(subset))
    ], axis=1)
    directions = []
    centers = []
    for view, camera_index in enumerate(subset):
        intrinsic, rotation, center = camera_parameters(records[camera_index])
        directions.append(pixels_to_rays(selected[:, view], intrinsic, rotation))
        centers.append(center)
    directions = np.stack(directions, axis=1)
    centers = np.broadcast_to(
        np.asarray(centers, dtype=np.float64)[None],
        (len(choices), len(subset), 3),
    )
    hypotheses = solve_ray_intersection(
        centers, directions,
        np.ones((len(choices), len(subset)), dtype=np.float64),
    )
    error_3d = float(np.min(np.linalg.norm(hypotheses - target[None], axis=-1)) * 1000.0)
    return error_2d, error_3d, pixel_error


def oracle_2d_batch(
    modes_by_joint: list[list[np.ndarray]], records: list[dict],
    subset: tuple[int, ...], targets: np.ndarray, gt2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 2-D oracle for all 17 joints of one camera subset.

    The earlier scalar diagnostic solved one 3x3 system per joint.  That is
    numerically identical but needlessly dominated by Python/BLAS call
    overhead.  Keeping the joint dimension in the batched ray solve makes the
    full validation diagnostic practical without changing any metric.
    """
    selected = np.stack([
        np.stack([
            modes_by_joint[joint][view][np.argmin(
                np.linalg.norm(
                    modes_by_joint[joint][view] - gt2d[camera_index, joint],
                    axis=-1,
                )
            )]
            for view, camera_index in enumerate(subset)
        ])
        for joint in range(17)
    ])
    directions = []
    centers = []
    for view, camera_index in enumerate(subset):
        intrinsic, rotation, center = camera_parameters(records[camera_index])
        directions.append(pixels_to_rays(selected[:, view], intrinsic, rotation))
        centers.append(center)
    directions = np.stack(directions, axis=1)
    centers = np.broadcast_to(
        np.asarray(centers, dtype=np.float64)[None],
        (17, len(subset), 3),
    )
    predictions = solve_ray_intersection(
        centers, directions,
        np.ones((17, len(subset)), dtype=np.float64),
    )
    errors = np.linalg.norm(predictions - targets, axis=-1) * 1000.0
    pixel_errors = np.asarray([
        np.mean([
            np.min(np.linalg.norm(
                modes_by_joint[joint][view] - gt2d[camera_index, joint], axis=-1
            ))
            for view, camera_index in enumerate(subset)
        ])
        for joint in range(17)
    ], dtype=np.float64)
    return errors, pixel_errors


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    means = []
    for action in sorted(set(int(item) for item in actions)):
        if action in ACTION_NAMES:
            means.append(float(np.mean(values[actions == action])))
    return float(np.mean(means))


def main() -> None:
    args = parse_args()
    if any(view not in (2, 3, 4) for view in args.views):
        raise ValueError("--views must be drawn from 2,3,4")
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_four_view_groups(records)
    with np.load(args.h76_cache) as cache_file:
        cache = {key: cache_file[key].copy() for key in cache_file.files}
    if len(groups) != len(cache["targets"]):
        raise ValueError(f"group/cache mismatch: {len(groups)} vs {len(cache['targets'])}")
    dense = DenseStore(args.dense_shards)
    limit = int(args.limit_groups) if args.limit_groups else len(groups)
    groups = groups[:limit]
    actions = cache["actions"][:limit].astype(np.int64)
    targets = cache["targets"][:limit].astype(np.float64)
    predictions = cache["predictions"][:limit].astype(np.float64)

    results = {}
    diagnostics = {}
    all_combos = {
        views: tuple(combo for combo in COMBINATIONS if len(combo) == views)
        for views in args.views
    }
    for radius in args.radii:
        radius_key = f"r{radius:g}"
        stores = {
            f"V{views}": {
                "baseline": [], "oracle_2d": [], "oracle_3d": [],
                "candidate_pixel_error": [], "baseline_per_joint": [],
                "oracle_2d_per_joint": [], "oracle_3d_per_joint": [],
            }
            for views in args.views
        }
        for group_index, group in enumerate(groups):
            group_records = [records[index] for index in group]
            gt2d = np.stack([h36m_2d(record) for record in group_records])
            # Cache the projected H76 predictions once per view/cardinality.
            heatmap_rows = [dense.get(index) for index in group]
            for views in args.views:
                for combo in all_combos[views]:
                    candidate_index = COMBINATION_INDEX[combo]
                    baseline = predictions[group_index, candidate_index]
                    baseline_error = np.linalg.norm(baseline - targets[group_index], axis=-1) * 1000.0
                    modes_by_joint = []
                    oracle_3d_joint = []
                    projected_by_view = [
                        project_world(baseline, group_records[camera_index])
                        for camera_index in combo
                    ]
                    for joint in range(17):
                        per_view = []
                        for local_view, camera_index in enumerate(combo):
                            row = heatmap_rows[camera_index]
                            modes, _ = extract_local_modes(
                                row["joint_maps"][joint], joint,
                                projected_by_view[local_view][joint],
                                row["input_center"], row["input_scale"],
                                row["input_size"], radius, args.topk,
                            )
                            per_view.append(modes)
                        modes_by_joint.append(per_view)
                        if args.oracle_mode == "both":
                            _, e3d, _ = oracle_for_joint(
                                per_view, group_records, combo,
                                targets[group_index, joint], gt2d[:, joint],
                                compute_3d=True,
                            )
                            oracle_3d_joint.append(e3d)
                    oracle_2d_joint, pixel_joint = oracle_2d_batch(
                        modes_by_joint, group_records, combo,
                        targets[group_index], gt2d,
                    )
                    stage = stores[f"V{views}"]
                    stage["baseline"].append(float(np.mean(baseline_error)))
                    stage["oracle_2d"].append(float(np.mean(oracle_2d_joint)))
                    if args.oracle_mode == "both":
                        stage["oracle_3d"].append(float(np.mean(oracle_3d_joint)))
                    stage["candidate_pixel_error"].append(float(np.mean(pixel_joint)))
                    stage["baseline_per_joint"].append(baseline_error)
                    stage["oracle_2d_per_joint"].append(oracle_2d_joint)
                    if args.oracle_mode == "both":
                        stage["oracle_3d_per_joint"].append(oracle_3d_joint)
            if (group_index + 1) % 100 == 0:
                print(f"radius={radius:g}: {group_index + 1}/{len(groups)} groups", flush=True)

        radius_result = {}
        for views in args.views:
            stage = stores[f"V{views}"]
            # Each group contributes one value per camera combination.  The
            # action-equal protocol must therefore repeat its group action for
            # every combination; otherwise the boolean mask has the wrong
            # length and silently biases or crashes the metric.
            stage_actions = np.repeat(actions, len(all_combos[views]))
            if len(stage_actions) != len(stage["baseline"]):
                raise RuntimeError(
                    f"V{views}: action/value length mismatch "
                    f"{len(stage_actions)} vs {len(stage['baseline'])}"
                )
            stage_result = {}
            methods = ["baseline", "oracle_2d", "candidate_pixel_error"]
            if args.oracle_mode == "both":
                methods.insert(2, "oracle_3d")
            for method in methods:
                values = np.asarray(stage[method], dtype=np.float64)
                stage_result[method] = {
                    "action_equal_mm": action_equal(values, stage_actions),
                    "frame_weighted_mm": float(values.mean()),
                }
            per_joint_methods = ["baseline_per_joint", "oracle_2d_per_joint"]
            if args.oracle_mode == "both":
                per_joint_methods.append("oracle_3d_per_joint")
            for method in per_joint_methods:
                values = np.asarray(stage[method], dtype=np.float64)
                stage_result[method] = {
                    "action_equal_mm": [
                        action_equal(values[:, joint], stage_actions)
                        for joint in range(17)
                    ],
                }
            radius_result[f"V{views}"] = stage_result
        results[radius_key] = radius_result

    payload = {
        "method": "RIGR P0 local HRNet heatmap oracle diagnostic",
        "paper_basis": [
            "Epipolar Transformer (CVPR 2020)",
            "MVGFormer (CVPR 2024)",
            "AdaFuse (IJCV 2021)",
        ],
        "input_pkl": str(Path(args.input_pkl).resolve()),
        "h76_cache": str(Path(args.h76_cache).resolve()),
        "dense_shards": [str(Path(path).resolve()) for path in args.dense_shards],
        "groups": len(groups), "topk": args.topk, "oracle_mode": args.oracle_mode,
        "radii_heatmap_px": args.radii,
        "gt_note": "GT is used only for oracle diagnostics; no model or inference path reads GT.",
        "results": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}", flush=True)
    for radius_key, radius_result in results.items():
        print(f"[{radius_key}]", flush=True)
        for stage, values in radius_result.items():
            print(
                stage,
                "H76={:.3f} 2D-oracle={:.3f} 3D-oracle={:.3f} px={:.3f}".format(
                values["baseline"]["action_equal_mm"],
                    values["oracle_2d"]["action_equal_mm"],
                    values.get("oracle_3d", {}).get("action_equal_mm", float("nan")),
                    values["candidate_pixel_error"]["action_equal_mm"],
                ), flush=True,
            )


if __name__ == "__main__":
    main()
