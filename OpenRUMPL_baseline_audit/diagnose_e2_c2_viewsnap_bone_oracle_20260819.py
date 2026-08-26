#!/usr/bin/env python3
"""Zero-training oracle for view-snap and bone-ray candidates on E2-C2.

The 22-candidate V2 pool only contains the same two-view triangulation twice
(H76 and confidence).  Degenerate pairs 1-4 / 2-3 still sit at ~45 mm oracle
while GT is only ~32 mm from the nearer ray.  This diagnostic adds, for each
task and without extra cameras:

1. snap the task H76 baseline onto each task view's ray (AdaFuse-style trust
   one observation, H76 provides depth);
2. reconstruct the skeleton along each task view's ray with train-set mean
   bone lengths (anatomy-aware fallback when triangulation is ill-posed).

GT is not used to generate candidates.  No model is trained.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES

ORIGINAL = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
CAMERA_LABEL = {0: "1", 1: "2", 2: "3", 3: "4"}
PARENTS = np.asarray([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int64)
CHILD_ORDER = tuple(index for index in range(17) if index != 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/train_c2_22c.npz"
        ),
    )
    parser.add_argument(
        "--validation-cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/validation_c2_22c.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819/"
            "e2_c2_viewsnap_bone_oracle"
        ),
    )
    return parser.parse_args()


def combo_name(combo: tuple[int, ...]) -> str:
    return "-".join(CAMERA_LABEL[index] for index in combo)


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    frame = values if values.ndim == 1 else values.mean(axis=-1)
    return float(np.mean([
        frame[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def unit(directions: np.ndarray) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-8)


def snap_pose_to_view(pose: np.ndarray, rays: np.ndarray, view: int) -> np.ndarray:
    direction = unit(rays[:, :, view, :3])
    point = rays[:, :, view, 3:6]
    depth = np.einsum("njd,njd->nj", pose - point, direction)
    return point + depth[..., None] * direction


def bone_pose_on_view(
    pose_prior: np.ndarray,
    rays: np.ndarray,
    view: int,
    bone_lengths: np.ndarray,
) -> np.ndarray:
    direction = unit(rays[:, :, view, :3])
    point = rays[:, :, view, 3:6]
    prior_depth = np.einsum("njd,njd->nj", pose_prior - point, direction)
    out = np.empty_like(pose_prior)
    out[:, 0] = point[:, 0] + prior_depth[:, 0, None] * direction[:, 0]
    for joint in CHILD_ORDER:
        parent = out[:, PARENTS[joint]]
        rel = point[:, joint] - parent
        vector = direction[:, joint]
        b = 2.0 * np.einsum("nd,nd->n", vector, rel)
        c = np.einsum("nd,nd->n", rel, rel) - bone_lengths[joint] ** 2
        disc = b * b - 4.0 * c
        sqrt = np.sqrt(np.clip(disc, 0.0, None))
        t_plus = (-b + sqrt) * 0.5
        t_minus = (-b - sqrt) * 0.5
        use_plus = np.abs(t_plus - prior_depth[:, joint]) <= np.abs(t_minus - prior_depth[:, joint])
        depth = np.where(use_plus, t_plus, t_minus)
        depth = np.where(disc >= 0.0, depth, prior_depth[:, joint])
        out[:, joint] = point[:, joint] + depth[:, None] * vector
    return out


def task_base_oracle(predictions: np.ndarray, targets: np.ndarray, combo: tuple[int, ...]) -> np.ndarray:
    available = [
        index for index, candidate in enumerate(list(ORIGINAL) + list(ORIGINAL))
        if set(candidate).issubset(combo)
    ]
    error = np.linalg.norm(predictions[:, available] - targets[:, None], axis=-1)
    return error.min(axis=1)


def extra_candidates(
    predictions: np.ndarray,
    rays: np.ndarray,
    combo: tuple[int, ...],
    bone_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline = predictions[:, ORIGINAL.index(combo)]
    snaps = np.stack([snap_pose_to_view(baseline, rays, view) for view in combo], axis=1)
    bones = np.stack(
        [bone_pose_on_view(baseline, rays, view, bone_lengths) for view in combo],
        axis=1,
    )
    return {"snap": snaps, "bone": bones}


def summarize(error_m: np.ndarray, actions: np.ndarray) -> dict:
    error_mm = error_m * 1000.0
    return {
        "action_equal_all17_mm": action_equal(error_mm, actions),
        "frame_weighted_all17_mm": float(error_mm.mean()),
        "per_joint_mm": {
            name: action_equal(error_mm[:, index], actions)
            for index, name in enumerate(JOINT_NAMES)
        },
    }


def main() -> None:
    args = parse_args()
    train = np.load(args.train_cache)
    val = np.load(args.validation_cache)
    train_targets = train["targets"].astype(np.float64)
    bone_lengths = np.linalg.norm(
        train_targets - train_targets[:, PARENTS], axis=-1
    ).mean(axis=0)
    bone_lengths[0] = 0.0

    predictions = val["predictions"].astype(np.float64)
    targets = val["targets"].astype(np.float64)
    rays = val["rays"].astype(np.float64)
    actions = val["actions"]

    rows = []
    for combo in ORIGINAL:
        extras = extra_candidates(predictions, rays, combo, bone_lengths)
        base_err = np.linalg.norm(
            predictions[:, ORIGINAL.index(combo)] - targets, axis=-1
        )
        pool_err = task_base_oracle(predictions, targets, combo)
        snap_err = np.linalg.norm(extras["snap"] - targets[:, None], axis=-1).min(axis=1)
        bone_err = np.linalg.norm(extras["bone"] - targets[:, None], axis=-1).min(axis=1)
        union = np.minimum(np.minimum(pool_err, snap_err), bone_err)
        snap_union = np.minimum(pool_err, snap_err)
        bone_union = np.minimum(pool_err, bone_err)
        rows.append({
            "combo": combo_name(combo),
            "views": int(len(combo)),
            "baseline": summarize(base_err, actions),
            "pool22_oracle": summarize(pool_err, actions),
            "snap_only_oracle": summarize(snap_err, actions),
            "bone_only_oracle": summarize(bone_err, actions),
            "pool_plus_snap": summarize(snap_union, actions),
            "pool_plus_bone": summarize(bone_union, actions),
            "pool_plus_snap_bone": summarize(union, actions),
        })

    def mean_rows(view_count: int, key: str) -> float:
        return float(np.mean([
            row[key]["action_equal_all17_mm"]
            for row in rows if row["views"] == view_count
        ]))

    headline = {
        stage: {
            key: mean_rows(views, key)
            for key in (
                "baseline", "pool22_oracle", "snap_only_oracle", "bone_only_oracle",
                "pool_plus_snap", "pool_plus_bone", "pool_plus_snap_bone",
            )
        }
        for stage, views in (("V2", 2), ("V3", 3), ("V4", 4))
    }
    degenerate = [row for row in rows if row["combo"] in {"1-4", "2-3"}]
    healthy = [row for row in rows if row["views"] == 2 and row["combo"] not in {"1-4", "2-3"}]

    def group(rows_, key):
        return float(np.mean([row[key]["action_equal_all17_mm"] for row in rows_]))

    decision = {
        "degenerate_pool22_oracle_mm": group(degenerate, "pool22_oracle"),
        "degenerate_plus_snap_mm": group(degenerate, "pool_plus_snap"),
        "degenerate_plus_bone_mm": group(degenerate, "pool_plus_bone"),
        "degenerate_plus_both_mm": group(degenerate, "pool_plus_snap_bone"),
        "healthy_pool22_oracle_mm": group(healthy, "pool22_oracle"),
        "healthy_plus_both_mm": group(healthy, "pool_plus_snap_bone"),
        "v2_pool22_oracle_mm": headline["V2"]["pool22_oracle"],
        "v2_plus_both_oracle_mm": headline["V2"]["pool_plus_snap_bone"],
    }
    drop = (
        decision["degenerate_pool22_oracle_mm"] - decision["degenerate_plus_both_mm"]
    )
    if drop >= 2.0 and headline["V2"]["pool_plus_snap_bone"] <= headline["V2"]["pool22_oracle"] - 0.5:
        next_step = "train_e2_with_viewsnap_bone: expanded oracle is large enough on degenerate pairs"
    elif drop >= 0.5:
        next_step = "screen_e2_seed0_only: modest oracle drop, one-seed training screen"
    else:
        next_step = "stop: new candidates do not lower the degenerate-pair oracle enough"
    decision["degenerate_oracle_drop_mm"] = drop
    decision["next_step"] = next_step

    payload = {
        "method": "zero-training view-snap + bone-ray oracle on frozen E2-C2 cache",
        "protocol": "clean H36M S9/S11, action-equal All-17, no extra cameras, no occlusion",
        "train_cache": str(Path(args.train_cache).resolve()),
        "validation_cache": str(Path(args.validation_cache).resolve()),
        "mean_train_bone_lengths_m": bone_lengths.tolist(),
        "headline": headline,
        "decision": decision,
        "combos": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "headline": headline,
        "decision": decision,
        "v2": [
            {
                "combo": row["combo"],
                "baseline": row["baseline"]["action_equal_all17_mm"],
                "pool22": row["pool22_oracle"]["action_equal_all17_mm"],
                "plus_snap": row["pool_plus_snap"]["action_equal_all17_mm"],
                "plus_bone": row["pool_plus_bone"]["action_equal_all17_mm"],
                "plus_both": row["pool_plus_snap_bone"]["action_equal_all17_mm"],
            }
            for row in rows if row["views"] == 2
        ],
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
