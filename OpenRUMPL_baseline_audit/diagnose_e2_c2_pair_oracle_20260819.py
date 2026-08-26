#!/usr/bin/env python3
"""Per-camera-combo E2-C2 oracle vs model diagnostic on clean H36M.

This is not a paper number and does not train.  It only splits the frozen
E2-C2 22-candidate cache and the calibrated soft-fusion scorer by camera
combination, so we can tell whether 1-4 / 2-3 still have no usable 3D in the
pool (need a new candidate family) or have a low oracle that the scorer misses
(need a scoring change).  GT is used only to report upper bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_v234_universal_20260812 as trainer
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES
from train_h76_set_transformer_utility_20260811 import (
    ArrayDataset,
    SetTransformerJointUtility,
)

CAMERA_LABEL = {0: "1", 1: "2", 2: "3", 3: "4"}
TEMPERATURES = {"V2": 0.4, "V3": 1.8, "V4": 1.8}


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/validation_c2_22c.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_training_protocol_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819/"
            "e2_c2_pair_oracle"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="1")
    return parser.parse_args()


def combo_name(combo: tuple[int, ...]) -> str:
    return "-".join(CAMERA_LABEL[index] for index in combo)


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def summarize_error(error_mm: np.ndarray, actions: np.ndarray) -> dict:
    frame = error_mm.mean(axis=-1)
    return {
        "action_equal_all17_mm": action_equal(frame, actions),
        "frame_weighted_all17_mm": float(frame.mean()),
        "per_joint_mm": {
            name: action_equal(error_mm[:, index], actions)
            for index, name in enumerate(JOINT_NAMES)
        },
    }


def closest_point_on_ray(points: np.ndarray, directions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    offset = targets - points
    depth = np.einsum("...d,...d->...", offset, directions)
    return points + depth[..., None] * directions


def ray_geometry(rays: np.ndarray, combo: tuple[int, ...], targets: np.ndarray) -> dict:
    """GT-only geometry bounds: how close GT sits to the observed rays."""
    subset = list(combo)
    directions = rays[:, :, subset, :3].astype(np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-8)
    points = rays[:, :, subset, 3:6].astype(np.float64)
    closest = closest_point_on_ray(points, directions, targets[:, :, None, :])
    lateral_mm = np.linalg.norm(closest - targets[:, :, None, :], axis=-1) * 1000.0
    best_view = lateral_mm.min(axis=-1)
    mean_view = lateral_mm.mean(axis=-1)
    # Angle between the two primary task cameras, if the task has at least two.
    if len(subset) >= 2:
        d0 = directions[:, :, 0]
        d1 = directions[:, :, 1]
        cosine = np.einsum("njd,njd->nj", d0, d1).clip(-1.0, 1.0)
        angle_deg = np.degrees(np.arccos(np.abs(cosine)))
    else:
        angle_deg = np.full(best_view.shape, np.nan)
    return {
        "best_view_lateral_mm": best_view.astype(np.float32),
        "mean_view_lateral_mm": mean_view.astype(np.float32),
        "ray_angle_deg": angle_deg.astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.cache], 22)
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    checkpoint = Path(args.checkpoint_root) / f"seed{args.seed}" / "model_best.pth.tar"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SetTransformerJointUtility(
        state["mean"], state["std"], state["attention_depth"],
        stage_heads=state.get("stage_heads", False),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    stores = {
        combo: {
            "baseline": [],
            "hard": [],
            "soft": [],
            "oracle": [],
            "hard_hits_oracle": [],
            "oracle_from_h76": [],
            "n_candidates": None,
        }
        for combo in trainer.TASK_COMBINATIONS
    }
    action_chunks = []

    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            action_chunks.append(actions.numpy().copy())
            for task_combo in trainer.TASK_COMBINATIONS:
                predicted, _, true_error, candidates, baseline_local = trainer.predict_task(
                    model, predictions, targets, rays, task_combo
                )
                stage = f"V{len(task_combo)}"
                hard_index = predicted.argmin(dim=-1)
                oracle_index = true_error.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                weights = F.softmax(-predicted / TEMPERATURES[stage], dim=-1)
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                baseline = candidates[:, baseline_local]
                oracle = true_error.min(dim=-1).values
                n_cand = candidates.shape[1]
                h76_count = n_cand // 2
                stores[task_combo]["n_candidates"] = int(n_cand)
                stores[task_combo]["baseline"].append(
                    torch.linalg.vector_norm(baseline - targets, dim=-1).cpu().numpy() * 1000.0
                )
                stores[task_combo]["hard"].append(
                    torch.linalg.vector_norm(hard - targets, dim=-1).cpu().numpy() * 1000.0
                )
                stores[task_combo]["soft"].append(
                    torch.linalg.vector_norm(soft - targets, dim=-1).cpu().numpy() * 1000.0
                )
                stores[task_combo]["oracle"].append(oracle.cpu().numpy() * 1000.0)
                stores[task_combo]["hard_hits_oracle"].append(
                    (hard_index == oracle_index).cpu().numpy()
                )
                stores[task_combo]["oracle_from_h76"].append(
                    (oracle_index < h76_count).cpu().numpy()
                )

    actions = np.concatenate(action_chunks)
    rays_np = arrays["rays"].astype(np.float64)
    targets_np = arrays["targets"].astype(np.float64)
    combo_rows = []
    for task_combo in trainer.TASK_COMBINATIONS:
        packed = {
            mode: np.concatenate(stores[task_combo][mode], axis=0)
            for mode in ("baseline", "hard", "soft", "oracle")
        }
        hits = np.concatenate(stores[task_combo]["hard_hits_oracle"], axis=0)
        from_h76 = np.concatenate(stores[task_combo]["oracle_from_h76"], axis=0)
        geometry = ray_geometry(rays_np, task_combo, targets_np)
        row = {
            "combo": combo_name(task_combo),
            "views": int(len(task_combo)),
            "n_candidates": stores[task_combo]["n_candidates"],
            "baseline": summarize_error(packed["baseline"], actions),
            "hard": summarize_error(packed["hard"], actions),
            "soft": summarize_error(packed["soft"], actions),
            "oracle": summarize_error(packed["oracle"], actions),
            "soft_minus_oracle_mm": (
                summarize_error(packed["soft"], actions)["action_equal_all17_mm"]
                - summarize_error(packed["oracle"], actions)["action_equal_all17_mm"]
            ),
            "hard_oracle_hit_rate": float(hits.mean()),
            "oracle_from_h76_rate": float(from_h76.mean()),
            "best_view_lateral_mm": summarize_error(
                geometry["best_view_lateral_mm"], actions
            ),
            "mean_view_lateral_mm": summarize_error(
                geometry["mean_view_lateral_mm"], actions
            ),
            "mean_abs_ray_angle_deg": float(np.nanmean(geometry["ray_angle_deg"])),
        }
        combo_rows.append(row)

    def stage_mean(rows, views, key):
        selected = [row for row in rows if row["views"] == views]
        return float(np.mean([row[key]["action_equal_all17_mm"] for row in selected]))

    headline = {
        stage: {
            "baseline": stage_mean(combo_rows, views, "baseline"),
            "soft": stage_mean(combo_rows, views, "soft"),
            "hard": stage_mean(combo_rows, views, "hard"),
            "oracle": stage_mean(combo_rows, views, "oracle"),
            "best_view_lateral": stage_mean(combo_rows, views, "best_view_lateral_mm"),
        }
        for stage, views in (("V2", 2), ("V3", 3), ("V4", 4))
    }
    v2_rows = [row for row in combo_rows if row["views"] == 2]
    degenerate = [row for row in v2_rows if row["combo"] in {"1-4", "2-3"}]
    healthy = [row for row in v2_rows if row["combo"] not in {"1-4", "2-3"}]

    def group_mean(rows, key):
        return float(np.mean([row[key]["action_equal_all17_mm"] for row in rows]))

    decision = {
        "degenerate_pairs": [row["combo"] for row in degenerate],
        "degenerate_baseline_mm": group_mean(degenerate, "baseline"),
        "degenerate_soft_mm": group_mean(degenerate, "soft"),
        "degenerate_oracle_mm": group_mean(degenerate, "oracle"),
        "degenerate_best_view_lateral_mm": group_mean(degenerate, "best_view_lateral_mm"),
        "healthy_baseline_mm": group_mean(healthy, "baseline"),
        "healthy_soft_mm": group_mean(healthy, "soft"),
        "healthy_oracle_mm": group_mean(healthy, "oracle"),
        "healthy_best_view_lateral_mm": group_mean(healthy, "best_view_lateral_mm"),
    }
    # If the bad-pair oracle is still near the model, the pool has no 3D to pick.
    # If GT still lies close to at least one ray, a depth/lifting candidate can help.
    degenerate_oracle = decision["degenerate_oracle_mm"]
    degenerate_soft = decision["degenerate_soft_mm"]
    lateral = decision["degenerate_best_view_lateral_mm"]
    if degenerate_oracle > 50.0:
        if lateral < 25.0:
            next_step = (
                "pool_empty_but_rays_ok: add a depth/lifting/bone-length candidate "
                "that does not triangulate the degenerate pair; do not change the scorer"
            )
        else:
            next_step = (
                "pool_empty_and_2d_bad: 2D rays already miss GT; a new 3D scorer "
                "cannot recover these pairs from the current detections"
            )
    elif (degenerate_soft - degenerate_oracle) > 3.0:
        next_step = (
            "oracle_has_headroom: keep the same 22 candidates and change scoring/"
            "loss on the degenerate pairs"
        )
    else:
        next_step = (
            "little_headroom: E2 already extracts the 22-candidate pool on these "
            "pairs; need a new candidate family or a different spatial model"
        )
    decision["next_step"] = next_step

    payload = {
        "method": "E2-C2 per-camera-combo oracle diagnostic",
        "protocol": "clean H36M S9/S11, action-equal All-17 absolute MPJPE, no occlusion",
        "cache": str(Path(args.cache).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "temperatures": TEMPERATURES,
        "seed": args.seed,
        "headline": headline,
        "decision": decision,
        "combos": combo_rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "headline": headline,
        "decision": decision,
        "v2_combos": [
            {
                "combo": row["combo"],
                "baseline": row["baseline"]["action_equal_all17_mm"],
                "soft": row["soft"]["action_equal_all17_mm"],
                "oracle": row["oracle"]["action_equal_all17_mm"],
                "best_view_lateral": row["best_view_lateral_mm"]["action_equal_all17_mm"],
                "ray_angle_deg": row["mean_abs_ray_angle_deg"],
            }
            for row in v2_rows
        ],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
