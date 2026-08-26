#!/usr/bin/env python3
"""Diagnose whether the expanded E2 candidate oracle is structurally usable.

This script keeps the frozen candidate generator and trained E2 unary scorer.
It reports three progressively less constrained hindsight bounds:

* pose oracle: one candidate label for the complete skeleton;
* limb oracle: one label for each of five connected body groups;
* joint oracle: an independent label for every joint (historical oracle).

It also evaluates a GT-free exact min-sum inference baseline on the H36M
skeleton tree.  Its unary term is the E2 score and its pairwise term penalizes
candidate-label combinations whose bone length is implausible under the train
set.  Lambda selection must be done on an internal holdout, not on S9/S11;
the grid here is therefore a diagnostic, and the output labels every entry as
such.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


PARENTS = extra.PARENTS.tolist()
GROUPS = {
    "torso_head": (0, 7, 8, 9, 10),
    "right_leg": (1, 2, 3),
    "left_leg": (4, 5, 6),
    "left_arm": (11, 12, 13),
    "right_arm": (14, 15, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
    )
    parser.add_argument(
        "--unary-temperatures", nargs="+", type=float,
        default=(0.4, 0.8, 1.8),
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--group-modulo", nargs=2, type=int, metavar=("DIVISOR", "REMAINDER"),
        help="Evaluate only group_indices %% DIVISOR == REMAINDER (for train holdout tuning).",
    )
    return parser.parse_args()


def train_bone_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    targets = np.load(path, allow_pickle=False)["targets"].astype(np.float64)
    parents = np.asarray(PARENTS)
    lengths = np.linalg.norm(targets - targets[:, parents], axis=-1)
    mean = lengths.mean(axis=0)
    std = lengths.std(axis=0)
    mean[0] = 0.0
    # Inter-subject bone variation is real; a 5 mm floor avoids a nearly hard
    # constraint for low-variance joints while retaining physical scale.
    std = np.maximum(std, 0.005)
    std[0] = 1.0
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def children_of_tree() -> list[list[int]]:
    children = [[] for _ in range(17)]
    for child in range(1, 17):
        children[PARENTS[child]].append(child)
    return children


def tree_map(
    candidates: torch.Tensor,
    unary: torch.Tensor,
    bone_mean: torch.Tensor,
    bone_std: torch.Tensor,
    pair_weight: float,
) -> torch.Tensor:
    """Exact hard MAP on the 17-joint tree.

    Args:
        candidates: B,C,J,3
        unary: B,J,C, lower is better
    Returns:
        B,J,3 pose assembled from the MAP candidate labels.
    """
    batch, count, joints, _ = candidates.shape
    children = children_of_tree()
    dp: list[torch.Tensor | None] = [None] * joints
    back: dict[tuple[int, int], torch.Tensor] = {}

    def visit(joint: int) -> torch.Tensor:
        score = unary[:, joint].clone()
        for child in children[joint]:
            child_score = visit(child)
            parent_xyz = candidates[:, :, joint, :]
            child_xyz = candidates[:, :, child, :]
            # B,parent_label,child_label
            length = torch.linalg.vector_norm(
                parent_xyz[:, :, None, :] - child_xyz[:, None, :, :], dim=-1
            )
            z2 = ((length - bone_mean[child]) / bone_std[child]).square()
            transition = child_score[:, None, :] + pair_weight * z2
            best, index = transition.min(dim=-1)
            score = score + best
            back[(joint, child)] = index
        dp[joint] = score
        return score

    root_score = visit(0)
    labels = torch.empty(batch, joints, dtype=torch.long, device=candidates.device)
    labels[:, 0] = root_score.argmin(dim=-1)

    def decode(parent: int) -> None:
        for child in children[parent]:
            table = back[(parent, child)]
            labels[:, child] = table.gather(1, labels[:, parent, None]).squeeze(1)
            decode(child)

    decode(0)
    pose = candidates.permute(0, 2, 1, 3).gather(
        2, labels[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)
    return pose


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def append_error(store, key, pose, target) -> None:
    error = torch.linalg.vector_norm(pose - target, dim=-1) * 1000.0
    store[key].append(error.detach().cpu().numpy())


def main() -> None:
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.cache], 22)
    if args.group_modulo is not None:
        divisor, remainder = args.group_modulo
        keep = arrays["group_indices"] % divisor == remainder
        arrays = {key: value[keep] for key, value in arrays.items()}
        if not len(arrays["targets"]):
            raise ValueError("--group-modulo selected no examples")
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SetTransformerJointUtility(
        state["mean"], state["std"], state["attention_depth"],
        stage_heads=state.get("stage_heads", False),
        neutralize_subset_penalty=state.get("neutralize_subset_penalty", True),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean = bone_mean.to(device)
    bone_std = bone_std.to(device)

    task_combos = tuple(
        combo for count in (2, 3, 4)
        for combo in itertools.combinations(range(4), count)
    )
    stores = {f"V{count}": defaultdict(list) for count in (2, 3, 4)}
    stage_actions = {f"V{count}": [] for count in (2, 3, 4)}
    with torch.inference_mode():
        for batch_index, (predictions, targets, rays, actions) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            for combo in task_combos:
                stage = f"V{len(combo)}"
                predicted, _, true_error, candidates, baseline_local = extra.predict_task(
                    model, predictions, targets, rays, combo
                )
                append_error(stores[stage], "baseline", candidates[:, baseline_local], targets)

                pose_index = true_error.mean(dim=1).argmin(dim=-1)
                pose_oracle = candidates.gather(
                    1, pose_index[:, None, None, None].expand(-1, 1, 17, 3)
                ).squeeze(1)
                append_error(stores[stage], "pose_oracle", pose_oracle, targets)

                limb_oracle = candidates[:, baseline_local].clone()
                for joints in GROUPS.values():
                    indices = torch.as_tensor(joints, device=device)
                    group_index = true_error[:, indices].mean(dim=1).argmin(dim=-1)
                    chosen = candidates.gather(
                        1, group_index[:, None, None, None].expand(-1, 1, 17, 3)
                    ).squeeze(1)
                    limb_oracle[:, indices] = chosen[:, indices]
                append_error(stores[stage], "limb_oracle", limb_oracle, targets)

                joint_index = true_error.argmin(dim=-1)
                joint_oracle = candidates.permute(0, 2, 1, 3).gather(
                    2, joint_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                append_error(stores[stage], "joint_oracle", joint_oracle, targets)

                for temperature in args.unary_temperatures:
                    unary = predicted / temperature
                    soft = torch.einsum(
                        "bjc,bcjd->bjd", F.softmax(-unary, dim=-1), candidates
                    )
                    append_error(stores[stage], f"soft_T{temperature:g}", soft, targets)
                    # Part-consistent E2 control: all joints in a kinematic
                    # group share one candidate distribution.  This directly
                    # tests the large limb-oracle gap without training a new
                    # scorer or using GT at inference.
                    limb_soft = candidates[:, baseline_local].clone()
                    limb_hard = candidates[:, baseline_local].clone()
                    for joints_in_group in GROUPS.values():
                        group_index = torch.as_tensor(joints_in_group, device=device)
                        group_unary = unary[:, group_index].mean(dim=1)
                        group_weight = F.softmax(-group_unary, dim=-1)
                        group_candidates = candidates[:, :, group_index]
                        limb_soft[:, group_index] = torch.einsum(
                            "bc,bcjd->bjd", group_weight, group_candidates
                        )
                        hard_index = group_unary.argmin(dim=-1)
                        chosen = candidates.gather(
                            1, hard_index[:, None, None, None].expand(-1, 1, 17, 3)
                        ).squeeze(1)
                        limb_hard[:, group_index] = chosen[:, group_index]
                    append_error(
                        stores[stage], f"limb_soft_T{temperature:g}", limb_soft, targets
                    )
                    append_error(
                        stores[stage], f"limb_hard_T{temperature:g}", limb_hard, targets
                    )
                    for pair_weight in args.lambdas:
                        pose = tree_map(
                            candidates, unary, bone_mean, bone_std, pair_weight
                        )
                        append_error(
                            stores[stage],
                            f"tree_T{temperature:g}_L{pair_weight:g}", pose, targets,
                        )
                stage_actions[stage].append(actions.numpy().copy())

    result = {}
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(stage_actions[stage])
        result[stage] = {}
        for key, chunks in stores[stage].items():
            values = np.concatenate(chunks, axis=0)
            result[stage][key] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    payload = {
        "method": "expanded E2 candidates: structural oracle and exact skeleton-tree MAP",
        "protocol": "H36M S9/S11, all camera subsets, action-equal All-17",
        "warning": "lambda/temperature grid on test is diagnostic only; select final hyperparameters on train holdout",
        "group_modulo": args.group_modulo,
        "groups": {key: list(value) for key, value in GROUPS.items()},
        "bone_mean_m": bone_mean.cpu().tolist(),
        "bone_std_m": bone_std.cpu().tolist(),
        "result": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
