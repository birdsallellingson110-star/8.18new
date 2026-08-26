#!/usr/bin/env python3
"""Decompose the selected K96 model into translation/articulation/pair errors."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_limb_utility_20260820 as limb
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import train_bone_stats
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def summarize(chunks, action_chunks):
    values = np.concatenate(chunks)
    actions = np.concatenate(action_chunks)
    return {
        "action_equal_mm": action_equal(values, actions),
        "frame_weighted_mm": float(values.mean()),
    }


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = SimpleNamespace(**state["args"])
    scorer = dsac.PoseScoreMLP(
        saved.features, getattr(saved, "sigmoid_score", False)
    ).to(device)
    scorer.load_state_dict(state["state_dict"], strict=True)
    scorer.eval()
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    proposal_state = torch.load(
        args.proposal_checkpoint, map_location=device, weights_only=False
    )
    proposal = limb.LimbUtility(coord_mean, coord_std).to(device)
    proposal.load_state_dict(proposal_state["state_dict"], strict=True)
    proposal.eval()
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    arrays = trainer.load_arrays([args.cache], 22)
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    store = {
        combo: defaultdict(list) for combo in TASKS
    }
    actions = {combo: [] for combo in TASKS}
    torch.manual_seed(10000 + saved.seed)
    torch.cuda.manual_seed_all(10000 + saved.seed)
    with torch.inference_mode():
        for predictions, targets, rays, action in loader:
            predictions, targets, rays = (
                predictions.to(device), targets.to(device), rays.to(device)
            )
            for combo in TASKS:
                scores, weights, hypotheses, pose, _, _, _ = dsac.forward_task(
                    scorer, e2, predictions, targets, rays, combo, saved,
                    coord_mean, coord_std, bone_mean, bone_std, proposal,
                )
                absolute = torch.linalg.vector_norm(pose - targets, dim=-1)
                pose_rr = pose - pose[:, :1]
                target_rr = targets - targets[:, :1]
                relative = torch.linalg.vector_norm(pose_rr - target_rr, dim=-1)
                root = torch.linalg.vector_norm(pose[:, 0] - targets[:, 0], dim=-1)
                direction = torch.nn.functional.normalize(
                    rays[..., list(combo), :3], dim=-1
                )
                point = rays[..., list(combo), 3:6]
                for name, xyz in (("prediction_ray", pose), ("target_ray", targets)):
                    offset = xyz[:, :, None] - point
                    distance = torch.linalg.vector_norm(
                        torch.cross(offset, direction, dim=-1), dim=-1
                    ).mean(dim=(1, 2))
                    store[combo][name].append(distance.cpu().numpy() * 1000.0)
                store[combo]["absolute"].append(absolute.mean(dim=1).cpu().numpy() * 1000.0)
                store[combo]["root"].append(root.cpu().numpy() * 1000.0)
                store[combo]["root_relative_all17"].append(
                    relative.mean(dim=1).cpu().numpy() * 1000.0
                )
                store[combo]["root_relative_nonroot"].append(
                    relative[:, 1:].mean(dim=1).cpu().numpy() * 1000.0
                )
                store[combo]["per_joint_absolute"].append(absolute.cpu().numpy() * 1000.0)
                store[combo]["per_joint_relative"].append(relative.cpu().numpy() * 1000.0)
                actions[combo].append(action.numpy().copy())
    result = {"by_combo": {}, "by_stage": {}}
    for combo in TASKS:
        key = "-".join(str(view + 1) for view in combo)
        result["by_combo"][key] = {
            name: summarize(chunks, actions[combo])
            for name, chunks in store[combo].items()
            if not name.startswith("per_joint")
        }
    for count in (2, 3, 4):
        combos = [combo for combo in TASKS if len(combo) == count]
        stage_actions = [chunk for combo in combos for chunk in actions[combo]]
        stage = {}
        for name in ("absolute", "root", "root_relative_all17",
                     "root_relative_nonroot", "prediction_ray", "target_ray"):
            chunks = [chunk for combo in combos for chunk in store[combo][name]]
            stage[name] = summarize(chunks, stage_actions)
        per_joint = np.concatenate([
            chunk for combo in combos for chunk in store[combo]["per_joint_absolute"]
        ])
        per_joint_rr = np.concatenate([
            chunk for combo in combos for chunk in store[combo]["per_joint_relative"]
        ])
        action_array = np.concatenate(stage_actions)
        stage["per_joint_absolute_action_equal_mm"] = [
            action_equal(per_joint[:, joint], action_array) for joint in range(17)
        ]
        stage["per_joint_relative_action_equal_mm"] = [
            action_equal(per_joint_rr[:, joint], action_array) for joint in range(17)
        ]
        result["by_stage"][f"V{count}"] = stage
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
