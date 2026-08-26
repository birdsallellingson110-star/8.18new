#!/usr/bin/env python3
"""Zero-training PointDSC-style compatibility test for limb candidate labels.

The input protocol is unchanged: frozen HRNet coordinates/confidence, camera
parameters, derived rays and RUMPL/E2 3D candidates only.  We treat the five
kinematic-part candidate labels as correspondence nodes.  Pairwise spatial
compatibility is the boundary-bone likelihood, and exact sum-product on the
torso-centred star graph replaces independent group softmax.

This deliberately tests the hand-defined compatibility signal before adding
PointDSC's non-local encoder.  Lambda is selected only on the train-subject
internal holdout; S9/S11 must be evaluated once afterwards.
"""
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
# torso is group 0.  Each tuple is (limb_group, torso_joint, limb_joint).
BOUNDARIES = ((1, 0, 1), (2, 0, 4), (3, 8, 11), (4, 8, 14))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1))
    parser.add_argument("--group-modulo", nargs=2, type=int, default=(10, 0))
    parser.add_argument("--no-group-filter", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def beliefs(cost, candidates, temperature, pair_weight, bone_mean, bone_std):
    """Exact sum-product marginals on torso -> four-limb star."""
    phi = torch.softmax(-cost / temperature, dim=-1)  # B,G,C
    messages = []
    compatibilities = []
    for _, torso_joint, limb_joint in BOUNDARIES:
        torso_xyz = candidates[:, :, torso_joint]
        limb_xyz = candidates[:, :, limb_joint]
        length = torch.linalg.vector_norm(
            torso_xyz[:, :, None] - limb_xyz[:, None, :], dim=-1
        )
        z2 = ((length - bone_mean[limb_joint]) / bone_std[limb_joint]).square()
        compatibility = torch.exp(-pair_weight * z2).clamp_min(1e-12)
        compatibilities.append(compatibility)
    for (limb_group, _, _), compatibility in zip(BOUNDARIES, compatibilities):
        messages.append(torch.einsum("bd,bcd->bc", phi[:, limb_group], compatibility))
    root_unnormalized = phi[:, 0].clone()
    for message in messages:
        root_unnormalized = root_unnormalized * message
    root = root_unnormalized / root_unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    output = [root]
    for index, ((limb_group, _, _), compatibility) in enumerate(
        zip(BOUNDARIES, compatibilities)
    ):
        cavity = phi[:, 0].clone()
        for other, message in enumerate(messages):
            if other != index:
                cavity = cavity * message
        root_to_leaf = torch.einsum("bc,bcd->bd", cavity, compatibility)
        marginal = phi[:, limb_group] * root_to_leaf
        marginal = marginal / marginal.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        output.append(marginal)
    return torch.stack(output, dim=1)


def fuse(group_weight, candidates, baseline_local):
    pose = candidates[:, baseline_local].clone()
    for group_id, joints in enumerate(limb.GROUP_LIST):
        idx = torch.as_tensor(joints, device=candidates.device)
        pose[:, idx] = torch.einsum(
            "bc,bcjd->bjd", group_weight[:, group_id], candidates[:, :, idx]
        )
    return pose


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = SimpleNamespace(**state["args"])
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    model = limb.LimbUtility(coord_mean, coord_std).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    arrays = trainer.load_arrays([args.cache], 22)
    if args.no_group_filter:
        selected = np.arange(len(arrays["targets"]))
    else:
        divisor, remainder = args.group_modulo
        selected = np.flatnonzero(arrays["group_indices"] % divisor == remainder)
    if args.max_examples:
        selected = selected[: args.max_examples]
    arrays = {key: value[selected] for key, value in arrays.items()}
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    store = {
        f"V{x}": {weight: [] for weight in args.lambdas} for x in (2, 3, 4)
    }
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions, targets, rays = predictions.to(device), targets.to(device), rays.to(device)
            for combo in TASKS:
                stage = f"V{len(combo)}"
                unary, _, _, candidates, baseline_local = extra.predict_task(
                    e2, predictions, targets, rays, combo
                )
                cost = model(candidates, unary, rays, combo, bone_mean, bone_std)
                temperature = saved.v2_temperature if len(combo) == 2 else saved.v34_temperature
                for pair_weight in args.lambdas:
                    marginal = beliefs(
                        cost, candidates, temperature, pair_weight, bone_mean, bone_std
                    )
                    pose = fuse(marginal, candidates, baseline_local)
                    error = torch.linalg.vector_norm(pose - targets, dim=-1)
                    store[stage][pair_weight].append(error.cpu().numpy() * 1000.0)
                action_store[stage].append(actions.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        stage_actions = np.concatenate(action_store[stage])
        result[stage] = {}
        for pair_weight, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][str(pair_weight)] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    rows = []
    for pair_weight in args.lambdas:
        values = {
            stage: result[stage][str(pair_weight)]["action_equal_all17_mm"]
            for stage in ("V2", "V3", "V4")
        }
        rows.append({"pair_weight": pair_weight,
                     "headline_mm": float(np.mean(list(values.values()))),
                     "weighted_mm": values})
    rows.sort(key=lambda row: row["headline_mm"])
    payload = {
        "method": "PointDSC-style spatial compatibility, exact star sum-product",
        "input_protocol": "HRNet coordinates/confidence + cameras + derived rays/candidates only",
        "source": "reference/PointDSC-official/models/PointDSC.py",
        "num_examples": len(arrays["targets"]),
        "rows": rows,
        "best": rows[0],
        "result": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
