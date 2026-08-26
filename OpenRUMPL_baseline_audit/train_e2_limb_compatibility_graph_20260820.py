#!/usr/bin/env python3
"""Learn a PointDSC-style compatibility graph over frozen limb candidates.

The observation protocol is deliberately unchanged: frozen HRNet keypoint
coordinates/confidences, cameras, and deterministic ray-derived 3D candidates.
The established E2 and limb-unary networks are frozen.  A shared pairwise MLP
scores torso/limb candidate compatibility and exact sum-product on the
torso-centred star produces joint candidate marginals.

This is an isolated test of PointDSC's local-score + spatial-consistency idea,
implemented at the kinematic-part level and combined with the exact inference
used by pictorial-structure models.  The final pairwise layer is zero
initialized, so epoch zero exactly reduces to the frozen limb-unary model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# (limb group, torso joint, limb joint)
BOUNDARIES = ((1, 0, 1), (2, 0, 4), (3, 8, 11), (4, 8, 14))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--unary-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--pair-regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


class LimbCompatibilityGraph(nn.Module):
    """Shared torso/limb pair energy with exact star-graph inference."""

    def __init__(self, frozen_unary: limb.LimbUtility):
        super().__init__()
        self.unary = frozen_unary.eval().requires_grad_(False)
        self.group_embedding = nn.Parameter(torch.randn(4, 8) * 0.02)
        # contexts32 + boundary vector3 + length z1 + unary2 + pose delta4
        # + same-candidate1 + cardinality3 + limb embedding8 = 54
        self.pair_scorer = nn.Sequential(
            nn.Linear(54, 64), nn.ReLU6(),
            nn.Linear(64, 32), nn.ReLU6(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.pair_scorer[-1].weight)
        nn.init.zeros_(self.pair_scorer[-1].bias)

    def pair_energy(
        self, candidates, cost, combo, bone_mean, bone_std,
        limb_index, torso_joint, limb_joint,
    ):
        batch, count = candidates.shape[:2]
        normalized = (candidates - self.unary.coord_mean) / self.unary.coord_std
        root_relative = normalized - normalized[:, :, :1]
        context = self.unary.pose_encoder(root_relative.flatten(2))
        torso_xyz = candidates[:, :, torso_joint]
        limb_xyz = candidates[:, :, limb_joint]
        vector = limb_xyz[:, None] - torso_xyz[:, :, None]
        scale = bone_mean[limb_joint].clamp_min(1e-3)
        vector_feature = vector / scale
        length = torch.linalg.vector_norm(vector, dim=-1)
        length_z = ((length - bone_mean[limb_joint]) / bone_std[limb_joint]).unsqueeze(-1)
        context_feature = torch.cat((
            context[:, :, None].expand(-1, -1, count, -1),
            context[:, None].expand(-1, count, -1, -1),
        ), dim=-1)
        unary_feature = torch.stack((
            cost[:, 0, :, None].expand(-1, -1, count),
            cost[:, limb_index, None, :].expand(-1, count, -1),
        ), dim=-1)
        pose_delta = torch.linalg.vector_norm(
            candidates[:, :, None] - candidates[:, None, :], dim=-1
        ) / 0.01
        delta_feature = torch.stack((
            pose_delta.mean(dim=-1),
            pose_delta.std(dim=-1, unbiased=False),
            pose_delta.amin(dim=-1),
            pose_delta.amax(dim=-1),
        ), dim=-1)
        same = torch.eye(count, device=candidates.device, dtype=candidates.dtype)
        same = same[None, :, :, None].expand(batch, -1, -1, -1)
        cardinality = F.one_hot(
            torch.tensor(len(combo) - 2, device=candidates.device), num_classes=3
        ).to(candidates.dtype)[None, None, None].expand(batch, count, count, -1)
        embedding = self.group_embedding[limb_index - 1][None, None, None]
        embedding = embedding.expand(batch, count, count, -1)
        feature = torch.cat((
            context_feature, vector_feature, length_z, unary_feature,
            delta_feature, same, cardinality, embedding,
        ), dim=-1)
        if feature.shape[-1] != 54:
            raise RuntimeError(f"bad pair feature size {feature.shape}")
        # Dimensionless energy. Zero is the identity/no-compatibility case.
        return self.pair_scorer(feature).squeeze(-1)

    @staticmethod
    def exact_marginals(cost, temperature, pair_energies):
        unary_log = -cost / temperature
        leaf_messages = []
        for (limb_index, _, _), energy in zip(BOUNDARIES, pair_energies):
            leaf_messages.append(torch.logsumexp(
                unary_log[:, limb_index, None, :] - energy, dim=-1
            ))
        root_log = unary_log[:, 0].clone()
        for message in leaf_messages:
            root_log = root_log + message
        root_log = root_log - torch.logsumexp(root_log, dim=-1, keepdim=True)
        marginals = [root_log.exp()]
        for edge_index, ((limb_index, _, _), energy) in enumerate(
            zip(BOUNDARIES, pair_energies)
        ):
            cavity = unary_log[:, 0].clone()
            for other_index, message in enumerate(leaf_messages):
                if edge_index != other_index:
                    cavity = cavity + message
            root_to_leaf = torch.logsumexp(cavity[:, :, None] - energy, dim=1)
            leaf_log = unary_log[:, limb_index] + root_to_leaf
            leaf_log = leaf_log - torch.logsumexp(leaf_log, dim=-1, keepdim=True)
            marginals.append(leaf_log.exp())
        return torch.stack(marginals, dim=1)

    def forward(self, candidates, cost, combo, bone_mean, bone_std, temperature):
        pair_energies = [
            self.pair_energy(
                candidates, cost, combo, bone_mean, bone_std,
                limb_index, torso_joint, limb_joint,
            )
            for limb_index, torso_joint, limb_joint in BOUNDARIES
        ]
        marginal = self.exact_marginals(cost, temperature, pair_energies)
        return marginal, pair_energies


def fuse(marginal, candidates, baseline_local):
    pose = candidates[:, baseline_local].clone()
    for group_id, joints in enumerate(limb.GROUP_LIST):
        idx = torch.as_tensor(joints, device=candidates.device)
        pose[:, idx] = torch.einsum(
            "bc,bcjd->bjd", marginal[:, group_id], candidates[:, :, idx]
        )
    return pose


def forward_task(model, e2, predictions, targets, rays, combo, bone_mean, bone_std, saved):
    with torch.no_grad():
        unary, _, true_error, candidates, baseline_local = extra.predict_task(
            e2, predictions, targets, rays, combo
        )
        cost = model.unary(candidates, unary, rays, combo, bone_mean, bone_std)
    temperature = saved.v2_temperature if len(combo) == 2 else saved.v34_temperature
    marginal, pair_energies = model(
        candidates, cost, combo, bone_mean, bone_std, temperature
    )
    pose = fuse(marginal, candidates, baseline_local)
    group_error = []
    for joints in limb.GROUP_LIST:
        idx = torch.as_tensor(joints, device=candidates.device)
        group_error.append(true_error[:, idx].mean(dim=1))
    return marginal, torch.stack(group_error, dim=1), pose, pair_energies


def loss_fn(marginal, group_error, pose, targets, pair_energies, args):
    target = F.softmax(
        -group_error / (args.target_temperature_mm / 1000.0), dim=-1
    )
    ranking = -(target * marginal.clamp_min(1e-8).log()).sum(dim=-1).mean()
    pose_error = torch.linalg.vector_norm(pose - targets, dim=-1).mean() / 0.01
    regularization = torch.stack([energy.square().mean() for energy in pair_energies]).mean()
    return pose_error + 0.5 * ranking + args.pair_regularization * regularization


def evaluate(model, e2, loader, device, bone_mean, bone_std, saved):
    model.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    actions = {f"V{x}": [] for x in (2, 3, 4)}
    with torch.inference_mode():
        for predictions, targets, rays, action in loader:
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            for combo in TASKS:
                stage = f"V{len(combo)}"
                _, _, pose, _ = forward_task(
                    model, e2, predictions, targets, rays, combo,
                    bone_mean, bone_std, saved,
                )
                error = torch.linalg.vector_norm(pose - targets, dim=-1)
                store[stage]["soft"].append(error.cpu().numpy() * 1000.0)
                actions[stage].append(action.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        stage_actions = np.concatenate(actions[stage])
        values = np.concatenate(store[stage]["soft"])
        result[stage] = {"soft": {
            "action_equal_all17_mm": action_equal(values, stage_actions),
            "frame_weighted_all17_mm": float(values.mean()),
        }}
    return result


def headline(result):
    return float(np.mean([
        result[stage]["soft"]["action_equal_all17_mm"]
        for stage in ("V2", "V3", "V4")
    ]))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    train = trainer.load_arrays([args.train_cache], 22)
    validation = trainer.load_arrays([args.validation_cache], 22)
    holdout = train["group_indices"] % 10 == 0
    train_idx, holdout_idx = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    if args.smoke_batches:
        train_idx = train_idx[:args.batch_size * args.smoke_batches]
        holdout_idx = holdout_idx[:args.batch_size]
        validation = {key: value[:args.batch_size] for key, value in validation.items()}
    loaders = {
        "train": DataLoader(
            ArrayDataset(train, train_idx), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        ),
        "holdout": DataLoader(
            ArrayDataset(train, holdout_idx), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers,
        ),
        "validation": DataLoader(
            ArrayDataset(validation, np.arange(len(validation["targets"]))),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        ),
    }
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    unary_state = torch.load(args.unary_checkpoint, map_location=device, weights_only=False)
    saved = SimpleNamespace(**unary_state["args"])
    unary = limb.LimbUtility(coord_mean, coord_std).to(device)
    unary.load_state_dict(unary_state["state_dict"], strict=True)
    model = LimbCompatibilityGraph(unary).to(device)
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        model.unary.eval()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(loaders["train"]):
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            combo = TASKS[(batch_index + epoch * 7) % len(TASKS)]
            optimizer.zero_grad(set_to_none=True)
            marginal, group_error, pose, pair_energies = forward_task(
                model, e2, predictions, targets, rays, combo,
                bone_mean, bone_std, saved,
            )
            loss = loss_fn(marginal, group_error, pose, targets, pair_energies, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout_result = evaluate(
            model, e2, loaders["holdout"], device, bone_mean, bone_std, saved
        )
        score = headline(holdout_result)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "holdout_headline_mm": score, "holdout": holdout_result}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save({
                "state_dict": {
                    key: value for key, value in model.state_dict().items()
                    if not key.startswith("unary.")
                },
                "args": vars(args), "epoch": epoch + 1,
                "holdout_headline_mm": score,
            }, output / "model_best.pth.tar")
        if args.smoke_batches and epoch >= 1:
            break
    checkpoint = torch.load(
        output / "model_best.pth.tar", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    result = evaluate(
        model, e2, loaders["validation"], device, bone_mean, bone_std, saved
    )
    payload = {
        "method": "frozen RUMPL/E2 limb unary + learned spatial compatibility graph",
        "input_protocol": (
            "frozen HRNet coordinates/confidence + cameras + deterministic "
            "ray-derived candidates only"
        ),
        "sources": [
            "PointDSC CVPR 2021: feature/spatial compatibility",
            "3D pictorial structures: exact kinematic-part inference",
        ],
        "best_epoch": checkpoint["epoch"],
        "best_holdout_headline_mm": checkpoint["holdout_headline_mm"],
        "validation": result,
        "history": history,
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
