#!/usr/bin/env python3
"""Train a kinematic-part utility head over frozen RUMPL/E2 candidates.

This is the deterministic counterpart to the PoseDSAC sampling experiment.
It combines three released-method ideas at module level: learned hypothesis
scoring (Generalizable Human Pose Triangulation), part-consistent inference
(3D pictorial structures), and E2/RUMPL ray-conditioned candidate utility.
No image features, GT 2D, or GT-dependent inference are introduced.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import GROUPS, train_bone_stats
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
GROUP_LIST = tuple(GROUPS.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument(
        "--train-cardinalities", nargs="+", type=int, choices=(2, 3, 4),
        default=(2, 3, 4),
        help="Training tasks only; evaluation always reports V2/V3/V4.",
    )
    parser.add_argument("--v2-temperature", type=float, default=0.2)
    parser.add_argument("--v34-temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def statistics(value: torch.Tensor, dim) -> torch.Tensor:
    return torch.cat((
        value.mean(dim=dim, keepdim=True),
        value.std(dim=dim, keepdim=True, unbiased=False),
        value.amin(dim=dim, keepdim=True),
        value.amax(dim=dim, keepdim=True),
    ), dim=-1)


class LimbUtility(nn.Module):
    def __init__(self, coord_mean, coord_std):
        super().__init__()
        self.register_buffer("coord_mean", coord_mean.clone().float())
        self.register_buffer("coord_std", coord_std.clone().float())
        self.pose_encoder = nn.Sequential(
            nn.Linear(51, 64), nn.ReLU6(), nn.Linear(64, 16), nn.ReLU6()
        )
        self.group_embedding = nn.Parameter(torch.randn(5, 8) * 0.02)
        # unary4 + ray4 + delta4 + bone4 + center/spread6 + context16
        # + cardinality3 + group8 = 49
        self.scorer = nn.Sequential(
            nn.Linear(49, 96), nn.ReLU6(),
            nn.Linear(96, 64), nn.ReLU6(),
            nn.Linear(64, 1),
        )
        # Identity-preserving residual initialization: before learning, the
        # module exactly reduces to mean E2 utility within each body part.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, candidates, unary, rays, combo, bone_mean, bone_std):
        batch, count, _, _ = candidates.shape
        normalized = (candidates - self.coord_mean) / self.coord_std
        root_relative = normalized - normalized[:, :, :1]
        context = self.pose_encoder(root_relative.flatten(2))
        direction = F.normalize(rays[..., :3], dim=-1)
        point = rays[..., 3:6]
        offset = candidates[:, :, :, None] - point[:, None]
        residual = torch.linalg.vector_norm(
            torch.cross(offset, direction[:, None], dim=-1), dim=-1
        )
        residual = torch.log1p(residual[..., list(combo)] / 0.005)
        consensus = candidates.mean(dim=1, keepdim=True)
        delta = torch.linalg.vector_norm(candidates - consensus, dim=-1) / 0.01
        parents = extra.PARENTS.to(candidates.device)
        lengths = torch.linalg.vector_norm(candidates - candidates[:, :, parents], dim=-1)
        bone_z = (lengths - bone_mean) / bone_std
        cardinality = F.one_hot(
            torch.tensor(len(combo) - 2, device=candidates.device), num_classes=3
        ).to(candidates.dtype)[None, None].expand(batch, count, -1)
        outputs = []
        for group_id, joints in enumerate(GROUP_LIST):
            idx = torch.as_tensor(joints, device=candidates.device)
            u = unary[:, idx].permute(0, 2, 1)
            unary_feature = statistics(u, dim=2)
            r = residual[:, :, idx].flatten(2)
            ray_feature = statistics(r, dim=2)
            delta_feature = statistics(delta[:, :, idx], dim=2)
            bone_feature = statistics(bone_z[:, :, idx], dim=2)
            coordinates = root_relative[:, :, idx]
            coord_feature = torch.cat(
                (coordinates.mean(dim=2), coordinates.std(dim=2, unbiased=False)), dim=-1
            )
            embedding = self.group_embedding[group_id][None, None].expand(batch, count, -1)
            feature = torch.cat((
                unary_feature, ray_feature, delta_feature, bone_feature,
                coord_feature, context, cardinality, embedding,
            ), dim=-1)
            if feature.shape[-1] != 49:
                raise RuntimeError(f"bad limb feature size {feature.shape}")
            outputs.append(u.mean(dim=2) + self.scorer(feature).squeeze(-1))
        # B,G,C; lower is better.
        return torch.stack(outputs, dim=1)


def fuse_groups(cost, candidates, baseline_local, temperature):
    batch = candidates.shape[0]
    soft = candidates[:, baseline_local].clone()
    hard = candidates[:, baseline_local].clone()
    for group_id, joints in enumerate(GROUP_LIST):
        idx = torch.as_tensor(joints, device=candidates.device)
        weight = F.softmax(-cost[:, group_id] / temperature, dim=-1)
        soft[:, idx] = torch.einsum("bc,bcjd->bjd", weight, candidates[:, :, idx])
        label = cost[:, group_id].argmin(dim=-1)
        chosen = candidates.gather(
            1, label[:, None, None, None].expand(-1, 1, 17, 3)
        ).squeeze(1)
        hard[:, idx] = chosen[:, idx]
    return soft, hard


def forward_task(model, e2, predictions, targets, rays, combo, bone_mean, bone_std, args):
    with torch.no_grad():
        unary, _, true_error, candidates, baseline_local = extra.predict_task(
            e2, predictions, targets, rays, combo
        )
    cost = model(candidates, unary, rays, combo, bone_mean, bone_std)
    temperature = args.v2_temperature if len(combo) == 2 else args.v34_temperature
    soft, hard = fuse_groups(cost, candidates, baseline_local, temperature)
    group_error = []
    for joints in GROUP_LIST:
        idx = torch.as_tensor(joints, device=candidates.device)
        group_error.append(true_error[:, idx].mean(dim=1))
    return cost, torch.stack(group_error, dim=1), soft, hard, candidates[:, baseline_local]


def training_loss(cost, group_error, soft, targets, args):
    target_distribution = F.softmax(
        -group_error / (args.target_temperature_mm / 1000.0), dim=-1
    )
    ranking = -(target_distribution * F.log_softmax(-cost, dim=-1)).sum(dim=-1).mean()
    target_cost = group_error.detach() / 0.01
    regression = F.smooth_l1_loss(
        cost - cost.mean(dim=-1, keepdim=True),
        target_cost - target_cost.mean(dim=-1, keepdim=True),
    )
    pose_error = torch.linalg.vector_norm(soft - targets, dim=-1).mean() / 0.01
    return pose_error + 0.5 * ranking + 0.5 * regression


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def evaluate(model, e2, loader, device, bone_mean, bone_std, args):
    model.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    actions = {f"V{x}": [] for x in (2, 3, 4)}
    with torch.inference_mode():
        for predictions, targets, rays, action in loader:
            predictions, targets, rays = predictions.to(device), targets.to(device), rays.to(device)
            for combo in TASKS:
                stage = f"V{len(combo)}"
                cost, group_error, soft, hard, baseline = forward_task(
                    model, e2, predictions, targets, rays, combo, bone_mean, bone_std, args
                )
                for mode, pose in (("baseline", baseline), ("soft", soft), ("hard", hard)):
                    error = torch.linalg.vector_norm(pose - targets, dim=-1)
                    store[stage][mode].append(error.cpu().numpy() * 1000.0)
                # This oracle is diagnostic only and never enters inference.
                group_weight = torch.as_tensor(
                    [len(group) / 17.0 for group in GROUP_LIST],
                    device=group_error.device, dtype=group_error.dtype,
                )
                oracle = (group_error.min(dim=-1).values * group_weight).sum(dim=-1)
                store[stage]["limb_oracle"].append(
                    oracle.cpu().numpy()[:, None] * 1000.0
                )
                actions[stage].append(action.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        result[stage] = {}
        stage_actions = np.concatenate(actions[stage])
        for mode, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def headline(result):
    return float(np.mean([
        result[stage]["soft"]["action_equal_all17_mm"] for stage in ("V2", "V3", "V4")
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
        train_idx = train_idx[: args.batch_size * args.smoke_batches]
        holdout_idx = holdout_idx[: args.batch_size]
        validation = {key: value[: args.batch_size] for key, value in validation.items()}
    loaders = {
        "train": DataLoader(
            ArrayDataset(train, train_idx), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        ),
        "holdout": DataLoader(
            ArrayDataset(train, holdout_idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        ),
        "validation": DataLoader(
            ArrayDataset(validation, np.arange(len(validation["targets"]))),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        ),
    }
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    model = LimbUtility(coord_mean, coord_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    training_tasks = tuple(
        combo for combo in TASKS if len(combo) in set(args.train_cardinalities)
    )
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(loaders["train"]):
            predictions, targets, rays = predictions.to(device), targets.to(device), rays.to(device)
            combo = training_tasks[(batch_index + epoch * 7) % len(training_tasks)]
            optimizer.zero_grad(set_to_none=True)
            cost, group_error, soft, _, _ = forward_task(
                model, e2, predictions, targets, rays, combo, bone_mean, bone_std, args
            )
            loss = training_loss(cost, group_error, soft, targets, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout_result = evaluate(
            model, e2, loaders["holdout"], device, bone_mean, bone_std, args
        )
        score = float(np.mean([
            holdout_result[f"V{count}"]["soft"]["action_equal_all17_mm"]
            for count in sorted(set(args.train_cardinalities))
        ]))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "holdout_headline_mm": score, "holdout": holdout_result}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "epoch": epoch + 1, "holdout_headline_mm": score},
                       output / "model_best.pth.tar")
    checkpoint = torch.load(output / "model_best.pth.tar", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    result = evaluate(model, e2, loaders["validation"], device, bone_mean, bone_std, args)
    payload = {
        "method": "RUMPL/E2 ray candidates + kinematic-part utility",
        "sources": [
            "general-3d-humans-official/src/dsac.py::PoseDSAC",
            "3D Pictorial Structures (CVPR 2013/2014)",
        ],
        "best_epoch": checkpoint["epoch"],
        "best_holdout_headline_mm": checkpoint["holdout_headline_mm"],
        "validation": result,
        "history": history,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
