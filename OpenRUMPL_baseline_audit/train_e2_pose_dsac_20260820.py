#!/usr/bin/env python3
"""Fuse RUMPL/E2 candidates with a PoseDSAC-style hypothesis scorer.

The implementation follows the released Generalizable Human Pose
Triangulation idea at the module boundary that fits this project:

1. independently sample a candidate label for each joint (or each limb);
2. assemble full 3D pose hypotheses;
3. score root-centred pose coordinates and bone lengths with a small MLP;
4. use a differentiable weighted estimate and expected pose risk.

The ray candidate generator and E2 utility network remain frozen.  The
``conditioned`` feature variant additionally exposes the selected E2 unary
scores and absolute pelvis to the hypothesis scorer, preserving useful RUMPL
geometry instead of stacking another complete pose model.
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
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import GROUPS, train_bone_stats
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling", choices=("joint", "limb"), default="joint")
    parser.add_argument(
        "--features", choices=("official", "conditioned", "bone_only"),
        default="official",
    )
    parser.add_argument(
        "--loss-mode", choices=("hybrid", "pose_dsac", "regression"),
        default="hybrid",
        help="pose_dsac mirrors the released normalized expectation + entropy objective.",
    )
    parser.add_argument(
        "--sigmoid-score", action="store_true",
        help="Use the released PoseDSAC scorer's final sigmoid.",
    )
    parser.add_argument("--hypotheses", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--proposal-temperature", type=float, default=0.8)
    parser.add_argument(
        "--proposal-source", choices=("e2", "limb_utility"), default="e2",
        help="Hypothesis sampling distribution; default preserves historical runs.",
    )
    parser.add_argument(
        "--proposal-checkpoint", default=None,
        help="Frozen deterministic limb-utility checkpoint for limb_utility proposal.",
    )
    parser.add_argument("--score-temperature", type=float, default=0.5)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class PoseScoreMLP(nn.Module):
    """The released PoseDSAC scorer is a 3x50 ReLU6 MLP."""

    def __init__(self, feature_mode: str, sigmoid_score: bool = False):
        super().__init__()
        # Released code supports xyz, xyz+bone, or bone-only modes.  ``official``
        # here denotes xyz+bone; ``bone_only`` reproduces its default mode=2.
        input_size = 16 if feature_mode == "bone_only" else 67
        if feature_mode == "conditioned":
            # selected E2 unary for 17 joints + absolute pelvis xyz
            input_size += 20
        self.feature_mode = feature_mode
        self.sigmoid_score = sigmoid_score
        self.network = nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 1),
        )

    def forward(
        self, hypotheses: torch.Tensor, selected_unary: torch.Tensor,
        coord_mean: torch.Tensor, coord_std: torch.Tensor,
        bone_mean: torch.Tensor, bone_std: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (hypotheses - coord_mean) / coord_std
        centred = normalized - normalized[:, :, :1]
        parents = extra.PARENTS.to(hypotheses.device)
        lengths = torch.linalg.vector_norm(
            hypotheses - hypotheses[:, :, parents], dim=-1
        )[:, :, 1:]
        bone = (lengths - bone_mean[1:]) / bone_std[1:]
        pieces = (bone,) if self.feature_mode == "bone_only" else (centred.flatten(2), bone)
        if self.feature_mode == "conditioned":
            pieces += (selected_unary, normalized[:, :, 0],)
        score = self.network(torch.cat(pieces, dim=-1)).squeeze(-1)
        return torch.sigmoid(score) if self.sigmoid_score else score


def load_e2(path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = SetTransformerJointUtility(
        state["mean"], state["std"], state["attention_depth"],
        stage_heads=state.get("stage_heads", False),
        neutralize_subset_penalty=True,
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, state["mean"].to(device), state["std"].to(device)


def gather_hypotheses(candidates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch, count, joints, xyz = candidates.shape
    hypotheses = labels.shape[1]
    source = candidates.permute(0, 2, 1, 3)[:, None].expand(
        batch, hypotheses, joints, count, xyz
    )
    return source.gather(
        3, labels[..., None, None].expand(-1, -1, -1, 1, xyz)
    ).squeeze(3)


def gather_unary(unary: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch, joints, count = unary.shape
    hypotheses = labels.shape[1]
    source = unary[:, None].expand(batch, hypotheses, joints, count)
    return source.gather(3, labels[..., None]).squeeze(3)


def sample_pool(
    candidates: torch.Tensor, unary: torch.Tensor, baseline_local: int,
    hypotheses: int, sampling: str, proposal_temperature: float,
    group_cost: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create GT-free mixed hypotheses, including identity and E2 controls."""
    batch, count, joints, _ = candidates.shape
    if hypotheses < 4:
        raise ValueError("--hypotheses must be >= 4")
    probability = 0.8 * F.softmax(-unary / proposal_temperature, dim=-1)
    probability = probability + 0.2 / count
    labels = torch.empty(
        batch, hypotheses - 1, joints, dtype=torch.long, device=candidates.device
    )
    if sampling == "joint":
        draw = torch.multinomial(
            probability.reshape(-1, count), hypotheses - 1, replacement=True
        )
        labels.copy_(draw.reshape(batch, joints, hypotheses - 1).permute(0, 2, 1))
    else:
        for group_id, group in enumerate(GROUPS.values()):
            idx = torch.as_tensor(group, device=candidates.device)
            if group_cost is None:
                group_probability = probability[:, idx].mean(dim=1)
            else:
                group_probability = (
                    0.8 * F.softmax(
                        -group_cost[:, group_id] / proposal_temperature, dim=-1
                    )
                    + 0.2 / count
                )
            group_probability = group_probability / group_probability.sum(dim=-1, keepdim=True)
            draw = torch.multinomial(
                group_probability, hypotheses - 1, replacement=True
            )
            labels[:, :, idx] = draw[:, :, None]

    # Hypothesis 0: the unchanged task-local RUMPL baseline.
    labels[:, 0] = baseline_local
    # Hypothesis 1: E2 per-joint hard selection.
    labels[:, 1] = unary.argmin(dim=-1)
    mixed = gather_hypotheses(candidates, labels)
    selected = gather_unary(unary, labels)

    # Last hypothesis: established E2 soft fusion.  Its unary feature is the
    # probability-weighted E2 score and it gives the new scorer a safe control.
    weight = F.softmax(-unary / proposal_temperature, dim=-1)
    soft_pose = torch.einsum("bjc,bcjd->bjd", weight, candidates)
    soft_unary = (weight * unary).sum(dim=-1)
    mixed = torch.cat((mixed, soft_pose[:, None]), dim=1)
    selected = torch.cat((selected, soft_unary[:, None]), dim=1)
    return mixed, selected


def forward_task(
    scorer, e2, predictions, targets, rays, combo, args,
    coord_mean, coord_std, bone_mean, bone_std, proposal_model=None,
):
    with torch.no_grad():
        unary, _, _, candidates, baseline_local = extra.predict_task(
            e2, predictions, targets, rays, combo
        )
        group_cost = None
        if proposal_model is not None:
            group_cost = proposal_model(
                candidates, unary, rays, combo, bone_mean, bone_std
            )
        hypotheses, selected = sample_pool(
            candidates, unary, baseline_local, args.hypotheses,
            args.sampling, args.proposal_temperature, group_cost,
        )
    scores = scorer(
        hypotheses, selected, coord_mean, coord_std, bone_mean, bone_std
    )
    weights = F.softmax(scores / args.score_temperature, dim=-1)
    fused = torch.einsum("bk,bkjd->bjd", weights, hypotheses)
    hard = hypotheses.gather(
        1, scores.argmax(dim=-1)[:, None, None, None].expand(-1, 1, 17, 3)
    ).squeeze(1)
    error = torch.linalg.vector_norm(hypotheses - targets[:, None], dim=-1).mean(dim=-1)
    return scores, weights, hypotheses, fused, hard, error, candidates[:, baseline_local]


def batch_loss(scores, weights, hypotheses, fused, target, hyp_error, args):
    if args.loss_mode == "pose_dsac":
        # Released PoseDSAC: normalized hypothesis expectation + entropy + a
        # small weighted-estimate term (defaults exp=1, entropy=1, est=.05).
        normalized = hyp_error / hyp_error.detach().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        expectation = (weights * normalized).sum(dim=-1).mean()
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        fused_error = torch.linalg.vector_norm(fused - target, dim=-1).mean()
        return expectation + entropy + 0.05 * fused_error / 0.01
    target_distribution = F.softmax(
        -hyp_error / (args.target_temperature_mm / 1000.0), dim=-1
    )
    ranking = -(target_distribution * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
    fused_error = torch.linalg.vector_norm(fused - target, dim=-1).mean()
    expected_error = (weights * hyp_error).sum(dim=-1).mean()
    regression_target = -hyp_error.detach() / 0.01
    regression = F.smooth_l1_loss(
        scores - scores.mean(dim=-1, keepdim=True),
        regression_target - regression_target.mean(dim=-1, keepdim=True),
    )
    # Direct 3D risk is primary; listwise or regression supervision stabilizes
    # hypothesis scores.  Regression is an explicit scoring-capacity control.
    if args.loss_mode == "regression":
        return fused_error / 0.01 + 0.2 * expected_error / 0.01 + regression
    return (
        fused_error / 0.01 + 0.2 * expected_error / 0.01
        + 0.5 * ranking + 0.25 * regression
    )


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def evaluate(
    scorer, e2, loader, device, args,
    coord_mean, coord_std, bone_mean, bone_std, proposal_model=None,
):
    scorer.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    # Fixed sampling stream gives reproducible model comparisons.
    torch.manual_seed(10000 + args.seed)
    torch.cuda.manual_seed_all(10000 + args.seed)
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            for combo in TASKS:
                stage = f"V{len(combo)}"
                scores, weights, hypotheses, fused, hard, hyp_error, baseline = forward_task(
                    scorer, e2, predictions, targets, rays, combo, args,
                    coord_mean, coord_std, bone_mean, bone_std, proposal_model,
                )
                modes = {
                    "baseline": baseline,
                    "weighted": fused,
                    "hard": hard,
                }
                for mode, pose in modes.items():
                    values = torch.linalg.vector_norm(pose - targets, dim=-1)
                    store[stage][mode].append(values.cpu().numpy() * 1000.0)
                store[stage]["sampled_oracle"].append(
                    hyp_error.min(dim=-1).values.cpu().numpy()[:, None] * 1000.0
                )
                action_store[stage].append(actions.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        result[stage] = {}
        actions = np.concatenate(action_store[stage])
        for mode, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def headline(result) -> float:
    return float(np.mean([
        result[stage]["weighted"]["action_equal_all17_mm"]
        for stage in ("V2", "V3", "V4")
    ]))


def main() -> None:
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
    holdout_mask = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout_mask)
    holdout_indices = np.flatnonzero(holdout_mask)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: args.batch_size]
        validation = {key: value[: args.batch_size] for key, value in validation.items()}
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )
    e2, coord_mean, coord_std = load_e2(args.e2_checkpoint, device)
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    proposal_model = None
    if args.proposal_source == "limb_utility":
        if args.sampling != "limb" or not args.proposal_checkpoint:
            raise ValueError(
                "limb_utility proposal requires --sampling limb and "
                "--proposal-checkpoint"
            )
        # Local import avoids a module cycle: limb utility itself imports this
        # file for shared E2/PoseDSAC helpers.
        import train_e2_limb_utility_20260820 as limb_utility
        proposal_state = torch.load(
            args.proposal_checkpoint, map_location=device, weights_only=False
        )
        proposal_model = limb_utility.LimbUtility(coord_mean, coord_std).to(device)
        proposal_model.load_state_dict(proposal_state["state_dict"], strict=True)
        proposal_model.eval().requires_grad_(False)
    scorer = PoseScoreMLP(args.features, args.sigmoid_score).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        scorer.train()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            # One balanced task per minibatch; all 11 tasks are covered each epoch.
            combo = TASKS[(batch_index + epoch * 7) % len(TASKS)]
            optimizer.zero_grad(set_to_none=True)
            scores, weights, hypotheses, fused, _, hyp_error, _ = forward_task(
                scorer, e2, predictions, targets, rays, combo, args,
                coord_mean, coord_std, bone_mean, bone_std, proposal_model,
            )
            loss = batch_loss(
                scores, weights, hypotheses, fused, targets, hyp_error, args
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout_result = evaluate(
            scorer, e2, holdout_loader, device, args,
            coord_mean, coord_std, bone_mean, bone_std, proposal_model,
        )
        score = headline(holdout_result)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "holdout_headline_mm": score,
            "holdout": holdout_result,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save({
                "state_dict": scorer.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "holdout_headline_mm": score,
            }, output_dir / "model_best.pth.tar")

    checkpoint = torch.load(
        output_dir / "model_best.pth.tar", map_location=device, weights_only=False
    )
    scorer.load_state_dict(checkpoint["state_dict"])
    validation_result = evaluate(
        scorer, e2, validation_loader, device, args,
        coord_mean, coord_std, bone_mean, bone_std, proposal_model,
    )
    payload = {
        "method": "RUMPL/E2 candidates + Generalizable-HPT PoseDSAC-style hypothesis scorer",
        "source_code": "reference/general-3d-humans-official/src/dsac.py::PoseDSAC",
        "sampling": args.sampling,
        "proposal_source": args.proposal_source,
        "features": args.features,
        "hypotheses": args.hypotheses,
        "best_epoch": checkpoint["epoch"],
        "best_holdout_headline_mm": checkpoint["holdout_headline_mm"],
        "validation": validation_result,
        "history": history,
        "protocol": "standard H36M S1/S5/S6/S7/S8 train, S9/S11 test; all camera subsets",
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
