#!/usr/bin/env python3
"""Stage-II frozen discrete-continuous scorer for the established K96 pool.

The Stage-I PCT/SimVQ/FSQ tokenizer is completely frozen.  Its continuous
tokens query their quantized counterparts through cross-attention, following
the transferable DCSA principle from UniCodebook.  A zero-initialized residual
is added to the frozen K96 score, guaranteeing exact baseline behavior before
training.  No hypothesis coordinates, 2-D inputs, or camera parameters change.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
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
from train_e4_pct_3d_tokenizer_stage1_20260821 import PCT3DTokenizer
from train_failure_informed_map_20260820 import FrozenK96Anchor
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--relative-score-limit", type=float, default=2.0)
    parser.add_argument("--gate-mm", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def load_tokenizer(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    args = state["args"]
    weights = state["state_dict"]
    model = PCT3DTokenizer(
        weights["pose_mean"], weights["pose_std"],
        args["hidden_dim"], args["token_dim"], args["token_num"],
        args["codebook_size"], args["encoder_depth"], args["decoder_depth"],
        args["mask_rate"], args["ema_decay"],
        args.get("quantizer", "ema"), args.get("fsq_levels", (8, 5, 5, 5)),
        args.get("simvq_beta", 0.25),
    ).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    return model, state


class DiscreteContinuousK96Scorer(nn.Module):
    """Attribute-preserving continuous-query/discrete-key residual scorer."""

    def __init__(self, token_dim, d_model, heads, depth, dropout, score_limit):
        super().__init__()
        if d_model % heads:
            raise ValueError("d-model must be divisible by attention heads")
        self.score_limit = float(score_limit)
        self.continuous_projection = nn.Linear(token_dim, d_model)
        self.discrete_projection = nn.Linear(token_dim, d_model)
        self.continuous_norm = nn.LayerNorm(d_model)
        self.discrete_norm = nn.LayerNorm(d_model)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, heads, dropout=dropout, batch_first=True
            ) for _ in range(depth)
        ])
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(depth)
        ])
        # mean/std of continuous, quantized, DCSA and quantization gap = 8D;
        # reconstruction/quantization/unary statistics = 12; base score = 1.
        self.output = nn.Sequential(
            nn.LayerNorm(8 * d_model + 13),
            nn.Linear(8 * d_model + 13, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def stats(value, dim):
        return torch.cat((
            value.mean(dim=dim, keepdim=True),
            value.std(dim=dim, keepdim=True, unbiased=False),
            value.amin(dim=dim, keepdim=True),
            value.amax(dim=dim, keepdim=True),
        ), dim=-1)

    def forward(
        self, tokenizer, hypotheses, selected_unary, base_scores,
    ):
        batch, count = hypotheses.shape[:2]
        flat_pose = hypotheses.flatten(0, 1)
        with torch.no_grad():
            normalized = tokenizer.normalize(flat_pose)
            continuous = tokenizer.encode(normalized, apply_mask=False)
            quantized, _, _ = tokenizer.nearest(continuous)
            reconstructed = tokenizer.decode(quantized)
        continuous = continuous.clone()
        quantized = quantized.clone()
        reconstructed = reconstructed.clone()
        projected_continuous = self.continuous_projection(continuous)
        projected_discrete = self.discrete_projection(quantized)
        token = self.continuous_norm(projected_continuous)
        memory = self.discrete_norm(projected_discrete)
        for attention, norm in zip(self.attention, self.attention_norm):
            update, _ = attention(token, memory, memory, need_weights=False)
            token = norm(token + update)
        gap = continuous - quantized
        projected_gap = projected_continuous - projected_discrete
        token_feature = torch.cat((
            projected_continuous.mean(dim=1),
            projected_continuous.std(dim=1, unbiased=False),
            projected_discrete.mean(dim=1),
            projected_discrete.std(dim=1, unbiased=False),
            token.mean(dim=1), token.std(dim=1, unbiased=False),
            projected_gap.mean(dim=1),
            projected_gap.std(dim=1, unbiased=False),
        ), dim=-1)
        joint_reconstruction = torch.linalg.vector_norm(
            reconstructed - normalized, dim=-1
        )
        token_gap = torch.linalg.vector_norm(gap, dim=-1)
        unary = selected_unary.flatten(0, 1)
        scalar_feature = torch.cat((
            self.stats(joint_reconstruction, 1),
            self.stats(token_gap, 1),
            self.stats(unary, 1),
            base_scores.flatten()[:, None],
        ), dim=-1)
        feature = torch.cat((token_feature, scalar_feature), dim=-1)
        residual = self.score_limit * torch.tanh(self.output(feature)).view(
            batch, count
        )
        return base_scores + residual


def forward_task(model, tokenizer, frozen, predictions, targets, rays, combo):
    with torch.inference_mode():
        hypotheses, selected, base_scores = frozen.hypothesis_evidence(
            predictions, rays, combo
        )
    hypotheses = hypotheses.clone()
    selected = selected.clone()
    base_scores = base_scores.clone()
    scores = model(tokenizer, hypotheses, selected, base_scores)
    weights = F.softmax(scores / frozen.score_temperature, dim=-1)
    base_weights = F.softmax(base_scores / frozen.score_temperature, dim=-1)
    fused = torch.einsum("bk,bkjd->bjd", weights, hypotheses)
    baseline = torch.einsum("bk,bkjd->bjd", base_weights, hypotheses)
    errors = torch.linalg.vector_norm(
        hypotheses - targets[:, None], dim=-1
    ).mean(dim=-1)
    return scores, weights, hypotheses, fused, baseline, errors


def loss_function(scores, weights, fused, target, errors):
    target_distribution = F.softmax(-errors / 0.005, dim=-1)
    ranking = -(
        target_distribution * F.log_softmax(scores, dim=-1)
    ).sum(dim=-1).mean()
    fused_error = torch.linalg.vector_norm(fused - target, dim=-1).mean()
    expected = (weights * errors).sum(dim=-1).mean()
    regression_target = -errors.detach() / 0.01
    regression = F.smooth_l1_loss(
        scores - scores.mean(dim=-1, keepdim=True),
        regression_target - regression_target.mean(dim=-1, keepdim=True),
    )
    return (
        fused_error / 0.01 + 0.2 * expected / 0.01
        + 0.5 * ranking + 0.25 * regression
    )


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


@torch.inference_mode()
def evaluate(model, tokenizer, frozen, loader, device, seed):
    model.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    torch.manual_seed(10000 + seed)
    torch.cuda.manual_seed_all(10000 + seed)
    for predictions, targets, rays, actions in loader:
        predictions, targets, rays = (
            predictions.to(device), targets.to(device), rays.to(device)
        )
        for combo in TASKS:
            stage = f"V{len(combo)}"
            _, _, _, fused, baseline, errors = forward_task(
                model, tokenizer, frozen, predictions, targets, rays, combo
            )
            for name, pose in (("baseline", baseline), ("discrete", fused)):
                value = torch.linalg.vector_norm(pose - targets, dim=-1)
                store[stage][name].append(value.cpu().numpy() * 1000.0)
            store[stage]["oracle"].append(
                errors.amin(dim=-1).cpu().numpy()[:, None] * 1000.0
            )
            action_store[stage].append(actions.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(action_store[stage])
        result[stage] = {}
        for name, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][name] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def mean_stage(result, mode):
    return float(np.mean([
        result[stage][mode]["action_equal_all17_mm"]
        for stage in ("V2", "V3", "V4")
    ]))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.train_cache], 22)
    train_idx = np.flatnonzero(arrays["subjects"] != args.holdout_subject)
    holdout_idx = np.flatnonzero(arrays["subjects"] == args.holdout_subject)[::args.holdout_stride]
    if args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(
            train_idx, min(args.max_train_samples, len(train_idx)), replace=False
        )
    if args.smoke_batches:
        train_idx = train_idx[: args.batch_size * args.smoke_batches]
        holdout_idx = holdout_idx[: args.batch_size]
    train_loader = DataLoader(
        ArrayDataset(arrays, train_idx), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        ArrayDataset(arrays, holdout_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    frozen = FrozenK96Anchor(args, device)
    tokenizer, tokenizer_state = load_tokenizer(args.tokenizer_checkpoint, device)
    token_dim = tokenizer_state["args"]["token_dim"]
    model = DiscreteContinuousK96Scorer(
        token_dim, args.d_model, args.heads, args.depth, args.dropout,
        args.relative_score_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses, grad_norms = [], []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions, targets, rays = (
                predictions.to(device), targets.to(device), rays.to(device)
            )
            combo = TASKS[(batch_index + 7 * epoch) % len(TASKS)]
            optimizer.zero_grad(set_to_none=True)
            scores, weights, _, fused, _, errors = forward_task(
                model, tokenizer, frozen, predictions, targets, rays, combo
            )
            loss = loss_function(scores, weights, fused, targets, errors)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
        holdout = evaluate(
            model, tokenizer, frozen, holdout_loader, device, args.seed
        )
        baseline = mean_stage(holdout, "baseline")
        score = mean_stage(holdout, "discrete")
        row = {
            "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "holdout_baseline_mm": baseline,
            "holdout_discrete_mm": score,
            "holdout_gain_mm": baseline - score,
            "holdout": holdout,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save({
                "state_dict": model.state_dict(), "args": vars(args),
                "epoch": epoch + 1, "holdout": holdout,
                "tokenizer_epoch": tokenizer_state["epoch"],
                "tokenizer_args": tokenizer_state["args"],
            }, output_dir / "model_best.pth.tar")
    checkpoint = torch.load(
        output_dir / "model_best.pth.tar", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    selected = checkpoint["holdout"]
    gain = mean_stage(selected, "baseline") - mean_stage(selected, "discrete")
    payload = {
        "method": "frozen PCT/SimVQ/FSQ tokenizer + DCSA-style K96 residual scorer",
        "best_epoch": checkpoint["epoch"],
        "holdout_gain_mm": gain,
        "passes_gate": bool(gain >= args.gate_mm),
        "history": history,
        "validation": None,
    }
    if payload["passes_gate"] and not args.smoke_batches:
        validation = trainer.load_arrays([args.validation_cache], 22)
        validation_loader = DataLoader(
            ArrayDataset(validation, np.arange(len(validation["targets"]))),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        )
        payload["validation"] = evaluate(
            model, tokenizer, frozen, validation_loader, device, args.seed + 1000
        )
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "best_epoch", "holdout_gain_mm", "passes_gate"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
