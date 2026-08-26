#!/usr/bin/env python3
"""Score frozen K96 hypotheses with full joint-view observation evidence.

Unlike the failed pose-delta MAP model, this module cannot move a joint outside
the frozen K96 hypothesis distribution.  Unlike the frozen pose density, it is
conditioned on the current frame's complete ray residual pattern.  Axial
attention first reasons over the full body separately in every view, then over
views for every joint, so body context is available before view compression.
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
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from train_failure_informed_map_20260820 import FrozenK96Anchor
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--score-temperature", type=float, default=0.5)
    parser.add_argument("--relative-score-limit", type=float, default=2.0)
    parser.add_argument("--gate-mm", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--holdout-stride", type=int, default=1,
        help="Deterministic S8 subsampling for the short screen; full gate uses 1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class ObservationConditionedK96Scorer(nn.Module):
    """Zero-initialized residual scorer with body-before-view axial attention."""

    def __init__(self, d_model: int, heads: int, depth: int, score_limit: float):
        super().__init__()
        self.score_limit = float(score_limit)
        # direction(3), root-centred camera origin(3), confidence(1),
        # signed perpendicular residual(3), residual norm(1), depth(1),
        # root-relative hypothesis joint(3), selected E2 unary(1).
        self.input_projection = nn.Sequential(
            nn.Linear(16, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.joint_embedding = nn.Parameter(torch.zeros(17, d_model))
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

        def encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=2 * d_model,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=depth)

        self.within_view_body = encoder()
        self.cross_view_joint = encoder()
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, hypotheses: torch.Tensor, selected_unary: torch.Tensor,
        rays: torch.Tensor, combo: tuple[int, ...], base_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, joints, _ = hypotheses.shape
        selected = rays[:, :, list(combo)]
        views = selected.shape[2]
        direction = F.normalize(selected[..., :3], dim=-1)
        origin = selected[..., 3:6]
        confidence = selected[..., 6:7].clamp(0.01, 1.0)
        pose = hypotheses[:, :, :, None]
        offset = pose - origin[:, None]
        depth = (offset * direction[:, None]).sum(dim=-1, keepdim=True)
        perpendicular = offset - depth * direction[:, None]
        root = hypotheses[:, :, :1, None]
        relative = pose - root
        feature = torch.cat(
            (
                direction[:, None].expand(-1, count, -1, -1, -1),
                (origin[:, None] - root).expand(-1, count, -1, -1, -1) / 5.0,
                confidence[:, None].expand(-1, count, -1, -1, -1),
                perpendicular / 0.10,
                torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True) / 0.10,
                depth / 5.0,
                relative.expand(-1, -1, -1, views, -1),
                selected_unary[:, :, :, None, None].expand(-1, -1, -1, views, -1),
            ), dim=-1,
        )
        token = self.input_projection(feature)
        token = token + self.joint_embedding[None, None, :, None]
        # B,K,J,V,D -> (B*K*V),J,D: body context is formed before views mix.
        body = token.permute(0, 1, 3, 2, 4).reshape(
            batch * count * views, joints, -1
        )
        body = self.within_view_body(body)
        body = body.reshape(batch, count, views, joints, -1).permute(0, 1, 3, 2, 4)
        cross_view = body.reshape(batch * count * joints, views, -1)
        cross_view = self.cross_view_joint(cross_view)
        cross_view = cross_view.reshape(batch, count, joints, views, -1)
        reliability = confidence[:, None]
        pooled_view = (cross_view * reliability).sum(dim=3) / reliability.sum(
            dim=3
        ).clamp_min(1e-6)
        pooled = pooled_view.mean(dim=2)
        residual = self.score_limit * torch.tanh(self.output(pooled).squeeze(-1))
        return base_scores + residual


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def forward_task(model, frozen, predictions, targets, rays, combo, temperature):
    with torch.no_grad():
        hypotheses, selected, base_scores = frozen.hypothesis_evidence(
            predictions, rays, combo
        )
    # Values returned by an inference_mode-decorated frozen module must be
    # materialized as ordinary tensors before autograd saves them for the new
    # scorer's backward pass.
    hypotheses = hypotheses.clone()
    selected = selected.clone()
    base_scores = base_scores.clone()
    scores = model(hypotheses, selected, rays, combo, base_scores)
    weights = F.softmax(scores / temperature, dim=-1)
    fused = torch.einsum("bk,bkjd->bjd", weights, hypotheses)
    base_weights = F.softmax(base_scores / frozen.score_temperature, dim=-1)
    baseline = torch.einsum("bk,bkjd->bjd", base_weights, hypotheses)
    errors = torch.linalg.vector_norm(
        hypotheses - targets[:, None], dim=-1
    ).mean(dim=-1)
    return scores, weights, hypotheses, fused, baseline, errors


def loss_function(scores, weights, hypotheses, fused, target, errors):
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
    return fused_error / 0.01 + 0.2 * expected / 0.01 + 0.5 * ranking + 0.25 * regression


@torch.inference_mode()
def evaluate(model, frozen, loader, device, temperature, seed):
    model.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    pair_store = defaultdict(list)
    torch.manual_seed(10000 + seed)
    torch.cuda.manual_seed_all(10000 + seed)
    for predictions, targets, rays, actions in loader:
        predictions, targets, rays = (
            predictions.to(device), targets.to(device), rays.to(device)
        )
        for combo in TASKS:
            stage = f"V{len(combo)}"
            _, _, _, fused, baseline, _ = forward_task(
                model, frozen, predictions, targets, rays, combo, temperature
            )
            for name, pose in (("baseline", baseline), ("contextual", fused)):
                error = torch.linalg.vector_norm(pose - targets, dim=-1)
                store[stage][name].append(error.cpu().numpy() * 1000.0)
            if len(combo) == 2:
                pair_store[f"{combo[0]+1}-{combo[1]+1}"].append(
                    torch.linalg.vector_norm(fused - targets, dim=-1).cpu().numpy() * 1000.0
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
    result["V2_pairs_contextual_frame_weighted_mm"] = {
        key: float(np.concatenate(value).mean()) for key, value in pair_store.items()
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
    train_indices = np.flatnonzero(arrays["subjects"] != args.holdout_subject)
    holdout_indices = np.flatnonzero(arrays["subjects"] == args.holdout_subject)
    holdout_indices = holdout_indices[::args.holdout_stride]
    if args.max_train_samples:
        train_indices = train_indices[:args.max_train_samples]
    if args.smoke_batches:
        train_indices = train_indices[:args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[:args.batch_size]
    train_loader = DataLoader(
        ArrayDataset(arrays, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        ArrayDataset(arrays, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    frozen = FrozenK96Anchor(args, device)
    model = ObservationConditionedK96Scorer(
        args.d_model, args.heads, args.depth, args.relative_score_limit
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
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions, targets, rays = (
                predictions.to(device), targets.to(device), rays.to(device)
            )
            combo = TASKS[(batch_index + 7 * epoch) % len(TASKS)]
            optimizer.zero_grad(set_to_none=True)
            scores, weights, hypotheses, fused, _, errors = forward_task(
                model, frozen, predictions, targets, rays, combo,
                args.score_temperature,
            )
            loss = loss_function(scores, weights, hypotheses, fused, targets, errors)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        holdout = evaluate(
            model, frozen, holdout_loader, device, args.score_temperature, args.seed
        )
        score = mean_stage(holdout, "contextual")
        baseline = mean_stage(holdout, "baseline")
        row = {
            "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
            "holdout_baseline_mm": baseline,
            "holdout_contextual_mm": score,
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
            }, output_dir / "model_best.pth.tar")
    checkpoint = torch.load(
        output_dir / "model_best.pth.tar", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    selected_gain = mean_stage(checkpoint["holdout"], "baseline") - mean_stage(
        checkpoint["holdout"], "contextual"
    )
    payload = {
        "method": "K96 + observation-conditioned body-before-view axial scorer",
        "best_epoch": checkpoint["epoch"],
        "subject_holdout_gain_mm": selected_gain,
        "passes_gate": bool(selected_gain >= args.gate_mm),
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
            model, frozen, validation_loader, device,
            args.score_temperature, args.seed,
        )
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "best_epoch": payload["best_epoch"],
        "subject_holdout_gain_mm": payload["subject_holdout_gain_mm"],
        "passes_gate": payload["passes_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
