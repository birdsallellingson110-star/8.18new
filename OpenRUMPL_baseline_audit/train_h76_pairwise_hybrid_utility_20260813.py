#!/usr/bin/env python3
"""Hybrid GHT/E2 utility training on the frozen pairwise candidate pool.

The pairwise E2 model predicts a per-joint counterfactual risk.  This control
adds a second, whole-pose GHT-style score over the same candidates.  The new
head is zero-initialized at its last layer, so epoch-0 predictions are exactly
the E2 model; RUMPL, the 2-D observations, and the frozen candidate generator
are unchanged.  The experiment tests whether whole-pose coherence supplies
information that the per-joint utility misses without conflating the result
with a new solver or a new dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    evaluate_expanded,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


class HybridGHTE2Utility(nn.Module):
    """E2 per-joint utility plus a zero-initialized whole-pose score.

    The global branch follows the GHT principle of scoring a complete pose
    hypothesis, while the base branch retains E2's candidate-set attention and
    per-joint counterfactual supervision.  No candidate or camera identity is
    encoded.  Absolute normalized coordinates are deliberately retained in the
    global branch; an earlier standalone root-centered GHT control discarded
    world translation and was therefore not a fair global risk model here.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, attention_depth: int):
        super().__init__()
        self.base = SetTransformerJointUtility(mean, std, attention_depth)
        # normalized full pose (51), root-relative pose (51), candidate spread,
        # included confidence mean, and view fraction.
        self.global_head = nn.Sequential(
            nn.LayerNorm(105),
            nn.Linear(105, 64), nn.ReLU6(),
            nn.Linear(64, 32), nn.ReLU6(),
            nn.Linear(32, 1),
        )
        # Make the newly added branch exactly identity at initialization.
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def forward(self, candidates, rays, candidate_masks, task_mask):
        local = self.base(candidates, rays, candidate_masks, task_mask)
        normalized = (candidates - self.base.pose_mean) / self.base.pose_std
        root_relative = normalized - normalized[:, :, :1]
        consensus = candidates.mean(dim=1, keepdim=True)
        spread = torch.linalg.vector_norm(
            candidates - consensus, dim=-1
        ).mean(dim=2, keepdim=True) / 0.1

        # Candidate-view mask is C,V.  Confidence is measured only over the
        # views actually used by that candidate and then normalized to [0,1].
        # Average detector confidence over joints before applying the
        # candidate's view mask: rays are B,J,V,7 whereas the mask is C,V.
        confidence = rays[..., 6].clamp(0, 1).mean(dim=1)
        included = candidate_masks[None, :, :].bool()
        included_count = included.sum(dim=-1).clamp_min(1).to(confidence.dtype)
        included_conf = (
            (confidence[:, None] * included).sum(dim=-1) / included_count
        )[..., None]
        view_fraction = (
            candidate_masks.sum(dim=-1)[None, :, None]
            / task_mask.sum().clamp_min(1.0)
        ).expand(candidates.shape[0], -1, -1)

        global_features = torch.cat(
            (
                normalized.flatten(2),
                root_relative.flatten(2),
                spread,
                included_conf,
                view_fraction,
            ),
            dim=-1,
        )
        if global_features.shape[-1] != 105:
            raise RuntimeError(f"unexpected global feature size {global_features.shape}")
        global_score = self.global_head(global_features).squeeze(-1)
        return local + global_score[:, None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-depth", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def task_loss(model, predictions, targets, rays, phase, temperature,
              target_temperature_m, oracle_weight):
    direct, oracle, ght = [], [], []
    for task in TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, _ = predict_delta_expanded(
            model, predictions, targets, rays, task
        )
        direct.append(training_loss(predicted, true_delta, "balanced_rank"))
        target = F.softmax(-true_error / target_temperature_m, dim=-1)
        log_prob = F.log_softmax(-predicted / temperature, dim=-1)
        oracle.append(-(target * log_prob).sum(dim=-1).mean())
        if phase == "ght":
            weights = F.softmax(-predicted / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            ght.append((expected + 0.05 * fused_error) / 0.01)
    loss = torch.stack(direct).mean() + oracle_weight * torch.stack(oracle).mean()
    if ght:
        loss = loss + torch.stack(ght).mean()
    return loss


def main() -> None:
    args = parse_args()
    if args.target_temperature_mm <= 0 or args.oracle_weight < 0:
        raise ValueError("target temperature must be positive and oracle weight non-negative")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = HybridGHTE2Utility(mean, std, args.attention_depth).to(device)

    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model_best.pth.tar"
    phases = (
        [("direct", 5e-4)] * args.pretrain_epochs
        + [("ght", 1e-4)] * args.finetune_epochs
    )
    optimizer = None
    previous_phase = None
    best_metric, best_epoch, history = math.inf, -1, []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous_phase:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous_phase = phase
        model.train()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, phase, args.temperature,
                args.target_temperature_mm / 1000.0, args.oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout_result = evaluate_expanded(model, holdout_loader, device, args.temperature)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch, "phase": phase,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature,
                "method": "hybrid_ght_e2_zero_init_global_head",
            }, checkpoint)

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_expanded(model, test_loader, device, args.temperature)
    result = {
        "method": "pairwise E2 + zero-initialized GHT whole-pose score",
        "paper_basis": (
            "GHT-style whole-hypothesis scoring combined with E2 counterfactual "
            "candidate-set utility; absolute world pose retained"
        ),
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": test_result,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
