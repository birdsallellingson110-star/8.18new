#!/usr/bin/env python3
"""Pairwise E2 with an explicit source prior for heterogeneous hypotheses."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    COMBINATIONS,
    TASK_COMBINATIONS,
)
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    load_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--attention-depth", type=int, default=2)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


class SourceAwareUtility(nn.Module):
    """E2 scorer plus a tiny source-by-joint prior.

    Source 0 is an original H76 hypothesis; source 1 is one of the six frozen
    pairwise ray hypotheses.  The bias is initialized to zero, so the model
    starts exactly at E2 and can only learn a source-dependent calibration.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, depth: int):
        super().__init__()
        self.base = SetTransformerJointUtility(mean, std, depth)
        self.source_joint_bias = nn.Parameter(torch.zeros(2, 17))

    def forward(self, candidates, rays, candidate_masks, task_mask, source):
        score = self.base(candidates, rays, candidate_masks, task_mask)
        # score is B,J,C; source_joint_bias[source] is C,J.
        bias = self.source_joint_bias[source.long()].transpose(0, 1)
        return score + bias[None]


def task_spec(task_combo, device):
    available = [
        i for i, combo in enumerate(EXPANDED_COMBINATIONS)
        if set(combo).issubset(task_combo)
    ]
    masks = torch.zeros(len(available), 4, device=device, dtype=torch.float32)
    for row, index in enumerate(available):
        masks[row, list(EXPANDED_COMBINATIONS[index])] = 1.0
    task_mask = torch.zeros(4, device=device, dtype=torch.float32)
    task_mask[list(task_combo)] = 1.0
    source = torch.tensor(
        [1 if index >= len(COMBINATIONS) else 0 for index in available],
        device=device, dtype=torch.long,
    )
    return available, masks, task_mask, source


def predict_delta(model, predictions, targets, rays, task_combo):
    available, masks, task_mask, source = task_spec(task_combo, predictions.device)
    candidates = predictions[:, available]
    raw = model(candidates, rays, masks, task_mask, source)
    baseline_local = available.index(COMBINATIONS.index(task_combo))
    true_error = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    baseline_error = true_error[..., baseline_local:baseline_local + 1]
    return raw - raw[..., baseline_local:baseline_local + 1], (
        true_error - baseline_error
    ), true_error, candidates, baseline_local


def task_loss(model, predictions, targets, rays, phase, temperature):
    direct, ght = [], []
    for task in TASK_COMBINATIONS:
        pred, true_delta, true_error, candidates, _ = predict_delta(
            model, predictions, targets, rays, task
        )
        direct.append(training_loss(pred, true_delta, "balanced_rank"))
        if phase == "ght":
            weights = F.softmax(-pred / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            ght.append((expected + 0.05 * fused_error) / 0.01)
    loss = torch.stack(direct).mean()
    return loss + (torch.stack(ght).mean() if ght else 0.0)


def evaluate(model, loader, device, temperature):
    model.eval()
    stores = {s: {m: [] for m in ("baseline", "hard", "soft", "oracle")}
              for s in ("V3", "V4")}
    actions = {s: [] for s in ("V3", "V4")}
    predicted_values, true_values = [], []
    with torch.inference_mode():
        for predictions, targets, rays, batch_actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = batch_actions.numpy()
            for task in TASK_COMBINATIONS:
                stage = f"V{len(task)}"
                pred, true_delta, true_error, candidates, baseline_local = predict_delta(
                    model, predictions, targets, rays, task
                )
                hard_index = pred.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                weights = F.softmax(-pred / temperature, dim=-1)
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                baseline = candidates[:, baseline_local]
                oracle = true_error.min(dim=-1).values
                for mode, pose in (("baseline", baseline), ("hard", hard), ("soft", soft)):
                    stores[stage][mode].append(
                        torch.linalg.vector_norm(pose - targets, dim=-1).cpu().numpy() * 1000.0
                    )
                stores[stage]["oracle"].append(oracle.cpu().numpy() * 1000.0)
                actions[stage].append(batch_actions.copy())
                predicted_values.append(pred.cpu().numpy().reshape(-1))
                true_values.append((true_delta.cpu().numpy() / 0.01).reshape(-1))
    result = {}
    for stage in ("V3", "V4"):
        act = np.concatenate(actions[stage])
        result[stage] = {}
        for mode, chunks in stores[stage].items():
            values = np.concatenate(chunks, axis=0)
            result[stage][mode] = {
                "action_equal_all17_mm": float(np.mean([
                    values[act == action].mean()
                    for action in ACTION_NAMES if np.any(act == action)
                ])),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    pred = np.concatenate(predicted_values)
    true = np.concatenate(true_values)
    finite = np.isfinite(pred) & np.isfinite(true)
    result["delta_pearson"] = float(np.corrcoef(pred[finite], true[finite])[0, 1])
    return result


def main() -> None:
    args = parse_args()
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
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = SourceAwareUtility(mean, std, args.attention_depth).to(device)
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
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model_best.pth.tar"
    phases = ([('direct', 5e-4)] * args.pretrain_epochs
              + [('ght', 1e-4)] * args.finetune_epochs)
    optimizer = None
    previous = None
    best_metric, best_epoch, history = math.inf, -1, []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous = phase
        model.train()
        losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(model, predictions, targets, rays, phase, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device, args.temperature)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {"epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
                  "holdout_selection_metric_mm": float(metric), "holdout": holdout_result}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate(model, test_loader, device, args.temperature)
    result = {
        "method": "E2 pairwise Set Transformer + explicit candidate-source prior",
        "paper_basis": "heterogeneous multi-hypothesis reliability calibration; frozen GHT-style candidate scoring",
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": test_result, "args": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
