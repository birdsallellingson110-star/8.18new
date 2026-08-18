#!/usr/bin/env python3
"""Train E2 utility scoring after adding deterministic pairwise ray hypotheses."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    COMBINATIONS,
    JOINT_NAMES,
    TASK_COMBINATIONS,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


PAIRWISE_COMBINATIONS = tuple(itertools.combinations(range(4), 2))
EXPANDED_COMBINATIONS = COMBINATIONS + PAIRWISE_COMBINATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_expanded(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if arrays["predictions"].shape[1:] != (17, 17, 3):
        raise ValueError(f"bad expanded prediction shape {arrays['predictions'].shape}")
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate train group indices")
    return arrays


def task_spec_expanded(task_combo: tuple[int, ...], device: torch.device):
    available = [
        index for index, combo in enumerate(EXPANDED_COMBINATIONS)
        if set(combo).issubset(task_combo)
    ]
    masks = torch.zeros(len(available), 4, device=device, dtype=torch.float32)
    for row, candidate_index in enumerate(available):
        masks[row, list(EXPANDED_COMBINATIONS[candidate_index])] = 1.0
    task_mask = torch.zeros(4, device=device, dtype=torch.float32)
    task_mask[list(task_combo)] = 1.0
    return available, masks, task_mask


def predict_delta_expanded(model, predictions, targets, rays, task_combo):
    available, masks, task_mask = task_spec_expanded(task_combo, predictions.device)
    candidates = predictions[:, available]
    raw = model(candidates, rays, masks, task_mask)
    baseline_local = available.index(EXPANDED_COMBINATIONS.index(task_combo))
    true_error = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    baseline_error = true_error[..., baseline_local:baseline_local + 1]
    return raw - raw[..., baseline_local:baseline_local + 1], (
        true_error - baseline_error
    ), true_error, candidates, baseline_local


def task_loss_expanded(model, predictions, targets, rays, phase, temperature):
    direct_losses = []
    ght_losses = []
    for task_combo in TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, _ = predict_delta_expanded(
            model, predictions, targets, rays, task_combo
        )
        direct_losses.append(training_loss(predicted, true_delta, "balanced_rank"))
        if phase == "ght":
            weights = F.softmax(-predicted / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            ght_losses.append((expected + 0.05 * fused_error) / 0.01)
    loss = torch.stack(direct_losses).mean()
    if ght_losses:
        loss = loss + torch.stack(ght_losses).mean()
    return loss


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def evaluate_expanded(model, loader, device, temperature):
    model.eval()
    stores = {
        stage: {mode: [] for mode in ("baseline", "hard", "soft", "oracle")}
        for stage in ("V3", "V4")
    }
    actions_by_stage = {"V3": [], "V4": []}
    predicted_values = []
    true_values = []
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = actions.numpy()
            for task_combo in TASK_COMBINATIONS:
                stage = "V3" if len(task_combo) == 3 else "V4"
                predicted, true_delta, true_error, candidates, baseline_local = (
                    predict_delta_expanded(model, predictions, targets, rays, task_combo)
                )
                hard_index = predicted.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                weights = F.softmax(-predicted / temperature, dim=-1)
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                baseline = candidates[:, baseline_local]
                oracle = true_error.min(dim=-1).values
                for mode, pose in (
                    ("baseline", baseline), ("hard", hard), ("soft", soft)
                ):
                    error = torch.linalg.vector_norm(
                        pose - targets, dim=-1
                    ).cpu().numpy() * 1000.0
                    stores[stage][mode].append(error)
                stores[stage]["oracle"].append(oracle.cpu().numpy() * 1000.0)
                actions_by_stage[stage].append(batch_actions.copy())
                predicted_values.append(predicted.cpu().numpy().reshape(-1))
                true_values.append((true_delta.cpu().numpy() / 0.01).reshape(-1))

    result = {}
    for stage in ("V3", "V4"):
        stage_actions = np.concatenate(actions_by_stage[stage])
        result[stage] = {}
        for mode, chunks in stores[stage].items():
            values = np.concatenate(chunks, axis=0)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
            if mode in ("baseline", "hard", "soft"):
                result[stage][mode]["per_joint_mm"] = {
                    name: action_equal(values[:, index], stage_actions)
                    for index, name in enumerate(JOINT_NAMES)
                }
    predicted = np.concatenate(predicted_values)
    true = np.concatenate(true_values)
    finite = np.isfinite(predicted) & np.isfinite(true)
    result["delta_pearson"] = float(np.corrcoef(predicted[finite], true[finite])[0, 1])
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
    model = SetTransformerJointUtility(mean, std, args.attention_depth).to(device)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    phases = (
        [("direct", 5e-4)] * args.pretrain_epochs
        + [("ght", 1e-4)] * args.finetune_epochs
    )
    optimizer = None
    previous_phase = None
    best_metric = math.inf
    best_epoch = -1
    history = []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous_phase:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous_phase = phase
        model.train()
        losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss_expanded(
                model, predictions, targets, rays, phase, args.temperature
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate_expanded(
            model, holdout_loader, device, args.temperature
        )
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
            best_metric = float(metric)
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature,
            }, checkpoint_path)

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_expanded(model, test_loader, device, args.temperature)
    result = {
        "method": "E2 Set Transformer utility + deterministic pairwise ray hypotheses",
        "paper_basis": "GHT-style multi-hypothesis scoring; pairwise ray hypotheses are a frozen geometry extension",
        "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS],
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": test_result,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
