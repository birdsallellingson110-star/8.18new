#!/usr/bin/env python3
"""Directly supervise per-joint counterfactual gain for H76 hypotheses."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    COMBINATIONS,
    JOINT_NAMES,
    JointUtilityScorer,
    TASK_COMBINATIONS,
    action_equal,
    load_arrays,
    task_spec,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--variant", choices=("regression", "balanced_rank"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--selection-mode", choices=("hard", "soft"), default="soft"
    )
    parser.add_argument("--holdout-subject", type=int, default=0)
    return parser.parse_args()


def predict_delta(model, predictions, targets, rays, task_combo):
    available, candidate_masks, task_mask = task_spec(task_combo, predictions.device)
    candidates = predictions[:, available]
    raw = model(candidates, rays, candidate_masks, task_mask)
    baseline_local = available.index(COMBINATIONS.index(task_combo))
    predicted_delta = raw - raw[..., baseline_local:baseline_local + 1]
    true_error = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    true_delta = true_error - true_error[..., baseline_local:baseline_local + 1]
    return predicted_delta, true_delta, true_error, candidates, baseline_local


def training_loss(predicted_delta, true_delta, variant):
    scale = 0.01  # 10 mm; keeps direct regression targets numerically stable.
    predicted = predicted_delta
    target = true_delta / scale
    regression = F.smooth_l1_loss(predicted, target, beta=1.0)
    if variant == "regression":
        return regression

    # Directly supervise whether a candidate improves the current full-input
    # H76 hypothesis. Balance helpful/harmful candidates per batch so the
    # trivial "always keep baseline" solution cannot minimize this term.
    nonbaseline = true_delta != 0
    helpful = (true_delta < 0) & nonbaseline
    harmful = (true_delta > 0) & nonbaseline
    positives = helpful.sum().clamp_min(1)
    negatives = harmful.sum().clamp_min(1)
    pos_weight = (negatives / positives).detach().clamp(0.25, 4.0)
    labels = helpful.float()
    classification = F.binary_cross_entropy_with_logits(
        -predicted[nonbaseline], labels[nonbaseline], pos_weight=pos_weight
    )

    # Pairwise ordering is the counterfactual analogue of GHT hypothesis
    # ranking. Ignore near-ties below 1 mm to avoid fitting annotation noise.
    true_pair = true_delta.unsqueeze(-1) - true_delta.unsqueeze(-2)
    pred_pair = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    selected = true_pair.abs() > 0.001
    sign = true_pair.sign()
    ranking = F.softplus(-sign[selected] * pred_pair[selected]).mean()
    return regression + 0.5 * classification + 0.25 * ranking


def evaluate(model, loader, device, soft_temperature=1.0):
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
                predicted_delta, true_delta, true_error, candidates, baseline_local = (
                    predict_delta(model, predictions, targets, rays, task_combo)
                )
                hard_index = predicted_delta.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                soft_weight = F.softmax(
                    -predicted_delta / soft_temperature, dim=-1
                )
                soft = torch.einsum("bjc,bcjd->bjd", soft_weight, candidates)
                baseline = candidates[:, baseline_local]
                oracle_error = true_error.min(dim=-1).values
                for mode, prediction in (
                    ("baseline", baseline), ("hard", hard), ("soft", soft)
                ):
                    error = torch.linalg.vector_norm(
                        prediction - targets, dim=-1
                    ).cpu().numpy() * 1000.0
                    stores[stage][mode].append(error)
                stores[stage]["oracle"].append(oracle_error.cpu().numpy() * 1000.0)
                actions_by_stage[stage].append(batch_actions.copy())
                predicted_values.append(predicted_delta.cpu().numpy().reshape(-1))
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


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    if args.holdout_subject:
        holdout = train["subjects"] == args.holdout_subject
        if not np.any(holdout) or np.all(holdout):
            raise ValueError("holdout subject is absent or consumes all training data")
        holdout_protocol = f"leave-subject-{args.holdout_subject}-out"
    else:
        holdout = train["group_indices"] % 10 == 0
        holdout_protocol = "group-index-modulo-10"
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = JointUtilityScorer(mean, std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=generator, num_workers=args.workers,
        pin_memory=True,
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
    checkpoint = output_dir / "model_best.pth.tar"
    history = []
    best_metric = math.inf
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for task_combo in TASK_COMBINATIONS:
                predicted, true, _, _, _ = predict_delta(
                    model, predictions, targets, rays, task_combo
                )
                loss = loss + training_loss(predicted, true, args.variant)
            loss = loss / len(TASK_COMBINATIONS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device)
        # Select the checkpoint only on training-subject holdout using the
        # predeclared inference rule. S9/S11 remains final-once evaluation.
        metric = 0.5 * (
            holdout_result["V3"][args.selection_mode]["action_equal_all17_mm"]
            + holdout_result["V4"][args.selection_mode]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric,
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "variant": args.variant, "epoch": epoch,
            }, checkpoint)

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate(model, test_loader, device)
    result = {
        "variant": args.variant,
        "method": "direct per-joint counterfactual delta supervision",
        "train_subjects": sorted(set(train["subjects"].tolist())),
        "test_subjects": sorted(set(validation["subjects"].tolist())),
        "best_epoch": best_epoch,
        "selection_mode": args.selection_mode,
        "holdout_protocol": holdout_protocol,
        "history": history,
        "S9_S11_final_once": test_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
