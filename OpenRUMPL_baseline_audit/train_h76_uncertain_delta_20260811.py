#!/usr/bin/env python3
"""Heteroscedastic counterfactual-delta scoring (UPose3D/LOSTU ablation)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import (
    ArrayDataset, COMBINATIONS, JOINT_NAMES, JointUtilityScorer,
    TASK_COMBINATIONS, action_equal, load_arrays, task_spec,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--variant", choices=("nll", "nll_rank"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_uncertain_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = JointUtilityScorer(
        checkpoint["mean"], checkpoint["std"], output_dim=2
    )
    source = checkpoint["state_dict"]
    target = model.state_dict()
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            target[key] = value
    # Preserve the learned C2 mean-delta output exactly in channel 0; initialize
    # log variance to zero (unit variance on the normalized 10 mm scale).
    target["utility.4.weight"][0] = source["utility.4.weight"][0]
    target["utility.4.bias"][0] = source["utility.4.bias"][0]
    target["utility.4.weight"][1].zero_()
    target["utility.4.bias"][1].zero_()
    model.load_state_dict(target, strict=True)
    return model.to(device), checkpoint


def uncertain_outputs(model, predictions, targets, rays, task_combo):
    available, candidate_masks, task_mask = task_spec(task_combo, predictions.device)
    candidates = predictions[:, available]
    output = model(candidates, rays, candidate_masks, task_mask)
    baseline_local = available.index(COMBINATIONS.index(task_combo))
    mean = output[..., 0] - output[..., baseline_local:baseline_local + 1, 0]
    log_variance = output[..., 1].clamp(-5, 5)
    error = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    true_delta = (error - error[..., baseline_local:baseline_local + 1]) / 0.01
    return mean, log_variance, true_delta, error, candidates, baseline_local


def loss_for(mean, log_variance, target, baseline_local, variant):
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., baseline_local] = False
    residual = mean[mask] - target[mask]
    logvar = log_variance[mask]
    nll = 0.5 * (torch.exp(-logvar) * residual.square() + logvar).mean()
    if variant == "nll":
        return nll
    helpful = target[mask] < 0
    harmful = target[mask] > 0
    pos_weight = (
        harmful.sum().clamp_min(1) / helpful.sum().clamp_min(1)
    ).detach().clamp(0.25, 4.0)
    classification = F.binary_cross_entropy_with_logits(
        -mean[mask], helpful.float(), pos_weight=pos_weight
    )
    true_pair = target.unsqueeze(-1) - target.unsqueeze(-2)
    mean_pair = mean.unsqueeze(-1) - mean.unsqueeze(-2)
    selected = true_pair.abs() > 0.1  # 1 mm on normalized 10 mm targets.
    ranking = F.softplus(-true_pair.sign()[selected] * mean_pair[selected]).mean()
    return nll + 0.5 * classification + 0.25 * ranking


def evaluate(model, loader, device):
    model.eval()
    stores = {
        stage: {mode: [] for mode in ("baseline", "mean_soft", "risk_soft", "oracle")}
        for stage in ("V3", "V4")
    }
    action_store = {"V3": [], "V4": []}
    nll_values = []
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for task_combo in TASK_COMBINATIONS:
                stage = "V3" if len(task_combo) == 3 else "V4"
                mean, logvar, true_delta, error, candidates, baseline_local = (
                    uncertain_outputs(model, predictions, targets, rays, task_combo)
                )
                mean_risk = mean
                conservative_risk = mean + torch.exp(0.5 * logvar)
                conservative_risk[..., baseline_local] = 0.0
                for mode, risk in (
                    ("mean_soft", mean_risk), ("risk_soft", conservative_risk)
                ):
                    weights = F.softmax(-risk, dim=-1)
                    fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                    value = torch.linalg.vector_norm(
                        fused - targets, dim=-1
                    ).cpu().numpy() * 1000
                    stores[stage][mode].append(value)
                stores[stage]["baseline"].append(
                    error[..., baseline_local].cpu().numpy() * 1000
                )
                stores[stage]["oracle"].append(
                    error.min(dim=-1).values.cpu().numpy() * 1000
                )
                action_store[stage].append(actions.numpy())
                mask = torch.ones_like(true_delta, dtype=torch.bool)
                mask[..., baseline_local] = False
                nll_values.append(float((
                    0.5 * (
                        torch.exp(-logvar[mask])
                        * (mean[mask] - true_delta[mask]).square()
                        + logvar[mask]
                    ).mean()
                ).item()))
    result = {"nll": float(np.mean(nll_values))}
    for stage in ("V3", "V4"):
        actions = np.concatenate(action_store[stage])
        result[stage] = {}
        for mode, chunks in stores[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices, holdout_indices = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    model, initial = load_uncertain_model(args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(0),
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
    best_metric, best_epoch, history = math.inf, -1, []
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
                mean, logvar, target, _, _, baseline_local = uncertain_outputs(
                    model, predictions, targets, rays, task_combo
                )
                loss = loss + loss_for(
                    mean, logvar, target, baseline_local, args.variant
                )
            loss = loss / len(TASK_COMBINATIONS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device)
        metric = 0.5 * (
            holdout_result["V3"]["risk_soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["risk_soft"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric, "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": initial["mean"],
                "std": initial["std"], "variant": args.variant,
                "epoch": epoch, "output_dim": 2,
            }, checkpoint_path)
    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate(model, test_loader, device)
    payload = {
        "variant": args.variant, "best_epoch": best_epoch,
        "paper_basis": "UPose3D uncertainty MLE; LOSTU measurement uncertainty",
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "history": history, "S9_S11_final_once": test_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
