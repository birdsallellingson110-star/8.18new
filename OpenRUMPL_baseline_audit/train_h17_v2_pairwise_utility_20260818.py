#!/usr/bin/env python3
"""H17: pairwise V2 candidate utility residual on the current C2 pool.

Only the two hypotheses available for a fixed two-view task are scored: the
frozen H76 RUMPL hypothesis and the confidence-weighted triangulation
hypothesis.  The residual head is zero initialized, so the first evaluation is
exactly the calibrated E2-C2 baseline.  Features are label-free coordinates,
confidence and ray geometry; labels enter only the ordinary training error
delta/ranking loss on S1/S5/S6/S7.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES
from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS


V2_TASKS = tuple(c for c in ORIGINAL_COMBINATIONS if len(c) == 2)
TASK_INDICES = tuple(ORIGINAL_COMBINATIONS.index(c) for c in V2_TASKS)
TEMPERATURE = 0.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-scores", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-scores", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ght-weight", type=float, default=0.1)
    p.add_argument("--identity-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def load_cache(path: str) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=False)
    required = {"predictions", "targets", "rays", "actions", "subjects"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    arrays = {key: source[key] for key in required}
    if arrays["predictions"].shape[1:] != (22, 17, 3):
        raise ValueError(f"bad prediction shape {arrays['predictions'].shape}")
    return arrays


class PairwiseUtilityResidual(nn.Module):
    def __init__(self, feature_dim: int = 11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU6(),
            nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU6(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # B,C,J,F -> B,J,C
        return self.network(features).squeeze(-1).permute(0, 2, 1)


def task_features(
    predictions: torch.Tensor, rays: torch.Tensor, scores: torch.Tensor,
    task_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # For a V2 task, the original and confidence hypotheses are exactly two
    # candidate entries: task_index and task_index + 11.
    indices = [task_index, task_index + 11]
    candidate = predictions[:, indices]  # B,C,J,3
    root_relative = candidate - candidate[..., :1, :]
    consensus = candidate.mean(dim=1, keepdim=True)
    displacement = (candidate - consensus) / 0.1
    displacement_norm = torch.linalg.vector_norm(
        displacement, dim=-1, keepdim=True
    )
    combo = V2_TASKS[TASK_INDICES.index(task_index)]
    mask = torch.zeros(2, 4, device=predictions.device, dtype=predictions.dtype)
    mask[:, list(combo)] = 1.0
    confidence = rays[..., 6]  # B,J,V
    confidence = (
        confidence[:, None] * mask[None, :, None]
    ).sum(dim=-1) / len(combo)
    confidence = confidence.unsqueeze(-1)

    direction = F.normalize(rays[..., :3], dim=-1)
    point = rays[..., 3:6]
    offset = candidate[:, :, :, None, :] - point[:, None, :, :, :]
    perpendicular = torch.linalg.vector_norm(
        torch.cross(offset, direction[:, None], dim=-1), dim=-1
    )
    ray_residual = (
        perpendicular * mask[None, :, None, :]
    ).sum(dim=-1) / len(combo)
    ray_residual = torch.log1p(ray_residual.unsqueeze(-1) / 0.005)

    score = scores[:, task_index][:, :, indices].permute(0, 2, 1).unsqueeze(-1) / 5.0
    score_delta = score - score[:, :1]
    features = torch.cat(
        (root_relative, displacement, displacement_norm, confidence,
         ray_residual, score, score_delta), dim=-1
    )
    return features, candidate


def forward_task(
    model: PairwiseUtilityResidual | None, predictions: torch.Tensor,
    targets: torch.Tensor, rays: torch.Tensor, scores: torch.Tensor,
    task_index: int,
):
    features, candidate = task_features(predictions, rays, scores, task_index)
    correction = (
        model(features).permute(0, 2, 1) if model is not None else
        torch.zeros(predictions.shape[0], 2, 17, device=predictions.device)
    )
    base = scores[:, task_index][
        :, :, [task_index, task_index + 11]
    ].permute(0, 2, 1)
    total = base + correction
    predicted_delta = (total - total[:, :1]).permute(0, 2, 1)
    errors = torch.linalg.vector_norm(
        candidate - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    true_delta = errors - errors[..., :1]
    weights = F.softmax(-predicted_delta / TEMPERATURE, dim=-1)
    fused = torch.einsum("bjc,bcjd->bjd", weights, candidate)
    fused_error = torch.linalg.vector_norm(fused - targets, dim=-1)
    return predicted_delta, true_delta, errors, fused_error, errors[..., 0]


def loss_batch(model, pred, target, rays, scores, args):
    losses, risk, identity = [], [], []
    for task_index in TASK_INDICES:
        predicted, true_delta, errors, fused_error, baseline_error = forward_task(
            model, pred, target, rays, scores, task_index
        )
        losses.append(training_loss(predicted, true_delta, "balanced_rank"))
        weights = F.softmax(-predicted / TEMPERATURE, dim=-1)
        expected = (weights * errors).sum(dim=-1).mean()
        risk.append((expected + 0.05 * fused_error.mean()) / 0.01)
        identity.append(
            args.identity_weight * F.relu(fused_error - baseline_error).mean() / 0.01
        )
    return torch.stack(losses).mean() + args.ght_weight * torch.stack(risk).mean() \
        + torch.stack(identity).mean()


def evaluate(model, arrays, score_memmap, indices, device, batch_size):
    # Keep each of the six camera-pair tasks separate; action-equal MPJPE is
    # computed per task and then averaged, matching the main E2 evaluator.
    task_values = [
        {"baseline": [], "soft": [], "actions": []}
        for _ in TASK_INDICES
    ]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            pred = torch.from_numpy(arrays["predictions"][rows]).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(arrays["targets"][rows]).to(
                device=device, dtype=torch.float32
            )
            rays = torch.from_numpy(arrays["rays"][rows]).to(
                device=device, dtype=torch.float32
            )
            scores = torch.from_numpy(np.asarray(score_memmap[rows])).to(
                device=device, dtype=torch.float32
            )
            batch_actions = arrays["actions"][rows].copy()
            for task_pos, task_index in enumerate(TASK_INDICES):
                _, _, _, fused_error, baseline_error = forward_task(
                    model, pred, target, rays, scores, task_index
                )
                task_values[task_pos]["baseline"].append(
                    baseline_error.cpu().numpy() * 1000.0
                )
                task_values[task_pos]["soft"].append(
                    fused_error.cpu().numpy() * 1000.0
                )
    result = {"V2": {}}
    per_task = []
    for task_pos, task in enumerate(task_values):
        task_actions = []
        # The action labels are identical for each task and can be reconstructed
        # from the row batches retained alongside the values.
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            task_actions.append(arrays["actions"][rows].copy())
        action = np.concatenate(task_actions)
        task_result = {}
        for name in ("baseline", "soft"):
            vals = np.concatenate(task[name], axis=0)
            task_result[name] = {
                "action_equal_all17_mm": action_equal(vals, action),
                "frame_weighted_all17_mm": float(vals.mean()),
                "per_joint_mm": {
                    joint: action_equal(vals[:, j], action)
                    for j, joint in enumerate(JOINT_NAMES)
                },
            }
        task_result["task"] = list(V2_TASKS[task_pos])
        per_task.append(task_result)
    for name in ("baseline", "soft"):
        metrics = [task[name]["action_equal_all17_mm"] for task in per_task]
        frames = [task[name]["frame_weighted_all17_mm"] for task in per_task]
        joints = {
            joint: float(np.mean([task[name]["per_joint_mm"][joint]
                                  for task in per_task]))
            for joint in JOINT_NAMES
        }
        result["V2"][name] = {
            "action_equal_all17_mm": float(np.mean(metrics)),
            "frame_weighted_all17_mm": float(np.mean(frames)),
            "per_joint_mm": joints,
        }
    result["V2"]["per_task"] = per_task
    result["V2"]["delta_soft_minus_baseline_mm"] = (
        result["V2"]["soft"]["action_equal_all17_mm"]
        - result["V2"]["baseline"]["action_equal_all17_mm"]
    )
    result["mean_soft_mm"] = result["V2"]["soft"]["action_equal_all17_mm"]
    result["samples"] = int(len(indices))
    return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_cache(args.train_cache)
    validation = load_cache(args.validation_cache)
    train_scores = np.load(args.train_scores, mmap_mode="r")
    validation_scores = np.load(args.validation_scores, mmap_mode="r")
    if train_scores.shape[1:] != (11, 17, 22):
        raise ValueError(f"bad train score shape {train_scores.shape}")
    if validation_scores.shape[1:] != (11, 17, 22):
        raise ValueError(f"bad validation score shape {validation_scores.shape}")
    train_idx = np.flatnonzero(np.isin(train["subjects"], [1, 5, 6, 7]))
    holdout_idx = np.flatnonzero(train["subjects"] == 8)
    val_idx = np.arange(len(validation["targets"]), dtype=np.int64)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "H17 pairwise V2 candidate utility residual",
        "candidate_generator": "frozen C2 H76 + confidence-weighted triangulation",
        "features": "root-relative candidate pose, pair displacement, confidence, ray residual, frozen E2 score",
        "train_subjects": [1, 5, 6, 7], "holdout_subjects": [8],
        "v2_tasks": [list(x) for x in V2_TASKS],
        "identity_init": True, "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline_holdout = evaluate(
        None, train, train_scores, holdout_idx, device, args.batch_size
    )
    baseline_val = evaluate(
        None, validation, validation_scores, val_idx, device, args.batch_size
    )
    model = PairwiseUtilityResidual().to(device)
    best_metric = baseline_holdout["mean_soft_mm"]
    best_epoch = -1
    history = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            pred = torch.from_numpy(train["predictions"][rows]).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(train["targets"][rows]).to(
                device=device, dtype=torch.float32
            )
            rays = torch.from_numpy(train["rays"][rows]).to(
                device=device, dtype=torch.float32
            )
            scores = torch.from_numpy(np.asarray(train_scores[rows])).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(model, pred, target, rays, scores, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout = evaluate(
            model, train, train_scores, holdout_idx, device, args.batch_size
        )
        metric = holdout["mean_soft_mm"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "holdout_selection_metric_mm": metric, "holdout": holdout}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "best_holdout_metric_mm": best_metric},
                       out / "model_best.pth.tar")
    if best_epoch >= 0:
        checkpoint = torch.load(out / "model_best.pth.tar", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        final_val = evaluate(model, validation, validation_scores, val_idx,
                              device, args.batch_size)
    else:
        final_val = baseline_val
    result = {
        **manifest, "baseline_holdout": baseline_holdout,
        "baseline_validation": baseline_val, "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric, "history": history,
        "S9_S11_final_once": final_val,
        "decision": (
            "retain V2 specialist for combined E2 table"
            if best_epoch >= 0 and final_val["mean_soft_mm"] < baseline_val["mean_soft_mm"] - 1.0
            else "no V2 specialist gain; retain calibrated E2-C2 and stop H17"
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps({"S9_S11_final_once": final_val,
                      "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
