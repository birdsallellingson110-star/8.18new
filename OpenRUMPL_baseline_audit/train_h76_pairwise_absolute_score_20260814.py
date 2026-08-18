#!/usr/bin/env python3
"""Train an absolute candidate-error scorer on the frozen H76 hypothesis pool.

The GHT paper trains a ScoreNN to rank geometric hypotheses by their absolute
3-D reconstruction quality.  The current E2 model instead predicts a
counterfactual error *delta* relative to the full-view H76 candidate.  This
script is a controlled objective-only comparison: candidate generation,
camera/ray inputs, Set Transformer, split, and final soft fusion stay fixed.

No GT is used at inference.  GT is used only to form the training target, as in
the supervised GHT score-network training protocol.
"""

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

import train_h76_pairwise_set_transformer_20260812 as base
from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


# The learned-triangulation extension cache contains the original 11 H76
# hypotheses, six pairwise hypotheses, and five learned ray hypotheses.
LEARNED_COMBINATIONS = tuple(
    combo for combo in base.COMBINATIONS if len(combo) >= 3
)
EXPANDED_COMBINATIONS = base.EXPANDED_COMBINATIONS + LEARNED_COMBINATIONS
base.EXPANDED_COMBINATIONS = EXPANDED_COMBINATIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--attention-depth", type=int, default=2)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--target-scale-mm", type=float, default=10.0)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--loss-mode", choices=("absolute", "absolute_rank"),
                    default="absolute_rank")
    p.add_argument(
        "--include-v2", action="store_true",
        help="also train/evaluate all six two-view tasks; default keeps the H11B V3/V4 protocol",
    )
    return p.parse_args()


def task_combinations(include_v2: bool) -> tuple[tuple[int, ...], ...]:
    """Return the task protocol without changing the historical H11B default."""
    if include_v2:
        return tuple(itertools.combinations(range(4), 2)) + TASK_COMBINATIONS
    return TASK_COMBINATIONS


def load_expanded(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0)
              for key in keys}
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate train group indices")
    expected = (len(EXPANDED_COMBINATIONS), 17, 3)
    if arrays["predictions"].shape[1:] != expected:
        raise ValueError(
            f"bad expanded prediction shape {arrays['predictions'].shape}; "
            f"expected {expected}"
        )
    return arrays


def predict_absolute(model, predictions, targets, rays, task_combo):
    available, masks, task_mask = base.task_spec_expanded(
        task_combo, predictions.device
    )
    candidates = predictions[:, available]
    predicted = model(candidates, rays, masks, task_mask)
    true_error = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    return predicted, true_error, candidates


def absolute_loss(model, predictions, targets, rays, task_combo, phase, args):
    predicted, true_error, candidates = predict_absolute(
        model, predictions, targets, rays, task_combo
    )
    scale_m = args.target_scale_mm / 1000.0
    target = true_error / scale_m
    loss = F.smooth_l1_loss(predicted, target, beta=1.0)
    if args.loss_mode == "absolute_rank":
        true_pair = true_error.unsqueeze(-1) - true_error.unsqueeze(-2)
        pred_pair = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
        selected = true_pair.abs() > (0.001 / scale_m)
        if selected.any():
            rank = F.softplus(-true_pair[selected].sign() * pred_pair[selected]).mean()
            loss = loss + args.ranking_weight * rank
    if phase == "ght":
        # GHT's expected-hypothesis and weighted-estimate losses, with lower
        # predicted absolute error receiving larger soft weights.
        weights = F.softmax(-predicted / args.temperature, dim=-1)
        expected = (weights * true_error).sum(dim=-1).mean()
        fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
        fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
        loss = loss + (expected + 0.05 * fused_error) / scale_m
    return loss


def train_one_epoch(model, loader, device, phase, args, task_combos):
    model.train()
    optimizer = train_one_epoch.optimizer
    values = []
    for predictions, targets, rays, _ in loader:
        predictions = predictions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        rays = rays.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([
            absolute_loss(model, predictions, targets, rays, combo, phase, args)
            for combo in task_combos
        ]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        values.append(float(loss.item()))
    return float(np.mean(values))


def evaluate_with_v2(model, loader, device, temperature, include_v2):
    """Extend the historical expanded evaluator with the six V2 tasks.

    The V3/V4 numbers are obtained from the unchanged official-style evaluator;
    V2 uses exactly the same candidate scoring, hard/soft fusion and
    action-equal aggregation.  Keeping the old path intact makes H11B and H11C
    directly comparable.
    """
    result = base.evaluate_expanded(model, loader, device, temperature)
    if not include_v2:
        return result

    stages = tuple(itertools.combinations(range(4), 2))
    stores = {
        mode: [] for mode in ("baseline", "hard", "soft", "oracle")
    }
    actions_by_task = []
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = actions.numpy()
            for task_combo in stages:
                available, _, _ = base.task_spec_expanded(
                    task_combo, predictions.device
                )
                predicted, true_error, candidates = predict_absolute(
                    model, predictions, targets, rays, task_combo
                )
                hard_index = predicted.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                weights = F.softmax(-predicted / temperature, dim=-1)
                soft = torch.einsum(
                    "bjc,bcjd->bjd", weights, candidates
                )
                baseline_local = available.index(
                    EXPANDED_COMBINATIONS.index(task_combo)
                )
                baseline = candidates[:, baseline_local]
                oracle = true_error.min(dim=-1).values
                for mode, pose in (
                    ("baseline", baseline), ("hard", hard), ("soft", soft)
                ):
                    stores[mode].append(
                        torch.linalg.vector_norm(
                            pose - targets, dim=-1
                        ).cpu().numpy() * 1000.0
                    )
                stores["oracle"].append(oracle.cpu().numpy() * 1000.0)
                actions_by_task.append(batch_actions.copy())

    stage_actions = np.concatenate(actions_by_task)
    v2 = {}
    for mode, chunks in stores.items():
        values = np.concatenate(chunks, axis=0)
        v2[mode] = {
            "action_equal_all17_mm": base.action_equal(values, stage_actions),
            "frame_weighted_all17_mm": float(values.mean()),
        }
        if mode in ("baseline", "hard", "soft"):
            v2[mode]["per_joint_mm"] = {
                name: base.action_equal(values[:, index], stage_actions)
                for index, name in enumerate(base.JOINT_NAMES)
            }
    result["V2"] = v2
    return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    task_combos = task_combinations(args.include_v2)

    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    expected = (len(EXPANDED_COMBINATIONS), 17, 3)
    if validation["predictions"].shape[1:] != expected:
        raise ValueError(
            f"bad validation prediction shape {validation['predictions'].shape}; "
            f"expected {expected}"
        )

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
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model_best.pth.tar"

    phases = (["direct"] * args.pretrain_epochs
              + ["ght"] * args.finetune_epochs)
    optimizer = None
    previous = None
    best_metric, best_epoch, history = math.inf, -1, []
    for epoch, phase in enumerate(phases):
        if phase != previous:
            lr = 5e-4 if phase == "direct" else 1e-4
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                          weight_decay=1e-4)
            train_one_epoch.optimizer = optimizer
            previous = phase
        train_loss = train_one_epoch(
            model, train_loader, device, phase, args, task_combos
        )
        holdout_result = evaluate_with_v2(
            model, holdout_loader, device, args.temperature, args.include_v2
        )
        stages = ("V2", "V3", "V4") if args.include_v2 else ("V3", "V4")
        metric = float(np.mean([
            holdout_result[stage]["soft"]["action_equal_all17_mm"]
            for stage in stages
        ]))
        record = {
            "epoch": epoch, "phase": phase, "train_loss": train_loss,
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(),
                "mean": mean, "std": std,
                "attention_depth": args.attention_depth,
                "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature,
                "loss_mode": args.loss_mode,
                "epoch": epoch,
            }, checkpoint)

    best = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_with_v2(
        model, test_loader, device, args.temperature, args.include_v2
    )
    result = {
        "method": (
            "absolute candidate-error ScoreNN on pairwise+learned H76 pool"
            + ("; V2 task included" if args.include_v2 else "")
        ),
        "paper_basis": "Generalizable Human Pose Triangulation (CVPR 2022) absolute hypothesis scoring",
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "task_combinations": [list(c) for c in task_combos],
        "history": history, "S9_S11_final_once": test_result,
        "args": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
