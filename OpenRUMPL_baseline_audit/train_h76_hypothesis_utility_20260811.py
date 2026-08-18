#!/usr/bin/env python3
"""Train paper-backed hypothesis scorers on frozen H76 subset predictions.

``pose`` is the Generalizable Human Pose Triangulation control: one score per
whole-pose hypothesis from a 3-layer ReLU6 MLP. ``joint`` extends the same
hypothesis-scoring principle to per-joint counterfactual utility, conditioned
on candidate pose context and permutation-invariant ray residual statistics.
Neither variant changes 2D points or the frozen H76 candidate generator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


JOINT_NAMES = (
    "root", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle",
    "belly", "neck", "nose", "head", "lshoulder", "lelbow", "lwrist",
    "rshoulder", "relbow", "rwrist",
)
ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet",
    6: "Phone", 7: "Photo", 8: "Pose", 9: "Purchase",
    10: "Sitting", 11: "SittingDown", 12: "Smoke", 13: "Wait",
    14: "WalkDog", 15: "Walk", 16: "WalkTwo",
}
COMBINATIONS = tuple(
    combo
    for views in (2, 3, 4)
    for combo in itertools.combinations(range(4), views)
)
TASK_COMBINATIONS = tuple(itertools.combinations(range(4), 3)) + ((0, 1, 2, 3),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--variant", choices=("pose", "joint"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class ArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source = self.indices[index]
        return (
            torch.from_numpy(self.arrays["predictions"][source]),
            torch.from_numpy(self.arrays["targets"][source]),
            torch.from_numpy(self.arrays["rays"][source]),
            int(self.arrays["actions"][source]),
        )


def load_arrays(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate train group indices")
    if arrays["predictions"].shape[1:] != (11, 17, 3):
        raise ValueError(f"bad prediction shape {arrays['predictions'].shape}")
    return arrays


class PoseHypothesisScorer(nn.Module):
    """Structural adaptation of GHT's official 51-50-50-50-1 ScoreNN."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std.clamp_min(1e-6))
        self.network = nn.Sequential(
            nn.Linear(51, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 1),
        )

    def forward(
        self,
        candidates: torch.Tensor,
        rays: torch.Tensor,
        candidate_masks: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        del rays, candidate_masks, task_mask
        normalized = (candidates - self.pose_mean) / self.pose_std
        normalized = normalized - normalized[:, :, :1]
        return self.network(normalized.flatten(2)).squeeze(-1)


def masked_statistics(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return masked mean/min/max for values B,C,J,V and mask C,V."""
    expanded = mask[None, :, None, :]
    count = expanded.sum(dim=-1).clamp_min(1.0)
    mean = (values * expanded).sum(dim=-1) / count
    large = torch.finfo(values.dtype).max
    minimum = values.masked_fill(~expanded.bool(), large).min(dim=-1).values
    maximum = values.masked_fill(~expanded.bool(), -large).max(dim=-1).values
    has_value = expanded.any(dim=-1)
    minimum = torch.where(has_value, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
    return torch.stack((mean, minimum, maximum), dim=-1)


class JointUtilityScorer(nn.Module):
    """GHT-style scoring extended to joint-level, geometry-aware utility."""

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, output_dim: int = 1
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std.clamp_min(1e-6))
        self.pose_encoder = nn.Sequential(
            nn.Linear(51, 64), nn.ReLU6(),
            nn.Linear(64, 64), nn.ReLU6(),
        )
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        # local xyz(3), candidate-root xyz(3), consensus delta xyz+norm(4),
        # included/excluded residual stats(6), confidence stats(6),
        # view fraction(1), normal-matrix spectrum(3), pose context(64), joint(16).
        self.utility = nn.Sequential(
            nn.Linear(106, 96), nn.ReLU6(),
            nn.Linear(96, 64), nn.ReLU6(),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        candidates: torch.Tensor,
        rays: torch.Tensor,
        candidate_masks: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, joints, _ = candidates.shape
        normalized = (candidates - self.pose_mean) / self.pose_std
        root_relative = normalized - normalized[:, :, :1]
        context = self.pose_encoder(root_relative.flatten(2))
        context = context[:, :, None].expand(-1, -1, joints, -1)

        consensus = candidates.mean(dim=1, keepdim=True)
        delta = candidates - consensus
        delta_feature = torch.cat(
            (delta / 0.1, torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / 0.1),
            dim=-1,
        )
        local = torch.cat(
            (root_relative, normalized[:, :, :1].expand(-1, -1, joints, -1)),
            dim=-1,
        )

        direction = F.normalize(rays[..., :3], dim=-1)
        point = rays[..., 3:6]
        offset = candidates[:, :, :, None, :] - point[:, None]
        residual = torch.linalg.vector_norm(
            torch.cross(offset, direction[:, None], dim=-1), dim=-1
        )
        residual = torch.log1p(residual / 0.005)
        input_mask = task_mask[None, None, None, :].bool()
        included = candidate_masks.bool()
        excluded = (~included) & task_mask[None].bool()
        included_residual = masked_statistics(residual, included)
        excluded_residual = masked_statistics(residual, excluded)

        confidence = rays[..., 6].clamp(0, 1)
        confidence = confidence[:, None].expand(-1, count, -1, -1)
        included_conf = masked_statistics(confidence, included)
        excluded_conf = masked_statistics(confidence, excluded)

        projection = (
            torch.eye(3, device=rays.device, dtype=rays.dtype)
            - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        )
        weight = rays[..., 6:7].clamp(0, 1) + 0.05
        normal_per_view = weight.unsqueeze(-1) * projection
        candidate_normal = torch.einsum(
            "cv,bjvxy->bcjxy", candidate_masks, normal_per_view
        )
        eigenvalues = torch.linalg.eigvalsh(candidate_normal).clamp_min(1e-7)
        spectrum = torch.log(eigenvalues / eigenvalues.sum(dim=-1, keepdim=True))
        view_fraction = (
            candidate_masks.sum(dim=-1)[None, :, None, None] / task_mask.sum()
        ).expand(batch, -1, joints, -1)

        joint = self.joint_embedding[None, None].expand(batch, count, -1, -1)
        features = torch.cat(
            (
                local, delta_feature, included_residual, excluded_residual,
                included_conf, excluded_conf, view_fraction, spectrum,
                context, joint,
            ),
            dim=-1,
        )
        if features.shape[-1] != 106:
            raise RuntimeError(f"joint utility feature size {features.shape[-1]}")
        logits = self.utility(features)
        if self.output_dim == 1:
            return logits.squeeze(-1).permute(0, 2, 1)  # B,J,C
        return logits.permute(0, 2, 1, 3)  # B,J,C,K


def task_spec(task_combo: tuple[int, ...], device: torch.device):
    available = [
        index for index, combo in enumerate(COMBINATIONS)
        if set(combo).issubset(task_combo)
    ]
    candidate_masks = torch.zeros(
        len(available), 4, device=device, dtype=torch.float32
    )
    for row, candidate_index in enumerate(available):
        candidate_masks[row, list(COMBINATIONS[candidate_index])] = 1.0
    task_mask = torch.zeros(4, device=device, dtype=torch.float32)
    task_mask[list(task_combo)] = 1.0
    return available, candidate_masks, task_mask


def fuse_task(
    model: nn.Module,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rays: torch.Tensor,
    task_combo: tuple[int, ...],
    temperature: float,
):
    available, candidate_masks, task_mask = task_spec(task_combo, predictions.device)
    candidates = predictions[:, available]
    logits = model(candidates, rays, candidate_masks, task_mask)
    if logits.ndim == 2:
        weights = F.softmax(logits / temperature, dim=1)
        fused = torch.einsum("bc,bcjd->bjd", weights, candidates)
        candidate_error = torch.linalg.vector_norm(
            candidates - targets[:, None], dim=-1
        ).mean(dim=-1)
        expected = (weights * candidate_error).sum(dim=1).mean()
    else:
        weights = F.softmax(logits / temperature, dim=-1)
        fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
        candidate_error = torch.linalg.vector_norm(
            candidates - targets[:, None], dim=-1
        ).permute(0, 2, 1)
        expected = (weights * candidate_error).sum(dim=-1).mean()
    fused_error = torch.linalg.vector_norm(fused - targets, dim=-1)
    # GHT defaults: expected-hypothesis loss coefficient 1 and weighted
    # estimate coefficient 0.05. Candidates/H76 stay frozen.
    loss = expected + 0.05 * fused_error.mean()
    return loss, fused, weights, candidates


def action_equal(errors: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        errors[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    temperature: float,
    device: torch.device,
) -> dict:
    model.eval()
    errors = {"V3": [], "V4": []}
    baselines = {"V3": [], "V4": []}
    oracles = {"V3": [], "V4": []}
    action_values = []
    v2_errors = []
    v2_actions = []
    losses = []
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = actions.numpy()
            current_v2_error = torch.linalg.vector_norm(
                predictions[:, :6] - targets[:, None], dim=-1
            ).reshape(-1, 17).cpu().numpy() * 1000.0
            v2_errors.append(current_v2_error)
            v2_actions.append(np.repeat(batch_actions, 6))
            for task_combo in TASK_COMBINATIONS:
                loss, fused, _, candidates = fuse_task(
                    model, predictions, targets, rays, task_combo, temperature
                )
                stage = "V3" if len(task_combo) == 3 else "V4"
                joint_error = torch.linalg.vector_norm(
                    fused - targets, dim=-1
                ).cpu().numpy() * 1000.0
                baseline_prediction = predictions[:, COMBINATIONS.index(task_combo)]
                baseline_error = torch.linalg.vector_norm(
                    baseline_prediction - targets, dim=-1
                ).cpu().numpy() * 1000.0
                oracle_error = torch.linalg.vector_norm(
                    candidates - targets[:, None], dim=-1
                ).min(dim=1).values.cpu().numpy() * 1000.0
                errors[stage].append(joint_error)
                baselines[stage].append(baseline_error)
                oracles[stage].append(oracle_error)
                losses.append(float(loss.item()))
                action_values.append((stage, batch_actions.copy()))

    all_v2_errors = np.concatenate(v2_errors, axis=0)
    all_v2_actions = np.concatenate(v2_actions)
    result = {
        "loss": float(np.mean(losses)),
        "V2": {
            "action_equal_all17_mm": action_equal(
                all_v2_errors, all_v2_actions
            ),
            "note": "unchanged frozen H76; no valid single-view subset hypothesis",
        },
    }
    for stage in ("V3", "V4"):
        stage_errors = np.concatenate(errors[stage], axis=0)
        stage_baselines = np.concatenate(baselines[stage], axis=0)
        stage_oracles = np.concatenate(oracles[stage], axis=0)
        stage_actions = np.concatenate([
            values for name, values in action_values if name == stage
        ])
        result[stage] = {
            "action_equal_all17_mm": action_equal(stage_errors, stage_actions),
            "baseline_action_equal_all17_mm": action_equal(
                stage_baselines, stage_actions
            ),
            "candidate_oracle_action_equal_all17_mm": action_equal(
                stage_oracles, stage_actions
            ),
            "frame_weighted_all17_mm": float(stage_errors.mean()),
            "per_joint_mm": {
                name: action_equal(stage_errors[:, index], stage_actions)
                for index, name in enumerate(JOINT_NAMES)
            },
        }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu}")

    train_arrays = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation_arrays = {key: validation_npz[key] for key in validation_npz.files}
    # Internal checkpoint selection uses only held-out training frames.
    holdout = train_arrays["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    mean = torch.from_numpy(train_arrays["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train_arrays["targets"][train_indices].std(axis=(0, 1)))

    if args.variant == "pose":
        model = PoseHypothesisScorer(mean, std)
    else:
        model = JointUtilityScorer(mean, std)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ArrayDataset(train_arrays, train_indices),
        batch_size=args.batch_size, shuffle=True, generator=train_generator,
        num_workers=args.workers, pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train_arrays, holdout_indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    validation_loader = DataLoader(
        ArrayDataset(validation_arrays, np.arange(len(validation_arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = math.inf
    best_epoch = -1
    checkpoint = output_dir / "model_best.pth.tar"
    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for task_combo in TASK_COMBINATIONS:
                task_loss, _, _, _ = fuse_task(
                    model, predictions, targets, rays,
                    task_combo, args.temperature,
                )
                loss = loss + task_loss
            loss = loss / len(TASK_COMBINATIONS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout_result = evaluate(
            model, holdout_loader, args.temperature, device
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "holdout": holdout_result,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if holdout_result["loss"] < best_loss:
            best_loss = holdout_result["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": args.variant,
                    "epoch": epoch,
                    "temperature": args.temperature,
                    "mean": mean,
                    "std": std,
                },
                checkpoint,
            )

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    # S9/S11 is touched exactly once, after internal train-subject selection.
    test_result = evaluate(model, validation_loader, args.temperature, device)
    result = {
        "variant": args.variant,
        "paper_basis": (
            "Generalizable Human Pose Triangulation, CVPR 2022, official MIT code"
        ),
        "train_subjects": sorted(set(train_arrays["subjects"].tolist())),
        "test_subjects": sorted(set(validation_arrays["subjects"].tolist())),
        "train_groups": len(train_indices),
        "internal_holdout_groups": len(holdout_indices),
        "best_epoch": best_epoch,
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "temperature": args.temperature, "seed": args.seed,
            "expected_loss_beta": 1.0, "weighted_estimate_beta": 0.05,
        },
        "history": history,
        "S9_S11_final_once": test_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["S9_S11_final_once"], indent=2), flush=True)


if __name__ == "__main__":
    main()
