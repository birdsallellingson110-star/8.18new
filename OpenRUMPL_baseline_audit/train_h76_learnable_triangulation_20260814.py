#!/usr/bin/env python3
"""Learnable view reliability followed by differentiable ray triangulation.

This is a controlled experiment based on Learnable Triangulation (ICCV 2019),
AdaFuse-style view reliability, and the weighted line-intersection already used
as RUMPL's tri-anchor.  It deliberately does not consume H76 candidate poses:
the only input at inference is the calibrated ray [direction, origin] and the
frozen 2-D detector confidence.  The experiment therefore tests whether the
remaining error can be reduced by learning per-joint, per-view reliability
before a differentiable triangulation solve.

Two variants are supported:
  --variant independent: each ray gets an independent reliability logit;
  --variant cross: one Transformer layer lets the active views compare their
                   geometry before producing reliability logits.

The script reports both direct learned triangulation and two non-learned DLT
controls.  It does not alter the RUMPL checkpoint or claim an improvement until
the learned output is compared with the H76 baseline on the same combinations.
"""

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
from torch.utils.data import DataLoader, Dataset

from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES


ALL_COMBINATIONS = tuple(
    combo
    for view_count in (2, 3, 4)
    for combo in itertools.combinations(range(4), view_count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("independent", "cross"), required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attention-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class RayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source = self.indices[index]
        return (
            torch.from_numpy(self.arrays["rays"][source]),
            torch.from_numpy(self.arrays["targets"][source]),
            torch.from_numpy(self.arrays["predictions"][source]),
            int(self.arrays["actions"][source]),
        )


def load_arrays(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate group indices after shard concatenation")
    if arrays["rays"].shape[1:] != (17, 4, 7):
        raise ValueError(f"unexpected ray shape {arrays['rays'].shape}")
    if arrays["targets"].shape[1:] != (17, 3):
        raise ValueError(f"unexpected target shape {arrays['targets'].shape}")
    return arrays


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def safe_solve(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a regularized 3x3 system and fall back to pinv for bad geometry."""
    try:
        return torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return (torch.linalg.pinv(lhs) @ rhs.unsqueeze(-1)).squeeze(-1)


def triangulate(
    rays: torch.Tensor,
    view_indices: tuple[int, ...],
    weights: torch.Tensor,
    regularization: float = 1e-4,
) -> torch.Tensor:
    """Weighted least-squares intersection of per-joint calibrated rays.

    rays: B,J,4,7, with [direction(3), origin(3), confidence].
    weights: B,J,V for the selected views; they are normalized internally.
    returns: B,J,3.
    """
    selected = rays[:, :, list(view_indices)]
    direction = F.normalize(selected[..., :3], dim=-1)
    origin = selected[..., 3:6]
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    normalized_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    lhs = (normalized_weights.unsqueeze(-1).unsqueeze(-1) * projection).sum(dim=2)
    rhs = (
        normalized_weights.unsqueeze(-1).unsqueeze(-1)
        * (projection @ origin.unsqueeze(-1))
    ).sum(dim=2).squeeze(-1)
    lhs = lhs + regularization * eye
    return safe_solve(lhs.reshape(-1, 3, 3), rhs.reshape(-1, 3)).reshape(
        rays.shape[0], rays.shape[1], 3
    )


def fixed_weights(
    rays: torch.Tensor, view_indices: tuple[int, ...], kind: str
) -> torch.Tensor:
    selected = rays[:, :, list(view_indices)]
    if kind == "uniform":
        weights = torch.ones_like(selected[..., 6])
    elif kind == "confidence":
        weights = selected[..., 6].clamp(0, 1) + 0.05
    else:
        raise ValueError(kind)
    return weights


class LearnableTriangulation(nn.Module):
    def __init__(self, variant: str, attention_depth: int = 1):
        super().__init__()
        self.variant = variant
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        # direction, origin/5, confidence, direction norm and origin norm are
        # all available without camera IDs or GT.  The duplicated norms make
        # the scale information explicit after direction normalization.
        self.encoder = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 1 + 1 + 16, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.ReLU6(),
        )
        if variant == "cross":
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=64,
                    nhead=4,
                    dim_feedforward=128,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(attention_depth)
            ])
            self.norm = nn.LayerNorm(64)
        else:
            self.blocks = nn.ModuleList()
            self.norm = nn.Identity()
        self.logit = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 32), nn.ReLU6(), nn.Linear(32, 1)
        )

    def forward(self, rays: torch.Tensor, view_indices: tuple[int, ...]):
        selected = rays[:, :, list(view_indices)]
        direction = F.normalize(selected[..., :3], dim=-1)
        origin = selected[..., 3:6]
        confidence = selected[..., 6:7].clamp(0, 1)
        joint = self.joint_embedding[None, :, None].expand(
            rays.shape[0], rays.shape[1], len(view_indices), -1
        )
        features = torch.cat(
            (
                direction,
                origin / 5.0,
                confidence,
                torch.linalg.vector_norm(selected[..., :3], dim=-1, keepdim=True),
                torch.linalg.vector_norm(origin, dim=-1, keepdim=True) / 5.0,
                joint,
            ),
            dim=-1,
        )
        tokens = self.encoder(features)
        if self.blocks:
            flat = tokens.reshape(-1, len(view_indices), 64)
            for block in self.blocks:
                flat = block(flat)
            tokens = self.norm(flat).reshape_as(tokens)
        logits = self.logit(tokens).squeeze(-1)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return triangulate(rays, view_indices, weights), weights

    @property
    def temperature(self) -> float:
        # Kept as a property so a checkpoint is not dependent on a global.
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = float(value)


def set_model_temperature(model: LearnableTriangulation, temperature: float) -> None:
    model.temperature = temperature


def evaluate(
    model: LearnableTriangulation | None,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict:
    if model is not None:
        model.eval()
        model.temperature = temperature
    stores = {
        f"V{count}": {"uniform": [], "confidence": [], "learned": [], "h76": []}
        for count in (2, 3, 4)
    }
    actions_by_stage = {f"V{count}": [] for count in (2, 3, 4)}
    with torch.inference_mode():
        for rays, targets, predictions, actions in loader:
            rays = rays.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = predictions.to(device, non_blocking=True)
            for combo in ALL_COMBINATIONS:
                stage = f"V{len(combo)}"
                uniform = triangulate(rays, combo, fixed_weights(rays, combo, "uniform"))
                confidence = triangulate(rays, combo, fixed_weights(rays, combo, "confidence"))
                if model is None:
                    learned = confidence
                else:
                    learned, _ = model(rays, combo)
                # H76 exact task candidate is the first 11 entries in the
                # expanded cache, ordered by COMBINATIONS.
                baseline_index = ALL_COMBINATIONS.index(combo)
                h76 = predictions[:, baseline_index]
                for name, pose in (
                    ("uniform", uniform),
                    ("confidence", confidence),
                    ("learned", learned),
                    ("h76", h76),
                ):
                    stores[stage][name].append(
                        torch.linalg.vector_norm(pose - targets, dim=-1).cpu().numpy()
                        * 1000.0
                    )
                actions_by_stage[stage].append(actions.numpy().copy())
    result = {}
    for stage in stores:
        stage_actions = np.concatenate(actions_by_stage[stage])
        result[stage] = {}
        for name, chunks in stores[stage].items():
            values = np.concatenate(chunks, axis=0)
            result[stage][name] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
                "per_joint_mm": {
                    joint: action_equal(values[:, index], stage_actions)
                    for index, joint in enumerate(JOINT_NAMES)
                },
            }
    return result


def train_loss(model, rays, targets) -> torch.Tensor:
    losses = []
    for combo in ALL_COMBINATIONS:
        prediction, _ = model(rays, combo)
        # Coordinate-wise robust loss avoids a few ill-conditioned rays
        # dominating the reliability head while retaining absolute 3D scale.
        losses.append(F.smooth_l1_loss(prediction, targets, beta=0.01))
    return torch.stack(losses).mean()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    train_loader = DataLoader(
        RayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        RayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        RayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    model = LearnableTriangulation(args.variant, args.attention_depth).to(device)
    model.temperature = args.temperature
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    best_metric = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        model.temperature = args.temperature
        losses = []
        for batch_index, (rays, targets, _, _) in enumerate(train_loader):
            rays = rays.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, rays, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout_result = evaluate(model, holdout_loader, device, args.temperature)
        metric = 0.5 * (
            holdout_result["V3"]["learned"]["action_equal_all17_mm"]
            + holdout_result["V4"]["learned"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch,
            "variant": args.variant,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = float(metric)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": args.variant,
                    "attention_depth": args.attention_depth,
                    "temperature": args.temperature,
                    "epoch": epoch,
                    "holdout_selection_metric_mm": best_metric,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(model, test_loader, device, args.temperature)
    result = {
        "method": "learnable view reliability + differentiable weighted ray triangulation",
        "paper_basis": [
            "Learnable Triangulation (ICCV 2019)",
            "AdaFuse (IJCV 2021) view reliability",
            "RUMPL tri-anchor weighted line intersection",
        ],
        "variant": args.variant,
        "attention_depth": args.attention_depth,
        "temperature": args.temperature,
        "train_groups": int(len(train_indices)),
        "holdout_groups": int(len(holdout_indices)),
        "validation_groups": int(len(validation["targets"])),
        "best_epoch": int(checkpoint["epoch"]),
        "best_holdout_selection_metric_mm": float(checkpoint["holdout_selection_metric_mm"]),
        "history": history,
        "test": test_result,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"final_test": test_result, "best_epoch": checkpoint["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
