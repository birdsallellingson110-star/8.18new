#!/usr/bin/env python3
"""Retrain E2-C2 utility with view-snap and bone-ray extra candidates.

H76 and the 22-candidate cache stay frozen.  For each task, the scorer also
sees, using only that task's cameras:

- the task H76 pose snapped onto each task view's ray;
- a mean-bone-length reconstruction along each task view's ray.

Bone lengths are a train-set statistic.  No occlusion augmentation.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F

import train_current_e2_confidence_20260815 as wrapper
import train_e2_v234_universal_20260812 as trainer

PARENTS = torch.tensor(
    [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], dtype=torch.long
)
CHILD_ORDER = tuple(index for index in range(17) if index != 0)
BONE_LENGTHS = None


def train_bone_lengths(train_cache: str) -> torch.Tensor:
    targets = np.load(train_cache)["targets"].astype(np.float64)
    parents = PARENTS.numpy()
    lengths = np.linalg.norm(targets - targets[:, parents], axis=-1).mean(axis=0)
    lengths[0] = 0.0
    return torch.from_numpy(lengths.astype(np.float32))


def snap_pose(pose: torch.Tensor, rays: torch.Tensor, view: int) -> torch.Tensor:
    direction = F.normalize(rays[:, :, view, :3], dim=-1)
    point = rays[:, :, view, 3:6]
    depth = ((pose - point) * direction).sum(dim=-1)
    return point + depth.unsqueeze(-1) * direction


def bone_pose(pose: torch.Tensor, rays: torch.Tensor, view: int, bone_lengths: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(rays[:, :, view, :3], dim=-1)
    point = rays[:, :, view, 3:6]
    prior_depth = ((pose - point) * direction).sum(dim=-1)
    out = pose.new_empty(pose.shape)
    out[:, 0] = point[:, 0] + prior_depth[:, 0, None] * direction[:, 0]
    parents = PARENTS.to(pose.device)
    for joint in CHILD_ORDER:
        parent = out[:, parents[joint]]
        rel = point[:, joint] - parent
        vector = direction[:, joint]
        b = 2.0 * (vector * rel).sum(dim=-1)
        c = (rel * rel).sum(dim=-1) - bone_lengths[joint] ** 2
        disc = b * b - 4.0 * c
        sqrt = torch.sqrt(disc.clamp_min(0.0))
        t_plus = (-b + sqrt) * 0.5
        t_minus = (-b - sqrt) * 0.5
        use_plus = (t_plus - prior_depth[:, joint]).abs() <= (t_minus - prior_depth[:, joint]).abs()
        depth = torch.where(use_plus, t_plus, t_minus)
        depth = torch.where(disc >= 0.0, depth, prior_depth[:, joint])
        out[:, joint] = point[:, joint] + depth.unsqueeze(-1) * vector
    return out


def extra_candidates(baseline: torch.Tensor, rays: torch.Tensor, task_combo: tuple[int, ...]):
    if BONE_LENGTHS is None:
        raise RuntimeError("BONE_LENGTHS is not initialized")
    bone_lengths = BONE_LENGTHS.to(device=baseline.device, dtype=baseline.dtype)
    poses = []
    masks = []
    for view in task_combo:
        poses.append(snap_pose(baseline, rays, view))
        poses.append(bone_pose(baseline, rays, view, bone_lengths))
        mask = torch.zeros(4, device=baseline.device, dtype=baseline.dtype)
        mask[view] = 1.0
        masks.append(mask)
        masks.append(mask.clone())
    return torch.stack(poses, dim=1), torch.stack(masks, dim=0)


def predict_task(model, predictions, targets, rays, task_combo):
    available, masks, task_mask, baseline_local = trainer.task_spec(
        task_combo, predictions.device
    )
    candidates = predictions[:, available]
    extras, extra_masks = extra_candidates(candidates[:, baseline_local], rays, task_combo)
    candidates = torch.cat((candidates, extras), dim=1)
    masks = torch.cat((masks, extra_masks), dim=0)
    raw = model(candidates, rays, masks, task_mask)
    error = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1)
    true_error = error.permute(0, 2, 1)
    baseline_error = true_error[..., baseline_local:baseline_local + 1]
    return (
        raw - raw[..., baseline_local:baseline_local + 1],
        true_error - baseline_error,
        true_error,
        candidates,
        baseline_local,
    )


def main() -> None:
    import sys
    from pathlib import Path

    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    trainer.predict_task = predict_task
    train_cache = None
    for index, token in enumerate(sys.argv):
        if token == "--train-shards":
            train_cache = sys.argv[index + 1]
            break
    if train_cache is None:
        raise ValueError("--train-shards is required")
    global BONE_LENGTHS
    BONE_LENGTHS = train_bone_lengths(train_cache)
    trainer.main()
    output_dir = None
    for index, token in enumerate(sys.argv):
        if token == "--output-dir":
            output_dir = Path(sys.argv[index + 1])
            break
    if output_dir is not None:
        payload_path = output_dir / "result.json"
        if payload_path.exists():
            import json
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["method"] = (
                "E2-C2 utility with task-local view-snap and bone-ray candidates"
            )
            payload["extra_candidates"] = [
                "snap task H76 onto each task view ray",
                "mean-bone-length reconstruction along each task view ray",
            ]
            payload["mean_train_bone_lengths_m"] = BONE_LENGTHS.cpu().tolist()
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
