#!/usr/bin/env python3
"""Per-action and root-relative audit for the H3 fixed-lag seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_e2_temporal_residual_20260812 import build_model
from train_e2_temporal_residual_fast_20260812 import WindowArrays


COMBINATIONS = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3),
)
ACTION_NAMES = {
    2: "Directions", 3: "Discussion", 4: "Eating", 5: "Greeting",
    6: "Phone", 7: "Photo", 8: "Posing", 9: "Purchases",
    10: "Sitting", 11: "SittingDown", 12: "Smoking", 13: "Waiting",
    14: "WalkingDog", 15: "Walking", 16: "WalkingTogether",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--seed-label", nargs="+", required=True)
    p.add_argument("--validation-window-cache", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


@torch.inference_mode()
def audit_checkpoint(path, arrays, device, batch_size):
    state = torch.load(path, map_location="cpu")
    task_ids = np.asarray(state["task_ids"], dtype=np.int64)
    model = build_model(
        state["architecture"], state["mean"], state["std"],
        int(state["window_length"]), int(state["hidden_dim"]),
        int(state["layers"]),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    stores = {
        int(task): {"err": [], "base": [], "root": [], "root_base": [],
                    "actions": []}
        for task in task_ids
    }
    indices = np.arange(arrays.window_count * len(task_ids), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        poses, targets, task_batch, actions = arrays.gather(
            indices[start : start + batch_size]
        )
        pose_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
        target_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        task_t = torch.from_numpy(task_batch).to(device=device, dtype=torch.long)
        prediction = model(pose_t, task_t).cpu().numpy()
        center = poses[:, poses.shape[1] // 2]
        error = np.linalg.norm(prediction - targets, axis=-1) * 1000.0
        baseline = np.linalg.norm(center - targets, axis=-1) * 1000.0
        pred_rel = prediction - prediction[:, :1]
        target_rel = targets - targets[:, :1]
        center_rel = center - center[:, :1]
        root_error = np.linalg.norm(prediction[:, 0] - targets[:, 0], axis=-1) * 1000.0
        root_base = np.linalg.norm(center[:, 0] - targets[:, 0], axis=-1) * 1000.0
        root_relative_error = np.linalg.norm(pred_rel - target_rel, axis=-1) * 1000.0
        root_relative_base = np.linalg.norm(center_rel - target_rel, axis=-1) * 1000.0
        for task in task_ids.tolist():
            mask = task_batch == task
            if np.any(mask):
                store = stores[int(task)]
                store["err"].append(error[mask])
                store["base"].append(baseline[mask])
                store["root"].append(root_error[mask])
                store["root_base"].append(root_base[mask])
                store["actions"].append(actions[mask])
    result = {}
    for task in task_ids.tolist():
        store = stores[int(task)]
        err = np.concatenate(store["err"])
        base = np.concatenate(store["base"])
        root = np.concatenate(store["root"])
        root_base = np.concatenate(store["root_base"])
        actions = np.concatenate(store["actions"])
        action_result = {}
        for action in sorted(set(actions.tolist())):
            mask = actions == action
            action_result[str(action)] = {
                "name": ACTION_NAMES.get(int(action), str(action)),
                "temporal_all17_mm": float(err[mask].mean()),
                "center_all17_mm": float(base[mask].mean()),
                "delta_all17_mm": float(err[mask].mean() - base[mask].mean()),
                "temporal_root_mm": float(root[mask].mean()),
                "center_root_mm": float(root_base[mask].mean()),
                "delta_root_mm": float(root[mask].mean() - root_base[mask].mean()),
            }
        result[str(COMBINATIONS[task])] = {
            "temporal_all17_mm": float(err.mean()),
            "center_all17_mm": float(base.mean()),
            "delta_all17_mm": float(err.mean() - base.mean()),
            "temporal_root_mm": float(root.mean()),
            "center_root_mm": float(root_base.mean()),
            "delta_root_mm": float(root.mean() - root_base.mean()),
            "actions": action_result,
        }
    return result


def main():
    args = parse_args()
    if len(args.checkpoint) != len(args.seed_label):
        raise ValueError("checkpoint and seed-label counts must match")
    device = torch.device(f"cuda:{args.gpu}")
    arrays = WindowArrays(
        args.validation_window_cache, args.validation_index,
        np.asarray([6, 7, 8, 9, 10], dtype=np.int64),
    )
    payload = {
        label: audit_checkpoint(path, arrays, device, args.batch_size)
        for label, path in zip(args.seed_label, args.checkpoint)
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": list(payload)}, indent=2))


if __name__ == "__main__":
    main()
