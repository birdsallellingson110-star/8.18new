#!/usr/bin/env python3
"""Export pairwise-E2 candidate poses and utility logits for T-CVU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-shards", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def load_model(path, device):
    state = torch.load(path, map_location="cpu")
    model = SetTransformerJointUtility(
        state["mean"], state["std"], int(state["attention_depth"])
    )
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()


def export_split(name, arrays, model, device, batch_size, out_dir):
    predictions = arrays["predictions"].astype(np.float32, copy=False)
    utility = np.zeros(
        (len(predictions), len(TASK_COMBINATIONS), 17, len(EXPANDED_COMBINATIONS)),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            stop = min(start + batch_size, len(predictions))
            p = torch.from_numpy(predictions[start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            t = torch.from_numpy(arrays["targets"][start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            r = torch.from_numpy(arrays["rays"][start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for task_position, task in enumerate(TASK_COMBINATIONS):
                delta, _, _, _, _ = predict_delta_expanded(model, p, t, r, task)
                available = [
                    i for i, combo in enumerate(EXPANDED_COMBINATIONS)
                    if set(combo).issubset(task)
                ]
                utility_chunk = utility[start:stop, task_position]
                utility_chunk[:, :, available] = delta.cpu().numpy().astype(
                    np.float32
                )
            if start == 0 or stop == len(predictions) or start % (batch_size * 20) == 0:
                print(f"{name}: exported {stop}/{len(predictions)}", flush=True)
    output = out_dir / f"{name}_pairwise_e2_temporal_candidates.npz"
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=arrays["group_indices"], subjects=arrays["subjects"],
        actions=arrays["actions"], targets=arrays["targets"],
        candidate_poses=predictions, utility_delta=utility,
    )
    temporary.replace(output)
    return {"split": name, "groups": int(len(predictions)),
            "candidate_pose_shape": list(predictions.shape),
            "utility_shape": list(utility.shape), "output": str(output)}


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    model = load_model(args.checkpoint, device)
    train = load_expanded(args.train_shards)
    validation = load_expanded(args.validation_shards)
    reports = {
        "train": export_split("train", train, model, device, args.batch_size, out),
        "validation": export_split(
            "validation", validation, model, device, args.batch_size, out
        ),
    }
    manifest = {"args": vars(args), "reports": reports,
                "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS]}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
