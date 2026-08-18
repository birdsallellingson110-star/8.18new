#!/usr/bin/env python3
"""Export frozen E-2 candidate poses and per-frame utility logits.

The existing temporal pose cache stores only the fused pose for each view
count.  T-CVU needs the complete candidate set at every frame, plus the
already-trained E-2 counterfactual logits.  This exporter computes exactly the
same soft candidate fusion as ``build_e2_temporal_cache_20260812.py`` and
stores no ground-truth-derived quantity in the model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_h76_hypothesis_utility_20260811 import (
    COMBINATIONS,
    TASK_COMBINATIONS,
    load_arrays,
    task_spec,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-shards", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def load_model(path, device):
    state = torch.load(path, map_location="cpu")
    model = SetTransformerJointUtility(
        state["mean"], state["std"], int(state["attention_depth"]),
        bool(state.get("view_cross_attention", False)),
        state.get("joint_attention", "none"),
    )
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()


def export_split(name, arrays, model, device, batch_size, out_dir):
    predictions = arrays["predictions"]
    # Candidate pool must remain the raw frozen H76 hypotheses.  E-2 scores
    # and fuses these hypotheses independently for each target task; using an
    # already fused 3/4-view output as another candidate would double-fuse the
    # set and break the exact identity check.
    candidate_poses = predictions.copy().astype(np.float32, copy=False)
    fused = predictions.copy().astype(np.float32, copy=False)
    # Five target tasks (four V3 combinations and one V4 combination), padded
    # to the global 11-combination index.  Unavailable entries remain zero and
    # are masked by the fixed task candidate list in the trainer.
    utility = np.zeros(
        (len(predictions), len(TASK_COMBINATIONS), 17, len(COMBINATIONS)),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            stop = min(start + batch_size, len(predictions))
            prediction = torch.from_numpy(predictions[start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            rays = torch.from_numpy(arrays["rays"][start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for task_position, task_combo in enumerate(TASK_COMBINATIONS):
                available, candidate_masks, task_mask = task_spec(
                    task_combo, device
                )
                candidates = prediction[:, available]
                raw = model(candidates, rays, candidate_masks, task_mask)
                baseline_local = available.index(COMBINATIONS.index(task_combo))
                delta = raw - raw[..., baseline_local:baseline_local + 1]
                weights = F.softmax(-delta, dim=-1)
                fused_batch = torch.einsum(
                    "bjc,bcjd->bjd", weights, candidates
                )
                fused[start:stop, COMBINATIONS.index(task_combo)] = (
                    fused_batch.cpu().numpy().astype(np.float32)
                )
                padded = np.zeros(
                    (stop - start, 17, len(COMBINATIONS)), dtype=np.float32
                )
                padded[:, :, np.asarray(available, dtype=np.int64)] = (
                    delta.cpu().numpy().astype(np.float32)
                )
                utility[start:stop, task_position] = padded
            if start == 0 or stop == len(predictions) or start % (batch_size * 20) == 0:
                print(f"{name}: exported {stop}/{len(predictions)}", flush=True)

    output = out_dir / f"{name}_e2_temporal_candidates.npz"
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=arrays["group_indices"],
        subjects=arrays["subjects"],
        actions=arrays["actions"],
        targets=arrays["targets"],
        candidate_poses=candidate_poses,
        e2_fused_poses=fused,
        utility_delta=utility,
    )
    temporary.replace(output)
    return {
        "split": name,
        "groups": int(len(predictions)),
        "candidate_pose_shape": list(fused.shape),
        "utility_shape": list(utility.shape),
        "output": str(output),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    model = load_model(args.checkpoint, device)
    train = load_arrays(args.train_shards)
    validation = load_arrays(args.validation_shards)
    reports = {
        "train": export_split("train", train, model, device, args.batch_size, out_dir),
        "validation": export_split(
            "validation", validation, model, device, args.batch_size, out_dir
        ),
    }
    manifest = {"args": vars(args), "reports": reports}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
