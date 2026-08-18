#!/usr/bin/env python3
"""Evaluate robust, training-free aggregation of frozen pairwise hypotheses.

The utility logits and 17 candidate poses stay frozen.  Only the final
candidate aggregation is changed, allowing a clean test of whether the large
oracle-versus-soft-fusion gap is caused by the arithmetic mean being pulled by
outlier hypotheses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_h76_hypothesis_utility_20260811 import TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--irls-steps", type=int, default=8)
    p.add_argument("--eps-mm", type=float, default=0.05)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def action_equal(values, actions):
    values = np.asarray(values)
    actions = np.asarray(actions)
    return float(np.mean([
        values[actions == action].mean()
        for action in sorted(set(actions.tolist()))
    ]) * 1000.0)


def aggregate(candidates, weights, method, steps, eps):
    # candidates: B,C,J,3; weights: B,J,C
    if method == "mean":
        return torch.einsum("bjc,bcjd->bjd", weights, candidates)
    if method == "medoid":
        pairwise = torch.linalg.vector_norm(
            candidates[:, :, None, :, :] - candidates[:, None, :, :, :],
            dim=-1,
        )  # B,C,C,J
        candidate_weights = weights.permute(0, 2, 1)  # B,C,J
        costs = (pairwise * candidate_weights[:, None, :, :]).sum(dim=2)  # B,C,J
        chosen = costs.argmin(dim=1)  # B,J
        by_joint = candidates.permute(0, 2, 1, 3)  # B,J,C,3
        gather = chosen[:, :, None, None].expand(
            -1, -1, 1, candidates.shape[-1]
        )
        return by_joint.gather(2, gather).squeeze(2)

    # Weiszfeld iterations: utility weights are retained and each iteration
    # adds the standard inverse-distance robust factor.  eps is in meters.
    fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
    for _ in range(steps):
        distances = torch.linalg.vector_norm(
            candidates - fused[:, None], dim=-1
        ).clamp_min(eps)
        robust_weights = weights.permute(0, 2, 1) / distances  # B,C,J
        robust_weights = robust_weights / robust_weights.sum(dim=1, keepdim=True)
        fused = torch.einsum("bcj,bcjd->bjd", robust_weights, candidates)
    return fused


@torch.inference_mode()
def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}")
    state = torch.load(args.checkpoint, map_location="cpu")
    model = SetTransformerJointUtility(
        state["mean"], state["std"], int(state["attention_depth"])
    ).to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    cache = np.load(args.validation_cache)
    predictions = cache["predictions"].astype(np.float32, copy=False)
    targets = cache["targets"].astype(np.float32, copy=False)
    rays = cache["rays"].astype(np.float32, copy=False)
    actions = cache["actions"].astype(np.int16, copy=False)
    stores = {
        method: {"V3": [], "V4": [], "base": {"V3": [], "V4": []}}
        for method in ("mean", "geomedian", "medoid")
    }
    for start in range(0, len(targets), args.batch_size):
        stop = min(start + args.batch_size, len(targets))
        p = torch.from_numpy(predictions[start:stop]).to(device)
        t = torch.from_numpy(targets[start:stop]).to(device)
        r = torch.from_numpy(rays[start:stop]).to(device)
        for task_index, task in enumerate(TASK_COMBINATIONS):
            # The candidate tensor and logits come from one frozen forward.
            predicted_delta, _, _, candidates, _ = predict_delta_expanded(
                model, p, t, r, task
            )
            weights = F.softmax(-predicted_delta / args.temperature, dim=-1)
            # The baseline is the frozen E2 soft fusion.  Keep it separate
            # from robust methods for an exact same-batch comparison.
            baseline = aggregate(candidates, weights, "mean", 0, 0.0)
            robust = {
                "mean": baseline,
                "geomedian": aggregate(
                    candidates, weights, "geomedian", args.irls_steps,
                    args.eps_mm / 1000.0,
                ),
                "medoid": aggregate(candidates, weights, "medoid", 0, 0.0),
            }
            stage = "V3" if task_index < 4 else "V4"
            for method, fused in robust.items():
                error = torch.linalg.vector_norm(fused - t, dim=-1)
                stores[method][stage].append(error.cpu().numpy())
                stores[method]["base"][stage].append(
                    torch.linalg.vector_norm(baseline - t, dim=-1).cpu().numpy()
                )
        if start == 0 or stop == len(targets) or start % (args.batch_size * 20) == 0:
            print(f"evaluated {stop}/{len(targets)}", flush=True)

    result = {}
    for method, stage_store in stores.items():
        result[method] = {}
        for stage in ("V3", "V4"):
            values = np.concatenate(stage_store[stage])
            base = np.concatenate(stage_store["base"][stage])
            # Chunks are stored batch-major with all stage tasks inside each
            # batch.  Rebuild exactly that order (the evaluator uses the same
            # order), rather than tiling the whole dataset task-major.
            task_repeats = len(values) // len(actions)
            action_chunks = []
            for start in range(0, len(actions), args.batch_size):
                stop = min(start + args.batch_size, len(actions))
                action_chunks.append(np.tile(actions[start:stop], task_repeats))
            stage_actions = np.concatenate(action_chunks)
            result[method][stage] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean() * 1000.0),
                "relative_to_mean_action_equal_mm": float(
                    action_equal(values, stage_actions)
                    - action_equal(base, stage_actions)
                ),
            }
    return result


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    payload = {"args": vars(args), "results": result}
    (out / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
