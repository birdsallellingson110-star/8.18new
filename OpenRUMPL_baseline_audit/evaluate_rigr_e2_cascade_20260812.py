#!/usr/bin/env python3
"""Evaluate a candidate-level RIGR -> E2 cascade on the exact H76 cache.

The HRNet feature refiner is applied independently to the original eleven
H76 subset candidates.  Those refined candidates replace only the first eleven
entries of the already validated 22-candidate E2 pool; pairwise and learned
triangulation candidates remain frozen.  This is an inference-only diagnostic
for complementarity, not a claim that the two modules have been jointly
trained.  It is intentionally written against the same ``evaluate_expanded``
routine used by the formal E2 experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_h76_learned_candidate_e2_20260814 as learned_e2
from train_rigr_hrnet_feature_20260812 import (
    COMBINATIONS,
    RIGRHRNetFeature,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-tokens", required=True)
    p.add_argument("--e2-cache", required=True)
    p.add_argument("--rigr-checkpoint", required=True)
    p.add_argument("--e2-checkpoints", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--e2-batch-size", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--seed-name", default="seed")
    return p.parse_args()


def load_npz(path: str) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=False)
    return {key: source[key] for key in source.files}


def load_rigr(path: str, token_shape: tuple[int, ...], device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    channels = int(checkpoint.get("channels", token_shape[-3]))
    patch = int(checkpoint.get("patch", token_shape[-1]))
    model = RIGRHRNetFeature(
        channels,
        patch,
        gated_residual=bool(checkpoint.get("gated_residual", False)),
        aux_dim=int(checkpoint.get("aux_dim", 0) or 0),
        attention_pooling=bool(checkpoint.get("attention_pooling", False)),
        cross_view_relation=bool(checkpoint.get("cross_view_relation", False)),
        patch_attention=bool(checkpoint.get("patch_attention", False)),
        geometry_biased_attention=bool(
            checkpoint.get("geometry_biased_attention", False)
        ),
        explicit_view_block=bool(checkpoint.get("explicit_view_block", False)),
        correspondence_attention=bool(
            checkpoint.get("correspondence_attention", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def pad_batch(
    rays: np.ndarray,
    features: np.ndarray,
    views: tuple[int, ...],
    device: torch.device,
):
    batch, joints = rays.shape[0], rays.shape[1]
    channels, height, width = features.shape[-3:]
    padded_rays = np.zeros((batch, joints, 4, 7), dtype=np.float32)
    padded_features = np.zeros(
        (batch, 4, joints, channels, height, width), dtype=np.float32
    )
    padded_rays[:, :, : len(views)] = rays[:, :, list(views)]
    padded_features[:, : len(views)] = features[:, list(views)]
    mask = np.zeros((batch, 4), dtype=np.bool_)
    mask[:, : len(views)] = True
    return (
        torch.from_numpy(padded_rays).to(device),
        torch.from_numpy(padded_features).to(device),
        torch.from_numpy(mask).to(device),
    )


def refine_candidates(
    cache: dict[str, np.ndarray],
    tokens_path: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return [G,11,J,3] refined candidates in the cache's group order."""
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 7 or tokens.shape[1] != len(COMBINATIONS):
        raise ValueError(f"expected [G,11,4,J,C,P,P] tokens, got {tokens.shape}")
    predictions = cache["predictions"]
    if predictions.shape[1:] != (len(COMBINATIONS), 17, 3):
        raise ValueError(f"expected H76 predictions [G,11,17,3], got {predictions.shape}")
    if len(tokens) != len(predictions):
        raise ValueError(f"token/cache length mismatch: {tokens.shape} vs {predictions.shape}")
    rays_all = cache["rays"]
    refined = np.empty_like(predictions, dtype=np.float32)
    with torch.inference_mode():
        for combo_id, views in enumerate(COMBINATIONS):
            views = tuple(views)
            for start in range(0, len(predictions), batch_size):
                stop = min(start + batch_size, len(predictions))
                prediction = torch.from_numpy(
                    predictions[start:stop, combo_id].astype(np.float32)
                ).to(device)
                rays, features, mask = pad_batch(
                    rays_all[start:stop],
                    np.asarray(tokens[start:stop, combo_id], dtype=np.float32),
                    views,
                    device,
                )
                output = model(prediction, rays, features, mask, None)
                refined[start:stop, combo_id] = output.cpu().numpy()
    return refined


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    unique = sorted(set(int(x) for x in actions))
    return float(np.mean([values[actions == action].mean() for action in unique]))


def evaluate_rigr_only(
    refined: np.ndarray, targets: np.ndarray, actions: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    for count in (2, 3, 4):
        values = []
        for combo_id, combo in enumerate(COMBINATIONS):
            if len(combo) == count:
                values.append(
                    np.linalg.norm(refined[:, combo_id] - targets, axis=-1).mean(axis=-1)
                )
        result[f"V{count}"] = action_equal(np.stack(values, axis=1).mean(axis=1), actions) * 1000.0
    return result


def build_e2_arrays(
    e2_cache: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    refined: np.ndarray | None,
) -> dict[str, np.ndarray]:
    required = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    for key in required:
        if key not in e2_cache:
            raise ValueError(f"missing {key} in E2 cache")
    if not np.array_equal(e2_cache["group_indices"], original["group_indices"]):
        raise ValueError("E2/H76 group ordering mismatch")
    if not np.allclose(e2_cache["targets"], original["targets"], atol=2e-5):
        raise ValueError("E2/H76 target mismatch")
    if not np.allclose(e2_cache["rays"], original["rays"], atol=2e-5):
        raise ValueError("E2/H76 ray mismatch")
    arrays = {key: e2_cache[key] for key in required}
    arrays["predictions"] = np.array(e2_cache["predictions"], copy=True)
    if refined is not None:
        arrays["predictions"][:, : len(COMBINATIONS)] = refined
    if arrays["predictions"].shape[1] != len(learned_e2.EXPANDED_COMBINATIONS):
        raise ValueError(f"unexpected E2 candidate count: {arrays['predictions'].shape}")
    return arrays


def eval_e2(
    arrays: dict[str, np.ndarray], checkpoint_path: str, device: torch.device,
    batch_size: int, temperature: float,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean = checkpoint["mean"].to(device)
    std = checkpoint["std"].to(device)
    model = learned_e2.SetTransformerJointUtility(
        mean, std, int(checkpoint.get("attention_depth", 2))
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    dataset = learned_e2.ArrayDataset(arrays, np.arange(len(arrays["targets"])))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    result = learned_e2.base.evaluate_expanded(model, loader, device, temperature)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    original = load_npz(args.validation_cache)
    e2_cache = load_npz(args.e2_cache)
    tokens = np.load(args.validation_tokens, mmap_mode="r")
    model = load_rigr(args.rigr_checkpoint, tokens.shape, device)
    refined = refine_candidates(
        original, args.validation_tokens, model, device, args.batch_size
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rigr_refined_candidates.npy", refined)
    rigr_only = evaluate_rigr_only(refined, original["targets"], original["actions"])
    results = {
        "method": "inference-only RIGR candidate refinement followed by E2",
        "rigr_checkpoint": str(Path(args.rigr_checkpoint).resolve()),
        "e2_cache": str(Path(args.e2_cache).resolve()),
        "rigr_only": rigr_only,
        "e2": {},
        "args": vars(args),
    }
    baseline_arrays = build_e2_arrays(e2_cache, original, None)
    cascade_arrays = build_e2_arrays(e2_cache, original, refined)
    for checkpoint in args.e2_checkpoints:
        name = Path(checkpoint).parent.name
        print(json.dumps({"stage": "e2", "checkpoint": checkpoint}), flush=True)
        results["e2"][name] = {
            "e2_only": eval_e2(
                baseline_arrays, checkpoint, device, args.e2_batch_size, args.temperature
            ),
            "rigr_to_e2": eval_e2(
                cascade_arrays, checkpoint, device, args.e2_batch_size, args.temperature
            ),
        }
    (output_dir / "result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
