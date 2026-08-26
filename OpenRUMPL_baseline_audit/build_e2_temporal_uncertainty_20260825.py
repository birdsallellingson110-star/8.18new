#!/usr/bin/env python3
"""Materialize label-free E2 uncertainty features for temporal refinement.

The features retain information that is lost after E2 soft fusion: candidate
dispersion, score ambiguity, generator/triangulation disagreement, ray
confidence, and the number of available views.  No target is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_e2_fused_temporal_cache_20260818 import TASKS, task_spec


FEATURE_NAMES = (
    "score_entropy",
    "score_ambiguity",
    "candidate_dispersion_50mm",
    "generator_triangulation_disagreement_50mm",
    "mean_ray_confidence",
    "min_ray_confidence",
    "view_count_fraction",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature-v2", type=float, default=0.4)
    p.add_argument("--temperature-v3", type=float, default=1.8)
    p.add_argument("--temperature-v4", type=float, default=1.8)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    predictions = cache["predictions"]
    rays = cache["rays"]
    scores = np.load(args.scores, mmap_mode="r")
    expected = (len(predictions), len(TASKS), 17, 22)
    if tuple(scores.shape) != expected:
        raise ValueError(f"score shape {scores.shape} != {expected}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(predictions), len(TASKS), 17, len(FEATURE_NAMES)),
    )
    temperatures = {
        2: args.temperature_v2,
        3: args.temperature_v3,
        4: args.temperature_v4,
    }
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for start in range(0, len(predictions), args.chunk_size):
            stop = min(start + args.chunk_size, len(predictions))
            pred = torch.from_numpy(np.asarray(predictions[start:stop])).to(device)
            ray = torch.from_numpy(np.asarray(rays[start:stop])).to(device)
            score = torch.from_numpy(np.asarray(scores[start:stop])).to(device)
            for task_index, task in enumerate(TASKS):
                _, available_np, baseline_local = task_spec(task_index)
                available = torch.as_tensor(available_np, device=device)
                candidate = pred[:, available]  # B,C,J,3
                task_score = score[:, task_index, :, available]  # B,J,C
                delta = task_score - task_score[:, :, baseline_local:baseline_local + 1]
                weights = torch.softmax(-delta / temperatures[len(task)], dim=-1)
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
                entropy = entropy / max(float(np.log(len(available_np))), 1.0)
                top2 = torch.topk(weights, min(2, len(available_np)), dim=-1).values
                ambiguity = 1.0 - (
                    top2[..., 0] - top2[..., 1]
                    if top2.shape[-1] == 2 else top2[..., 0]
                )
                fused = torch.einsum("bjc,bcjd->bjd", weights, candidate)
                squared = ((candidate - fused[:, None]) ** 2).sum(-1)
                dispersion = torch.sqrt(
                    torch.einsum("bjc,bcj->bj", weights, squared).clamp_min(0.0)
                ) / 0.05

                generator_local = torch.nonzero(available < len(TASKS)).flatten()
                triangulation_local = torch.nonzero(available >= len(TASKS)).flatten()
                gen_w = weights[..., generator_local]
                tri_w = weights[..., triangulation_local]
                gen_w = gen_w / gen_w.sum(-1, keepdim=True).clamp_min(1e-8)
                tri_w = tri_w / tri_w.sum(-1, keepdim=True).clamp_min(1e-8)
                gen_pose = torch.einsum(
                    "bjc,bcjd->bjd", gen_w, candidate[:, generator_local]
                )
                tri_pose = torch.einsum(
                    "bjc,bcjd->bjd", tri_w, candidate[:, triangulation_local]
                )
                gen_tri = torch.linalg.vector_norm(gen_pose - tri_pose, dim=-1) / 0.05

                selected_conf = ray[:, :, list(task), 6].clamp(0.0, 1.0)
                mean_conf = selected_conf.mean(-1)
                min_conf = selected_conf.min(-1).values
                view_fraction = torch.full_like(mean_conf, len(task) / 4.0)
                task_features = torch.stack((
                    entropy,
                    ambiguity,
                    dispersion.clamp_max(4.0),
                    gen_tri.clamp_max(4.0),
                    mean_conf,
                    min_conf,
                    view_fraction,
                ), dim=-1)
                features[start:stop, task_index] = task_features.cpu().numpy()
            if start == 0 or (start // args.chunk_size) % 20 == 0:
                print(f"features {stop}/{len(predictions)}", flush=True)
    features.flush()
    manifest = {
        "method": "label-free E2 temporal uncertainty",
        "cache": str(Path(args.cache).resolve()),
        "scores": str(Path(args.scores).resolve()),
        "output": str(output),
        "shape": list(features.shape),
        "feature_names": list(FEATURE_NAMES),
        "metric_scale_m": 0.05,
        "temperatures": {f"V{k}": v for k, v in temperatures.items()},
        "uses_targets": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
