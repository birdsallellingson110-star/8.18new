#!/usr/bin/env python3
"""Small, deterministic SMPL/MHP random-camera audit for frozen H76+C2+E2.

This is a diagnostic only.  It compares the learned generator and scorer with
confidence-weighted ray triangulation on an existing AMASS/SMPL validation set
rendered from 20 random cameras.  It does not train or alter checkpoints.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RUMPL_REPO = HERE.parent / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUMPL_REPO / "lib"))

from audit_camera_coordinate_equivariance_20260824 import (  # noqa: E402
    configure_generator,
    e2_outputs,
    load_e2,
)


COMBINATIONS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument(
        "--amass-dataset-type",
        default="paper_single_h36m/datasets/strict_split90_9",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--camera-selection", choices=("random4", "quality4"),
        default="random4",
        help=(
            "quality4 loads all 20 cameras and selects four using only detector "
            "confidence and ray conditioning (never 3D labels)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--h36m-cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/validation_c2_22c.npz"
        ),
    )
    return parser.parse_args()


def confidence_triangulation(rays: torch.Tensor, combo: tuple[int, ...]):
    subset = rays[:, :, list(combo)]
    direction = torch.nn.functional.normalize(subset[..., :3], dim=-1, eps=1e-8)
    point = subset[..., 3:6]
    confidence = subset[..., 6].clamp(0, 1) + 0.05
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    matrix = torch.einsum("bjv,bjvxy->bjxy", confidence, projection)
    rhs = torch.einsum("bjv,bjvxy,bjvy->bjx", confidence, projection, point)
    matrix = matrix + 1e-8 * eye
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def select_quality_four(rays: torch.Tensor):
    """Select a label-free, well-conditioned four-camera subset per frame."""
    if rays.shape[2] < 4:
        raise ValueError("quality selection requires at least four cameras")
    eye = torch.eye(3, dtype=rays.dtype)
    selected_rays, selected_indices = [], []
    all_combos = tuple(itertools.combinations(range(rays.shape[2]), 4))
    for item in rays:
        direction = torch.nn.functional.normalize(item[..., :3], dim=-1, eps=1e-8)
        confidence = item[..., 6].clamp(0, 1)
        best_score, best_combo = -float("inf"), None
        for combo in all_combos:
            indices = list(combo)
            d = direction[:, indices]
            w = confidence[:, indices] + 0.05
            projection = eye - d.unsqueeze(-1) * d.unsqueeze(-2)
            normal = (w.unsqueeze(-1).unsqueeze(-1) * projection).sum(dim=1)
            eigenvalues = torch.linalg.eigvalsh(normal).clamp_min(1e-8)
            conditioning = (eigenvalues[:, 0] / eigenvalues.sum(dim=-1)).mean()
            confidence_score = confidence[:, indices].mean()
            # Both terms are dimensionless and label-free.  Multiplication
            # rejects either well-spread but invisible cameras or confident
            # yet nearly collinear cameras.
            score = float(conditioning * confidence_score)
            if score > best_score:
                best_score, best_combo = score, combo
        selected_rays.append(item[:, list(best_combo)])
        selected_indices.append(list(best_combo))
    return torch.stack(selected_rays), selected_indices


def error_values(prediction: torch.Tensor, target: torch.Tensor):
    absolute = torch.linalg.vector_norm(prediction - target, dim=-1)
    pred_relative = prediction - prediction[:, :1]
    target_relative = target - target[:, :1]
    relative = torch.linalg.vector_norm(pred_relative - target_relative, dim=-1)
    root = torch.linalg.vector_norm(prediction[:, 0] - target[:, 0], dim=-1)
    return absolute, relative, root


def summarize_stage(predictions, targets):
    absolute, relative, root = zip(
        *(error_values(prediction, targets) for prediction in predictions)
    )
    absolute = torch.cat([value.flatten() for value in absolute])
    relative = torch.cat([value.flatten() for value in relative])
    root = torch.cat([value.flatten() for value in root])
    return {
        "absolute_mpjpe_mm": float(absolute.mean() * 1000),
        "root_relative_mpjpe_mm": float(relative.mean() * 1000),
        "root_translation_error_mm": float(root.mean() * 1000),
        "absolute_p95_mm": float(torch.quantile(absolute, 0.95) * 1000),
    }


@torch.inference_mode()
def evaluate_pool(generator, e2, candidate_combinations, temperature,
                  rays, targets):
    generated = torch.stack(
        [generator(rays[:, :, list(combo)], is_training=False)
         for combo in COMBINATIONS],
        dim=1,
    )
    triangulated = torch.stack(
        [confidence_triangulation(rays, combo) for combo in COMBINATIONS], dim=1
    )
    candidates = torch.cat((generated, triangulated), dim=1)
    if candidates.shape[1] != len(candidate_combinations):
        raise ValueError(
            f"built {candidates.shape[1]} candidates, E2 expects "
            f"{len(candidate_combinations)}"
        )
    scorer = e2_outputs(
        e2, candidates, rays, candidate_combinations, temperature
    )
    result = {}
    for count in (2, 3, 4):
        stage_combos = [combo for combo in COMBINATIONS if len(combo) == count]
        stage_indices = [COMBINATIONS.index(combo) for combo in stage_combos]
        result[f"V{count}"] = {
            "generator": summarize_stage(
                [generated[:, index] for index in stage_indices], targets
            ),
            "triangulation": summarize_stage(
                [triangulated[:, index] for index in stage_indices], targets
            ),
            "e2_soft": summarize_stage(
                [scorer[combo]["soft"] for combo in stage_combos], targets
            ),
            "e2_hard": summarize_stage(
                [scorer[combo]["hard"] for combo in stage_combos], targets
            ),
        }
    return result, generated, triangulated


def h36m_cache_control(e2, candidate_combinations, temperature, path, samples,
                       device):
    source = np.load(path, allow_pickle=False)
    count = min(samples, len(source["targets"]))
    predictions = torch.from_numpy(source["predictions"][:count]).to(device)
    targets = torch.from_numpy(source["targets"][:count]).to(device)
    rays = torch.from_numpy(source["rays"][:count]).to(device)
    scorer = e2_outputs(e2, predictions, rays, candidate_combinations, temperature)
    result = {}
    for count_views in (2, 3, 4):
        stage_combos = [combo for combo in COMBINATIONS if len(combo) == count_views]
        stage_indices = [COMBINATIONS.index(combo) for combo in stage_combos]
        result[f"V{count_views}"] = {
            "generator": summarize_stage(
                [predictions[:, index] for index in stage_indices], targets
            ),
            "triangulation": summarize_stage(
                [predictions[:, 11 + index] for index in stage_indices], targets
            ),
            "e2_soft": summarize_stage(
                [scorer[combo]["soft"] for combo in stage_combos], targets
            ),
            "e2_hard": summarize_stage(
                [scorer[combo]["hard"] for combo in stage_combos], targets
            ),
        }
    return result


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_model, architecture_environment = configure_generator(args.cfg, 1e-4)
    from core.config import config
    import dataset

    config.DATASET.TEST_DATASET = "multiview_amass_rumpl"
    config.DATASET.AMASS_DATASET_TYPE = args.amass_dataset_type
    config.DATASET.TEST_AMASS_WITH_RANDOM_CAMERAS = True
    config.DATASET.TEST_ON_ALL_CAMERAS = False
    config.DATASET.TEST_VIEWS = (
        list(range(20)) if args.camera_selection == "quality4"
        else [0, 1, 2, 3]
    )
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.TEST_N_SAMPLES = args.samples
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.USE_MMPOSE_TEST = True
    config.DATASET.FLIP_LOWER_BODY_KP_TEST = False
    config.DATASET.APPLY_NOISE_MISSING = False
    config.DATASET.APPLY_NOISE_MISSING_TEST = False
    config.DATASET.NO_AUGMENTATION = True
    config.DATASET.NO_AUGMENTATION_3D = True

    validation = dataset.multiview_amass_rumpl(
        config, "validation", False, transform=None
    )
    count = min(args.samples, len(validation))
    ray_items, target_items, camera_sets = [], [], []
    # Direct indexing keeps the random camera selection deterministic.
    for index in range(count):
        _, _, target, rays, metadata, _ = validation[index]
        ray_items.append(rays)
        target_items.append(target)
        camera_sets.append(metadata["fname_camera_ids"])
    rays = torch.stack(ray_items)
    selected_indices = None
    if args.camera_selection == "quality4":
        rays, selected_indices = select_quality_four(rays)
        camera_sets = [
            "_" + "_".join(str(view) for view in views)
            for views in selected_indices
        ]
    rays = rays.to(device)
    targets = torch.stack(target_items).to(device)

    generator = load_model(Path(args.generator_checkpoint).resolve(), device)
    e2, candidate_combinations, temperature = load_e2(
        Path(args.e2_checkpoint).resolve(), device
    )
    smpl_result, generated, triangulated = evaluate_pool(
        generator, e2, candidate_combinations, temperature, rays, targets
    )
    h36m_result = h36m_cache_control(
        e2, candidate_combinations, temperature, args.h36m_cache, count, device
    )

    confidence = rays[..., 6].detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    checkpoint = torch.load(
        args.e2_checkpoint, map_location="cpu", weights_only=False
    )
    payload = {
        "purpose": "small SMPL/MHP unseen random-camera distribution audit",
        "samples": count,
        "seed": args.seed,
        "camera_selection": args.camera_selection,
        "amass_dataset_type": args.amass_dataset_type,
        "camera_sets": camera_sets,
        "generator_checkpoint": str(Path(args.generator_checkpoint).resolve()),
        "e2_checkpoint": str(Path(args.e2_checkpoint).resolve()),
        "architecture_environment": architecture_environment,
        "smpl_random_camera": smpl_result,
        "h36m_first_n_control": h36m_result,
        "distribution": {
            "smpl_root_mean_m": target_np[:, 0].mean(axis=0).tolist(),
            "smpl_root_std_m": target_np[:, 0].std(axis=0).tolist(),
            "e2_h36m_pose_mean_m": checkpoint["mean"].numpy().tolist(),
            "e2_h36m_pose_std_m": checkpoint["std"].numpy().tolist(),
            "smpl_confidence_mean": float(confidence.mean()),
            "smpl_confidence_p10": float(np.quantile(confidence, 0.10)),
            "smpl_confidence_p50": float(np.quantile(confidence, 0.50)),
            "smpl_confidence_p90": float(np.quantile(confidence, 0.90)),
            "generator_vs_triangulation_disagreement_mm": float(
                torch.linalg.vector_norm(generated - triangulated, dim=-1).mean()
                * 1000
            ),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
