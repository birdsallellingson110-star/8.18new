#!/usr/bin/env python3
"""Zero-training diagnostic for geometry-generated H76 candidates.

The script keeps the frozen RUMPL/H76 candidates untouched and asks only whether
additional, deterministic ray-intersection hypotheses can lower the candidate
oracle.  It is intentionally a validation/test diagnostic, not a learned
method: no parameter or checkpoint is selected from the result.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    COMBINATIONS,
    JOINT_NAMES,
    TASK_COMBINATIONS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--irls-iters", type=int, default=3)
    parser.add_argument("--huber-threshold-m", type=float, default=0.03)
    return parser.parse_args()


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def ray_solver(
    rays: np.ndarray,
    subset: tuple[int, ...],
    mode: str,
    irls_iters: int,
    huber_threshold_m: float,
) -> np.ndarray:
    """Solve weighted closest-point triangulation for every frame/joint."""
    # Cache layout is N,J,V,7: unit-like direction, point on ray, confidence.
    subset_indices = list(subset)
    directions = rays[:, :, subset_indices, :3].astype(np.float64)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-8)
    points = rays[:, :, subset_indices, 3:6].astype(np.float64)
    identity = np.eye(3, dtype=np.float64)
    projections = (
        identity[None, None, None]
        - directions[..., :, None] * directions[..., None, :]
    )
    detector_conf = rays[:, :, subset_indices, 6].astype(np.float64).clip(0.0, 1.0)

    if mode == "uniform":
        weights = np.ones_like(detector_conf)
    else:
        weights = detector_conf + 0.05

    def solve(current_weights: np.ndarray) -> np.ndarray:
        matrix = np.einsum("njv,njvab->njab", current_weights, projections)
        rhs = np.einsum(
            "njv,njvab,njvb->nja", current_weights, projections, points
        )
        # The geometry is generally well-conditioned for 3/4 views.  The small
        # Tikhonov term only protects nearly parallel rays in diagnostic frames.
        matrix = matrix + 1e-8 * identity[None, None]
        # NumPy treats a 3-D right-hand side as ``(..., M, K)`` rather than
        # as a vector for each (N,J) system.  Add an explicit singleton K
        # dimension so the per-joint 3-vector is solved correctly across the
        # whole cache (this also keeps the function valid for large temporal
        # validation shards).
        return np.linalg.solve(matrix, rhs[..., None])[..., 0]

    estimate = solve(weights)
    if mode != "irls":
        return estimate.astype(np.float32)

    for _ in range(max(0, irls_iters)):
        residual = np.linalg.norm(
            np.cross(estimate[:, :, None, :] - points, directions), axis=-1
        )
        robust = np.minimum(1.0, huber_threshold_m / (residual + 1e-8))
        weights = (detector_conf + 0.05) * robust
        estimate = solve(weights)
    return estimate.astype(np.float32)


def error_matrix(candidates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return N,C,J Euclidean errors in metres."""
    return np.linalg.norm(candidates - targets[:, None], axis=-1)


def summarize(
    errors: np.ndarray, actions: np.ndarray, label: str
) -> dict[str, float | str]:
    # errors is N,J and metrics are reported in mm.
    mm = errors * 1000.0
    return {
        "label": label,
        "action_equal_all17_mm": action_equal(mm, actions),
        "frame_weighted_all17_mm": float(mm.mean()),
    }


def main() -> None:
    args = parse_args()
    cache = np.load(args.validation_cache)
    predictions = cache["predictions"]
    targets = cache["targets"]
    rays = cache["rays"]
    actions = cache["actions"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    raw_candidates: dict[str, list[dict]] = {}
    for task in TASK_COMBINATIONS:
        stage = f"V{len(task)}"
        task_name = "".join(str(v) for v in task)
        available = [
            index for index, combo in enumerate(COMBINATIONS)
            if set(combo).issubset(task)
        ]
        existing = predictions[:, available]
        existing_errors = error_matrix(existing, targets)
        candidate_sets: dict[str, list[np.ndarray]] = {
            "uniform_all": [ray_solver(
                rays, task, "uniform", args.irls_iters, args.huber_threshold_m
            )],
            "confidence_all": [ray_solver(
                rays, task, "confidence", args.irls_iters, args.huber_threshold_m
            )],
            "irls_all": [ray_solver(
                rays, task, "irls", args.irls_iters, args.huber_threshold_m
            )],
            "pairwise_uniform": [ray_solver(
                rays, pair, "uniform", args.irls_iters, args.huber_threshold_m
            ) for pair in itertools.combinations(task, 2)],
            "pairwise_confidence": [ray_solver(
                rays, pair, "confidence", args.irls_iters, args.huber_threshold_m
            ) for pair in itertools.combinations(task, 2)],
            "pairwise_irls": [ray_solver(
                rays, pair, "irls", args.irls_iters, args.huber_threshold_m
            ) for pair in itertools.combinations(task, 2)],
        }

        task_results: dict[str, dict] = {
            "task": list(task),
            "num_existing_candidates": len(available),
            "existing": summarize(
                existing_errors.min(axis=1), actions, "existing_oracle"
            ),
            "baseline": summarize(
                error_matrix(
                    predictions[:, COMBINATIONS.index(task)][:, None], targets
                )[:, 0],
                actions,
                "full_input_h76",
            ),
        }
        existing_per_joint_oracle = existing_errors.min(axis=1)
        for variant, additions in candidate_sets.items():
            addition_errors = [error_matrix(item[:, None], targets)[:, 0] for item in additions]
            direct = np.stack(addition_errors, axis=1)
            pool = np.concatenate([existing_errors, direct], axis=1)
            pool_oracle = pool.min(axis=1)
            task_results[variant] = {
                "num_added_candidates": len(additions),
                "direct_best": summarize(direct.min(axis=1), actions, "direct_best"),
                "pool_oracle": summarize(pool_oracle, actions, "pool_oracle"),
                "oracle_gain_mm": float(
                    (existing_per_joint_oracle.mean() - pool_oracle.mean()) * 1000.0
                ),
                "oracle_improved_joint_fraction": float(
                    (pool_oracle < existing_per_joint_oracle - 1e-9).mean()
                ),
            }
        results[stage] = results.get(stage, {"tasks": {}})
        results[stage]["tasks"][task_name] = task_results

        raw_candidates.setdefault(stage, []).append({
            "task": list(task),
            "existing_errors": existing_errors.astype(np.float32),
            "candidate_sets": {
                key: [item.astype(np.float32) for item in value]
                for key, value in candidate_sets.items()
            },
        })

    # Aggregate the same action-equal protocol used by E2: concatenate each
    # nested task's per-frame/per-joint errors before averaging by action.
    for stage in ("V3", "V4"):
        stage_summary: dict[str, dict] = {}
        for variant in (
            "existing", "uniform_all", "confidence_all", "irls_all",
            "pairwise_uniform", "pairwise_confidence", "pairwise_irls",
        ):
            oracle_chunks = []
            for item in raw_candidates[stage]:
                existing_errors = item["existing_errors"]
                if variant == "existing":
                    oracle_chunks.append(existing_errors.min(axis=1))
                else:
                    additions = item["candidate_sets"][variant]
                    direct = np.stack([
                        error_matrix(candidate[:, None], targets)[:, 0]
                        for candidate in additions
                    ], axis=1)
                    oracle_chunks.append(
                        np.concatenate([existing_errors, direct], axis=1).min(axis=1)
                    )
            errors = np.concatenate(oracle_chunks, axis=0)
            repeated_actions = np.tile(actions, len(raw_candidates[stage]))
            stage_summary[variant] = summarize(
                errors, repeated_actions, f"{stage}_{variant}_oracle"
            )
        results[stage]["aggregate_oracle"] = stage_summary

    # Arrays are not needed after computing the JSON; retain compact metadata
    # only so the output stays small and reproducible.
    results["config"] = {
        "validation_cache": args.validation_cache,
        "num_frames": int(len(targets)),
        "irls_iters": args.irls_iters,
        "huber_threshold_m": args.huber_threshold_m,
        "ray_layout": "N,J,V,(direction[3],point[3],confidence)",
        "note": "zero-training candidate-pool oracle diagnostic; no checkpoint selection",
    }
    (output_dir / "result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# H76 zero-training candidate-pool diagnostic (2026-08-12)",
        "",
        "新增候选只由四条冻结世界射线计算，未训练任何参数；oracle 仅用于判断候选池是否有上限，不能作为模型结果。",
        "",
        "| stage | existing oracle | +uniform | +confidence | +IRLS | +pairwise-u | +pairwise-c | +pairwise-IRLS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("V3", "V4"):
        values = results[stage]["aggregate_oracle"]
        lines.append(
            f"| {stage} | {values['existing']['action_equal_all17_mm']:.4f} "
            f"| {values['uniform_all']['action_equal_all17_mm']:.4f} "
            f"| {values['confidence_all']['action_equal_all17_mm']:.4f} "
            f"| {values['irls_all']['action_equal_all17_mm']:.4f} "
            f"| {values['pairwise_uniform']['action_equal_all17_mm']:.4f} "
            f"| {values['pairwise_confidence']['action_equal_all17_mm']:.4f} "
            f"| {values['pairwise_irls']['action_equal_all17_mm']:.4f} |"
        )
    lines.extend([
        "",
        "解释：只有新增候选显著降低 oracle，才值得把 E2 utility scorer 接到扩展候选池。",
        "完整 JSON：`result.json`。",
    ])
    (output_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
