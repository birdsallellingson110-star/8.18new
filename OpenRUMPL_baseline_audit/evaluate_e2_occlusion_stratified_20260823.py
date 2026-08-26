#!/usr/bin/env python3
"""Evaluate frozen E2 and stratify errors by occluded views in each subset."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_v234_universal_20260812 as trainer
from train_h76_set_transformer_utility_20260811 import (
    ArrayDataset,
    SetTransformerJointUtility,
)


COMBINATIONS = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--validation-pkl", required=True, type=Path)
    parser.add_argument("--selection-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--v2-temperature", type=float, default=0.4)
    parser.add_argument("--v3-temperature", type=float, default=1.8)
    parser.add_argument("--v4-temperature", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def key_from_record(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(
        int(record[name]) for name in ("subject", "action", "subaction", "image_id")
    )


def ordered_pickle_group_keys(path: Path) -> list[tuple[int, int, int, int]]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    groups: OrderedDict[tuple[int, int, int, int], set[int]] = OrderedDict()
    for record in records:
        groups.setdefault(key_from_record(record), set()).add(int(record["camera_id"]))
    keys = [key for key, cameras in groups.items() if cameras == {0, 1, 2, 3}]
    if len(keys) != len(groups):
        raise RuntimeError("validation pickle contains incomplete camera groups")
    return keys


def load_selections(path: Path) -> dict[tuple[int, int, int, int], set[int]]:
    selections = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = tuple(
                int(row[name]) for name in ("subject", "action", "subaction", "image_id")
            )
            selections[key] = set(int(value) for value in row["occluded_camera_ids"])
    return selections


def action_equal(errors: np.ndarray, actions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                errors[actions == action].mean()
                for action in trainer.ACTION_NAMES
                if np.any(actions == action)
            ]
        )
    )


def summarize(errors: np.ndarray, actions: np.ndarray) -> dict[str, float | int]:
    return {
        "action_equal_all17_mm": action_equal(errors, actions),
        "frame_weighted_all17_mm": float(errors.mean()),
        "samples": int(len(errors)),
    }


def evaluate_seed(
    model: SetTransformerJointUtility,
    loader: DataLoader,
    device: torch.device,
    temperatures: dict[str, float],
) -> dict[tuple[int, ...], np.ndarray]:
    model.eval()
    stores = {combo: [] for combo in COMBINATIONS}
    with torch.inference_mode():
        for predictions, targets, rays, _ in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for combo in COMBINATIONS:
                predicted, _, _, candidates, _ = trainer.predict_task(
                    model, predictions, targets, rays, combo
                )
                weights = F.softmax(
                    -predicted / temperatures[f"V{len(combo)}"], dim=-1
                )
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                errors = (
                    torch.linalg.vector_norm(soft - targets, dim=-1).cpu().numpy()
                    * 1000.0
                )
                stores[combo].append(errors)
    return {combo: np.concatenate(chunks) for combo, chunks in stores.items()}


def main() -> None:
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    trainer.TASK_COMBINATIONS = COMBINATIONS
    arrays = trainer.load_arrays([str(args.cache)], 22)
    raw_cache = np.load(args.cache)
    group_indices = raw_cache["group_indices"].astype(np.int64)
    actions = raw_cache["actions"].astype(np.int64)
    if len(group_indices) != len(arrays["targets"]):
        raise RuntimeError("cache group indices and loaded arrays differ")

    pkl_keys = ordered_pickle_group_keys(args.validation_pkl)
    selection_by_key = load_selections(args.selection_jsonl)
    selected_sets = []
    for index in group_indices:
        key = pkl_keys[int(index)]
        if key not in selection_by_key:
            raise KeyError(f"no occlusion selection for group {key}")
        selected_sets.append(selection_by_key[key])

    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    temperatures = {
        "V2": args.v2_temperature,
        "V3": args.v3_temperature,
        "V4": args.v4_temperature,
    }
    device = torch.device(f"cuda:{args.gpu}")
    per_seed: dict[str, Any] = {}
    seed_errors: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
    for seed in (0, 1):
        checkpoint = args.checkpoint_root / f"seed{seed}" / "model_best.pth.tar"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = SetTransformerJointUtility(
            state["mean"],
            state["std"],
            state["attention_depth"],
            stage_heads=state.get("stage_heads", False),
        ).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        errors_by_combo = evaluate_seed(model, loader, device, temperatures)
        seed_errors[seed] = errors_by_combo
        seed_payload = {}
        for count in (2, 3, 4):
            combos = [combo for combo in COMBINATIONS if len(combo) == count]
            errors = np.concatenate([errors_by_combo[combo] for combo in combos])
            stage_actions = np.tile(actions, len(combos))
            occ_counts = np.concatenate(
                [
                    np.asarray([len(set(combo) & selected) for selected in selected_sets])
                    for combo in combos
                ]
            )
            seed_payload[f"V{count}"] = {
                "all": summarize(errors, stage_actions),
                "by_occluded_views_in_selected_subset": {
                    str(value): summarize(errors[occ_counts == value], stage_actions[occ_counts == value])
                    for value in sorted(set(occ_counts.tolist()))
                },
                "per_camera_combination": {
                    "-".join(str(index + 1) for index in combo): summarize(
                        errors_by_combo[combo], actions
                    )
                    for combo in combos
                },
            }
        per_seed[str(seed)] = seed_payload

    mean_summary = {}
    for count in (2, 3, 4):
        stage = f"V{count}"
        strata = per_seed["0"][stage]["by_occluded_views_in_selected_subset"]
        mean_summary[stage] = {
            "all_action_equal_all17_mm": float(
                np.mean(
                    [per_seed[str(seed)][stage]["all"]["action_equal_all17_mm"] for seed in (0, 1)]
                )
            ),
            "by_occluded_views_action_equal_all17_mm": {
                value: float(
                    np.mean(
                        [
                            per_seed[str(seed)][stage]["by_occluded_views_in_selected_subset"][value][
                                "action_equal_all17_mm"
                            ]
                            for seed in (0, 1)
                        ]
                    )
                )
                for value in strata
            },
        }

    payload = {
        "method": "frozen E2 soft candidate fusion, occlusion-stratified",
        "protocol": "H36M S9/S11, all subsets, action-equal All-17 absolute MPJPE",
        "cache": str(args.cache.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "validation_pkl": str(args.validation_pkl.resolve()),
        "selection_jsonl": str(args.selection_jsonl.resolve()),
        "temperatures": temperatures,
        "groups": len(group_indices),
        "mean_over_two_seeds": mean_summary,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(mean_summary, indent=2))


if __name__ == "__main__":
    main()
