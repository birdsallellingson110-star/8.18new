#!/usr/bin/env python3
"""Report the frozen candidate generator on a H36M-Occl 11-candidate cache.

For each two-, three-, or four-camera task, the direct prediction is the
candidate generated from exactly that camera subset.  Results use the same
action-equal, All-17, absolute MPJPE convention as the Stage-I tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet",
    6: "Phone", 7: "Photo", 8: "Pose", 9: "Purchase",
    10: "Sitting", 11: "SittingDown", 12: "Smoke", 13: "Wait",
    14: "WalkDog", 15: "Walk", 16: "WalkTwo",
}
COMBINATIONS = tuple(
    combo
    for cardinality in (2, 3, 4)
    for combo in itertools.combinations(range(4), cardinality)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def action_equal(errors: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        errors[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def main() -> None:
    args = parse_args()
    source = Path(args.cache).resolve()
    arrays = np.load(source)
    predictions = arrays["predictions"]
    targets = arrays["targets"]
    actions = arrays["actions"]
    subjects = arrays["subjects"] if "subjects" in arrays else None
    if predictions.shape[1:] != (11, 17, 3):
        raise ValueError(f"unexpected predictions shape: {predictions.shape}")

    stages: dict[str, dict] = {}
    for cardinality in (2, 3, 4):
        combo_rows = []
        all_errors = []
        all_actions = []
        for combo in itertools.combinations(range(4), cardinality):
            candidate = predictions[:, COMBINATIONS.index(combo)]
            errors = np.linalg.norm(candidate - targets, axis=-1) * 1000.0
            combo_rows.append({
                "cameras_zero_based": list(combo),
                "action_equal_all17_mm": action_equal(errors, actions),
                "frame_weighted_all17_mm": float(errors.mean()),
            })
            all_errors.append(errors)
            all_actions.append(actions)
        stacked_errors = np.concatenate(all_errors, axis=0)
        stacked_actions = np.concatenate(all_actions, axis=0)
        stages[f"V{cardinality}"] = {
            "action_equal_all17_mm": action_equal(stacked_errors, stacked_actions),
            "frame_weighted_all17_mm": float(stacked_errors.mean()),
            "camera_combinations": combo_rows,
        }

    payload = {
        "method": "frozen candidate generator, exact-subset direct prediction",
        "protocol": "H36M; all camera combinations; action-equal All-17 absolute MPJPE",
        "subjects": (
            sorted(set(map(int, subjects.tolist()))) if subjects is not None else None
        ),
        "cache": str(source),
        "groups": int(len(targets)),
        "results": stages,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
