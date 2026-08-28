#!/usr/bin/env python3
"""Select the frozen sparse scoring centers from a dense frontend cache.

The dense VOC frontend contains all synchronized H36M groups, while the
official Stage-2 score uses the 2,021 center groups from the sparse c2
benchmark.  This adapter keeps the dense frontend's current 2-D observations
and image names, but selects exactly the sparse center keys for a matched
reevaluation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def key(record: dict) -> tuple[int, ...]:
    return (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        int(record["image_id"]),
        int(record["camera_id"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-pkl", required=True, type=Path)
    parser.add_argument("--center-pkl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.dense_pkl.open("rb") as handle:
        dense = pickle.load(handle)
    with args.center_pkl.open("rb") as handle:
        centers = pickle.load(handle)

    center_keys = {key(record) for record in centers}
    selected = [record for record in dense if key(record) in center_keys]
    selected_keys = {key(record) for record in selected}
    if selected_keys != center_keys:
        missing = sorted(center_keys - selected_keys)
        raise RuntimeError(
            f"dense frontend is missing {len(missing)} center records; "
            f"first missing={missing[:3]}"
        )
    if len(selected) != len(centers):
        raise RuntimeError(
            f"record count mismatch: selected={len(selected)} centers={len(centers)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"selected_records={len(selected)} center_keys={len(center_keys)} "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
