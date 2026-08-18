#!/usr/bin/env python3
"""Select a deterministic subject/action-covered prefix for feature probing."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=20000)
    args = p.parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_four_view_groups(records)
    keys = []
    for group in groups:
        record = records[group[0]]
        keys.append((int(record["subject"]), int(record["action"]), int(record["subaction"]), int(record["image_id"])))
    # Round-robin over subjects, then actions.  This is a coverage probe, not
    # a claimed production sampling strategy; it prevents the first-prefix
    # subject-1 domain from masquerading as a model failure.
    buckets = defaultdict(list)
    for i, (subject, action, _, _) in enumerate(keys):
        buckets[(subject, action)].append(i)
    ordered_keys = sorted(buckets)
    chosen = []
    cursor = {key: 0 for key in ordered_keys}
    while len(chosen) < min(args.count, len(groups)):
        progressed = False
        for key in ordered_keys:
            index = cursor[key]
            if index < len(buckets[key]):
                chosen.append(buckets[key][index])
                cursor[key] = index + 1
                progressed = True
                if len(chosen) >= min(args.count, len(groups)):
                    break
        if not progressed:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(chosen, dtype=np.int64))
    print({"groups": len(chosen), "subjects": sorted(set(keys[i][0] for i in chosen)),
           "actions": sorted(set(keys[i][1] for i in chosen)), "output": str(output)})


if __name__ == "__main__":
    main()
