#!/usr/bin/env python3
"""Create the auditable [::stride] complete-four-view H36M subset."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path


def complete_groups(records):
    groups = OrderedDict()
    for index, record in enumerate(records):
        key = (int(record["subject"]), int(record["action"]),
               int(record["subaction"]), int(record["image_id"]))
        groups.setdefault(key, [-1] * 4)[int(record["camera_id"])] = index
    return [group for group in groups.values() if min(group) >= 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    records = pickle.load(open(args.input, "rb"))
    groups = complete_groups(records)
    selected = groups[::args.stride]
    flattened = [records[index] for group in selected for index in group]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(flattened, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "source": str(Path(args.input).resolve()),
        "output": str(destination.resolve()),
        "stride": args.stride,
        "source_records": len(records),
        "source_complete_groups": len(groups),
        "selected_groups": len(selected),
        "output_records": len(flattened),
        "selection": "complete groups in insertion order, then groups[::stride]",
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
