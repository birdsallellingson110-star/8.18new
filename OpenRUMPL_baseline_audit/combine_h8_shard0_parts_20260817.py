#!/usr/bin/env python3
"""Combine the two half-shards used to accelerate H8 validation recovery."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs=2, required=True)
    parser.add_argument("--manifests", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected", type=int, default=26269)
    args = parser.parse_args()

    predictions = []
    manifests = []
    for part_name, manifest_name in zip(args.parts, args.manifests):
        with Path(part_name).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list):
            raise RuntimeError(f"{part_name} is not a prediction list")
        predictions.extend(payload)
        with Path(manifest_name).open("r", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    predictions.sort(key=lambda item: int(item["record_index"]))
    indices = [int(item["record_index"]) for item in predictions]
    if len(predictions) != args.expected or indices != list(range(0, args.expected * 4, 4)):
        raise RuntimeError(
            f"combined shard has {len(predictions)} entries or unexpected indices; "
            f"first={indices[:3]}, last={indices[-3:]}"
        )
    if any(int(item.get("error_count", -1)) != 0 for item in manifests):
        raise RuntimeError("a part manifest reports errors")

    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    merged = {
        "protocol": "GBT-aligned-HRNet-coordinate-only-v2",
        "command": "combine_h8_shard0_parts_20260817.py",
        "record_count_total": 105076,
        "record_count_selected": len(predictions),
        "prediction_count": len(predictions),
        "error_count": 0,
        "shard_id": 0,
        "num_shards": 4,
        "score_threshold": 0.01,
        "fallback_record_box": True,
        "fallback_count": sum(int(item.get("fallback_count", 0)) for item in manifests),
        "fallbacks": [
            fallback
            for item in manifests
            for fallback in item.get("fallbacks", [])
        ],
        "part_manifests": [str(Path(x).resolve()) for x in args.manifests],
    }
    manifest_path = Path(args.manifest)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    manifest_tmp.replace(manifest_path)
    print(json.dumps(merged, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
