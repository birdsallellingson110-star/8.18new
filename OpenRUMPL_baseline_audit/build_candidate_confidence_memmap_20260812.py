#!/usr/bin/env python3
"""Build per-candidate confidence summaries for the I3 T-CVU ablation.

The row order is the concatenation order used by the frozen H76/E-2 export.
For every joint and every 2/3/4-view H76 candidate, the feature is the mean
of the frozen detector confidences over the views in that candidate.  No
ground-truth quantity is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_h76_hypothesis_utility_20260811 import COMBINATIONS, load_arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()

    arrays = load_arrays(args.shards)
    rays = arrays["rays"]
    confidence = np.asarray(rays[..., 6], dtype=np.float32)  # N,J,V
    output_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "candidate_confidence.npy"
    shape = (len(confidence), len(COMBINATIONS), confidence.shape[1])
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=shape
    )
    for start in range(0, len(confidence), args.chunk):
        stop = min(start + args.chunk, len(confidence))
        block = confidence[start:stop]
        for candidate_index, combination in enumerate(COMBINATIONS):
            output[start:stop, candidate_index] = block[
                :, :, np.asarray(combination, dtype=np.int64)
            ].mean(axis=-1)
    output.flush()
    manifest_path = output_dir / "candidate_confidence_manifest.json"
    manifest = {
        "shards": args.shards,
        "output": str(destination),
        "shape": list(shape),
        "dtype": "float32",
        "definition": "mean ray confidence over views in each H76 candidate",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
