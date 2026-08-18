#!/usr/bin/env python3
"""Decompress frozen E-2 candidate exports into read-only ``.npy`` arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()
    source = np.load(args.input)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = (
        "group_indices", "subjects", "actions", "targets",
        "candidate_poses", "utility_delta",
    )
    manifests = {}
    for name in names:
        array = source[name]
        destination = out / f"{name}.npy"
        memmap = np.lib.format.open_memmap(
            destination, mode="w+", dtype=array.dtype, shape=array.shape
        )
        for start in range(0, len(array), args.chunk):
            stop = min(start + args.chunk, len(array))
            memmap[start:stop] = array[start:stop]
        memmap.flush()
        manifests[name] = {
            "path": str(destination),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    manifest = {"source": args.input, "arrays": manifests}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
