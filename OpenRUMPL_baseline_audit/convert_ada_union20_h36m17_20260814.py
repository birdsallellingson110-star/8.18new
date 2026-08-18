#!/usr/bin/env python3
"""Extract the 17 valid H36M channels from an AdaFuse-style 20-joint head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


H36M_TO_UNION20 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError("expected training checkpoint containing state_dict")
    state = dict(payload["state_dict"])
    for key in ("head.final_layer.weight", "head.final_layer.bias"):
        if key not in state:
            raise KeyError(key)
        value = state[key]
        if value.shape[0] != 20:
            raise ValueError(f"{key}: expected 20 channels, got {tuple(value.shape)}")
        state[key] = value[H36M_TO_UNION20].clone()
    out = {
        "state_dict": state,
        "source": str(Path(args.input).resolve()),
        "joint_mapping_union20_to_h36m": H36M_TO_UNION20,
        "source_channels": 20,
        "output_channels": 17,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, destination)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
