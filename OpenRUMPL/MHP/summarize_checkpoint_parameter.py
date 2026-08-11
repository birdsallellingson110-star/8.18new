#!/usr/bin/env python3
"""Print checkpoint parameters whose names end with requested suffixes."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("suffix", nargs="+")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    matched = False
    for name, value in sorted(state.items()):
        if any(name.endswith(suffix) for suffix in args.suffix):
            matched = True
            flat = value.detach().float().reshape(-1)
            print(f"{name}: {flat.tolist()}")
    if not matched:
        raise SystemExit(f"No parameters matched suffixes: {args.suffix}")


if __name__ == "__main__":
    main()
