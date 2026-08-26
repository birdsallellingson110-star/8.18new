#!/usr/bin/env python3
"""Average predeclared RUMPL state-dict checkpoints without test selection."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [Path(item) for item in args.inputs]
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required")
    states = [torch.load(path, map_location="cpu") for path in paths]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("checkpoint state-dict keys do not match")

    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if tensors[0].is_floating_point():
            accumulator = tensors[0].to(torch.float64)
            for tensor in tensors[1:]:
                accumulator.add_(tensor.to(torch.float64))
            averaged[key] = accumulator.div_(len(tensors)).to(tensors[0].dtype)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating tensor differs: {key}")
            averaged[key] = tensors[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output)
    print(f"averaged={len(paths)} output={output}")


if __name__ == "__main__":
    main()
