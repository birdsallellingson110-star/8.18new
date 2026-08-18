#!/usr/bin/env python3
"""Convert Learnable-Triangulation's H36M ResNet checkpoint to MMPose names.

The public LT checkpoint is a plain ``module.*`` state dict.  Its final
convolution has 33 channels although the published H36M configuration uses
17 joints; the first 17 channels are the H36M heatmap channels used by the
official LT model.  We keep only those channels and write a normal MMPose
``state_dict`` so the existing exporter can be reused unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    src = torch.load(args.input, map_location="cpu")
    if not isinstance(src, dict):
        raise TypeError(f"expected state dict, got {type(src)}")
    dst = {}
    for key, value in src.items():
        if not key.startswith("module."):
            continue
        key = key[len("module."):]
        if key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
            out_key = "backbone." + key
        elif key.startswith("deconv_layers.") or key.startswith("final_layer."):
            out_key = "head." + key
        else:
            continue
        if key == "final_layer.weight":
            if value.shape[0] < 17:
                raise ValueError(f"unexpected LT final layer shape: {tuple(value.shape)}")
            value = value[:17].clone()
        elif key == "final_layer.bias":
            value = value[:17].clone()
        dst[out_key] = value
    required = ["backbone.conv1.weight", "backbone.layer4.0.conv1.weight",
                "head.deconv_layers.0.weight", "head.final_layer.weight"]
    missing = [k for k in required if k not in dst]
    if missing:
        raise RuntimeError(f"missing converted keys: {missing}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dst, "source": str(Path(args.input).resolve()),
                "final_layer_source_channels": 33, "final_layer_kept_channels": 17}, args.output)
    print(f"saved {args.output}: {len(dst)} tensors")


if __name__ == "__main__":
    main()
