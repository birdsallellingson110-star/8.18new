"""Adapt view-dependent tensors when loading RUMPL checkpoints across view counts."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


def _adapt_weighted_mean_weight(
    target: torch.Tensor, value: torch.Tensor
) -> torch.Tensor | None:
    if target.shape == value.shape:
        return value
    if target.dim() == 3 and value.dim() == 3:
        if target.shape[0] != value.shape[0] or target.shape[2] != value.shape[2]:
            return None
        new_w = target.clone()
        old_in = value.shape[1]
        new_in = target.shape[1]
        copy_in = min(old_in, new_in)
        new_w[:, :copy_in, :] = value[:, :copy_in, :]
        if new_in > old_in:
            pad = value.mean(dim=1, keepdim=True).expand(
                target.shape[0], new_in - old_in, target.shape[2]
            )
            new_w[:, old_in:new_in, :] = pad
        return new_w
    if target.dim() == 2 and value.dim() == 2:
        if target.shape[0] != value.shape[0]:
            return None
        new_w = target.clone()
        old_in = value.shape[1]
        new_in = target.shape[1]
        copy_in = min(old_in, new_in)
        new_w[:, :copy_in] = value[:, :copy_in]
        if new_in > old_in:
            col_mean = value.mean(dim=1, keepdim=True)
            new_w[:, old_in:new_in] = col_mean.expand(target.shape[0], new_in - old_in)
        return new_w
    return None


def adapt_pretrained_state_dict(
    model_state: Dict[str, torch.Tensor],
    pretrained_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    adapted: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in pretrained_state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if target.shape == value.shape:
            adapted[key] = value
            continue
        if key.endswith("weighted_mean.weight"):
            merged = _adapt_weighted_mean_weight(target, value)
            if merged is not None:
                adapted[key] = merged
                logger.info(
                    "adapt_pretrained_state_dict: expanded/sliced %s %s -> %s",
                    key,
                    tuple(value.shape),
                    tuple(merged.shape),
                )
                continue
        skipped.append(key)
    return adapted, skipped


def merge_pretrained_into_model_state(
    model_state: Dict[str, torch.Tensor],
    pretrained_state: Dict[str, torch.Tensor],
    strict_shapes: bool,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Build a loadable state dict; optionally adapt view-dependent layers."""
    if strict_shapes:
        return pretrained_state, []
    view_adapted, view_skipped = adapt_pretrained_state_dict(
        model_state, pretrained_state
    )
    merged = dict(view_adapted)
    skipped = list(view_skipped)
    for key, value in pretrained_state.items():
        if key in merged:
            continue
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            merged[key] = value
        else:
            skipped.append(key)
    return merged, skipped
