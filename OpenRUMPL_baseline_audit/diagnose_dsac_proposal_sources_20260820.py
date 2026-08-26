#!/usr/bin/env python3
"""Compare fixed E2 and trained limb-utility proposal distributions for K96.

This is a train-subject holdout diagnostic.  It does not change observations:
both proposal sources use only frozen coordinate/confidence/camera-derived
RUMPL candidates.  The trained PoseDSAC scorer is held fixed so that sampled
oracle and downstream scoring effects can be separated before implementing a
differentiable proposal sampler.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_limb_utility_20260820 as limb
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import GROUPS, train_bone_stats
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--limb-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hypotheses", type=int, default=96)
    parser.add_argument("--proposal-temperatures", nargs="+", type=float,
                        default=(0.4, 0.8, 1.2))
    parser.add_argument("--score-temperature", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def sample_pool(candidates, unary, group_cost, baseline_local, hypotheses,
                source, temperature):
    batch, count, joints, _ = candidates.shape
    labels = torch.empty(
        batch, hypotheses - 1, joints, dtype=torch.long, device=candidates.device
    )
    for group_id, group in enumerate(GROUPS.values()):
        idx = torch.as_tensor(group, device=candidates.device)
        if source == "e2":
            logits = -unary[:, idx].mean(dim=1) / temperature
        else:
            logits = -group_cost[:, group_id] / temperature
        probability = 0.8 * torch.softmax(logits, dim=-1) + 0.2 / count
        draw = torch.multinomial(probability, hypotheses - 1, replacement=True)
        labels[:, :, idx] = draw[:, :, None]
    labels[:, 0] = baseline_local
    labels[:, 1] = unary.argmin(dim=-1)
    mixed = dsac.gather_hypotheses(candidates, labels)
    selected = dsac.gather_unary(unary, labels)
    weight = F.softmax(-unary / temperature, dim=-1)
    soft_pose = torch.einsum("bjc,bcjd->bjd", weight, candidates)
    soft_unary = (weight * unary).sum(dim=-1)
    return (
        torch.cat((mixed, soft_pose[:, None]), dim=1),
        torch.cat((selected, soft_unary[:, None]), dim=1),
    )


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def evaluate(source, proposal_temperature, scorer, limb_model, e2, loader,
             device, args, coord_mean, coord_std, bone_mean, bone_std):
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    torch.manual_seed(10000)
    torch.cuda.manual_seed_all(10000)
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions, targets, rays = (
                predictions.to(device), targets.to(device), rays.to(device)
            )
            for combo in TASKS:
                stage = f"V{len(combo)}"
                unary, _, _, candidates, baseline_local = extra.predict_task(
                    e2, predictions, targets, rays, combo
                )
                group_cost = limb_model(
                    candidates, unary, rays, combo, bone_mean, bone_std
                )
                hypotheses, selected = sample_pool(
                    candidates, unary, group_cost, baseline_local,
                    args.hypotheses, source, proposal_temperature,
                )
                scores = scorer(
                    hypotheses, selected, coord_mean, coord_std,
                    bone_mean, bone_std,
                )
                weights = F.softmax(scores / args.score_temperature, dim=-1)
                fused = torch.einsum("bk,bkjd->bjd", weights, hypotheses)
                hyp_error = torch.linalg.vector_norm(
                    hypotheses - targets[:, None], dim=-1
                ).mean(dim=-1)
                pose_error = torch.linalg.vector_norm(fused - targets, dim=-1)
                store[stage]["weighted"].append(pose_error.cpu().numpy() * 1000.0)
                store[stage]["sampled_oracle"].append(
                    hyp_error.min(dim=-1).values.cpu().numpy()[:, None] * 1000.0
                )
                action_store[stage].append(actions.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        result[stage] = {}
        actions = np.concatenate(action_store[stage])
        for mode, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = action_equal(values, actions)
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    scorer_state = torch.load(
        args.scorer_checkpoint, map_location=device, weights_only=False
    )
    saved = SimpleNamespace(**scorer_state["args"])
    scorer = dsac.PoseScoreMLP(
        saved.features, getattr(saved, "sigmoid_score", False)
    ).to(device)
    scorer.load_state_dict(scorer_state["state_dict"], strict=True)
    scorer.eval()
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    limb_state = torch.load(
        args.limb_checkpoint, map_location=device, weights_only=False
    )
    limb_model = limb.LimbUtility(coord_mean, coord_std).to(device)
    limb_model.load_state_dict(limb_state["state_dict"], strict=True)
    limb_model.eval()
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    arrays = trainer.load_arrays([args.cache], 22)
    keep = arrays["group_indices"] % 10 == 0
    arrays = {key: value[keep] for key, value in arrays.items()}
    if args.max_examples:
        arrays = {key: value[:args.max_examples] for key, value in arrays.items()}
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    rows = []
    for source in ("e2", "limb_utility"):
        for temperature in args.proposal_temperatures:
            result = evaluate(
                source, temperature, scorer, limb_model, e2, loader, device,
                args, coord_mean, coord_std, bone_mean, bone_std,
            )
            row = {
                "proposal_source": source,
                "proposal_temperature": temperature,
                "weighted_mm": {
                    stage: result[stage]["weighted"] for stage in ("V2", "V3", "V4")
                },
                "sampled_oracle_mm": {
                    stage: result[stage]["sampled_oracle"] for stage in ("V2", "V3", "V4")
                },
            }
            row["weighted_headline_mm"] = float(np.mean(list(row["weighted_mm"].values())))
            row["oracle_headline_mm"] = float(np.mean(list(row["sampled_oracle_mm"].values())))
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "protocol": "train-subject internal holdout only; fixed scorer and sampling seed",
        "input_protocol": "coordinate/confidence/cameras and derived candidates only",
        "source_code": "reference/differentiable-ransac-official/samplers/gumbel_sampler.py",
        "num_examples": len(arrays["targets"]),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
