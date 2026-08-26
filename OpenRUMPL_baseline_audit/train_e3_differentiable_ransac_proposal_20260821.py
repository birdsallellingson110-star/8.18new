#!/usr/bin/env python3
"""Train the K96 limb proposal with an official-style differentiable sampler.

This is the missing \N{NABLA}-RANSAC component in the current PoseDSAC line.
The forward pass uses hard, valid candidate selections while the backward pass
uses the Gumbel-Softmax relaxation, matching the straight-through construction
in ``reference/differentiable-ransac-official/samplers/gumbel_sampler.py``.

The E2 candidate generator and the established K96 hypothesis scorer are
frozen.  Only the proposal network is optimized, so the experiment changes no
2-D input, camera protocol, candidate geometry, or final scorer architecture.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import GROUPS
from train_failure_informed_map_20260820 import FrozenK96Anchor
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
GROUP_LIST = tuple(GROUPS.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--hypotheses", type=int, default=96)
    parser.add_argument("--tau-start", type=float, default=0.8)
    parser.add_argument("--tau-end", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--proposal-kl-weight", type=float, default=0.02)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument("--holdout-stride", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=24000)
    parser.add_argument("--gate-mm", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def straight_through_gumbel(logits, samples, tau):
    """Official \N{NABLA}-RANSAC hard-forward/soft-backward categorical draw.

    Args:
        logits: B,G,C proposal logits (larger is better).
        samples: number of independently sampled pose hypotheses.
        tau: Gumbel-Softmax relaxation temperature.
    Returns:
        Straight-through one-hot tensor B,K,G,C and its soft counterpart.
    """
    expanded = logits[:, None].expand(-1, samples, -1, -1)
    uniform = torch.rand_like(expanded).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    soft = F.softmax((expanded + gumbel) / tau, dim=-1)
    index = soft.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
    return hard - soft.detach() + soft, soft


def assemble_hypotheses(candidates, unary, selection, baseline_local):
    """Assemble limb-consistent hypotheses without an in-place autograd path."""
    batch, count, joints, xyz = candidates.shape
    sample_count = selection.shape[1]
    pieces = []
    unary_pieces = []
    for group_id, group in enumerate(GROUP_LIST):
        idx = torch.as_tensor(group, device=candidates.device)
        weight = selection[:, :, group_id]
        pose = torch.einsum("bkc,bcjd->bkjd", weight, candidates[:, :, idx])
        selected_unary = torch.einsum(
            "bkc,bjc->bkj", weight, unary[:, idx]
        )
        pieces.append((idx, pose))
        unary_pieces.append((idx, selected_unary))
    # Concatenation followed by inverse joint order avoids indexed assignment,
    # which caused the historical D3-PCT in-place backward failure.
    order = torch.cat([item[0] for item in pieces])
    inverse = torch.argsort(order)
    sampled_pose = torch.cat([item[1] for item in pieces], dim=2)[:, :, inverse]
    sampled_unary = torch.cat([item[1] for item in unary_pieces], dim=2)[:, :, inverse]

    baseline = candidates[:, baseline_local]
    hard_label = unary.argmin(dim=-1)
    hard_e2 = dsac.gather_hypotheses(candidates, hard_label[:, None]).squeeze(1)
    hard_unary = dsac.gather_unary(unary, hard_label[:, None]).squeeze(1)
    e2_weight = F.softmax(-unary / 0.8, dim=-1)
    soft_e2 = torch.einsum("bjc,bcjd->bjd", e2_weight, candidates)
    soft_unary = (e2_weight * unary).sum(dim=-1)
    baseline_unary = unary[:, :, baseline_local]
    hypotheses = torch.cat(
        (baseline[:, None], hard_e2[:, None], soft_e2[:, None], sampled_pose), dim=1
    )
    selected = torch.cat(
        (baseline_unary[:, None], hard_unary[:, None], soft_unary[:, None], sampled_unary),
        dim=1,
    )
    if hypotheses.shape[1] != sample_count + 3:
        raise RuntimeError("hypothesis assembly changed K")
    return hypotheses, selected


def make_hypotheses(frozen, proposal, predictions, rays, combo, count, tau):
    with torch.no_grad():
        unary, candidates, baseline_local = frozen.candidates_without_target(
            predictions, rays, combo
        )
    # Candidate tensors are fixed observations; proposal parameters retain grad.
    unary, candidates = unary.clone(), candidates.clone()
    cost = proposal(
        candidates, unary, rays, combo, frozen.bone_mean, frozen.bone_std
    )
    # Preserve the established K96 proposal exactly: 80% learned categorical
    # distribution plus 20% uniform exploration.  Gumbel-Max samples from the
    # log of that same probability; tau changes only the backward relaxation.
    probability = (
        0.8 * F.softmax(-cost / frozen.proposal_temperature, dim=-1)
        + 0.2 / cost.shape[-1]
    )
    selection, soft = straight_through_gumbel(
        probability.clamp_min(1e-8).log(), count - 3, tau
    )
    hypotheses, selected = assemble_hypotheses(
        candidates, unary, selection, baseline_local
    )
    scores = frozen.scorer(
        hypotheses, selected, frozen.coord_mean, frozen.coord_std,
        frozen.bone_mean, frozen.bone_std,
    )
    weight = F.softmax(scores / frozen.score_temperature, dim=-1)
    fused = torch.einsum("bk,bkjd->bjd", weight, hypotheses)
    return hypotheses, fused, cost, soft


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


@torch.inference_mode()
def evaluate(proposal, frozen, loader, device, args, seed_offset=0):
    proposal.eval()
    store = {f"V{x}": defaultdict(list) for x in (2, 3, 4)}
    action_store = {f"V{x}": [] for x in (2, 3, 4)}
    torch.manual_seed(10000 + args.seed + seed_offset)
    torch.cuda.manual_seed_all(10000 + args.seed + seed_offset)
    for predictions, targets, rays, actions in loader:
        predictions, targets, rays = (
            predictions.to(device), targets.to(device), rays.to(device)
        )
        for combo in TASKS:
            stage = f"V{len(combo)}"
            hypotheses, fused, _, _ = make_hypotheses(
                frozen, proposal, predictions, rays, combo,
                args.hypotheses, args.tau_end,
            )
            error = torch.linalg.vector_norm(fused - targets, dim=-1)
            oracle = torch.linalg.vector_norm(
                hypotheses - targets[:, None], dim=-1
            ).mean(dim=-1).amin(dim=-1)
            store[stage]["weighted"].append(error.cpu().numpy() * 1000.0)
            store[stage]["oracle"].append(oracle.cpu().numpy()[:, None] * 1000.0)
            action_store[stage].append(actions.numpy().copy())
    result = {}
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(action_store[stage])
        result[stage] = {}
        for mode, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def headline(result):
    return float(np.mean([
        result[stage]["weighted"]["action_equal_all17_mm"]
        for stage in ("V2", "V3", "V4")
    ]))


def main():
    args = parse_args()
    if args.hypotheses < 4:
        raise ValueError("--hypotheses must be >=4")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.train_cache], 22)
    train_idx = np.flatnonzero(arrays["subjects"] != args.holdout_subject)
    holdout_idx = np.flatnonzero(arrays["subjects"] == args.holdout_subject)[::args.holdout_stride]
    if args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(
            train_idx, size=min(args.max_train_samples, len(train_idx)), replace=False
        )
    if args.smoke_batches:
        train_idx = train_idx[: args.batch_size * args.smoke_batches]
        holdout_idx = holdout_idx[: args.batch_size]
    train_loader = DataLoader(
        ArrayDataset(arrays, train_idx), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        ArrayDataset(arrays, holdout_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    frozen = FrozenK96Anchor(args, device)
    proposal = copy.deepcopy(frozen.proposal).train().requires_grad_(True)
    initial = copy.deepcopy(proposal).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        proposal.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_result = evaluate(initial, frozen, holdout_loader, device, args)
    initial_score = headline(initial_result)
    best = initial_score
    best_epoch = 0
    history = [{"epoch": 0, "holdout_headline_mm": initial_score,
                "holdout": initial_result}]
    torch.save({"state_dict": proposal.state_dict(), "args": vars(args), "epoch": 0},
               output_dir / "model_best.pth.tar")

    initial_state = {name: value.detach().clone() for name, value in proposal.state_dict().items()}
    for epoch in range(args.epochs):
        proposal.train()
        tau = args.tau_start * (
            args.tau_end / args.tau_start
        ) ** (epoch / max(args.epochs - 1, 1))
        losses, grad_norms = [], []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions, targets, rays = (
                predictions.to(device), targets.to(device), rays.to(device)
            )
            combo = TASKS[(batch_index + 7 * epoch) % len(TASKS)]
            optimizer.zero_grad(set_to_none=True)
            hypotheses, fused, cost, soft = make_hypotheses(
                frozen, proposal, predictions, rays, combo,
                args.hypotheses, tau,
            )
            hyp_error = torch.linalg.vector_norm(
                hypotheses - targets[:, None], dim=-1
            ).mean(dim=-1)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            # The expected model loss is the training objective used by the
            # official differentiable RANSAC pipeline; fused risk preserves
            # compatibility with the established frozen K96 scorer.
            expected_model_risk = hyp_error[:, 3:].mean()
            entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=-1).mean()
            with torch.no_grad():
                old_unary, old_candidates, _ = frozen.candidates_without_target(
                    predictions, rays, combo
                )
                old_cost = initial(
                    old_candidates, old_unary, rays, combo,
                    frozen.bone_mean, frozen.bone_std,
                )
            old_prob = F.softmax(-old_cost, dim=-1)
            new_log_prob = F.log_softmax(-cost, dim=-1)
            proposal_kl = F.kl_div(new_log_prob, old_prob, reduction="batchmean")
            loss = (
                fused_error / 0.01 + 0.5 * expected_model_risk / 0.01
                - args.entropy_weight * entropy
                + args.proposal_kl_weight * proposal_kl
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(proposal.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
        holdout = evaluate(proposal, frozen, holdout_loader, device, args)
        score = headline(holdout)
        parameter_delta = math.sqrt(sum(
            float(((value.detach() - initial_state[name].to(value.device)) ** 2).sum())
            for name, value in proposal.state_dict().items()
        ))
        row = {
            "epoch": epoch + 1, "tau": tau,
            "train_loss": float(np.mean(losses)),
            "mean_grad_norm": float(np.mean(grad_norms)),
            "parameter_l2_delta": parameter_delta,
            "holdout_headline_mm": score,
            "holdout_gain_vs_initial_mm": initial_score - score,
            "holdout": holdout,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best, best_epoch = score, epoch + 1
            torch.save({"state_dict": proposal.state_dict(), "args": vars(args),
                        "epoch": best_epoch, "holdout": holdout},
                       output_dir / "model_best.pth.tar")

    checkpoint = torch.load(
        output_dir / "model_best.pth.tar", map_location=device, weights_only=False
    )
    proposal.load_state_dict(checkpoint["state_dict"])
    gain = initial_score - best
    payload = {
        "method": "official-style differentiable RANSAC proposal for frozen K96",
        "official_source": "reference/differentiable-ransac-official/samplers/gumbel_sampler.py",
        "best_epoch": best_epoch,
        "initial_holdout_headline_mm": initial_score,
        "best_holdout_headline_mm": best,
        "holdout_gain_mm": gain,
        "passes_gate": bool(gain >= args.gate_mm),
        "history": history,
        "validation": None,
    }
    if payload["passes_gate"] and not args.smoke_batches:
        validation = trainer.load_arrays([args.validation_cache], 22)
        validation_loader = DataLoader(
            ArrayDataset(validation, np.arange(len(validation["targets"]))),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        )
        payload["validation"] = evaluate(
            proposal, frozen, validation_loader, device, args, seed_offset=1000
        )
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "best_epoch", "holdout_gain_mm", "passes_gate"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
