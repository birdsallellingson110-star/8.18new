#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset.exact_temporal_clip import (
    ExactTemporalClipDataset,
    fixed_view_collate,
    random_view_collate,
)
from models.exact_temporal_rumpl import ExactTemporalRUMPL


KP_STAR = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-clips", type=int, default=200)
    parser.add_argument("--min-center-oks", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--residual-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(prediction, target):
    error = torch.linalg.vector_norm(prediction - target, dim=-1) * 1000.0
    return float(error.mean()), float(error[:, KP_STAR].mean())


def grouped_train_validation_split(dataset, validation_clips, seed):
    source_to_indexes = defaultdict(list)
    for index, clip in enumerate(dataset.clips):
        source_to_indexes[clip["source"]].append(index)

    sources = sorted(source_to_indexes)
    random.Random(seed).shuffle(sources)
    validation_sources = []
    validation_indexes = []
    for source in sources:
        if validation_indexes and len(validation_indexes) >= validation_clips:
            break
        validation_sources.append(source)
        validation_indexes.extend(source_to_indexes[source])

    validation_set = set(validation_indexes)
    train_indexes = [
        index for index in range(len(dataset)) if index not in validation_set
    ]
    return train_indexes, validation_indexes, validation_sources


@torch.no_grad()
def evaluate(model, loader, device, no_temporal):
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        prediction = model(
            batch["rays"].to(device),
            batch["confidence"].to(device),
            batch["delta_t"].to(device),
            no_temporal=no_temporal,
        )
        predictions.append(prediction.cpu())
        targets.append(batch["target"][:, batch["target"].shape[1] // 2])
    return metrics(torch.cat(predictions), torch.cat(targets))


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ["RUMPL_FIX_PFT_LAST_BLOCK"] = "0"
    os.environ["GBT_LEARNABLE_BIAS"] = "0"
    device = torch.device("cuda")
    model = ExactTemporalRUMPL(
        args.config,
        args.checkpoint,
        temporal_depth=args.depth,
        freeze_backbone=True,
        motion_only=args.motion_only,
    ).to(device)

    train_dataset = ExactTemporalClipDataset(
        args.data, num_frames=args.frames, min_center_oks=args.min_center_oks
    )
    validation_dataset = copy.copy(train_dataset)
    validation_dataset.fixed_window = True
    validation_target = min(args.validation_clips, len(train_dataset) // 5)
    train_indexes, validation_indexes, validation_sources = (
        grouped_train_validation_split(train_dataset, validation_target, args.seed)
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indexes),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=random_view_collate(seed=args.seed),
        drop_last=True,
    )
    validation_loaders = {
        views: DataLoader(
            Subset(validation_dataset, validation_indexes),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=fixed_view_collate(views, seed=args.seed + views),
        )
        for views in (2, 3, 4, 5)
    }

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    os.makedirs(args.output, exist_ok=True)
    split_manifest = {
        "seed": args.seed,
        "train_indexes": train_indexes,
        "validation_indexes": validation_indexes,
        "validation_sources": validation_sources,
    }
    with open(
        os.path.join(args.output, "split_manifest.json"), "w", encoding="ascii"
    ) as handle:
        json.dump(split_manifest, handle, indent=2)
    baseline = {
        views: evaluate(model, loader, device, no_temporal=True)
        for views, loader in validation_loaders.items()
    }
    print(
        f"DATA train={len(train_indexes)} validation={len(validation_indexes)} "
        f"validation_sources={len(validation_sources)}",
        flush=True,
    )
    print(f"TRAINABLE parameters={sum(p.numel() for p in trainable)}", flush=True)
    print(f"BASELINE {baseline}", flush=True)

    best = float("inf")
    history = []
    center_index = args.frames // 2
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            target = batch["target"][:, center_index].to(device)
            prediction, baseline_prediction = model(
                batch["rays"].to(device),
                batch["confidence"].to(device),
                batch["delta_t"].to(device),
                return_baseline=True,
            )
            distance = torch.linalg.vector_norm(prediction - target, dim=-1)
            loss = F.huber_loss(
                distance, torch.zeros_like(distance), delta=args.huber_delta
            )
            if args.residual_penalty > 0:
                residual = torch.linalg.vector_norm(
                    prediction - baseline_prediction.detach(), dim=-1
                ).mean()
                loss = loss + args.residual_penalty * residual
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            batch_size = target.shape[0]
            total_loss += float(loss) * batch_size
            samples += batch_size

        validation = {
            views: evaluate(model, loader, device, no_temporal=False)
            for views, loader in validation_loaders.items()
        }
        record = {
            "epoch": epoch,
            "train_mpjpe_mm": total_loss / samples * 1000.0,
            "validation": validation,
            "selection_mpjpe_mm": float(
                np.mean([result[0] for result in validation.values()])
            ),
            "seconds": time.time() - start,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["selection_mpjpe_mm"] < best:
            best = record["selection_mpjpe_mm"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "baseline": baseline, **record},
                os.path.join(args.output, "model_best.pth.tar"),
            )

    torch.save(
        {"model": model.state_dict(), "args": vars(args), "baseline": baseline, "history": history},
        os.path.join(args.output, "final_state.pth.tar"),
    )
    with open(os.path.join(args.output, "summary.json"), "w", encoding="ascii") as handle:
        json.dump({"args": vars(args), "baseline": baseline, "history": history}, handle, indent=2)


if __name__ == "__main__":
    main()
