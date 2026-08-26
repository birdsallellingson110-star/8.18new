#!/usr/bin/env python3
"""Stage-I PCT-style VQ tokenizer for a frozen 3-D human-pose prior.

The implementation follows the released PCT tokenizer at the transferable
module boundary: joint/channel Mixer encoder, 34 compositional tokens, hard
nearest-code assignment, EMA codebook updates, straight-through quantization,
Mixer decoder, random joint masking, Smooth-L1 reconstruction, and commitment
loss weight 15.  It is trained only on root-relative H36M training poses; it
does not see images, test subjects, camera identities, or validation labels.

Stage II will freeze this complete tokenizer and expose only code distances and
reconstruction diagnostics to the existing K96 scorer.  Keeping the stages
separate prevents the historical continuously moving memory-bank failure.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--token-num", type=int, default=34)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument(
        "--quantizer", choices=("ema", "simvq", "fsq"), default="ema",
        help="EMA=PCT; SimVQ=ICCV25 frozen basis+linear map; FSQ=ICLR24.",
    )
    parser.add_argument(
        "--fsq-levels", nargs="+", type=int, default=(8, 5, 5, 5),
        help="Implicit FSQ levels; token-dim must equal their count.",
    )
    parser.add_argument("--simvq-beta", type=float, default=0.25)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=1)
    parser.add_argument("--mask-rate", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--commitment-weight", type=float, default=15.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--holdout-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inter_dim), nn.GELU(), nn.Linear(inter_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerLayer(nn.Module):
    """Exact non-inplace equivalent of PCT ``models/modules.py::MixerLayer``."""

    def __init__(self, hidden_dim, hidden_inter_dim, token_dim, token_inter_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = MLPBlock(token_dim, token_inter_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = MLPBlock(hidden_dim, hidden_inter_dim)

    def forward(self, x):
        y = self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        z = self.channel_mlp(self.norm2(x + y))
        return x + y + z


class PCT3DTokenizer(nn.Module):
    def __init__(
        self, mean, std, hidden_dim=256, token_dim=128, token_num=34,
        codebook_size=512, encoder_depth=4, decoder_depth=1,
        mask_rate=0.2, ema_decay=0.9, quantizer="ema",
        fsq_levels=(8, 5, 5, 5), simvq_beta=0.25,
    ):
        super().__init__()
        self.register_buffer("pose_mean", mean.float())
        self.register_buffer("pose_std", std.float())
        self.mask_rate = float(mask_rate)
        self.ema_decay = float(ema_decay)
        self.token_num = int(token_num)
        self.quantizer = quantizer
        self.simvq_beta = float(simvq_beta)
        self.fsq_levels = tuple(int(level) for level in fsq_levels)
        if quantizer == "fsq":
            if token_dim != len(self.fsq_levels):
                raise ValueError("FSQ token-dim must equal len(fsq-levels)")
            codebook_size = int(np.prod(self.fsq_levels))
        self.codebook_size = int(codebook_size)
        self.input = nn.Linear(3, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = nn.ModuleList([
            MixerLayer(hidden_dim, hidden_dim, 17, 64)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.to_tokens = nn.Linear(17, token_num)
        self.to_code = nn.Linear(hidden_dim, token_dim)
        self.register_buffer("codebook", torch.randn(codebook_size, token_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_sum", self.codebook.clone())
        if quantizer == "simvq":
            # Official SimVQ: C is Gaussian and frozen; only the shared W is
            # learned, so every transformed code changes on every update.
            self.codebook.normal_(mean=0.0, std=token_dim ** -0.5)
            self.codebook_projection = nn.Linear(token_dim, token_dim)
        else:
            self.codebook_projection = None
        if quantizer == "fsq":
            self.register_buffer(
                "fsq_level_tensor", torch.tensor(self.fsq_levels).float()
            )
            basis = np.cumprod((1,) + self.fsq_levels[:-1])
            self.register_buffer("fsq_basis", torch.tensor(basis).long())
        self.from_tokens = nn.Linear(token_num, 17)
        self.from_code = nn.Linear(token_dim, 64)
        self.decoder = nn.ModuleList([
            MixerLayer(64, 64, 17, 64) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 3)

    def normalize(self, pose):
        relative = pose - pose[:, :1]
        return (relative - self.pose_mean) / self.pose_std

    def encode(self, normalized, apply_mask=False):
        feature = self.input(normalized)
        if apply_mask and self.mask_rate > 0:
            visible = torch.rand(
                normalized.shape[:2], device=normalized.device
            ) > self.mask_rate
            feature = torch.where(
                visible[..., None], feature,
                self.mask_token.expand(normalized.shape[0], 17, -1),
            )
        for layer in self.encoder:
            feature = layer(feature)
        feature = self.encoder_norm(feature)
        feature = self.to_tokens(feature.transpose(1, 2)).transpose(1, 2)
        return self.to_code(feature)

    def nearest(self, encoded):
        if self.quantizer == "fsq":
            levels = self.fsq_level_tensor.to(encoded)
            half_l = (levels - 1.0) * (1.0 - 1e-3) / 2.0
            offset = torch.where(levels.remainder(2) == 1, 0.0, 0.5)
            # Exact official JAX formula.  The paper writes tan(offset/half_l),
            # not atanh; retain it verbatim for reproducibility.
            shift = torch.tan(offset / half_l)
            bounded = torch.tanh(encoded + shift) * half_l - offset
            rounded = torch.round(bounded)
            straight = bounded + (rounded - bounded).detach()
            half_width = torch.floor(levels / 2.0)
            quantized = straight / half_width
            non_centered = quantized * half_width + half_width
            index = (
                non_centered.round().long() * self.fsq_basis
            ).sum(dim=-1)
            return quantized, index, encoded.new_zeros(
                encoded.numel() // encoded.shape[-1], 1
            )
        flat = encoded.flatten(0, 1)
        codebook = self.codebook
        if self.quantizer == "simvq":
            codebook = self.codebook_projection(codebook)
        distance = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1)[None]
            - 2.0 * flat @ codebook.t()
        )
        index = distance.argmin(dim=1)
        quantized = F.embedding(index, codebook).view_as(encoded)
        return quantized, index.view(encoded.shape[:2]), distance

    @torch.no_grad()
    def update_codebook(self, encoded, index):
        if self.quantizer != "ema":
            return
        flat = encoded.detach().flatten(0, 1)
        flat_index = index.flatten()
        count = torch.bincount(flat_index, minlength=self.codebook_size).to(flat.dtype)
        total = torch.zeros_like(self.ema_sum)
        total.index_add_(0, flat_index, flat)
        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(count, alpha=1.0 - decay)
        self.ema_sum.mul_(decay).add_(total, alpha=1.0 - decay)
        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.codebook_size * 1e-5) * n
        )
        active = smoothed > 1e-3
        updated = self.ema_sum / smoothed.clamp_min(1e-3)[:, None]
        self.codebook[active] = updated[active]

    def decode(self, quantized):
        feature = self.from_tokens(quantized.transpose(1, 2)).transpose(1, 2)
        feature = self.from_code(feature)
        for layer in self.decoder:
            feature = layer(feature)
        return self.output(self.decoder_norm(feature))

    def forward(self, pose, update_ema=False, apply_mask=False):
        target = self.normalize(pose)
        encoded = self.encode(target, apply_mask=apply_mask)
        quantized, index, distance = self.nearest(encoded)
        if update_ema:
            self.update_codebook(encoded, index)
        if self.quantizer == "simvq":
            # Official SimVQ legacy=False objective: encoder commitment plus
            # beta-weighted transformed-codebook alignment.
            commitment = (
                self.simvq_beta * F.mse_loss(encoded, quantized.detach())
                + F.mse_loss(encoded.detach(), quantized)
            )
        elif self.quantizer == "fsq":
            commitment = encoded.new_zeros(())
        else:
            commitment = F.mse_loss(encoded, quantized.detach())
        # FSQ already applies round-STE after the bounded scalar projection.
        # VQ/SimVQ use the conventional identity straight-through estimator.
        straight_through = (
            quantized if self.quantizer == "fsq"
            else encoded + (quantized - encoded).detach()
        )
        reconstruction = self.decode(straight_through)
        return reconstruction, target, commitment, index, distance


def smooth_l1_beta(prediction, target, beta=0.05):
    difference = (prediction - target).abs()
    return torch.where(
        difference < beta,
        0.5 * difference.square() / beta,
        difference - 0.5 * beta,
    ).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    reconstruction, assignment = [], []
    for (poses,) in loader:
        poses = poses.to(device)
        recovered, target, commitment, index, _ = model(poses)
        reconstruction.append(float(smooth_l1_beta(recovered, target)))
        assignment.append(index.cpu())
    index = torch.cat(assignment).flatten()
    counts = torch.bincount(index, minlength=model.codebook_size).float()
    probability = counts / counts.sum().clamp_min(1)
    perplexity = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()
    )
    return {
        "reconstruction_loss": float(np.mean(reconstruction)),
        "active_codes": int((counts > 0).sum()),
        "codebook_size": model.codebook_size,
        "perplexity": float(perplexity),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    arrays = np.load(args.train_cache)
    poses = arrays["targets"].astype(np.float32)
    subjects = arrays["subjects"]
    train_idx = np.flatnonzero(subjects != args.holdout_subject)
    holdout_idx = np.flatnonzero(subjects == args.holdout_subject)[::args.holdout_stride]
    if args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(
            train_idx, min(args.max_train_samples, len(train_idx)), replace=False
        )
    if args.smoke_batches:
        train_idx = train_idx[: args.batch_size * args.smoke_batches]
        holdout_idx = holdout_idx[: args.batch_size]
    train_pose = torch.from_numpy(poses[train_idx])
    holdout_pose = torch.from_numpy(poses[holdout_idx])
    relative = train_pose - train_pose[:, :1]
    mean = relative.mean(dim=0, keepdim=True)
    std = relative.std(dim=0, keepdim=True).clamp_min(0.02)
    model = PCT3DTokenizer(
        mean, std, args.hidden_dim, args.token_dim, args.token_num,
        args.codebook_size, args.encoder_depth, args.decoder_depth,
        args.mask_rate, args.ema_decay,
        args.quantizer, args.fsq_levels, args.simvq_beta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    train_loader = DataLoader(
        TensorDataset(train_pose), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        TensorDataset(holdout_pose), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses, reconstructions, commitments = [], [], []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recovered, target, commitment, _, _ = model(
                batch, update_ema=True, apply_mask=True
            )
            reconstruction = smooth_l1_beta(recovered, target)
            loss = reconstruction + args.commitment_weight * commitment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            reconstructions.append(float(reconstruction.detach()))
            commitments.append(float(commitment.detach()))
        holdout = evaluate(model, holdout_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
            "train_reconstruction": float(np.mean(reconstructions)),
            "train_commitment": float(np.mean(commitments)),
            "holdout": holdout,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if holdout["reconstruction_loss"] < best:
            best = holdout["reconstruction_loss"]
            torch.save({
                "state_dict": model.state_dict(), "args": vars(args),
                "epoch": epoch + 1, "holdout": holdout,
                "implementation_source": {
                    "repo": "https://github.com/Gengzigang/PCT",
                    "commit": "6f356f6",
                    "files": ["models/pct_tokenizer.py", "models/modules.py", "models/pct_loss.py"],
                    "quantizer": args.quantizer,
                    "quantizer_source": {
                        "ema": "PCT models/pct_tokenizer.py",
                        "simvq": "reference/SimVQ-official/taming/modules/vqvae/quantize.py@d8bd94d",
                        "fsq": "reference/fsq-google-official/fsq.ipynb",
                    }[args.quantizer],
                },
            }, output_dir / "tokenizer_best.pth.tar")
    payload = {
        "method": "PCT-faithful Stage-I root-relative 3D VQ tokenizer",
        "best_holdout_reconstruction": best,
        "history": history,
        "protocol": "H36M train cache only; whole-subject S8 holdout; no test labels",
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
