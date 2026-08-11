"""Quick CMU mechanism diagnostic for GBT noise sensitivity and view attention."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import _init_paths  # noqa: F401,E402
import dataset  # noqa: E402
import models  # noqa: E402
from core.config import config, update_config  # noqa: E402


JOINT_NAMES = [
    "Pelvis", "R-Hip", "R-Knee", "R-Ankle", "L-Hip", "L-Knee", "L-Ankle",
    "Spine", "Neck", "Nose", "Head", "L-Shoulder", "L-Elbow", "L-Wrist",
    "R-Shoulder", "R-Elbow", "R-Wrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--views", nargs="+", type=int, default=[3, 6, 13])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise-px", nargs="+", type=float, default=[0, 2, 5, 10, 20])
    parser.add_argument("--focal-px", type=float, default=1000.0)
    parser.add_argument("--corrupt-view-index", type=int, default=-1)
    return parser.parse_args()


def load_state(path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def build_model(checkpoint: str, device: torch.device):
    model = eval("models." + config.MODEL + ".get_multiview_rumpl_net")(config, is_train=False)
    incompatible = model.load_state_dict(load_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[load] missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    return model.to(device).eval()


def perturb_directions(
    rays: torch.Tensor,
    sigma_px: float,
    focal_px: float,
    seed: int,
    view_index: int | None = None,
) -> torch.Tensor:
    """Apply tangent-plane angular noise equivalent to small image-plane jitter."""
    if sigma_px <= 0:
        return rays.clone()
    result = rays.clone()
    direction = result[..., :3]
    generator = torch.Generator(device=direction.device)
    generator.manual_seed(seed)
    noise = torch.randn(direction.shape, device=direction.device, generator=generator)
    direction_unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = noise - (noise * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
    noise = noise * (sigma_px / focal_px)
    perturbed = direction_unit + noise
    perturbed = perturbed / perturbed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if view_index is None:
        result[..., :3] = perturbed
    else:
        result[:, :, view_index, :3] = perturbed[:, :, view_index, :]
    return result


def world_arrays(output: torch.Tensor, target: torch.Tensor, meta) -> tuple[np.ndarray, np.ndarray]:
    pred = output.detach().cpu().numpy().copy()
    gt = target.detach().cpu().numpy().copy()
    if "room_scaled" in meta:
        if "room_scaled_equal" in meta:
            scale = meta["room_x_scale"][0].item()
            center = meta["room_center"][0].cpu().numpy()
            pred = pred * scale + center
            gt = gt * scale + center
        else:
            sx = meta["room_x_scale"][0].item()
            sy = meta["room_y_scale"][0].item()
            pred[:, :, 0] *= sx
            pred[:, :, 1] *= sy
            gt[:, :, 0] *= sx
            gt[:, :, 1] *= sy
    if "shift_room_tri" in meta:
        shift = meta["shift_room_tri"].cpu().numpy()[:, None, :]
        pred -= shift
        gt -= shift
    return pred, gt


def attention_map(model, batch_size: int) -> np.ndarray:
    attn = model.features.blocks_view_fusion[-1].attn.last_attn
    num_heads, tokens = attn.shape[1], attn.shape[-1]
    values = attn.reshape(batch_size, len(JOINT_NAMES), num_heads, tokens, tokens)
    return values[:, :, :, 0, 1:].mean(dim=2).cpu().numpy()


def ray_inconsistency(rays: torch.Tensor) -> np.ndarray:
    batch, joints, views = rays.shape[:3]
    direction = rays[..., :3].reshape(batch * joints, views, 3)
    origin = rays[..., 3:6].reshape(batch * joints, views, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cross = torch.cross(direction[:, :, None, :], direction[:, None, :, :], dim=-1)
    diff = origin[:, None, :, :] - origin[:, :, None, :]
    distance = (diff * cross).sum(-1).abs() / cross.norm(dim=-1).clamp_min(1e-8)
    eye = torch.eye(views, device=rays.device, dtype=torch.bool)
    distance = distance.masked_fill(eye[None], 0.0)
    consistency = distance.sum(-1) / max(views - 1, 1)
    return consistency.reshape(batch, joints, views).cpu().numpy()


@torch.no_grad()
def evaluate_curve(model, loader, noise_levels, focal_px, max_samples, device):
    errors = {float(level): [] for level in noise_levels}
    seen = 0
    for batch_index, (_, _, target, rays, meta, _) in enumerate(loader):
        rays = rays.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        keep = min(rays.shape[0], max_samples - seen)
        if keep <= 0:
            break
        rays = rays[:keep]
        target = target[:keep]
        for level in noise_levels:
            noisy = perturb_directions(
                rays, float(level), focal_px, seed=20260719 + 1009 * batch_index + int(level * 10)
            )
            output = model(noisy, is_training=False)
            pred, gt = world_arrays(output, target, meta)
            errors[float(level)].extend(np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 1000.0)
        seen += keep
        if seen >= max_samples:
            break
    return {
        level: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }
        for level, values in errors.items()
    }


@torch.no_grad()
def capture_attention(model, loader, focal_px, corrupt_view, device):
    _, _, target, rays, meta, _ = next(iter(loader))
    rays = rays.to(device)
    target = target.to(device)
    batch_size = rays.shape[0]
    os.environ["GBT_SAVE_ATTN"] = "1"
    clean_output = model(rays, is_training=False)
    clean_attn = attention_map(model, batch_size)
    noisy = perturb_directions(rays, 20.0, focal_px, seed=20260719, view_index=corrupt_view)
    noisy_output = model(noisy, is_training=False)
    noisy_attn = attention_map(model, batch_size)
    clean_pred, gt = world_arrays(clean_output, target, meta)
    noisy_pred, _ = world_arrays(noisy_output, target, meta)
    return {
        "clean_attn": clean_attn,
        "noisy_attn": noisy_attn,
        "clean_inconsistency": ray_inconsistency(rays),
        "noisy_inconsistency": ray_inconsistency(noisy),
        "clean_mpjpe": float(np.linalg.norm(clean_pred - gt, axis=-1).mean() * 1000.0),
        "noisy_mpjpe": float(np.linalg.norm(noisy_pred - gt, axis=-1).mean() * 1000.0),
    }


def plot_curve(results, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = {"Hard-view baseline": "#397dc0", "Geometry-biased": "#278b76"}
    for name, curve in results.items():
        levels = sorted(float(level) for level in curve)
        means = [curve[str(level)]["mean"] for level in levels]
        clean = means[0]
        degradation = [value - clean for value in means]
        ax.plot(levels, degradation, marker="o", linewidth=2.5, label=name, color=colors[name])
        for x, y in zip(levels, degradation):
            ax.annotate(f"{y:+.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#8d96a6", linewidth=1)
    ax.set_xlabel("Equivalent 2D keypoint jitter (pixels)")
    ax.set_ylabel("MPJPE degradation from clean (mm)")
    ax.set_title("Noise sensitivity on CMU hard V3 configuration [3, 6, 13]")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_attention(data, views, corrupt_view, out_path: Path):
    clean = data["clean_attn"].mean(axis=0)
    noisy = data["noisy_attn"].mean(axis=0)
    delta = noisy - clean
    clean_inc = data["clean_inconsistency"].mean(axis=(0, 1))
    noisy_inc = data["noisy_inconsistency"].mean(axis=(0, 1))
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.25])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    vmax = max(clean.max(), noisy.max())
    images = [
        axes[0].imshow(clean, aspect="auto", cmap="viridis", vmin=0, vmax=vmax),
        axes[1].imshow(noisy, aspect="auto", cmap="viridis", vmin=0, vmax=vmax),
        axes[2].imshow(delta, aspect="auto", cmap="coolwarm", vmin=-abs(delta).max(), vmax=abs(delta).max()),
    ]
    titles = ["Clean fusion attention", f"Camera {views[corrupt_view]} jittered by 20 px", "Attention change (jitter - clean)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks(range(len(views)), [str(view) for view in views])
        ax.set_xlabel("Camera ID")
        ax.set_yticks(range(len(JOINT_NAMES)), JOINT_NAMES, fontsize=8)
    fig.colorbar(images[0], ax=axes[:2], fraction=0.025, pad=0.02, label="Fusion-token attention")
    fig.colorbar(images[2], ax=axes[2], fraction=0.05, pad=0.02, label="Delta attention")

    ax_bar = fig.add_subplot(grid[1, :2])
    x = np.arange(len(views))
    width = 0.34
    ax_bar.bar(x - width / 2, clean.mean(axis=0), width, label="clean", color="#397dc0")
    ax_bar.bar(x + width / 2, noisy.mean(axis=0), width, label="jitter", color="#dd9250")
    ax_bar.set_xticks(x, [str(view) for view in views])
    ax_bar.set_xlabel("Camera ID")
    ax_bar.set_ylabel("Mean attention")
    ax_bar.legend(frameon=False)
    ax_bar.set_title("Mean attention over 17 joints")

    ax_geo = fig.add_subplot(grid[1, 2])
    ax_geo.plot(views, clean_inc, marker="o", label="clean", color="#397dc0")
    ax_geo.plot(views, noisy_inc, marker="o", label="jitter", color="#d66b70")
    ax_geo.set_xlabel("Camera ID")
    ax_geo.set_ylabel("Mean ray inconsistency")
    ax_geo.set_title("Explicit geometry signal")
    ax_geo.legend(frameon=False)
    fig.suptitle(
        "Geometry-biased VFT mechanism diagnostic\n"
        f"batch MPJPE: {data['clean_mpjpe']:.2f} -> {data['noisy_mpjpe']:.2f} mm",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.09, top=0.88, wspace=0.3, hspace=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    update_config(args.cfg)
    config.DATASET.TEST_VIEWS = args.views
    config.DATASET.USE_MMPOSE_VAL = True
    config.WORKERS = 0
    device = torch.device(args.device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval("dataset." + config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    model_specs = [
        ("Hard-view baseline", args.baseline, {"GBT_GEOM_BIAS": "0", "GBT_VIEW_AWARE": "0"}),
        (
            "Geometry-biased",
            args.geometry,
            {
                "GBT_GEOM_BIAS": "0.12",
                "GBT_CONF_BIAS": "0",
                "GBT_VIEW_AWARE": "1",
                "GBT_V2_SCALE": "0",
                "GBT_V3_SCALE": "1",
                "GBT_V4_SCALE": "2",
            },
        ),
    ]
    results = {}
    attention_data = None
    for name, checkpoint, environment in model_specs:
        os.environ.update(environment)
        model = build_model(checkpoint, device)
        curve = evaluate_curve(model, loader, args.noise_px, args.focal_px, args.samples, device)
        results[name] = {str(level): values for level, values in curve.items()}
        if name == "Geometry-biased":
            corrupt_view = args.corrupt_view_index % len(args.views)
            attention_data = capture_attention(model, loader, args.focal_px, corrupt_view, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.out / "noise_metrics.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    with (args.out / "noise_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "noise_px", "mean_mpjpe_mm", "std_mm", "n"])
        for name, curve in results.items():
            for level, values in curve.items():
                writer.writerow([name, level, values["mean"], values["std"], values["n"]])
    plot_curve(results, args.out / "noise_robustness_curve.png")
    plot_attention(
        attention_data,
        args.views,
        args.corrupt_view_index % len(args.views),
        args.out / "attention_clean_vs_jitter.png",
    )
    np.savez_compressed(
        args.out / "attention_arrays.npz",
        clean_attention=attention_data["clean_attn"],
        noisy_attention=attention_data["noisy_attn"],
        clean_ray_inconsistency=attention_data["clean_inconsistency"],
        noisy_ray_inconsistency=attention_data["noisy_inconsistency"],
    )
    corrupt_index = args.corrupt_view_index % len(args.views)
    clean_target_attention = float(attention_data["clean_attn"][..., corrupt_index].mean())
    noisy_target_attention = float(attention_data["noisy_attn"][..., corrupt_index].mean())
    clean_target_inconsistency = float(
        attention_data["clean_inconsistency"][..., corrupt_index].mean()
    )
    noisy_target_inconsistency = float(
        attention_data["noisy_inconsistency"][..., corrupt_index].mean()
    )
    summary = {
        "views": args.views,
        "samples": args.samples,
        "noise_type": "tangent-plane ray-direction jitter, pixel-equivalent via sigma/focal",
        "attention_corrupted_camera": args.views[args.corrupt_view_index % len(args.views)],
        "attention_batch_clean_mpjpe_mm": attention_data["clean_mpjpe"],
        "attention_batch_noisy_mpjpe_mm": attention_data["noisy_mpjpe"],
        "corrupted_camera_attention_clean": clean_target_attention,
        "corrupted_camera_attention_noisy": noisy_target_attention,
        "corrupted_camera_attention_delta": noisy_target_attention - clean_target_attention,
        "corrupted_camera_ray_inconsistency_clean": clean_target_inconsistency,
        "corrupted_camera_ray_inconsistency_noisy": noisy_target_inconsistency,
        "corrupted_camera_ray_inconsistency_delta": (
            noisy_target_inconsistency - clean_target_inconsistency
        ),
    }
    with (args.out / "diagnostic_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
