#!/usr/bin/env python3
"""Train the public AdaFuse view-weight objective on frozen H36M heatmaps.

The official implementation freezes its 2D backbone and optimizes only the
per-joint/per-view ``ViewWeightNet`` with the fused 2D smooth loss.  This
script keeps that training target and sampling protocol while reading the
already-exported HRNet/ResNet heatmaps, so no image detector is accidentally
updated during the fusion experiment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from eval_h36m_dense_epipolar_heatmaps import (
    DenseHeatmapStore,
    LT_PRED_TO_RUMPL,
    epipolar_support,
)
from eval_h36m_sparse_epipolar_topk import (
    DIRECT_COCO_JOINTS,
    DIRECT_H36M_JOINTS,
    build_four_view_groups,
    camera_parameters,
)
from official_adafuse_heatmap_fusion import (
    OfficialAdaFuseHeatmapFusion,
    sampson_features_from_cameras,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--dense-shards", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3903)
    p.add_argument(
        "--group-stride", type=int, default=20,
        help="Use complete groups[::stride], matching public AdaFuse H36M training.",
    )
    p.add_argument("--depth-min-m", type=float, default=1.0)
    p.add_argument("--depth-max-m", type=float, default=5.0)
    p.add_argument("--depth-samples", type=int, default=2)
    p.add_argument(
        "--support-mode", choices=("depth", "line", "official_line"),
        default="official_line",
        help="Use official_line for nearest-neighbour AdaFuse line sampling.",
    )
    p.add_argument(
        "--joint-format", choices=("coco", "h36m", "lt_h36m"), default="coco",
        help=(
            "Heatmap/target channel order. Use h36m for the native Ada2D head "
            "or lt_h36m for the public Learnable Triangulation checkpoint."
        ),
    )
    p.add_argument(
        "--target-source", choices=("projected", "annotation"),
        default="projected",
        help=(
            "2D target for the official smooth loss. 'projected' reproduces "
            "AdaFuse's camera-projected GT target; 'annotation' is an "
            "explicit control for the prepared PKL labels."
        ),
    )
    p.add_argument(
        "--heatmap-mode", choices=("nonnegative", "signed"),
        default="nonnegative",
        help="Keep raw detector heatmaps nonnegative or preserve signed outputs.",
    )
    p.add_argument(
        "--train-views", type=int, choices=(0, 2, 3, 4), default=0,
        help="0 samples V2/V3/V4 by --view-probabilities; otherwise fixed cardinality.",
    )
    p.add_argument(
        "--view-probabilities", type=float, nargs=3, default=(0.0, 0.0, 1.0),
        metavar=("P_V2", "P_V3", "P_V4"),
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-groups", type=int, default=2)
    p.add_argument(
        "--sampling", choices=("official", "random"), default="official",
        help=(
            "official shuffles complete groups without replacement each epoch "
            "(matching AdaFuse DataLoader); random is the exploratory sampler."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def image_to_heatmap(
    image_xy: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    size = np.asarray([width, height], dtype=np.float32)
    return (image_xy - center[:, None] + 0.5 * scale[:, None]) / scale[:, None] * size


def soft_argmax_official(
    heatmaps: torch.Tensor,
    temperature: float = 0.05,
    window_width: float = 15.0,
) -> torch.Tensor:
    """AdaFuse SoftArgmax2D (uniform radius window, x/y output)."""
    n_views, n_joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(n_views * n_joints, height * width)
    maxv, argmax = flat.max(dim=1)
    peak_x = torch.remainder(argmax, width).float()
    peak_y = torch.div(argmax, width, rounding_mode="floor").float()
    y_indices = torch.arange(height, dtype=heatmaps.dtype, device=heatmaps.device)
    x_indices = torch.arange(width, dtype=heatmaps.dtype, device=heatmaps.device)
    ys, xs = torch.meshgrid(y_indices, x_indices, indexing="ij")
    distance = torch.sqrt(
        (xs[None] - peak_x[:, None, None]) ** 2
        + (ys[None] - peak_y[:, None, None]) ** 2
    )
    window = (distance <= window_width / 2.0).to(heatmaps.dtype)
    probability = torch.softmax(flat / temperature, dim=1).reshape(
        n_views * n_joints, height, width
    ) * window
    probability = probability / probability.flatten(1).sum(1).clamp_min(1e-8)[:, None, None]
    x = (probability.sum(1) * x_indices[None]).sum(1)
    y = (probability.sum(2) * y_indices[None]).sum(1)
    return torch.stack((x, y), dim=-1).reshape(n_views, n_joints, 2)


def smooth_2d_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact Joint2dSmoothLoss from the public AdaFuse code."""
    factor = pred.new_tensor(8.0)
    alpha = pred.new_tensor(-10.0)
    x = torch.sum(torch.abs(pred - target), dim=-1)
    x_scaled = ((x / factor) ** 2 / torch.abs(alpha - 2.0) + 1.0)
    x_scaled = x_scaled ** (alpha * 0.5) - 1.0
    return ((torch.abs(alpha) - 2.0) / alpha * x_scaled).mean() * 1000.0


def camera_projected_joints_2d(record: dict) -> np.ndarray:
    """Reproject stored camera-frame GT joints using the H36M K matrix.

    The public AdaFuse fusion loss builds its target by projecting GT 3D
    through the camera parameters.  The prepared PKL's ``joints_2d`` can
    retain a distortion/annotation convention, so this path is kept
    explicit instead of silently treating both targets as interchangeable.
    """
    camera = record["camera"]
    intrinsic = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    joints_camera = np.asarray(record["joints_3d_camera"], dtype=np.float64)
    projected = (intrinsic @ joints_camera.T).T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)


def select_group(
    groups: list[list[int]],
    n_views: int,
    rng: random.Random,
) -> list[int]:
    group = rng.choice(groups)
    if n_views == 4:
        return list(group)
    # Keep every subset equally likely, as the evaluator's combinations do.
    import itertools

    return [group[i] for i in rng.choice(list(itertools.combinations(range(4), n_views)))]


def make_official_batches(
    groups: list[list[int]],
    batch_groups: int,
    steps: int,
    rng: random.Random,
) -> list[list[list[int]]]:
    """Build AdaFuse-style shuffled, non-replacement group batches.

    The public H36M driver uses a DataLoader over ``grouping[::20]`` with
    batch size 2 and ``shuffle=True``.  Constructing batches explicitly keeps
    the same no-replacement epoch boundary in this frozen-heatmap trainer.
    """
    if batch_groups < 1:
        raise ValueError("batch_groups must be positive")
    batches: list[list[list[int]]] = []
    while len(batches) < steps:
        order = list(groups)
        rng.shuffle(order)
        for start in range(0, len(order), batch_groups):
            batches.append(order[start:start + batch_groups])
            if len(batches) >= steps:
                break
    return batches


def prepare_sample(
    records: list[dict],
    store: DenseHeatmapStore,
    indices: list[int],
    depths: torch.Tensor,
    device: torch.device,
    support_mode: str,
    joint_format: str,
    target_source: str,
    heatmap_mode: str,
    model: OfficialAdaFuseHeatmapFusion,
) -> tuple[torch.Tensor, dict[str, float]]:
    group_records = [records[index] for index in indices]
    data = store.get(indices)
    raw = torch.as_tensor(data["heatmaps"], dtype=torch.float32, device=device)
    if heatmap_mode != "signed":
        raw = raw.clamp_min(0.0)
    maximum = raw.flatten(-2).amax(dim=-1, keepdim=True)
    normalized = raw / maximum.clamp_min(1e-6)[..., None]
    camera_data = [camera_parameters(record) for record in group_records]
    intrinsics = [item[0] for item in camera_data]
    rotations = [item[1] for item in camera_data]
    centers = np.stack([item[2] for item in camera_data])
    with torch.no_grad():
        support = epipolar_support(
            normalized, intrinsics, rotations, centers,
            data["input_center"], data["input_scale"], depths,
            mode=support_mode,
        )
    if joint_format == "coco":
        # HRNet's COCO head has 13 joints directly observable in H36M.  The
        # public AdaFuse H36M model trains its reliability net on these
        # semantic channels after conversion.
        heatmap_channels = DIRECT_COCO_JOINTS
        xy = data["decoded_keypoints"][:, heatmap_channels].astype(np.float64)
        confidence = data["decoded_scores"][:, heatmap_channels].astype(np.float64)
    elif joint_format == "h36m":
        # Ada2D's union20-trained ResNet head is exported as native H36M-17;
        # it already contains root/belly/neck/head and must not be reduced to
        # the COCO subset.
        heatmap_channels = np.arange(17, dtype=np.int64)
        xy = data["decoded_keypoints"].astype(np.float64)
        confidence = data["decoded_scores"].astype(np.float64)
    else:
        # The public LT checkpoint emits all 17 H36M joints, but in its own
        # semantic order.  Reorder before both epipolar fusion and the
        # Sampson descriptor so the learned view-weight net sees the same
        # H36M order as the prepared target and the evaluator.
        heatmap_channels = LT_PRED_TO_RUMPL
        xy = data["decoded_keypoints"][:, LT_PRED_TO_RUMPL].astype(np.float64)
        confidence = data["decoded_scores"][:, LT_PRED_TO_RUMPL].astype(np.float64)
    distances, confidences = sampson_features_from_cameras(
        xy, confidence, intrinsics, rotations, centers
    )
    distances_t = torch.as_tensor(distances, dtype=torch.float32, device=device)
    confidences_t = torch.as_tensor(confidences, dtype=torch.float32, device=device)
    # The HRNet export is COCO-17.  AdaFuse's H36M head consumes the 13
    # observable COCO joints mapped to H36M semantics; the four synthetic
    # H36M joints are not allowed to leak a differently ordered heatmap into
    # the view-weight network.
    input_channels = torch.as_tensor(heatmap_channels, dtype=torch.long, device=device)
    direct_h36m = np.asarray(DIRECT_H36M_JOINTS, dtype=np.int64)
    fused_logits, aux = model(
        normalized[:, input_channels],
        support[:, :, input_channels],
        distances=distances_t,
        confidences=confidences_t,
    )
    fused = (
        fused_logits
        if heatmap_mode == "signed"
        else torch.exp(fused_logits)
    )
    pred_hm = soft_argmax_official(fused)
    _, _, height, width = fused.shape
    target_rows = []
    for record in group_records:
        if target_source == "projected":
            target_full = camera_projected_joints_2d(record)
        else:
            target_full = np.asarray(record["joints_2d"], dtype=np.float64)
        target_rows.append(
            target_full
            if joint_format in ("h36m", "lt_h36m")
            else target_full[direct_h36m]
        )
    target_image = np.stack(target_rows).astype(np.float32)
    target_hm = image_to_heatmap(
        target_image, data["input_center"], data["input_scale"], width, height
    )
    target = torch.as_tensor(target_hm, dtype=torch.float32, device=device)
    loss = smooth_2d_loss(pred_hm, target)
    metrics = {
        "loss": float(loss.detach()),
        "mean_weight": float(aux["view_weights"].detach().mean()),
        "min_weight": float(aux["view_weights"].detach().min()),
        "max_weight": float(aux["view_weights"].detach().max()),
        "mean_2d_error_hm": float(torch.linalg.norm(pred_hm - target, dim=-1).mean().detach()),
    }
    return loss, metrics


def save_checkpoint(
    model: OfficialAdaFuseHeatmapFusion,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    path: Path,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "model_kind": "official_adafuse",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if sum(args.view_probabilities) <= 0 or min(args.view_probabilities) < 0:
        raise ValueError("view probabilities must be non-negative and nonzero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    store = DenseHeatmapStore(args.dense_shards)
    groups = [g for g in build_four_view_groups(records) if all(i in store for i in g)]
    if args.group_stride > 1:
        groups = groups[:: args.group_stride]
    if not groups:
        raise RuntimeError("no complete groups in heatmap cache")
    model = OfficialAdaFuseHeatmapFusion(
        signed_heatmaps=args.heatmap_mode == "signed"
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    depths = torch.linspace(args.depth_min_m, args.depth_max_m, args.depth_samples, device=device)
    probs = np.asarray(args.view_probabilities, dtype=np.float64)
    probs /= probs.sum()
    rng = random.Random(args.seed)
    official_batches = (
        make_official_batches(groups, args.batch_groups, args.steps, rng)
        if args.sampling == "official" and args.train_views == 4
        else None
    )
    started = time.time()
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            rows = []
            total_loss = None
            if official_batches is not None:
                batch_groups = official_batches[step - 1]
            else:
                batch_groups = []
                for _ in range(max(1, args.batch_groups)):
                    if args.train_views:
                        n_views = args.train_views
                    else:
                        n_views = int(rng.choices((2, 3, 4), weights=probs, k=1)[0])
                    batch_groups.append(select_group(groups, n_views, rng))
            for base_group in batch_groups:
                if official_batches is not None:
                    indices = list(base_group)
                else:
                    indices = base_group
                loss, metrics = prepare_sample(
                    records, store, indices, depths, device, args.support_mode,
                    args.joint_format, args.target_source, args.heatmap_mode,
                    model
                )
                total_loss = loss if total_loss is None else total_loss + loss
                rows.append(metrics)
            total_loss = total_loss / float(max(1, len(batch_groups)))
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {
                    "step": step,
                    "train_views": args.train_views,
                    "loss": float(total_loss.detach()),
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": time.time() - started,
                    "mean_weight": float(np.mean([x["mean_weight"] for x in rows])),
                    "mean_2d_error_hm": float(np.mean([x["mean_2d_error_hm"] for x in rows])),
                }
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, args, output_dir / f"checkpoint_step{step:06d}.pth")
    save_checkpoint(model, optimizer, args.steps, args, output_dir / "final.pth")
    (output_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "groups": len(groups)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
