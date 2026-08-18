#!/usr/bin/env python3
"""Train the AdaFuse-style 2-D ResNet-152 front end on real H36M.

This is deliberately a small, auditable training driver rather than a second
3-D model.  It follows the public AdaFuse/Simple-Baselines geometry:

* square 384x384 person crop and 96x96 Gaussian heatmaps;
* real H36M images/2-D labels from the prepared PKL;
* H36M training subjects S1/S5/S6/S7/S8 and optional official [::20]
  multiview-group sampling;
* ResNet-152 weights for the backbone.  By default the 17-channel target head
  is reinitialized (the original COCO-to-H36M adaptation control); with
  ``--preserve-head`` an already H36M-trained public LT head is retained so
  this script can measure front-end domain adaptation without changing the
  detector's joint semantics.

The script intentionally does not use any 3-D labels or 3-D loss.  The output
checkpoint is therefore a 2-D front-end checkpoint and can be evaluated with
the existing heatmap exporter and all current fusion/triangulation controls.
Large outputs should be placed under /mnt/data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mmpose.apis import init_model


MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)

# AdaFuse uses a 20-joint union head.  H36M supplies these 17 union entries;
# thorax, upper-neck and head-top are absent and therefore receive zero target
# weight.  Keep this mapping explicit so the trained checkpoint can be
# exported back to the 17-joint H36M evaluation order without ambiguity.
H36M_TO_UNION20 = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19],
    dtype=np.int64,
)
UNION20_JOINTS = 20


def affine_transform(center, scale, rot, output_size, inv=False):
    """Exact affine helper used by the public Simple-Baselines code."""
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.asarray([scale, scale], dtype=np.float32)
    scale_tmp = scale * 200.0
    src_w = float(scale_tmp[0])
    dst_w, dst_h = float(output_size[0]), float(output_size[1])
    rot_rad = np.pi * float(rot) / 180.0
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    # This is the literal released AdaFuse/Simple-Baselines transform:
    # get_dir([0, -0.5*src_w], rot) = [0.5*src_w*sin(rot),
    # -0.5*src_w*cos(rot)].  Keeping the sign is important for rotation
    # augmentation; a hand-written equivalent with the opposite x sign
    # trains on a mirrored augmentation distribution.
    src_dir = np.asarray([0.5 * src_w * sn, -0.5 * src_w * cs], dtype=np.float32)
    dst_dir = np.asarray([0.0, -0.5 * dst_w], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    # Public Simple-Baselines helper: get_3rd_point(a, b) computes
    # ``b + [- (a-b)_y, (a-b)_x]``.  Keeping this literal matters for the
    # crop orientation when rotation augmentation is enabled.
    direct = src[0] - src[1]
    src[2] = src[1] + np.asarray([-direct[1], direct[0]], dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    direct = dst[0] - dst[1]
    dst[2] = dst[1] + np.asarray([-direct[1], direct[0]], dtype=np.float32)
    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def gaussian_heatmaps(joints, visible, image_size=(384, 384), heatmap_size=(96, 96), sigma=3.0):
    """Generate the same integer-centre Gaussian target as AdaFuse."""
    joints = np.asarray(joints, dtype=np.float32)
    visible = np.asarray(visible, dtype=np.float32)
    h, w = int(heatmap_size[1]), int(heatmap_size[0])
    out = np.zeros((joints.shape[0], h, w), dtype=np.float32)
    weights = np.zeros((joints.shape[0], 1), dtype=np.float32)
    stride = np.asarray(image_size, dtype=np.float32) / np.asarray(heatmap_size, dtype=np.float32)
    tmp = int(sigma * 3)
    size = 2 * tmp + 1
    xx = np.arange(size, dtype=np.float32)[None, :]
    yy = np.arange(size, dtype=np.float32)[:, None]
    g = np.exp(-((xx - tmp) ** 2 + (yy - tmp) ** 2) / (2.0 * sigma * sigma))
    for j in range(joints.shape[0]):
        if visible[j] <= 0.5:
            continue
        mu_x = int(joints[j, 0] / stride[0] + 0.5)
        mu_y = int(joints[j, 1] / stride[1] + 0.5)
        ul = [mu_x - tmp, mu_y - tmp]
        br = [mu_x + tmp + 1, mu_y + tmp + 1]
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            continue
        gx0, gx1 = max(0, -ul[0]), min(br[0], w) - ul[0]
        gy0, gy1 = max(0, -ul[1]), min(br[1], h) - ul[1]
        x0, x1 = max(0, ul[0]), min(br[0], w)
        y0, y1 = max(0, ul[1]), min(br[1], h)
        out[j, y0:y1, x0:x1] = g[gy0:gy1, gx0:gx1]
        weights[j, 0] = 1.0
    return out, weights


def complete_groups(records):
    groups = OrderedDict()
    for idx, rec in enumerate(records):
        key = (int(rec["subject"]), int(rec["action"]), int(rec["subaction"]), int(rec["image_id"]))
        groups.setdefault(key, [-1] * 4)[int(rec["camera_id"])] = idx
    return [g for g in groups.values() if min(g) >= 0]


class H36MHeatmapDataset(Dataset):
    def __init__(self, pkl_path, images_root, group_stride=20, max_groups=0,
                 training=True, augment=True, seed=20260814,
                 input_order="bgr", scale_factor=0.15,
                 rotation_factor=20.0):
        self.records = pickle.load(open(pkl_path, "rb"))
        groups = complete_groups(self.records)
        if group_stride > 1:
            groups = groups[::group_stride]
        if max_groups > 0:
            groups = groups[:max_groups]
        self.indices = [idx for group in groups for idx in group]
        self.images_root = Path(images_root)
        self.training = bool(training)
        self.augment = bool(augment and training)
        if input_order not in ("bgr", "rgb"):
            raise ValueError(f"unsupported input_order={input_order!r}")
        self.input_order = input_order
        self.scale_factor = float(scale_factor)
        self.rotation_factor = float(rotation_factor)
        self.rng = random.Random(seed)
        if not self.indices:
            raise ValueError("no H36M records selected")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        rec = self.records[self.indices[item]]
        image_path = self.images_root / rec["image"]
        # The compact stride-20 training PKL keeps the original flat source
        # filename alongside the evaluation-layout ``image`` path.  Prefer
        # that fallback when no materialized train symlink exists; validation
        # records continue to use their ordinary image field.
        if not image_path.is_file() and rec.get("source_image"):
            image_path = self.images_root / rec["source_image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(image_path)
        center = np.asarray(rec["center"], dtype=np.float32).copy()
        scale = np.asarray(rec["scale"], dtype=np.float32).copy()
        rotation = 0.0
        if self.augment:
            scale *= np.clip(
                np.random.randn() * self.scale_factor + 1.0,
                1.0 - self.scale_factor, 1.0 + self.scale_factor)
            rotation = float(np.clip(
                np.random.randn() * self.rotation_factor,
                -2.0 * self.rotation_factor, 2.0 * self.rotation_factor
            )) if random.random() <= 0.6 else 0.0
        # The released AdaFuse H36M dataset class applies scale/rotation
        # augmentation here but does not perform a horizontal flip.  Keep
        # that convention so this front-end remains an auditable control.
        joints = np.asarray(rec["joints_2d"], dtype=np.float32).copy()
        trans = affine_transform(center, scale, rotation, (384, 384), inv=False)
        crop = cv2.warpAffine(image, trans, (384, 384), flags=cv2.INTER_LINEAR)
        transformed = np.concatenate([joints, np.ones((len(joints), 1), dtype=np.float32)], axis=1)
        transformed = (trans @ transformed.T).T
        vis = np.asarray(rec.get("joints_vis", np.ones((17, 3))), dtype=np.float32)[:, 0]
        # The training pkl has all joints visible; still mask points outside
        # the crop just as the public dataset code does.
        vis = vis.copy()
        vis[(transformed[:, 0] < 0) | (transformed[:, 1] < 0) |
            (transformed[:, 0] >= 384) | (transformed[:, 1] >= 384)] = 0.0
        union_joints = np.zeros((UNION20_JOINTS, 2), dtype=np.float32)
        union_vis = np.zeros((UNION20_JOINTS,), dtype=np.float32)
        union_joints[H36M_TO_UNION20] = transformed
        union_vis[H36M_TO_UNION20] = vis
        target, weight = gaussian_heatmaps(union_joints, union_vis)
        # AdaFuse's released H36M loader reads with OpenCV and applies the
        # ImageNet statistics without an RGB swap.  Keep BGR as the auditable
        # default; RGB remains an explicit control for ablation.
        if self.input_order == "rgb":
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32)
        crop = (crop - MEAN) / STD
        crop = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float()
        return crop, torch.from_numpy(target), torch.from_numpy(weight), torch.from_numpy(transformed), torch.from_numpy(vis)


def init_h36m_head(model):
    head = model.head
    # The public checkpoint has a COCO 17-channel head.  AdaFuse removes the
    # final layer before H36M fine-tuning; do the same for H36M channel order.
    torch.nn.init.normal_(head.final_layer.weight, std=0.001)
    if head.final_layer.bias is not None:
        torch.nn.init.constant_(head.final_layer.bias, 0.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--group-stride", type=int, default=20)
    p.add_argument("--max-groups", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-milestones", type=int, nargs="*", default=[8])
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--input-order", choices=("bgr", "rgb"), default="bgr")
    p.add_argument("--scale-factor", type=float, default=0.15)
    p.add_argument("--rotation-factor", type=float, default=20.0)
    p.add_argument(
        "--preserve-head", action="store_true",
        help="Do not reinitialize the checkpoint's existing H36M 17-joint head.")
    p.add_argument("--max-steps", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = vars(args).copy()
    metadata["pkl"] = str(Path(args.pkl).resolve())
    metadata["checkpoint"] = str(Path(args.checkpoint).resolve())
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    dataset = H36MHeatmapDataset(
        args.pkl, args.images_root, group_stride=args.group_stride,
        max_groups=args.max_groups, training=True, augment=not args.no_augment,
        seed=args.seed, input_order=args.input_order,
        scale_factor=args.scale_factor,
        rotation_factor=args.rotation_factor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, drop_last=True)
    model = init_model(args.config, args.checkpoint, device="cpu")
    if not args.preserve_head:
        init_h36m_head(model)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    if torch.cuda.is_available() and gpu_ids:
        model = model.cuda(gpu_ids[0])
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    model.train()
    # The released AdaFuse 2D driver uses Adam without weight decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_milestones), gamma=args.lr_gamma)
    log_path = out / "train.log"
    with log_path.open("w") as log:
        def emit(msg):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        emit(json.dumps({"records": len(dataset), "groups": len(dataset) // 4,
                         "device": str(device), "gpus": gpu_ids}, sort_keys=True))
        global_step = 0
        for epoch in range(args.epochs):
            running = 0.0
            for step, batch in enumerate(loader):
                inputs, target, weight, _, _ = batch
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                weight = weight.to(device, non_blocking=True).view(
                    target.shape[0], UNION20_JOINTS, 1, 1)
                pred = model(inputs, None, mode="tensor")
                # Same reduction as the public JointsMSELoss: average each
                # joint heatmap independently, then sum the 20 joint losses.
                pred_split = pred.reshape(pred.shape[0], UNION20_JOINTS, -1)
                target_split = target.reshape(target.shape[0], UNION20_JOINTS, -1)
                weight_split = weight.reshape(pred.shape[0], UNION20_JOINTS, 1)
                loss = ((pred_split * weight_split - target_split * weight_split)
                        ** 2).mean(dim=(0, 2)).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                global_step += 1
                if step % 20 == 0:
                    emit(json.dumps({"epoch": epoch, "step": step,
                                     "steps": len(loader), "loss": running / (step + 1)}, sort_keys=True))
                if args.max_steps and global_step >= args.max_steps:
                    break
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            ckpt = out / f"res152_h36m_ep{epoch + 1}.pth"
            torch.save({"state_dict": state, "epoch": epoch + 1,
                        "loss": running / max(1, step + 1), "config": metadata}, ckpt)
            emit(json.dumps({"epoch_done": epoch + 1, "mean_loss": running / max(1, step + 1),
                             "checkpoint": str(ckpt),
                             "learning_rate": optimizer.param_groups[0]["lr"]}, sort_keys=True))
            scheduler.step()
            if args.max_steps and global_step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
