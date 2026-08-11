"""路B STVFTv2 训练: 时序在 VFT 去噪后特征空间; 无gate(zero-init残差); 帧质量扰动增强;
AdamW wd=0.1(强正则, 抄PoseFormer); Huber loss(防离群帧主导); 中心帧监督; 与12w同口径eval。
"""
import argparse, copy, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_fixed_views, make_collate_random_views
from models.stvft.stvft_v2 import STVFTv2
from utils.kp_star import kp_star_indices
from core.evaluate import calc_mpjpe                  # 官方 MPJPE, 直接用

COCO_NAMES = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb',
              'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
KP_IDX = kp_star_indices(COCO_NAMES)                  # [5,6,7,8,9,10,13,14,15,16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]    # [0-4,11,12] 头+髋, KP* 排除


def huber_dist(pred, gt, delta, rel=False):
    """per-joint 距离 的 Huber(防离群帧/关节主导)。pred,gt (B,J,3) 米。
    rel=False: absolute(同官方训练 L1)。rel=True: 先减 COCO mid-hip(pelvis-anchored), 对齐官方 eval 指标→
    梯度聚焦姿态、不浪费容量在全局平移上。"""
    if rel:
        pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0
        pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
        pred = pred - pv_p; gt = gt - pv_g
    d = (((pred - gt) ** 2).sum(-1) + 1e-8).sqrt()             # (B,J) 米
    return F.huber_loss(d, torch.zeros_like(d), delta=delta)


@torch.no_grad()
def eval_v2(model, val_dl, L, device, no_temporal):
    """官方口径(function_rumpl.evaluate, relative_evaluation=True): pelvis-anchored(COCO mid-hip)
    + MEAN(官方 calc_mpjpe) + cm(OUTPUT_IN_METER ×100)。返回 All-17(官方perf_indicator) + KP*(论文头条)。"""
    was = model.training; model.eval()
    ct = L // 2
    preds, gts = [], []
    for b in val_dl:
        rays = b['rays'].to(device); confs = b['confs'].to(device); dts = b['delta_ts'].to(device)
        pred = model(rays, confs, dts, no_temporal=no_temporal).cpu().numpy()  # (B,17,3) m
        preds.append(pred); gts.append(b['gt_3d'][:, ct].numpy())
    if was: model.train()
    pred = np.concatenate(preds) * 100.0                       # OUTPUT_IN_METER → cm
    gt = np.concatenate(gts) * 100.0
    pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0                 # 官方 relative: 减 COCO mid-hip
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0
    gt = gt - pv_g; pred = pred - pv_p
    _, all17 = calc_mpjpe(gt, pred, mode='absolute')                       # 已预中心化→absolute
    _, kp = calc_mpjpe(gt, pred, mode='absolute', not_consider_kp=NOT_KP)  # KP* 排除头+髋
    return {'all17': float(all17), 'kp': float(kp)}            # cm, rel-mean


def train(args):
    dev = args.device
    model = STVFTv2(args.pretrained_cfg, args.pretrained_ckpt, freeze_backbone=True,
                    temporal_layers=args.temporal_layers).to(dev)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] STVFTv2 temporal_layers={args.temporal_layers}, trainable={ntr/1e6:.2f}M (只Temporal)")

    ds_tr = STVFTClipDataset(args.data_glob, L_window=args.L, min_oks=args.min_oks,
                             perturb=args.perturb, perturb_offset_px=args.perturb_offset_px)
    ds_val = copy.copy(ds_tr); ds_val.perturb = 0.0            # 共享clips, val不扰动
    ds_val.eval_fixed_window = True                            # 修bug: val用确定性居中窗口, B/C同窗口可比
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds_tr), generator=g).tolist()
    nv = min(args.val_clips, len(ds_tr) // 5)
    val_idx, tr_idx = perm[:nv], perm[nv:]
    dl = DataLoader(Subset(ds_tr, tr_idx), batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate_random_views(args.min_train_views, args.max_train_views, seed=args.seed),
                    num_workers=args.workers, drop_last=True)
    val_dls = {v: DataLoader(Subset(ds_val, val_idx), batch_size=args.batch_size, shuffle=False,
                             collate_fn=make_collate_fixed_views(v, seed=args.seed + 1),
                             num_workers=args.workers) for v in args.eval_views}
    print(f"[data] train {len(tr_idx)} / val {nv}, {len(dl)} batches/ep, perturb={args.perturb}, "
          f"train_views=随机k∈[{args.min_train_views},{args.max_train_views}](照抄baseline per-batch)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    print(f"[opt] AdamW lr={args.lr} wd={args.weight_decay} (强正则), Huber delta={args.huber_delta}, loss=中心帧")

    os.makedirs(args.save_dir, exist_ok=True)
    ct = args.L // 2
    # baseline(no_temporal) 地板, 各视角 — 官方口径 rel-mean(cm)
    base = {v: eval_v2(model, val_dls[v], args.L, dev, no_temporal=True) for v in args.eval_views}
    for v in args.eval_views:
        print(f"[BASELINE no_temporal 官方rel-mean cm] V={v}: All-17={base[v]['all17']:.3f}  KP*={base[v]['kp']:.3f}")
    pv = args.eval_views[0]
    best = float('inf')
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); agg = {'l': 0.0, 'n': 0}
        for b in dl:
            rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev)
            gt = b['gt_3d'][:, ct].to(dev)
            pred = model(rays, confs, dts)
            loss = huber_dist(pred, gt, args.huber_delta, rel=args.rel_loss)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            tp = [p for p in model.parameters() if p.requires_grad]
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in tp):
                continue
            torch.nn.utils.clip_grad_norm_(tp, args.grad_clip)
            opt.step()
            bs = rays.shape[0]; agg['l'] += loss.item() * bs; agg['n'] += bs
        if agg['n'] == 0:
            print(f"[ep {ep}] 全batch跳过, 停"); break
        ev = {v: eval_v2(model, val_dls[v], args.L, dev, no_temporal=False) for v in args.eval_views}
        cur = ev[pv]['all17']                                  # 官方 perf_indicator = rel All-17 mean
        star = ""
        if math.isfinite(cur) and cur < best:
            best = cur
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep, 'eval': ev, 'base': base},
                       os.path.join(args.save_dir, 'stvft_v2_best.pth'))
            star = " *best"
        # 官方rel-mean(cm), 括号内 ΔA17 = C-B (负=时序改善)
        msg = "  ".join(f"V{v}[A17 {ev[v]['all17']:.3f} KP* {ev[v]['kp']:.3f} ΔA17 {ev[v]['all17']-base[v]['all17']:+.3f}]"
                        for v in args.eval_views)
        print(f"[ep {ep}] huber {agg['l']/agg['n']*1000:.2f} | {msg} "
              f"(baseV{pv}A17={base[pv]['all17']:.3f}, λ={model.temporal.conf_bias_scale.item():.2f}){star} | {time.time()-t0:.0f}s")
    torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(args.save_dir, 'stvft_v2_final.pth'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/v2")
    ap.add_argument('--pretrained-ckpt', required=True)
    ap.add_argument('--pretrained-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--temporal-layers', type=int, default=2)
    ap.add_argument('--min-oks', type=float, default=0.5)
    ap.add_argument('--perturb', type=float, default=0.0)
    ap.add_argument('--perturb-offset-px', type=float, default=120.0)
    ap.add_argument('--lr', type=float, default=4e-5)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--huber-delta', type=float, default=0.1)
    ap.add_argument('--rel-loss', action='store_true', help='用 pelvis-anchored 相对 loss(对齐官方eval指标)')
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--min-train-views', type=int, default=2, help='训练随机视角下界(照抄baseline)')
    ap.add_argument('--max-train-views', type=int, default=5, help='训练随机视角上界(照抄baseline)')
    ap.add_argument('--eval-views', type=int, nargs='+', default=[2, 5])
    ap.add_argument('--val-clips', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
