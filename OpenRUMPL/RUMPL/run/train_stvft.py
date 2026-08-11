"""ST-VFT Phase 1 — 训练脚本 (设计文档 v1 §4)。

per-timestep supervision (TEMPO 式): 对窗口每个 t_offset 都 forward 一次,
把该 t 设为 target (new_delta_ts = delta_ts - delta_ts[:,:,t]),监督每个时刻的 pose。
loss = lambda_per_t * MPJPE(全L帧) + lambda_vel * MPJVE(速度)。

超参 (设计文档 v1): lambda_per_t=1.0, lambda_vel=0.1, batch=16, lr=3e-5, 20 epochs, Adam。
gt_3d 是绝对世界坐标 (米), PFT 输出绝对3D, 直接 MPJPE (与 RUMPL baseline 同口径)。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views, make_collate_fixed_views
from models.stvft.stvft import STVFT
from models.stvft.stvft_pretrained import STVFTPretrained
from utils.kp_star import kp_star_indices

# COCO-17 关节名 (与 baseline evaluate() 同序), 给 KP* 用
COCO_NAMES = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb',
              'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
KP_IDX = kp_star_indices(COCO_NAMES)   # KP* 10 关节索引 (肩肘腕膝踝)


def mpjpe(pred, gt):
    """mean per-joint position error. pred/gt: (..., J, 3)。返回标量 (与输入同单位)。
    sqrt 内加 eps: 防 sqrt(0) 梯度=inf (预测贴近GT时距离→0→inf梯度→NaN) 的反复发散。"""
    return (((pred - gt) ** 2).sum(-1) + 1e-8).sqrt().mean()


def per_t_forward(model, rays, confs, delta_ts, L, gate_override=None):
    """per-timestep: 对每个 t_offset forward, 预测该帧 pose。返回 (B, L, J, 3)。
    gate_override: None→学到的gate; 0→旁路(baseline); 1→满接入TFT。"""
    preds = []
    for t in range(L):
        new_dts = delta_ts - delta_ts[:, :, t:t + 1]   # 把第 t 帧设为 target (Δt=0)
        preds.append(model(rays, confs, new_dts, t_target=t, gate_override=gate_override))
    return torch.stack(preds, dim=1)                     # (B, L, J, 3)


@torch.no_grad()
def eval_center(model, val_dl, L, device, gate_override=None):
    """与 12w/baseline **同口径**的评估: 固定视角 + **中心帧单帧** + **Abs MPJPE**。
    返回每clip的All-17/KP*的 mean 和 median。median 抗重尾(motion-sampling 有少数检测崩的
    灾难性clip拉高mean; median=典型clip≈12w)。(per-t/MPJVE 是时序训练加项, eval 用中心帧。)"""
    was_training = model.training
    model.eval()
    ct = L // 2
    all17, kp = [], []
    for b in val_dl:
        rays = b['rays'].to(device); confs = b['confs'].to(device)
        dts = b['delta_ts'].to(device); gt = b['gt_3d'].to(device)
        new_dts = dts - dts[:, :, ct:ct + 1]                         # 中心帧设为 target (Δt=0)
        pred = model(rays, confs, new_dts, t_target=ct, gate_override=gate_override)  # (B,J,3)
        pj = ((pred - gt[:, ct]) ** 2).sum(-1).sqrt() * 1000          # (B,J) mm, Abs per-joint
        all17 += list(pj.mean(-1).cpu().numpy())                     # (B,) 每clip全17
        kp += list(pj[:, KP_IDX].mean(-1).cpu().numpy())             # (B,) 每clip KP*
    if was_training:
        model.train()
    if not all17:
        return {k: float('nan') for k in ('all17_mean', 'all17_med', 'kp_mean', 'kp_med')}
    a, k = np.array(all17), np.array(kp)
    return {'all17_mean': float(a.mean()), 'all17_med': float(np.median(a)),
            'kp_mean': float(k.mean()), 'kp_med': float(np.median(k))}


def eval_all_views(model, val_dls, L, device, gate_override=None):
    """对每个固定视角数评估。val_dls: {n_views: dataloader}。返回 {n_views: {'all17','kp'}}。"""
    return {nv: eval_center(model, dl, L, device, gate_override) for nv, dl in val_dls.items()}


def train(args):
    device = torch.device(args.device)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("[anomaly] autograd 异常检测已开 (会变慢, 在产 nan 的 backward 算子处抛异常)")
    from torch.utils.data import Subset
    full_ds = STVFTClipDataset(args.data_glob, L_window=args.L, min_oks=args.min_oks)
    # 固定种子留出 held-out val (val 目录为空, 从 train 划)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_ds), generator=g).tolist()
    n_val = min(args.val_clips, len(full_ds) // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    collate = make_collate_random_views(args.min_views, args.max_views, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=args.workers, drop_last=True)
    # eval 与 12w 同口径: 固定视角数(默认 V=5 匹配 12w TEST_VIEWS, 另 V=2 对照 CMU)
    val_dls = {}
    for nv in args.eval_views:
        vc = make_collate_fixed_views(nv, seed=args.seed + 100 + nv)
        val_dls[nv] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=vc, num_workers=args.workers)
    primary_v = args.eval_views[0]   # best-ckpt 判据用第一个视角数(默认 V=2=研究口径)
    print(f"[data] train {len(ds)} / val {len(val_ds)} clips, {len(dl)} batches/epoch, "
          f"batch={args.batch_size}, eval固定视角={args.eval_views} (12w同口径: 固定视角+中心帧+Abs)")

    if args.pretrained_ckpt:
        model = STVFTPretrained(args.pretrained_cfg, args.pretrained_ckpt,
                                freeze_backbone=bool(args.freeze_backbone),
                                gate_init=args.gate_init).to(device)
        tag = f"pretrained(freeze={bool(args.freeze_backbone)})"
    else:
        model = STVFT().to(device)
        tag = "from-scratch"
    if args.fixed_gate is not None and hasattr(model, 'gate'):
        model.gate.requires_grad_(False)   # 固定gate不学, forward 用 gate_override 替换; VFT/PFT 已由 freeze_backbone 冻结
        print(f"[fixed-gate] gate 冻结, forward gate_override={args.fixed_gate} (满接入, 只训 TFT, 测时序天花板)")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {tag}: {n_total/1e6:.2f}M total, {n_train/1e6:.2f}M trainable, L={args.L}")
    if args.gate_lr is not None and hasattr(model, 'gate'):
        # ReZero 陷阱: gate=0 时 TFT 梯度被乘没(∂L/∂tft_out=gate·...=0), gate 必须先长大
        # TFT 才解冻。给 gate 单独大 lr 加速解冻。
        gate_p = [model.gate]
        other_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith('gate')]
        opt = torch.optim.Adam([
            {'params': other_p, 'lr': args.lr},
            {'params': gate_p, 'lr': args.gate_lr},
        ], weight_decay=1e-4)
        print(f"[opt] gate lr={args.gate_lr} (单独), other(TFT+dt) lr={args.lr}")
    else:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    base = eval_all_views(model, val_dls, args.L, device, gate_override=0.0)  # gate=0 不加时序地板
    mode = f"fixed_gate={args.fixed_gate}" if args.fixed_gate is not None else "learnable_gate"
    print(f"[BASELINE] gate=0 不加时序地板 (12w同口径: Abs 中心帧, mm; 训练模式={mode}):")
    for nv in args.eval_views:
        b = base[nv]
        print(f"   V={nv}: All-17 med={b['all17_med']:.1f}/mean={b['all17_mean']:.1f}  "
              f"KP* med={b['kp_med']:.1f}/mean={b['kp_mean']:.1f}  (med抗重尾≈12w; mean含motion-sampling灾难clip尾)")
    base_primary = base[primary_v]['all17_med']   # 判据用 median(抗重尾)
    best_val = float('inf')
    diag = {'n': 0}   # 非有限梯度诊断计数 (只打印前几次崩溃现场)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {'loss': 0.0, 'pos': 0.0, 'vel': 0.0, 'n': 0}
        for it, b in enumerate(dl):
            rays = b['rays'].to(device); confs = b['confs'].to(device)
            dts = b['delta_ts'].to(device); gt = b['gt_3d'].to(device)  # (B,L,J,3)

            preds = per_t_forward(model, rays, confs, dts, args.L, gate_override=args.fixed_gate)  # (B,L,J,3)
            pos = mpjpe(preds, gt)
            v_pred = preds[:, 1:] - preds[:, :-1]
            v_gt = gt[:, 1:] - gt[:, :-1]
            vel = mpjpe(v_pred, v_gt)
            loss = args.lambda_per_t * pos + args.lambda_vel * vel

            opt.zero_grad(); loss.backward()
            named_tp = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
            bad = [(nm, p) for nm, p in named_tp if p.grad is not None and not torch.isfinite(p.grad).all()]
            if (not torch.isfinite(loss)) or bad:
                # 诊断: 非有限 loss/梯度的崩溃现场 (打印前3次, 不改逻辑只多打印)
                if diag['n'] < 3:
                    diag['n'] += 1
                    print(f"\n[DIAG#{diag['n']}] ep{epoch} it{it}: loss={loss.item()} "
                          f"(finite={bool(torch.isfinite(loss))}) pos={pos.item()} vel={vel.item()}")
                    print(f"  dts: min={dts.min().item():.4f} max={dts.max().item():.4f} "
                          f"nan={bool(torch.isnan(dts).any())} inf={bool(torch.isinf(dts).any())}")
                    print(f"  preds: nan={bool(torch.isnan(preds).any())} inf={bool(torch.isinf(preds).any())} "
                          f"absmax={preds.abs().max().item():.3e}")
                    print(f"  gt: nan={bool(torch.isnan(gt).any())} inf={bool(torch.isinf(gt).any())} "
                          f"absmax={gt.abs().max().item():.3f}")
                    print(f"  confs: nan={bool(torch.isnan(confs).any())} min={confs.min().item():.3f} max={confs.max().item():.3f}")
                    for t in range(args.L):
                        pn, gn = bool(torch.isnan(preds[:, t]).any()), bool(torch.isnan(gt[:, t]).any())
                        if pn or gn:
                            print(f"    t={t}: pred_nan={pn} gt_nan={gn}")
                    for nm, p in bad[:10]:
                        fin = torch.isfinite(p.grad)
                        gmax = p.grad.abs()[fin].max().item() if fin.any() else float('nan')
                        print(f"    grad非有限: {nm} weight_absmax={p.abs().max().item():.3e} "
                              f"grad_finite_absmax={gmax:.3e}")
                    if not bad:
                        print("    (所有 grad 有限 → loss 本身非有限)")
                continue
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([p for _, p in named_tp], args.grad_clip)
            opt.step()

            bs = rays.shape[0]
            agg['loss'] += loss.item() * bs; agg['pos'] += pos.item() * bs
            agg['vel'] += vel.item() * bs; agg['n'] += bs

            if it == 0 and epoch == 0:
                # sanity: per-t 张量 shape + 显存
                print(f"[sanity] preds {tuple(preds.shape)} gt {tuple(gt.shape)} "
                      f"(应 B,L={args.L},17,3); v_pred {tuple(v_pred.shape)}")
                if device.type == 'cuda':
                    print(f"[sanity] 显存 {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        n = agg['n']
        if n == 0:
            print(f"[epoch {epoch}] 全 batch 被跳过(梯度非有限)→ 权重可能已坏, 停止训练")
            break
        ev = eval_all_views(model, val_dls, args.L, device, gate_override=args.fixed_gate)
        gate_v = args.fixed_gate if args.fixed_gate is not None else (model.gate.item() if hasattr(model, 'gate') else 0.0)
        cur = ev[primary_v]['all17_med']             # best 判据 = 主视角 All-17 median(抗重尾)
        import math
        if math.isfinite(cur) and cur < best_val:
            best_val = cur
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch,
                        'eval': ev, 'base': base},
                       os.path.join(args.save_dir, 'stvft_best.pth'))
            star = " *best"
        else:
            star = ""
        vmsg = "  ".join(f"V{nv}[med {ev[nv]['all17_med']:.1f}/{ev[nv]['kp_med']:.1f} mn {ev[nv]['all17_mean']:.1f}]"
                         for nv in args.eval_views)
        verdict = "↓时序有用" if cur < base_primary else "↑未降"
        print(f"[epoch {epoch}] train_pos {agg['pos']/n*1000:.2f}mm "
              f"(loss {agg['loss']/n*1000:.2f}, vel {agg['vel']/n*1000:.2f}) gate={gate_v:.4f} | "
              f"{vmsg} (med All17/KP* + mn All17; baseV{primary_v}med={base_primary:.1f}, {verdict}){star} | {time.time()-t0:.1f}s")

    ckpt = os.path.join(args.save_dir, 'stvft_final.pth')
    torch.save({'model': model.state_dict(), 'args': vars(args)}, ckpt)
    print(f"[save] {ckpt}")
    if device.type == 'cuda':
        print(f"[peak mem] {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/sanity/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/sanity")
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--lambda-per-t', type=float, default=1.0)
    ap.add_argument('--lambda-vel', type=float, default=0.1)
    ap.add_argument('--min-views', type=int, default=2)
    ap.add_argument('--max-views', type=int, default=5)
    ap.add_argument('--val-clips', type=int, default=200, help='从 train 留出的 held-out val clip 数')
    ap.add_argument('--eval-views', type=int, nargs='+', default=[2, 5],
                    help='eval 固定视角数(与12w同口径); 第一个=best判据(默认V2研究口径,V5对照12w的14mm)')
    ap.add_argument('--min-oks', type=float, default=0.0,
                    help='>0: 滤掉中心帧中位OKS<阈值的clip(HRNet检测崩的垃圾, motion-sampling产物); 0=不过滤(=12w)')
    ap.add_argument('--fixed-gate', type=float, default=None, help='固定 gate 值(如1.0=满接入); None=学习gate')
    ap.add_argument('--grad-clip', type=float, default=None, help='梯度裁剪 max-norm; None=不裁剪')
    ap.add_argument('--gate-init', type=float, default=0.0, help='gate 初始值; 0=ReZero(冷启动陷阱), 0.1=破冷启动')
    ap.add_argument('--detect-anomaly', action='store_true', help='开 autograd 异常检测, 在产 nan 的 backward 算子处抛异常+定位 forward 行')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pretrained-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--pretrained-ckpt', default=None, help='给路径=路A(加载baseline); None=从头训')
    ap.add_argument('--freeze-backbone', type=int, default=1)
    ap.add_argument('--gate-lr', type=float, default=None, help='gate单独lr(ReZero解冻); None=同lr')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    train(args)
