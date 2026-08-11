"""层次1: 在RUMPL基础上"加"骨长一致性约束(几何/解剖普适, 契合RUMPL几何优先哲学, 补Fig.4 V=2深度歧义缺口)。
保留RUMPL全部强项(ray/VFT融合冻结), 只微调出3D的PFT+head, 加 L_bone。
L = L_mpjpe(原) + λ·L_bone(||预测骨长-GT骨长|| + 左右对称)。单帧。eval CMU V=2(官方口径 abs KP* + median)。
"""
import argparse, os, sys, time, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views, make_collate_fixed_views
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe
from run.cmu_eval_v2 import windows_for_seq, win_tensors

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
SYM = [(0, 2), (1, 3), (4, 6), (5, 7), (10, 11)]  # 左右对称骨对(臂上/臂下/腿上/腿下/躯干侧)在BONES里的index


def bones_of(x):  # (B,17,3) → (B,nbones)
    return torch.stack([(x[:, a] - x[:, b]).norm(dim=-1) for a, b in BONES], 1)


def bone_loss(pred, gt):
    blp, blg = bones_of(pred), bones_of(gt)
    l_len = F.l1_loss(blp, blg)                                  # 骨长 vs GT骨长
    l_sym = sum(F.l1_loss(blp[:, i], blp[:, j]) for i, j in SYM) / len(SYM)  # 左右对称
    return l_len + 0.3 * l_sym


def huber_dist(pred, gt, delta=0.1):
    d = (((pred - gt) ** 2).sum(-1) + 1e-8).sqrt()
    return F.huber_loss(d, torch.zeros_like(d), delta=delta)


def load_cmu(pkl, views, L=9, bs=32):  # L=9 中心帧 = 官方可比子集(baseline abs=40.37=官方); 模型仍 no_temporal 单帧
    d = pickle.load(open(pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    ct = L // 2; items = []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, views, L):
            items.append(win_tensors(fr, w, views, L, ct))
    batches = [(torch.stack([c[0] for c in items[i:i+bs]]), torch.stack([c[1] for c in items[i:i+bs]]),
                torch.stack([c[2] for c in items[i:i+bs]]), torch.stack([c[3] for c in items[i:i+bs]]))
               for i in range(0, len(items), bs)]
    return batches, ct


@torch.no_grad()
def eval_cmu(model, batches, ct, dev):
    preds, gts = [], []
    for rays, confs, dts, gt in batches:
        p = model(rays.to(dev), confs.to(dev), dts.to(dev), t_target=ct, no_temporal=True).cpu().numpy()
        preds.append(p); gts.append(gt.numpy())
    pred = np.stack([x for b in preds for x in b]) * 100.0; g = np.stack([x for b in gts for x in b]) * 100.0  # cm
    out = {}
    # 绝对(paper headline)
    da = np.sqrt(((pred - g) ** 2).sum(-1))[:, KP_IDX].mean(-1) * 10
    out['abs_mean'] = da.mean(); out['abs_med'] = float(np.median(da))
    # pelvis-relative MEAN(官方对比口径, 骨长约束作用空间)
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0; pv_g = (g[:, 11:12] + g[:, 12:13]) / 2.0
    pr = pred - pv_p; gr = g - pv_g
    dr = np.sqrt(((pr - gr) ** 2).sum(-1))[:, KP_IDX].mean(-1) * 10
    out['rel_mean'] = dr.mean(); out['rel_med'] = float(np.median(dr))
    # 骨长误差(看约束有没有真的改善解剖)
    bp = np.stack([np.linalg.norm(pred[:, a] - pred[:, b], axis=-1) for a, b in BONES], 1)
    bg = np.stack([np.linalg.norm(g[:, a] - g[:, b], axis=-1) for a, b in BONES], 1)
    out['bone'] = np.abs(bp - bg).mean() * 10  # mm
    return out


def train(args):
    dev = args.device
    model = STVFTv2(args.cfg, args.pretrained_ckpt, freeze_backbone=True).to(dev)
    bb = model.backbone
    tp = list(bb.blocks.parameters()) + list(bb.Spatial_norm.parameters()) + list(bb.head.parameters())  # PFT+head
    for p in model.parameters():
        p.requires_grad = False
    for p in tp:
        p.requires_grad = True
    print(f"[model] RUMPL + 骨长约束, 微调 PFT+head = {sum(p.numel() for p in tp)/1e6:.2f}M (encode/VFT冻结=保留ray融合)")

    ds = STVFTClipDataset(args.data_glob, L_window=1, min_oks=args.min_oks, perturb=0.0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate_random_views(2, 5, seed=args.seed), num_workers=args.workers, drop_last=True)
    cmu2, ct2 = load_cmu(args.cmu_pkl, [3, 6]); cmu5, ct5 = load_cmu(args.cmu_pkl, [3, 6, 12, 13, 23])
    print(f"[data] AMASS {len(ds)} clips; CMU V2={sum(b[0].shape[0] for b in cmu2)} V5={sum(b[0].shape[0] for b in cmu5)} | λ_bone={args.lam}")

    opt = torch.optim.AdamW(tp, lr=args.lr, weight_decay=args.weight_decay)
    B2 = eval_cmu(model, cmu2, ct2, dev); B5 = eval_cmu(model, cmu5, ct5, dev)   # 原始baseline(微调前)
    print(f"[原始baseline] V2 KP*: rel-mean={B2['rel_mean']:.2f} abs-mean={B2['abs_mean']:.2f} bone={B2['bone']:.2f}mm | V5 rel-mean={B5['rel_mean']:.2f} abs-mean={B5['abs_mean']:.2f}")
    os.makedirs(args.save_dir, exist_ok=True); best = float('inf')
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); agg = defaultdict(float); n = 0
        for b in dl:
            rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev); gt = b['gt_3d'][:, 0].to(dev)
            pred = model(rays, confs, dts, t_target=0, no_temporal=True)
            lm = huber_dist(pred, gt); lb = bone_loss(pred, gt); loss = lm + args.lam * lb
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tp, 0.5); opt.step()
            agg['m'] += lm.item(); agg['b'] += lb.item(); n += 1
        C2 = eval_cmu(model, cmu2, ct2, dev); C5 = eval_cmu(model, cmu5, ct5, dev)
        star = ""
        if C2['rel_mean'] < best:
            best = C2['rel_mean']; torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep}, os.path.join(args.save_dir, 'bone_best.pth')); star = " *best"
        print(f"[ep {ep}] mpjpe {agg['m']/max(n,1)*1000:.1f} bone {agg['b']/max(n,1):.3f} | "
              f"CMU V2 KP* rel={C2['rel_mean']:.2f}(ΔB{C2['rel_mean']-B2['rel_mean']:+.2f}) abs={C2['abs_mean']:.2f}(ΔB{C2['abs_mean']-B2['abs_mean']:+.2f}) bone={C2['bone']:.2f}(ΔB{C2['bone']-B2['bone']:+.2f}) "
              f"| V5 rel={C5['rel_mean']:.2f}(ΔB{C5['rel_mean']-B5['rel_mean']:+.2f}){star} | {time.time()-t0:.0f}s")
    print(f"\n[判定] V2 KP* rel-mean best={best:.2f} vs 原始baseline={B2['rel_mean']:.2f} ({best-B2['rel_mean']:+.2f}mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/bone_run1")
    ap.add_argument('--pretrained-ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--cmu-pkl', default="/mnt/data/cjydata/cmu_temporal_centered/full/cmu_panoptic_validation.pkl")
    ap.add_argument('--min-oks', type=float, default=0.7)
    ap.add_argument('--lam', type=float, default=1.0, help='骨长loss权重')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
