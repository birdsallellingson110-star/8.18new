"""贡献1: 质量感知时序训练。AMASS连续+帧扰动(mode1低conf/mode2几何错→监督质量分)训时序;
eval 连续CMU V=2 absolute KP*(论文口径): B(无时序) vs C(质量时序) vs 朴素平均(go/no-go基准)。
成功线: C 在 absolute KP* 上稳定 < B 且 < 朴素平均(我们上限实验~0.6mm), 目标 ~2mm。
"""
import argparse, os, sys, time, pickle, math
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views, make_collate_fixed_views
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe
from run.train_stvft_v2 import huber_dist
from run.cmu_eval_v2 import windows_for_seq, win_tensors, ALLV

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]


def metric(preds, gts):  # (N,17,3) m → [abs_kp_mean, abs_kp_med, rel_kp_mean, rel_kp_med] mm
    pred = np.stack(preds) * 100.0; gt = np.stack(gts) * 100.0
    out = []
    for rel in (False, True):
        p, g = pred.copy(), gt.copy()
        if rel:
            p = p - (p[:, 11:12] + p[:, 12:13]) / 2; g = g - (g[:, 11:12] + g[:, 12:13]) / 2
        d = np.sqrt(((p - g) ** 2).sum(-1))[:, KP_IDX].mean(-1) * 10   # (N,) 每样本 KP* mm
        out += [d.mean(), float(np.median(d))]                          # mean + median(典型, 尾巴鲁棒)
    return out  # [abs_kp_mean, abs_kp_med, rel_kp_mean, rel_kp_med]


def load_cmu(pkl, views, L, bs=12):
    d = pickle.load(open(pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    ct = L // 2; items = []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, views, L):
            items.append(win_tensors(fr, w, views, L, ct))
    return [(torch.stack([c[0] for c in items[i:i+bs]]), torch.stack([c[1] for c in items[i:i+bs]]),
             torch.stack([c[2] for c in items[i:i+bs]]), torch.stack([c[3] for c in items[i:i+bs]]))
            for i in range(0, len(items), bs)]


@torch.no_grad()
def eval_amass(model, batches, L, dev, no_temporal):
    """缓存的扰动AMASS val: temporal vs baseline 的 abs KP*(mm)。看是否单调下降=在学。"""
    ct = L // 2; preds, gts = [], []
    for rays, confs, dts, gtc in batches:
        p = model(rays.to(dev), confs.to(dev), dts.to(dev), t_target=ct, no_temporal=no_temporal).cpu().numpy()
        preds.append(p); gts.append(gtc.numpy())
    pred = np.concatenate(preds) * 100.0; g = np.concatenate(gts) * 100.0
    d = np.sqrt(((pred - g) ** 2).sum(-1))[:, KP_IDX].mean(-1) * 10   # (N,) 每样本 KP* mm
    return float(np.median(d))                                        # median(典型, 尾巴鲁棒)


@torch.no_grad()
def eval_cmu(model, batches, L, dev, mode):
    """mode: 'base'(无时序中心帧) / 'temporal'(质量时序) / 'mean'(逐帧baseline平均)。"""
    ct = L // 2; preds, gts = [], []
    for rays, confs, dts, gt in batches:
        rays, confs, dts = rays.to(dev), confs.to(dev), dts.to(dev)
        if mode == 'mean':
            pf = [model(rays, confs, dts, t_target=l, no_temporal=True).cpu().numpy() for l in range(L)]
            p = np.stack(pf, 1).mean(1)
        else:
            p = model(rays, confs, dts, t_target=ct, no_temporal=(mode == 'base')).cpu().numpy()
        preds.append(p); gts += [g.numpy() for g in gt]
    preds = np.concatenate(preds)
    return metric(list(preds), gts)


def train(args):
    dev = args.device
    model = STVFTv2(args.cfg, args.pretrained_ckpt, freeze_backbone=True, temporal_layers=args.temporal_layers).to(dev)
    model.avg_residual = args.avg_residual
    tp = list(model.temporal.parameters()) + (list(model.res_head.parameters()) if args.avg_residual else [])
    for p in model.parameters():
        p.requires_grad = False
    for p in tp:
        p.requires_grad = True
    print(f"[model] 质量感知时序 avg_residual={args.avg_residual}, 训={sum(p.numel() for p in tp)/1e6:.2f}M")

    ds = STVFTClipDataset(args.data_glob, L_window=args.L, min_oks=args.min_oks,
                          perturb=args.perturb, perturb_offset_px=args.perturb_offset_px)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    nv = 200; val_idx, tr_idx = perm[:nv], perm[nv:]
    dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate_random_views(2, 5, seed=args.seed),
                    num_workers=args.workers, drop_last=True)
    # 缓存固定的扰动 AMASS val(每epoch同样本, 看temporal是否单调改善=在学)
    import numpy as _np; _np.random.seed(123)
    vdl = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False,
                     collate_fn=make_collate_fixed_views(2, seed=args.seed + 1), num_workers=0)
    amass_val = [(b['rays'], b['confs'], b['delta_ts'], b['gt_3d'][:, args.L // 2]) for b in vdl]
    cmu2 = load_cmu(args.cmu_pkl, [3, 6], args.L)
    print(f"[data] AMASS train {len(tr_idx)}/val {nv}(扰动缓存), perturb={args.perturb}; CMU V2 windows={sum(b[0].shape[0] for b in cmu2)}")

    opt = torch.optim.AdamW(tp, lr=args.lr, weight_decay=args.weight_decay)
    # 固定参照(baseline不变): B(无时序) + 朴素平均
    B = eval_cmu(model, cmu2, args.L, dev, 'base')
    MN = eval_cmu(model, cmu2, args.L, dev, 'mean')
    # 时序该发力处=relative(对标论文); abs作旁参。metric=[abs_a17,abs_kp,rel_a17,rel_kp]
    aB = eval_amass(model, amass_val, args.L, dev, no_temporal=True)   # AMASS val baseline KP* median
    # metric=[abs_mean,abs_med,rel_mean,rel_med]。主看 median(典型, 不被极端姿态尾巴主导)
    print(f"[参照 KP* median] CMU rel: B={B[3]:.2f} 朴素平均={MN[3]:.2f}({MN[3]-B[3]:+.2f}) | abs: B={B[1]:.2f} 平均={MN[1]:.2f}({MN[1]-B[1]:+.2f}) | AMASS-val abs B={aB:.2f}")
    print(f"  (mean旁参: CMU rel B={B[2]:.1f} abs B={B[0]:.1f}) | 目标: CMU median 降1-2mm")
    os.makedirs(args.save_dir, exist_ok=True)
    best = float('inf'); ct = args.L // 2
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); agg = {'l': 0.0, 'n': 0}
        for b in dl:
            rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev)
            gt = b['gt_3d'][:, ct].to(dev)
            pred = model(rays, confs, dts)
            loss = huber_dist(pred, gt, args.huber_delta, rel=False)   # absolute(论文口径)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in tp):
                continue
            torch.nn.utils.clip_grad_norm_(tp, args.grad_clip); opt.step()
            bs = rays.shape[0]; agg['l'] += loss.item() * bs; agg['n'] += bs
        C = eval_cmu(model, cmu2, args.L, dev, 'temporal')
        star = ""
        if C[3] < best:                              # 按 relKP* 选(时序该发力处)
            best = C[3]
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep}, os.path.join(args.save_dir, 'tq_best.pth'))
            star = " *best"
        aC = eval_amass(model, amass_val, args.L, dev, no_temporal=False)   # AMASS val temporal KP* median
        print(f"[ep {ep}] loss {agg['l']/max(agg['n'],1)*1000:.2f} | AMASS-val med={aC:.2f}(ΔB{aC-aB:+.2f}) "
              f"|| CMU rel-med={C[3]:.2f}(Δmean{C[3]-MN[3]:+.2f} ΔB{C[3]-B[3]:+.2f}) abs-med={C[1]:.2f}(ΔB{C[1]-B[1]:+.2f}) relmean={C[2]:.1f}{star} | {time.time()-t0:.0f}s")
    print(f"\n[判定 rel-median] C_best={best:.2f} vs B={B[3]:.2f}({best-B[3]:+.2f}) vs 朴素平均={MN[3]:.2f}({best-MN[3]:+.2f})")
    print("  → 时序有效(超过平均)" if best < MN[3] else "  → 仅达平均水平" if best <= B[3] else "  → ⚠️ 比baseline差")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/temporal_q")
    ap.add_argument('--pretrained-ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--cmu-pkl', default="/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl")
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--temporal-layers', type=int, default=2)
    ap.add_argument('--min-oks', type=float, default=0.5)
    ap.add_argument('--avg-residual', action='store_true', help='time输出=朴素平均+zero-init残差(稳定,保证≥平均)')
    ap.add_argument('--perturb', type=float, default=0.6)
    ap.add_argument('--perturb-offset-px', type=float, default=120.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--huber-delta', type=float, default=0.1)
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
