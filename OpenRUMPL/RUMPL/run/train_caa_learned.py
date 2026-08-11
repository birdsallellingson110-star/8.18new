"""A: 学习版自适应 conf→weight 训练(冻结VFT, 只训 conf2w ~113参数)。
view-perturbation(2模式50:50: mode1坏视角+低conf / mode2坏视角+conf不变)给 g 信号。
CMU pose5/pose6 做 val + 早停(防AMASS-specific)。回退保护: 比固定λ=0.5差则退回。
"""
import argparse, os, sys, time, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe
from run.train_stvft_v2 import huber_dist
from run.cmu_eval_v2 import windows_for_seq, win_tensors, ALLV

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]


def view_perturb(rays, confs, frac=0.5, offset=0.4, dev='cuda'):
    """训练时坏一个视角(每样本概率 frac)。rays(B,J,V,L,6) confs(B,J,V,L,1)。
    mode1(50%): 坏rays+conf×0.1; mode2(50%): 坏rays, conf不变(看着正常实则坏)。"""
    B, J, V, L, _ = rays.shape
    if V < 2:
        return rays, confs
    rays, confs = rays.clone(), confs.clone()
    sel = torch.rand(B, device=dev) < frac
    for b in torch.nonzero(sel).flatten():
        v = int(torch.randint(0, V, (1,)))
        rays[b, :, v] += torch.randn_like(rays[b, :, v]) * offset   # 坏射线→几何不一致
        if torch.rand(1).item() < 0.5:                             # mode1: 同时压低conf
            confs[b, :, v] *= 0.1
    return rays, confs


@torch.no_grad()
def eval_cmu(model, batches, dev, mode):
    """mode: 'base'(无CAA) / 'learned' / 'fixed'(λ=0.5)。返回 All-17 rel-mean(cm)。"""
    model.caa = (mode == 'fixed'); model.caa_learned = (mode == 'learned')
    if mode == 'fixed':
        model.caa_scale.data.fill_(0.5)
    preds, gts = [], []
    for rays, confs, dts, gt in batches:
        p = model(rays.to(dev), confs.to(dev), dts.to(dev), no_temporal=True).cpu().numpy()
        preds.append(p); gts.append(gt.numpy())
    pred = np.concatenate(preds) * 100.0; g = np.concatenate(gts) * 100.0
    pvp = (pred[:, 11:12] + pred[:, 12:13]) / 2; pvg = (g[:, 11:12] + g[:, 12:13]) / 2
    _, a = calc_mpjpe(g - pvg, pred - pvp, mode='absolute')
    return a


def load_cmu(pkl, views, L=1, bs=16):
    d = pickle.load(open(pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    ct = L // 2
    items = []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, views, L):
            items.append(win_tensors(fr, w, views, L, ct))
    batches = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        batches.append((torch.stack([c[0] for c in chunk]), torch.stack([c[1] for c in chunk]),
                        torch.stack([c[2] for c in chunk]), torch.stack([c[3] for c in chunk])))
    return batches


def train(args):
    dev = args.device
    model = STVFTv2(args.cfg, args.pretrained_ckpt, freeze_backbone=True, temporal_layers=2).to(dev)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.conf2w.parameters():
        p.requires_grad = True                       # 只训 conf2w(冻结VFT, 防微调陷阱)
    ntr = sum(p.numel() for p in model.conf2w.parameters())
    print(f"[model] A 学习版, 只训 conf2w = {ntr} 参数 (冻结VFT)")

    ds = STVFTClipDataset(args.data_glob, L_window=1, min_oks=args.min_oks, perturb=0.0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate_random_views(2, 5, seed=args.seed),
                    num_workers=args.workers, drop_last=True)
    cmu = {2: load_cmu(args.cmu_pkl, [3, 6]), 5: load_cmu(args.cmu_pkl, ALLV)}
    print(f"[data] AMASS train {len(ds)}; CMU val V2={sum(b[0].shape[0] for b in cmu[2])} V5={sum(b[0].shape[0] for b in cmu[5])}")

    opt = torch.optim.AdamW(model.conf2w.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 基线参照(CMU): baseline(无CAA) + fixed λ=0.5(回退基准)
    base2 = eval_cmu(model, cmu[2], dev, 'base'); base5 = eval_cmu(model, cmu[5], dev, 'base')
    fix2 = eval_cmu(model, cmu[2], dev, 'fixed'); fix5 = eval_cmu(model, cmu[5], dev, 'fixed')
    print(f"[CMU基准] V2: baseline={base2:.3f} λ0.5={fix2:.3f}({(fix2-base2)*10:+.2f}mm) | "
          f"V5: baseline={base5:.3f} λ0.5={fix5:.3f}({(fix5-base5)*10:+.2f}mm)")
    os.makedirs(args.save_dir, exist_ok=True)
    best = float('inf')
    for ep in range(args.epochs):
        model.train(); model.caa_learned = True; model.caa = False
        t0 = time.time(); agg = {'l': 0.0, 'n': 0}
        for b in dl:
            rays, confs = view_perturb(b['rays'].to(dev), b['confs'].to(dev), args.pert_frac, args.pert_offset, dev)
            dts = b['delta_ts'].to(dev); gt = b['gt_3d'][:, 0].to(dev)
            pred = model(rays, confs, dts, no_temporal=True)
            loss = huber_dist(pred, gt, args.huber_delta, rel=True)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.conf2w.parameters(), 1.0)
            opt.step()
            bs = rays.shape[0]; agg['l'] += loss.item() * bs; agg['n'] += bs
        L2 = eval_cmu(model, cmu[2], dev, 'learned'); L5 = eval_cmu(model, cmu[5], dev, 'learned')
        star = ""
        if L2 < best:
            best = L2
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep}, os.path.join(args.save_dir, 'caa_learned_best.pth'))
            star = " *best"
        print(f"[ep {ep}] loss {agg['l']/max(agg['n'],1)*1000:.2f} | "
              f"CMU V2 learned={L2:.3f}(Δbase{(L2-base2)*10:+.2f} Δλ0.5{(L2-fix2)*10:+.2f}) "
              f"V5 learned={L5:.3f}(Δbase{(L5-base5)*10:+.2f}){star} | {time.time()-t0:.0f}s")
    # 回退判定
    print(f"\n[回退判定] V2: learned_best={best:.3f} vs λ0.5={fix2:.3f} → "
          f"{'用学习版(更好)' if best < fix2 else '⚠️退回固定λ0.5(学习版没赢)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/caa_learned")
    ap.add_argument('--pretrained-ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--cmu-pkl', default="/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl")
    ap.add_argument('--min-oks', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--huber-delta', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--pert-frac', type=float, default=0.5)
    ap.add_argument('--pert-offset', type=float, default=0.4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
