"""CAA 训练: 单帧(L=1) conf-aware 跨视角融合。训 caa_scale + 微调 VFT(blocks_view_fusion),
让 VFT 适应 conf 加权(sweep 显示冻结VFT只能到λ~1, 要更强需微调)。rel loss(官方指标), 混合视角。
eval: B(caa off=baseline) vs C(caa on), AMASS V2/V5 官方 rel-mean。CMU 用 caa_sweep/cmu 脚本另测。
"""
import argparse, copy, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_fixed_views, make_collate_random_views
from models.stvft.stvft_v2 import STVFTv2
from run.train_stvft_v2 import huber_dist, eval_v2   # 复用 rel loss + 官方eval


@torch.no_grad()
def eval_bc(model, dl, L, dev):
    """返回 B(caa off) 和 C(caa on) 的官方 rel-mean。单帧 no_temporal。"""
    prev = model.caa
    model.caa = False; B = eval_v2(model, dl, L, dev, no_temporal=True)
    model.caa = True;  C = eval_v2(model, dl, L, dev, no_temporal=True)
    model.caa = prev
    return B, C


def train(args):
    dev = args.device
    model = STVFTv2(args.cfg, args.pretrained_ckpt, freeze_backbone=True, temporal_layers=2).to(dev)
    model.caa = True
    # 解冻: caa_scale(本就可训) + VFT(blocks_view_fusion) 微调
    vft_params = list(model.backbone.blocks_view_fusion.parameters()) + list(model.backbone.View_norm.parameters())
    for p in vft_params:
        p.requires_grad = True
    trainable = [model.caa_scale] + vft_params
    ntr = sum(p.numel() for p in trainable if p.requires_grad)
    print(f"[model] CAA 单帧, 可训: caa_scale + VFT = {ntr/1e6:.2f}M")

    ds_tr = STVFTClipDataset(args.data_glob, L_window=args.L, min_oks=args.min_oks, perturb=0.0)
    ds_val = copy.copy(ds_tr); ds_val.eval_fixed_window = True
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds_tr), generator=g).tolist()
    nv = min(args.val_clips, len(ds_tr) // 5)
    val_idx, tr_idx = perm[:nv], perm[nv:]
    dl = DataLoader(Subset(ds_tr, tr_idx), batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate_random_views(args.min_train_views, args.max_train_views, seed=args.seed),
                    num_workers=args.workers, drop_last=True)
    val_dls = {v: DataLoader(Subset(ds_val, val_idx), batch_size=args.batch_size, shuffle=False,
                             collate_fn=make_collate_fixed_views(v, seed=args.seed + 1), num_workers=args.workers)
               for v in args.eval_views}
    print(f"[data] train {len(tr_idx)}/val {nv}, L={args.L}, 混合视角 k∈[{args.min_train_views},{args.max_train_views}]")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[opt] AdamW lr={args.lr} wd={args.weight_decay}, rel-loss, Huber {args.huber_delta}")
    os.makedirs(args.save_dir, exist_ok=True)
    ct = args.L // 2

    base0 = {}   # 原始 baseline(训练前, 冻结), 作总增益的固定参照
    for v in args.eval_views:
        B, C = eval_bc(model, val_dls[v], args.L, dev)
        base0[v] = B['all17']
        print(f"[init] V={v}: 原始baseline A17={B['all17']:.3f}  C(CAAλ=0) A17={C['all17']:.3f} (应≈)")
    pv = args.eval_views[0]; best = float('inf')
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); agg = {'l': 0.0, 'n': 0}
        for b in dl:
            rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev)
            gt = b['gt_3d'][:, ct].to(dev)
            pred = model(rays, confs, dts, no_temporal=True)      # 单帧 + CAA
            loss = huber_dist(pred, gt, args.huber_delta, rel=True)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
                continue
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            bs = rays.shape[0]; agg['l'] += loss.item() * bs; agg['n'] += bs
        if agg['n'] == 0:
            print(f"[ep {ep}] 全跳过, 停"); break
        msg = []; cur = None
        for v in args.eval_views:
            B, C = eval_bc(model, val_dls[v], args.L, dev)
            dtot = (C['all17'] - base0[v]) * 10     # vs 原始baseline = 总增益(负=好)
            dcaa = (C['all17'] - B['all17']) * 10    # vs 微调后无CAA = CAA边际贡献
            msg.append(f"V{v}[C {C['all17']:.3f} Δtot {dtot:+.2f} Δcaa {dcaa:+.2f}mm]")
            if v == pv: cur = C['all17']
        star = ""
        if cur < best:
            best = cur
            torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': ep}, os.path.join(args.save_dir, 'caa_best.pth'))
            star = " *best"
        print(f"[ep {ep}] loss {agg['l']/agg['n']*1000:.2f} | {'  '.join(msg)} (λ={model.caa_scale.item():.2f}){star} | {time.time()-t0:.0f}s")
    torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(args.save_dir, 'caa_final.pth'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--save-dir', default="/mnt/data/cjyoutput/stvft/caa_run1")
    ap.add_argument('--pretrained-ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--L', type=int, default=1)
    ap.add_argument('--min-oks', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--huber-delta', type=float, default=0.1)
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--min-train-views', type=int, default=2)
    ap.add_argument('--max-train-views', type=int, default=5)
    ap.add_argument('--eval-views', type=int, nargs='+', default=[2, 5])
    ap.add_argument('--val-clips', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
