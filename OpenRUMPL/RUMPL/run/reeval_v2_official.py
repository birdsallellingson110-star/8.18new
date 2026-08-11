"""用官方口径重评 STVFTv2 ckpt: pelvis-anchored(mid-hip) + MEAN(官方主指标) vs absolute + median(我们之前用的)。
B(no_temporal=baseline) vs C(temporal), V=2/V=5, All-17 + KP*。同训练 val split。
"""
import argparse, copy, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_fixed_views
from models.stvft.stvft_v2 import STVFTv2
from utils.kp_star import kp_star_indices

COCO_NAMES = ['nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb',
              'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
KP_IDX = kp_star_indices(COCO_NAMES)


def pelvis_anchor(x):  # x (N,17,3) → 减 mid-hip (J11+J12)/2 (官方 COCO pelvis)
    pelvis = (x[:, 11:12, :] + x[:, 12:13, :]) / 2.0
    return x - pelvis


@torch.no_grad()
def collect(model, dl, L, dev, no_temporal):
    ct = L // 2
    preds, gts = [], []
    for b in dl:
        rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev)
        pred = model(rays, confs, dts, no_temporal=no_temporal).cpu().numpy()  # (B,17,3) m
        gt = b['gt_3d'][:, ct].numpy()                                          # (B,17,3) m
        preds.append(pred); gts.append(gt)
    return np.concatenate(preds), np.concatenate(gts)


def metrics(pred, gt):
    """返回 dict: abs/rel × all17/kp × mean/median (mm)。"""
    out = {}
    for align, fn in [('abs', lambda a: a), ('rel', pelvis_anchor)]:
        p, g = fn(pred.copy()), fn(gt.copy())
        pj = np.sqrt(((p - g) ** 2).sum(-1)) * 1000          # (N,17) mm per-joint
        per_sample_all = pj.mean(-1)                          # (N,)
        per_sample_kp = pj[:, KP_IDX].mean(-1)                # (N,)
        out[f'{align}_all_mean'] = per_sample_all.mean()      # ← 官方主指标(若align=rel)
        out[f'{align}_all_med'] = np.median(per_sample_all)
        out[f'{align}_kp_mean'] = per_sample_kp.mean()
        out[f'{align}_kp_med'] = np.median(per_sample_kp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--min-oks', type=float, default=0.5)
    ap.add_argument('--val-clips', type=int, default=200)
    ap.add_argument('--eval-views', type=int, nargs='+', default=[2, 5])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tl = ck.get('args', {}).get('temporal_layers', 2)
    rumpl_ckpt = ck.get('args', {}).get('pretrained_ckpt')
    model = STVFTv2(args.cfg, rumpl_ckpt, temporal_layers=tl).to(args.device)
    model.load_state_dict(ck['model']); model.eval()
    print(f"[ckpt] {args.ckpt} (ep={ck.get('epoch','?')}, temporal_layers={tl})")

    ds = STVFTClipDataset(args.data_glob, L_window=args.L, min_oks=args.min_oks, perturb=0.0)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    nv = min(args.val_clips, len(ds) // 5)
    val_idx = perm[:nv]
    print(f"[val] {nv} clips (同训练 split)\n")

    for v in args.eval_views:
        dl = DataLoader(Subset(ds, val_idx), batch_size=8, shuffle=False,
                        collate_fn=make_collate_fixed_views(v, seed=args.seed + 1), num_workers=4)
        pB, gB = collect(model, dl, args.L, args.device, no_temporal=True)   # baseline
        pC, gC = collect(model, dl, args.L, args.device, no_temporal=False)  # temporal
        mB, mC = metrics(pB, gB), metrics(pC, gC)
        print(f"===== V={v} =====")
        print(f"  {'指标':<22}{'B(baseline)':>14}{'C(temporal)':>14}{'C-B(↓好)':>12}")
        for key, label in [('rel_all_mean', '★官方 rel All-17 MEAN'), ('rel_kp_mean', '★官方 rel KP* MEAN'),
                           ('rel_all_med', '  rel All-17 median'), ('rel_kp_med', '  rel KP* median'),
                           ('abs_all_mean', '  abs All-17 MEAN'), ('abs_all_med', '  abs All-17 median(旧主看)')]:
            d = mC[key] - mB[key]
            print(f"  {label:<22}{mB[key]:>14.2f}{mC[key]:>14.2f}{d:>+12.2f}")
        print()


if __name__ == '__main__':
    main()
