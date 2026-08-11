"""时序"可学习上限"实验(零训练): 用 baseline 逐帧 V=2/V=5 预测, 几种非学习聚合估计中心帧,
看时序最多能降多少误差 → 区分 (A)多视角下本无时序空间 vs (B)我们模块没学到。
聚合: center-only(地板) / mean / median / savgol(尊重运动只去抖) / oracle借帧(用GT逐关节挑最准帧, 选择性上限)。
官方口径(pelvis-rel mean, cm)。AMASS val + CMU 各跑 V=2/V=5。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Subset
from dataset.stvft_dataset import STVFTClipDataset, make_collate_fixed_views
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe
from run.cmu_eval_v2 import windows_for_seq, win_tensors, ALLV

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]


def metric(pred, gt):  # (N,17,3) m → All-17, KP* rel-mean (cm)
    pred = pred * 100.0; gt = gt * 100.0
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0; pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
    pred = pred - pv_p; gt = gt - pv_g
    _, a = calc_mpjpe(gt, pred, mode='absolute')
    _, k = calc_mpjpe(gt, pred, mode='absolute', not_consider_kp=NOT_KP)
    return a, k


@torch.no_grad()
def per_frame_preds(model, rays, confs, dts, L, dev):
    # → (B, L, J, 3): 每帧单独 baseline 预测
    out = []
    for l in range(L):
        out.append(model(rays, confs, dts, t_target=l, no_temporal=True).cpu().numpy())
    return np.stack(out, axis=1)


def aggregate(pf, gt_c, conf, ct, L):
    """pf (B,L,J,3), gt_c (B,J,3), conf (B,L,J) → dict method→(pred B,J,3)。"""
    res = {'center': pf[:, ct], 'mean': pf.mean(1), 'median': np.median(pf, 1)}
    win = L if L % 2 == 1 else L - 1
    res['savgol'] = savgol_filter(pf, window_length=win, polyorder=2, axis=1)[:, ct]
    B, J = gt_c.shape[0], gt_c.shape[1]
    bi, ji = np.arange(B)[:, None], np.arange(J)[None]
    # conf_borrow: 用 conf 逐关节挑最高conf帧 (无GT, 可实现)
    res['conf_borrow'] = pf[bi, conf.argmax(1), ji]
    # oracle_borrow: 用 GT 逐关节挑最准帧 (上限, 不可实现)
    d = np.linalg.norm(pf - gt_c[:, None], axis=-1)             # (B,L,J)
    res['oracle_borrow'] = pf[bi, d.argmin(1), ji]
    return res


def run(model, win_iter, L, dev, label):
    ct = L // 2
    acc = defaultdict(lambda: {'p': [], 'g': []})
    for rays, confs, dts, gt_c in win_iter:
        rays = rays.to(dev); confs = confs.to(dev); dts = dts.to(dev)
        pf = per_frame_preds(model, rays, confs, dts, L, dev)
        gt_np = gt_c.numpy()
        conf = confs.mean(2).squeeze(-1).permute(0, 2, 1).cpu().numpy()   # (B,L,J) 视角平均conf
        for m, pred in aggregate(pf, gt_np, conf, ct, L).items():
            acc[m]['p'].append(pred); acc[m]['g'].append(gt_np)
    print(f"\n========== {label} ==========")
    print(f"  {'方法':<16}{'All-17':>9}{'KP*':>9}{'ΔAll17 vs center':>18}")
    base_a = None
    for m in ['center', 'mean', 'median', 'savgol', 'conf_borrow', 'oracle_borrow']:
        a, k = metric(np.concatenate(acc[m]['p']), np.concatenate(acc[m]['g']))
        if m == 'center': base_a = a
        print(f"  {m:<16}{a:>9.3f}{k:>9.3f}{(a-base_a)*10:>+15.2f}mm")


def amass_windows(glob, L, V, seed, val_clips, bs=8):
    ds = STVFTClipDataset(glob, L_window=L, min_oks=0.5, perturb=0.0)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    nv = min(val_clips, len(ds) // 5)
    dl = DataLoader(Subset(ds, perm[:nv]), batch_size=bs, shuffle=False,
                    collate_fn=make_collate_fixed_views(V, seed=seed + 1), num_workers=4)
    ct = L // 2
    for b in dl:
        yield b['rays'], b['confs'], b['delta_ts'], b['gt_3d'][:, ct]


def cmu_windows(pkl, L, views, bs=16):
    d = pickle.load(open(pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    ct = L // 2
    buf_r, buf_c, buf_d, buf_g = [], [], [], []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, views, L):
            rays, confs, dts, gt = win_tensors(fr, w, views, L, ct)
            buf_r.append(rays); buf_c.append(confs); buf_d.append(dts); buf_g.append(gt)
            if len(buf_r) == bs:
                yield torch.stack(buf_r), torch.stack(buf_c), torch.stack(buf_d), torch.stack(buf_g)
                buf_r, buf_c, buf_d, buf_g = [], [], [], []
    if buf_r:
        yield torch.stack(buf_r), torch.stack(buf_c), torch.stack(buf_d), torch.stack(buf_g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rumpl-ckpt', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--amass-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--cmu-pkl', default="/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl")
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--val-clips', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    model = STVFTv2(args.cfg, args.rumpl_ckpt, temporal_layers=2).to(args.device).eval()
    print(f"[model] baseline (no_temporal), L={args.L}")
    for V in [2, 5]:
        run(model, amass_windows(args.amass_glob, args.L, V, args.seed, args.val_clips), args.L, args.device, f"AMASS val  V={V}")
    for views in [[3, 6], ALLV]:
        run(model, cmu_windows(args.cmu_pkl, args.L, views), args.L, args.device, f"CMU  V={len(views)} {views}")


if __name__ == '__main__':
    main()
