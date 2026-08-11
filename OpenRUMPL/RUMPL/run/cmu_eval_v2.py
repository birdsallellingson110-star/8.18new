"""CMU 金标准迁移验证: STVFTv2, 官方口径(pelvis-relative MEAN, cm, calc_mpjpe)。
B(no_temporal=baseline) vs C(temporal)。真实 CMU 连续帧, 滑动 L 窗口, 中心帧。
验证 RUMPL 范式: AMASS 上的时序增益是否迁移到真实 CMU(且因更难而放大)。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from dataset.stvft_dataset import build_rays_one_view
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]
ALLV = [3, 6, 12, 13, 23]


def windows_for_seq(frames_seq, views, L):
    ids = sorted(frames_seq.keys())
    out = []
    for i in range(len(ids) - L + 1):
        w = ids[i:i + L]
        if w[-1] - w[0] != L - 1:
            continue
        if all(all(c in frames_seq[f] for c in views) for f in w):
            out.append(w)
    return out


def win_tensors(frames_seq, win, views, L, t_target, fps=30.0):
    J, V = 17, len(views)
    rays = np.zeros((J, V, L, 6), np.float32); confs = np.zeros((J, V, L, 1), np.float32)
    for li, fid in enumerate(win):
        for vi, c in enumerate(views):
            e = frames_seq[fid][c]
            cam = dict(e['camera']); cam = {**cam, 'T': np.asarray(cam['T'], np.float64).reshape(3) / 100.0}
            dirs, inters = build_rays_one_view(np.asarray(e['joints_2d'], np.float32), cam)
            rays[:, vi, li, 0:3] = dirs; rays[:, vi, li, 3:6] = inters
            confs[:, vi, li, 0] = np.asarray(e['joints_2d_conf']).reshape(-1)
    dt = (np.arange(L, dtype=np.float32) - t_target) / fps
    dts = np.broadcast_to(dt[None, :], (V, L)).copy()
    gt = np.asarray(frames_seq[win[t_target]][views[0]]['joints_3d'], np.float32) / 100.0
    return torch.from_numpy(rays), torch.from_numpy(confs), torch.from_numpy(dts), torch.from_numpy(gt)


def official_metric(preds, gts):  # lists of (17,3) m → All-17 + KP* rel-mean (cm)
    pred = np.stack(preds) * 100.0; gt = np.stack(gts) * 100.0
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0; pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
    pred = pred - pv_p; gt = gt - pv_g
    _, all17 = calc_mpjpe(gt, pred, mode='absolute')
    _, kp = calc_mpjpe(gt, pred, mode='absolute', not_consider_kp=NOT_KP)
    return all17, kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--ckpt', required=True, help='STVFTv2 训练好的 ckpt')
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--views', nargs='+', type=int, default=None, help='None=同时跑V2[3,6]和V5')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    ct = args.L // 2

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tl = ck.get('args', {}).get('temporal_layers', 2)
    rumpl_ckpt = ck.get('args', {}).get('pretrained_ckpt')
    model = STVFTv2(args.cfg, rumpl_ckpt, temporal_layers=tl).to(dev).eval()
    model.load_state_dict(ck['model'])
    print(f"[load] {args.ckpt} (ep={ck.get('epoch','?')}, L={args.L}, temporal_layers={tl})")

    d = pickle.load(open(args.pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    print(f"[data] 序列: {dict((k, len(v)) for k, v in seqs.items())}")

    viewsets = [[3, 6], ALLV] if args.views is None else [args.views]
    for views in viewsets:
        wins = []
        for pid, fr in seqs.items():
            wins += [(pid, w) for w in windows_for_seq(fr, views, args.L)]
        pB, gB, pC, gC = [], [], [], []
        with torch.no_grad():
            for pid, w in wins:
                rays, confs, dts, gt = win_tensors(seqs[pid], w, views, args.L, ct)
                rays = rays.unsqueeze(0).to(dev); confs = confs.unsqueeze(0).to(dev); dts = dts.unsqueeze(0).to(dev)
                oB = model(rays, confs, dts, t_target=ct, no_temporal=True)[0].cpu().numpy()
                oC = model(rays, confs, dts, t_target=ct, no_temporal=False)[0].cpu().numpy()
                pB.append(oB); gB.append(gt.numpy()); pC.append(oC); gC.append(gt.numpy())
        Ba, Bk = official_metric(pB, gB); Ca, Ck = official_metric(pC, gC)
        print(f"\n=== V={len(views)} {views} ({len(wins)} 窗口) 官方 rel-mean (cm) ===")
        print(f"  B(baseline): All-17={Ba:.3f}  KP*={Bk:.3f}")
        print(f"  C(时序)    : All-17={Ca:.3f}  KP*={Ck:.3f}")
        print(f"  增益 B-C   : All-17={Ba-Ca:+.3f}  KP*={Bk-Ck:+.3f}  (正=时序改善)")


if __name__ == '__main__':
    main()
