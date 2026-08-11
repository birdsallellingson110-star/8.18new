"""CMU 金标准: B(gate=0) vs C(时序) 在真实 CMU 连续帧上, 与 baseline 同口径。
读 swapv3 pkl (list of entries), 按(pose_id,image_id)分组(pose5/6 image_id重叠, 必须分序列),
滑动 L=5 窗口, 中心帧 Abs MPJPE (All-17 + KP*), mean+median, V=2/V=5。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from dataset.stvft_dataset import build_rays_one_view
from models.stvft.stvft_pretrained import STVFTPretrained

KP = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]   # KP* 肩肘腕膝踝
ALLV = [3, 6, 12, 13, 23]


def windows_for_seq(frames_seq, views, L=5):
    ids = sorted(frames_seq.keys())
    out = []
    for i in range(len(ids) - L + 1):
        w = ids[i:i + L]
        if w[-1] - w[0] != L - 1:
            continue
        if all(all(c in frames_seq[f] for c in views) for f in w):
            out.append(w)
    return out


def win_tensors(frames_seq, win, views, L=5, t_target=2, fps=30.0):
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


def abs_mpjpe(pred, gt):  # (J,3) m → per-joint mm
    return (pred - gt).norm(dim=-1).numpy() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--rumpl-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--rumpl-ckpt', required=True)
    ap.add_argument('--stvft-ckpt', required=True, help='训练好的 ST-VFT (含TFT+gate)')
    ap.add_argument('--views', nargs='+', type=int, default=None, help='None=同时跑V2[3,6]和V5')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    model = STVFTPretrained(args.rumpl_cfg, args.rumpl_ckpt, freeze_backbone=True).to(dev).eval()
    sd = torch.load(args.stvft_ckpt, map_location='cpu'); sd = sd['model'] if 'model' in sd else sd
    model.load_state_dict(sd, strict=False)
    print(f"[load] gate={model.gate.item():.4f}")
    d = pickle.load(open(args.pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e
    print(f"[data] 序列: {dict((k, len(v)) for k, v in seqs.items())}")

    viewsets = [[3, 6], ALLV] if args.views is None else [args.views]
    for views in viewsets:
        wins = []
        for pid, fr in seqs.items():
            wins += [(pid, w) for w in windows_for_seq(fr, views)]
        Ba, Bk, Ca, Ck = [], [], [], []
        with torch.no_grad():
            for pid, w in wins:
                rays, confs, dts, gt = win_tensors(seqs[pid], w, views)
                rays = rays.unsqueeze(0).to(dev); confs = confs.unsqueeze(0).to(dev); dts = dts.unsqueeze(0).to(dev)
                pB = model(rays, confs, dts, t_target=2, gate_override=0)[0].cpu()
                pC = model(rays, confs, dts, t_target=2, gate_override=None)[0].cpu()
                eB = abs_mpjpe(pB, gt); eC = abs_mpjpe(pC, gt)
                Ba.append(eB.mean()); Bk.append(eB[KP].mean()); Ca.append(eC.mean()); Ck.append(eC[KP].mean())
        Ba, Bk, Ca, Ck = map(np.array, (Ba, Bk, Ca, Ck))
        print(f"\n=== V={len(views)} {views} ({len(wins)} 窗口) Abs MPJPE mm ===")
        print(f"  B(gate=0)  : All-17 med={np.median(Ba):.1f}/mean={Ba.mean():.1f}  KP* med={np.median(Bk):.1f}/mean={Bk.mean():.1f}")
        print(f"  C(时序)    : All-17 med={np.median(Ca):.1f}/mean={Ca.mean():.1f}  KP* med={np.median(Ck):.1f}/mean={Ck.mean():.1f}")
        print(f"  增益 B-C   : All-17 med={np.median(Ba)-np.median(Ca):+.1f}/mean={Ba.mean()-Ca.mean():+.1f}  KP* med={np.median(Bk)-np.median(Ck):+.1f}/mean={Bk.mean()-Ck.mean():+.1f}")


if __name__ == '__main__':
    main()
