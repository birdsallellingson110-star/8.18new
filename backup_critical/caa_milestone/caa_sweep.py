"""CAA λ-sweep(零训练探针): CMU V=2/V=5 单帧(L=1), 手设 caa_scale, 看 conf 加权 VFT 帮不帮。
官方口径(pelvis-rel mean, cm)。λ=0 即 baseline。注意: baseline VFT 没用 conf_weights 训过,
手设 λ 是 OOD 探针——若有帮助是强证据, 若有害则模糊(可能需训练让 VFT 适应)。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
from models.stvft.stvft_v2 import STVFTv2
from core.evaluate import calc_mpjpe
from run.cmu_eval_v2 import windows_for_seq, win_tensors, ALLV

KP_IDX = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
NOT_KP = [i for i in range(17) if i not in KP_IDX]


def metric(preds, gts):
    pred = np.stack(preds) * 100.0; gt = np.stack(gts) * 100.0
    pv_p = (pred[:, 11:12] + pred[:, 12:13]) / 2.0; pv_g = (gt[:, 11:12] + gt[:, 12:13]) / 2.0
    pred = pred - pv_p; gt = gt - pv_g
    _, a = calc_mpjpe(gt, pred, mode='absolute')
    _, k = calc_mpjpe(gt, pred, mode='absolute', not_consider_kp=NOT_KP)
    return a, k


@torch.no_grad()
def eval_caa(model, seqs, views, L, lams, dev):
    ct = L // 2
    wins = []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, views, L):
            wins.append((pid, w))
    # 预存张量
    batch = [win_tensors(seqs[pid], w, views, L, ct) for pid, w in wins]
    out = {}
    for lam in lams:
        model.caa = (lam > 0); model.caa_scale.data.fill_(float(lam))
        preds, gts = [], []
        for rays, confs, dts, gt in batch:
            r = rays.unsqueeze(0).to(dev); c = confs.unsqueeze(0).to(dev); d = dts.unsqueeze(0).to(dev)
            p = model(r, c, d, t_target=ct, no_temporal=True)[0].cpu().numpy()
            preds.append(p); gts.append(gt.numpy())
        out[lam] = metric(preds, gts)
    return out, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rumpl-ckpt', required=True)
    ap.add_argument('--ckpt', default=None, help='微调后的CAA ckpt(载入VFT权重); 不给=原始baseline')
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--pkl', default="/mnt/data/cjydata/cmu_temporal/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_temporal_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl")
    ap.add_argument('--L', type=int, default=1)
    ap.add_argument('--lams', type=float, nargs='+', default=[0, 0.5, 1, 2, 5, 10])
    ap.add_argument('--per-seq', action='store_true', help='按 pose_id 分序列分别评(验稳定性)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    model = STVFTv2(args.cfg, args.rumpl_ckpt, temporal_layers=2).to(args.device).eval()
    if args.ckpt:   # 载入微调后的 VFT (CAA训练产物)
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(sd['model']); print(f"[load ckpt] {args.ckpt} (ep={sd.get('epoch','?')})")
    d = pickle.load(open(args.pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e

    groups = {'ALL': seqs}
    if args.per_seq:
        groups = {pid: {pid: seqs[pid]} for pid in sorted(seqs.keys())}
    for gname, gseqs in groups.items():
        for views in [[3, 6], ALLV]:
            out, nw = eval_caa(model, gseqs, views, args.L, args.lams, args.device)
            base_a = out[0][0]
            print(f"\n===== [{gname}] V={len(views)} ({nw}窗口) 官方rel-mean cm, 单帧 =====")
            print(f"  {'λ':>6}{'All-17':>9}{'KP*':>9}{'ΔAll17':>10}")
            for lam in args.lams:
                a, k = out[lam]
                print(f"  {lam:>6.1f}{a:>9.3f}{k:>9.3f}{(a-base_a)*10:>+8.2f}mm")


if __name__ == '__main__':
    main()
