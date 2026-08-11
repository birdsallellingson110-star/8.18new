"""层次2 最小验证(go/no-go闸门, 非最终方案):
推理期 ray-aware 骨长约束 refine 作用在 RUMPL 的 CMU V=2 真实输出上, 能否动 KP*。
机制: min_p  Σ conf·dist(p, ray)²  +  λ·Σ(‖骨‖−目标骨长)²    根(盆骨)锚定→不动全局, 只改相对结构。
骨长来源: GT(上界闸门, 标注) / V5(落地). 阈值: KP* rel 降>1.5有信号 / <0.5存疑。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from models.stvft.stvft_v2 import STVFTv2
from run.cmu_eval_v2 import windows_for_seq, win_tensors

KP = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
PV = [11, 12]


def metrics(P, G):  # (N,17,3) m → dict mm
    P, G = P * 100, G * 100  # cm
    da = np.sqrt(((P - G) ** 2).sum(-1))[:, KP].mean(-1) * 10
    pv_p = P[:, PV].mean(1, keepdims=True); pv_g = G[:, PV].mean(1, keepdims=True)
    dr = np.sqrt((((P - pv_p) - (G - pv_g)) ** 2).sum(-1))[:, KP].mean(-1) * 10
    bp = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=-1) for a, b in BONES], 1)
    bg = np.stack([np.linalg.norm(G[:, a] - G[:, b], axis=-1) for a, b in BONES], 1)
    return dict(abs_m=da.mean(), abs_md=np.median(da), rel_m=dr.mean(), rel_md=np.median(dr),
                bone=np.abs(bp - bg).mean() * 10)


def refine(pred, rays, confs, target_bones, lam, dev, steps=300, lr=5e-3):
    """pred(N,17,3) init; rays(N,17,V,6)[dir,inter]; confs(N,17,V); target_bones(N,nB). 返回 refined(N,17,3) 锚根."""
    p = torch.tensor(pred, dtype=torch.float32, device=dev, requires_grad=True)
    rays = torch.tensor(rays, dtype=torch.float32, device=dev)
    confs = torch.tensor(confs, dtype=torch.float32, device=dev)
    tb = torch.tensor(target_bones, dtype=torch.float32, device=dev)
    d = rays[..., 0:3]; inter = rays[..., 3:6]
    dhat = d / (d.norm(dim=-1, keepdim=True) + 1e-8)            # (N,17,V,3)
    root0 = p.detach()[:, PV].mean(1)                          # RUMPL 根
    bidx = torch.tensor(BONES, device=dev)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        vec = p.unsqueeze(2) - inter                           # (N,17,V,3)
        proj = (vec * dhat).sum(-1, keepdim=True)
        perp2 = ((vec - proj * dhat) ** 2).sum(-1)             # (N,17,V)
        ray_loss = (confs * perp2).sum(-1).mean()
        blen = (p[:, bidx[:, 0]] - p[:, bidx[:, 1]]).norm(dim=-1)  # (N,nB)
        bone_loss = ((blen - tb) ** 2).mean()
        loss = ray_loss + lam * bone_loss
        opt.zero_grad(); loss.backward(); opt.step()
    out = p.detach()
    out = out - out[:, PV].mean(1, keepdim=True) + root0.unsqueeze(1)  # 锚根: 平移使盆骨=RUMPL盆骨
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default="/mnt/data/cjydata/cmu_singleperson/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_filtered_1_1_mmpose_hrnet_coco_matched_swapv3/cmu_panoptic_validation.pkl")
    ap.add_argument('--ckpt', default="/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar")
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--bone-src', default='gt', choices=['gt', 'v5'], help='gt=上界闸门; v5=落地')
    ap.add_argument('--views', nargs='+', type=int, default=[3, 6])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    m = STVFTv2(args.cfg, args.ckpt, freeze_backbone=True).to(dev).eval()

    d = pickle.load(open(args.pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e

    def collect(views, L=1):
        items = []
        for pid, fr in seqs.items():
            for w in windows_for_seq(fr, views, L):
                items.append(win_tensors(fr, w, views, L, 0))
        rays = torch.stack([c[0] for c in items]); confs = torch.stack([c[1] for c in items])
        dts = torch.stack([c[2] for c in items]); gt = torch.stack([c[3] for c in items])
        preds = []
        with torch.no_grad():
            for i in range(0, len(items), 64):
                p = m(rays[i:i+64].to(dev), confs[i:i+64].to(dev), dts[i:i+64].to(dev), t_target=0, no_temporal=True)
                preds.append(p.cpu())
        return rays.numpy(), confs.numpy(), gt.numpy(), torch.cat(preds).numpy()

    rays, confs, gt, pred = collect(args.views)   # rays (N,17,V,1,6) L=1
    rays = rays[:, :, :, 0, :]; confs = confs[:, :, :, 0, 0]   # (N,17,V,6),(N,17,V)
    B = metrics(pred, gt)
    print(f"[B 原始RUMPL V={len(args.views)}] abs KP*={B['abs_m']:.2f}(md{B['abs_md']:.2f}) rel KP*={B['rel_m']:.2f}(md{B['rel_md']:.2f}) bone={B['bone']:.2f}mm  (N={len(gt)})")

    # 目标骨长
    if args.bone_src == 'gt':
        tb = np.stack([np.linalg.norm(gt[:, a] - gt[:, b], axis=-1) for a, b in BONES], 1)
        print("[骨长来源] GT (上界闸门, 标注: 用了测试GT骨长)")
    else:
        _, _, _, pred5 = collect([3, 6, 12, 13, 23])
        tb = np.stack([np.linalg.norm(pred5[:, a] - pred5[:, b], axis=-1) for a, b in BONES], 1)
        print("[骨长来源] V=5 RUMPL输出估 (落地, 数据集无关)")

    print(f"\n{'λ':>6} {'abs KP*':>9} {'ΔB':>7} {'rel KP*':>9} {'ΔB':>7} {'rel中位':>8} {'ΔB':>7} {'bone':>7}")
    for lam in [0.0, 0.5, 2.0, 5.0, 20.0]:
        C = metrics(refine(pred, rays, confs, tb, lam, dev), gt)
        print(f"{lam:6.1f} {C['abs_m']:9.2f} {C['abs_m']-B['abs_m']:+7.2f} {C['rel_m']:9.2f} {C['rel_m']-B['rel_m']:+7.2f} {C['rel_md']:8.2f} {C['rel_md']-B['rel_md']:+7.2f} {C['bone']:7.2f}")


if __name__ == '__main__':
    main()
