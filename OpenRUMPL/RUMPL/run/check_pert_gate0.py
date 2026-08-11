"""实证: gate=0 时 per-t 误差分项, 验"残差 center 写死 frame-2"缺陷。

预期(若缺陷成立): t=L//2(中心)≈baseline 低; t≠中心 显著更高, 且 ≈ 该帧 GT 相对
中心帧的运动量(因为 gate=0 时所有 per-t 预测都是 frame-2 姿态, 与目标帧 GT 错位)。
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views
from models.stvft.stvft_pretrained import STVFTPretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--rumpl-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--rumpl-ckpt', required=True)
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--n-batches', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device(args.device)
    ds = STVFTClipDataset(args.data_glob, L_window=args.L)
    collate = make_collate_random_views(2, 5, seed=0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    print(f"[data] {len(ds)} clips, batch={args.batch_size}, L={args.L}")

    model = STVFTPretrained(args.rumpl_cfg, args.rumpl_ckpt, freeze_backbone=True).to(device)
    model.eval()
    print(f"[model] gate={model.gate.item()} (init应0)")

    L = args.L
    per_t_err = np.zeros(L); gt_motion = np.zeros(L); nb = 0
    with torch.no_grad():
        for it, b in enumerate(dl):
            if it >= args.n_batches: break
            rays = b['rays'].to(device); confs = b['confs'].to(device)
            dts = b['delta_ts'].to(device); gt = b['gt_3d'].to(device)  # (B,L,J,3)
            # 复刻 train 的 per_t_forward, 但强制 gate_override=0
            preds = []
            for t in range(L):
                new_dts = dts - dts[:, :, t:t+1]
                preds.append(model(rays, confs, new_dts, t_target=t, gate_override=0))  # (B,J,3) 修复: 残差基随 t
            preds = torch.stack(preds, dim=1)  # (B,L,J,3)
            err = ((preds - gt) ** 2).sum(-1).sqrt().mean(dim=(0, 2))  # (L,) mm前
            ctr = L // 2
            motion = ((gt - gt[:, ctr:ctr+1]) ** 2).sum(-1).sqrt().mean(dim=(0, 2))  # (L,)
            per_t_err += err.cpu().numpy() * 1000
            gt_motion += motion.cpu().numpy() * 1000
            nb += 1
    per_t_err /= nb; gt_motion /= nb
    ctr = L // 2
    print(f"\n=== gate=0 per-t pos 误差 (mm), 中心 t={ctr} ===")
    for t in range(L):
        tag = "  <-- 中心(应≈baseline)" if t == ctr else ""
        print(f"  t={t}: pos_err={per_t_err[t]:7.2f}mm | GT相对中心运动={gt_motion[t]:7.2f}mm{tag}")
    print(f"\n  中心帧 t={ctr} 误差 = {per_t_err[ctr]:.2f}mm (= baseline 中心帧水平?)")
    print(f"  非中心均值 = {np.delete(per_t_err, ctr).mean():.2f}mm (错位则显著高)")
    print(f"  全窗口均值 = {per_t_err.mean():.2f}mm (= 训练初始 loss 量级?)")


if __name__ == '__main__':
    main()
