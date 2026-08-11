"""切分 46mm: 几何/摆位 OOD vs 2D 检测质量。
同一 clip 场景(同相机/同摆位/同 GT), 只换 2D 来源喂 baseline backbone(gate=0等价):
  - HRNet 2D (joints_2d_mmpose): = 之前测的 46mm (含 2D 噪声)
  - GT 2D    (joints_2d_amass) : 完美 2D, 只剩几何/摆位的影响
若 GT-2D≈14 → 几何 OK, 46 全是 HRNet 噪声; 若 GT-2D≈46 → 几何/摆位 OOD。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import glob, pickle
import numpy as np
import torch
from dataset.stvft_dataset import build_rays_one_view
from models.stvft.stvft_pretrained import STVFTPretrained

CKPT = "/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar"
CFG = "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"


def build_input(j2d, conf, cams, views):
    J, V = 17, len(views)
    x = np.zeros((J, V, 7), np.float32)
    for vi, v in enumerate(views):
        d, i = build_rays_one_view(j2d[v], cams[v])
        x[:, vi, 0:3] = d; x[:, vi, 3:6] = i; x[:, vi, 6] = conf[v, :, 0]
    return torch.from_numpy(x)


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = 'cuda'
    model = STVFTPretrained(CFG, CKPT, freeze_backbone=True).to(device).eval()
    bb = model.backbone
    files = sorted(glob.glob("/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl"))[:5]

    res = {'hrnet': [], 'gt': []}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for f in files:
            d = pickle.load(open(f, 'rb'))
            Nc = d['joints_3d'].shape[0]
            for ci in range(Nc):
                t = d['joints_3d'].shape[1] // 2          # 中心帧
                Vtot = d['joints_2d_mmpose'].shape[2]
                k = int(rng.integers(2, 6)); k = min(k, Vtot)
                views = sorted(rng.choice(Vtot, k, replace=False).tolist())
                cams = d['camera_parameters_all'][ci]
                conf = d['confs_2d_mmpose'][ci, t]        # (V,17,1) 两边同 conf, 只换2D坐标
                gt = torch.from_numpy(d['joints_3d'][ci, t].astype(np.float32)).to(device)  # (17,3)
                for src, key in [('joints_2d_mmpose', 'hrnet'), ('joints_2d_amass', 'gt')]:
                    j2d = d[src][ci, t]                   # (V,17,2)
                    x = build_input(j2d, conf, cams, views).unsqueeze(0).to(device)  # (1,J,V,7)
                    pred = bb(x, is_training=False)[0]    # (17,3)
                    err = (pred - gt).pow(2).sum(-1).sqrt().mean().item() * 1000
                    res[key].append(err)
    print(f"[n clips] {len(res['hrnet'])}")
    print(f"  HRNet 2D (joints_2d_mmpose): All-17 Abs = {np.mean(res['hrnet']):.2f} mm  (= 之前 46?)")
    print(f"  GT    2D (joints_2d_amass) : All-17 Abs = {np.mean(res['gt']):.2f} mm")
    print(f"  → HRNet-GT 差 = {np.mean(res['hrnet'])-np.mean(res['gt']):.2f} mm (2D噪声贡献)")
    print(f"  → GT 残留 = {np.mean(res['gt']):.2f} mm (几何/摆位贡献; ≈14则几何OK, ≈46则OOD)")


if __name__ == '__main__':
    main()
