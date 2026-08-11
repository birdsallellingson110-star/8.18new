"""CMU 时序评估 (ST-VFT 金标准: B vs C)。

CMU 测试集是单帧(skip64), ST-VFT 需连续 L=5 帧。本脚本:
  1. 读连续帧 CMU 数据(skip=1 + preprocess + HRNet + swap, 同 baseline eval 格式)
  2. 按 image_id 组连续 L=5 窗口(全 V 视角齐), 中心帧 t_target=2
  3. ray 构造复用 stvft_dataset.build_rays_one_view(CMU camera, cm→m)
  4. STVFTPretrained 评估:
     - B (gate_override=0): TFT 旁路 = baseline 中心帧 (隔离"窗口输入"影响)
     - C (gate_override=None): 完整 ST-VFT
  5. MPJPE(中心帧 pred vs gt), 报告 B vs C 增益

关键: Δt = (arange(L)-t_target)/30 (CMU 30fps); 训练是 120fps → Δt 尺度 4x,
DeltaTEncoder 正弦连续可外推, 但训练没见过大 Δt, 评估时观察。

用法: python run/cmu_temporal_eval.py --ckpt <stvft.pth> --cmu-pkl <连续帧pkl>
"""
import argparse
import os
import sys
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

import numpy as np
import torch

from dataset.stvft_dataset import build_rays_one_view
from models.stvft.stvft_pretrained import STVFTPretrained


CMU_VIEWS = [3, 6, 12, 13, 23]   # paper 5 相机
KP_STAR = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 占位; 实际 KP* 见下方按 COCO 子集
# 论文 KP* = 肩肘腕膝踝 (COCO idx): 5,6(肩) 7,8(肘) 9,10(腕) 13,14(膝) 15,16(踝)
KP_STAR_COCO = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]


def load_continuous_frames(pkl_path):
    """读 CMU 连续帧数据, 按 image_id 组多视角。返回 {image_id: {camera_id: sample}}"""
    d = pickle.load(open(pkl_path, 'rb'))
    frames = defaultdict(dict)
    for e in d:
        frames[e['image_id']][e['camera_id']] = e
    return frames


def build_windows(frames, L=5, views=CMU_VIEWS):
    """连续 image_id 组 L 窗口(帧号严格连续 + 全视角齐)。返回 list of [image_id×L]。"""
    ids = sorted(frames.keys())
    windows = []
    for i in range(len(ids) - L + 1):
        win = ids[i:i + L]
        if win[-1] - win[0] != L - 1:        # 严格连续
            continue
        if all(all(c in frames[f] for c in views) for f in win):  # 全视角齐
            windows.append(win)
    return windows


def window_to_tensors(frames, win, views=CMU_VIEWS, L=5, t_target=2, fps=30.0):
    """组 rays(J,V,L,6) + confs(J,V,L,1) + delta_ts(V,L) + gt(J,3)。CMU cm→m。"""
    J, V = 17, len(views)
    rays = np.zeros((J, V, L, 6), dtype=np.float32)
    confs = np.zeros((J, V, L, 1), dtype=np.float32)
    for li, fid in enumerate(win):
        for vi, c in enumerate(views):
            e = frames[fid][c]
            cam = dict(e['camera'])
            cam = {**cam, 'T': np.asarray(cam['T'], dtype=np.float64).reshape(3) / 100.0}  # cm→m
            j2d = np.asarray(e['joints_2d'], dtype=np.float32)        # HRNet 2D
            dirs, inters = build_rays_one_view(j2d, cam)
            rays[:, vi, li, 0:3] = dirs
            rays[:, vi, li, 3:6] = inters
            confs[:, vi, li, 0] = np.asarray(e['joints_2d_conf']).reshape(-1)
    dt = (np.arange(L, dtype=np.float32) - t_target) / fps
    delta_ts = np.broadcast_to(dt[None, :], (V, L)).copy()
    gt = np.asarray(frames[win[t_target]][views[0]]['joints_3d'], dtype=np.float32) / 100.0  # cm→m
    return (torch.from_numpy(rays), torch.from_numpy(confs),
            torch.from_numpy(delta_ts), torch.from_numpy(gt))


def mpjpe_mm(pred, gt, idx=None, pelvis_anchor=True):
    """pred/gt: (J,3) 米。返回 mm。pelvis(mid-hip COCO 11/12)锚定=KP* Rel口径。"""
    p, g = pred.clone(), gt.clone()
    if pelvis_anchor:
        pel_p = (p[11] + p[12]) / 2
        pel_g = (g[11] + g[12]) / 2
        p = p - pel_p; g = g - pel_g
    e = (p - g).norm(dim=-1)  # (J,)
    if idx is not None:
        e = e[idx]
    return e.mean().item() * 1000.0


@torch.no_grad()
def evaluate(model, frames, device, L=5, fps=30.0):
    windows = build_windows(frames, L=L)
    print(f"[eval] {len(windows)} 个连续 L={L} 窗口")
    res = {'B_all': [], 'C_all': [], 'B_kp': [], 'C_kp': [], 'A_all': [], 'A_kp': []}
    for win in windows:
        rays, confs, dts, gt = window_to_tensors(frames, win, L=L, fps=fps)
        rays = rays.unsqueeze(0).to(device); confs = confs.unsqueeze(0).to(device)
        dts = dts.unsqueeze(0).to(device); gt = gt.to(device)
        pred_C = model(rays, confs, dts, gate_override=None)[0]   # (J,3)
        pred_B = model(rays, confs, dts, gate_override=0)[0]
        res['C_all'].append(mpjpe_mm(pred_C, gt))
        res['B_all'].append(mpjpe_mm(pred_B, gt))
        res['C_kp'].append(mpjpe_mm(pred_C, gt, idx=KP_STAR_COCO))
        res['B_kp'].append(mpjpe_mm(pred_B, gt, idx=KP_STAR_COCO))
    m = {k: float(np.mean(v)) for k, v in res.items() if v}
    print("\n=== CMU 时序评估 (V=%d, L=%d, %dfps) ===" % (len(CMU_VIEWS), L, int(fps)))
    print(f"  B (gate=0, baseline中心帧):  全17={m['B_all']:.1f}mm  KP*={m['B_kp']:.1f}mm")
    print(f"  C (完整 ST-VFT):             全17={m['C_all']:.1f}mm  KP*={m['C_kp']:.1f}mm")
    print(f"  时序净增益 (B-C):            全17={m['B_all']-m['C_all']:+.1f}mm  KP*={m['B_kp']-m['C_kp']:+.1f}mm")
    print(f"  gate 学到值: {model.gate.item():.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='ST-VFT 训练权重 (含gate/tft)')
    ap.add_argument('--rumpl-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--rumpl-ckpt', required=True, help='baseline conf 权重 (STVFTPretrained backbone)')
    ap.add_argument('--cmu-pkl', required=True, help='连续帧 CMU eval pkl')
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = STVFTPretrained(args.rumpl_cfg, args.rumpl_ckpt, freeze_backbone=True).to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    state = sd['model'] if 'model' in sd else sd
    # 只加载 TFT/gate/dt_encoder (backbone 已从 rumpl_ckpt 加载)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] ST-VFT权重: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    frames = load_continuous_frames(args.cmu_pkl)
    print(f"[data] {len(frames)} 帧, 视角齐查中...")
    evaluate(model, frames, device, L=args.L, fps=args.fps)


if __name__ == "__main__":
    main()
