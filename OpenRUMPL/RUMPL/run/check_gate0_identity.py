"""路二: gate=0 数值恒等检验。

证明边界(只证这一条): gate_override=0 时, STVFTPretrained 的前向计算
是否在数值上等价于 baseline MultiView_RUMPL.forward。
  - 一致(diff~浮点尾数) → gate=0 旁路正确, baseline 40.40 可继承给对照 B。
  - 不一致(diff 大)     → forward 复刻有 bug(很可能是 sine 编码/通道顺序), 这是真问题。

不证明: 时序有用 / ST-VFT 在真实连续输入上 work / 训练 loss 随 gate 上升的原因。

方法: 同一份输入张量喂两边, 取中心帧对齐, 报告逐元素 max/mean abs diff。
backbone 真 baseline forward 吃 (B,J,V,7)=[dir3,inter3,conf1];
STVFT 吃 rays(B,J,V,L,6)+confs(B,J,V,L,1), 内部取 t_target=L//2 中心帧。
构造时令 STVFT 中心帧 == backbone 的那份输入。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

import numpy as np
import torch

from models.stvft.stvft_pretrained import STVFTPretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rumpl-cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--rumpl-ckpt', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--V', type=int, default=2, help='视角数 (baseline 40.40 是 V=2)')
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model = STVFTPretrained(args.rumpl_cfg, args.rumpl_ckpt, freeze_backbone=True).to(device)
    model.eval()
    bb = model.backbone
    print(f"[cfg] backbone.random_num_views={getattr(bb,'random_num_views',None)} "
          f"apply_view_fusion={getattr(bb,'apply_view_fusion',None)} "
          f"concat_confidence={getattr(bb,'concat_confidence',None)} "
          f"apply_sine_on_points={getattr(bb,'apply_sine_encoding_on_points',None)} "
          f"concat_dir_inter_first={getattr(bb,'concat_direction_and_intersection_first',None)} "
          f"D={model.D} gate={model.gate.item()}")

    B, J, V, L = args.B, 17, args.V, args.L
    t_target = L // 2

    # 随机输入(恒等性是 forward 的性质, 与具体输入无关; eval 模式无 dropout / 无随机子采样)
    rays = torch.randn(B, J, V, L, 6, device=device)
    confs = torch.rand(B, J, V, L, 1, device=device)
    dts = torch.randn(B, V, L, device=device)

    with torch.no_grad():
        # ② STVFT, gate=0 旁路
        out_stvft = model(rays, confs, dts, gate_override=0)         # (B,J,3)

        # ① 真 baseline forward: 喂 STVFT 的中心帧那份输入 (B,J,V,7)=[dir,inter,conf]
        center_rays = rays[:, :, :, t_target, :]                    # (B,J,V,6)
        center_conf = confs[:, :, :, t_target, :]                   # (B,J,V,1)
        x_center = torch.cat([center_rays, center_conf], dim=-1)    # (B,J,V,7)
        out_base = bb(x_center, is_training=False)                  # (B,J,3)

    diff = (out_stvft - out_base).abs()
    print(f"\n[shapes] stvft={tuple(out_stvft.shape)} base={tuple(out_base.shape)}")
    print(f"[magnitude] out_base abs mean={out_base.abs().mean().item():.6f} "
          f"max={out_base.abs().max().item():.6f}")
    print(f"[IDENTITY] max abs diff = {diff.max().item():.3e}")
    print(f"[IDENTITY] mean abs diff = {diff.mean().item():.3e}")
    rel = diff.max().item() / (out_base.abs().max().item() + 1e-12)
    print(f"[IDENTITY] max abs diff / max|out_base| = {rel:.3e}")
    verdict = "PASS (gate=0 旁路 == baseline forward)" if diff.max().item() < 1e-4 \
        else "FAIL (forward 复刻与 baseline 不一致 — 真 bug)"
    print(f"[VERDICT] {verdict}")


if __name__ == '__main__':
    main()
