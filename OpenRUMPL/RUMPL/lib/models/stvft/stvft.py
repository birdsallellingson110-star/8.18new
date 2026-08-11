"""ST-VFT Phase 1 — 顶层模型 (设计文档 v1 §2.1)。

方案B 三阶段串联: RayTokenEmbed -> TFT (时序) -> VFT (视角) -> PFT (跨关节 + 回归头)。

forward 输入 (设计文档 §2.1):
  rays:     (B, J, V, L, 6)   camera_T(3) + ray_direction(3)
  confs:    (B, J, V, L, 1)
  delta_ts: (B, V, L)         秒, 相对 t_target 的时间偏移 (不依赖关节, 广播到 J)
输出:
  pred:     (B, J, 3)         t_target 时刻的绝对3D坐标

mask 接口 (Phase1 传 None, Phase2 稀疏调度用):
  tft_mask: (B, J, V, L)  True=缺失帧
  vft_mask: (B, J, V)     True=缺失视角
"""
import torch
import torch.nn as nn

from .ray_embed import RayTokenEmbed
from .tft import TFT
from .vft import VFT
from .pft import PFT


class STVFT(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_joints=17, max_period=2.0,
                 tft_layers=4, vft_layers=6, pft_layers=6):
        super().__init__()
        self.token_embed = RayTokenEmbed(max_period=max_period)
        self.tft = TFT(d_model=d_model, n_heads=n_heads, n_layers=tft_layers)
        self.vft = VFT(d_model=d_model, n_heads=n_heads, n_layers=vft_layers)
        self.pft = PFT(num_joints=num_joints, d_model=d_model,
                       n_heads=n_heads, n_layers=pft_layers)

    def forward(self, rays, confs, delta_ts, tft_mask=None, vft_mask=None):
        B, J, V, L, _ = rays.shape
        # delta_ts (B,V,L) 不依赖关节 -> 广播到 (B,J,V,L)
        dt = delta_ts.unsqueeze(1).expand(B, J, V, L)
        tokens = self.token_embed(rays, confs, dt)        # (B,J,V,L,D)
        view_tokens = self.tft(tokens, key_padding_mask=tft_mask)   # (B,J,V,D)
        joint_tokens = self.vft(view_tokens, key_padding_mask=vft_mask)  # (B,J,D)
        pred = self.pft(joint_tokens)                     # (B,J,3)
        return pred


def test_stvft():
    model = STVFT()
    B, J, V, L = 2, 17, 5, 5
    rays = torch.randn(B, J, V, L, 6)
    confs = torch.randn(B, J, V, L, 1)
    delta_ts = torch.randn(B, V, L)
    pred = model(rays, confs, delta_ts)
    assert pred.shape == (B, J, 3), pred.shape
    # 全零输入也不应报错
    pred0 = model(torch.zeros(B, J, V, L, 6), torch.zeros(B, J, V, L, 1), torch.zeros(B, V, L))
    assert pred0.shape == (B, J, 3), pred0.shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"STVFT PASS  ({n_params/1e6:.2f}M params)")


if __name__ == "__main__":
    test_stvft()
