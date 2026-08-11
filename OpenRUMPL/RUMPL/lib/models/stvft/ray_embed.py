"""ST-VFT Phase 1 — 射线 token 编码 (设计文档 v1 §2.3 / §1.4)。

每条 token = concat[ Embed_ray(192), Embed_conf(32), Embed_Δt(32) ] = 256。
concat 而非 sum —— 保留各分信息独立性。
ray = camera_T(3) + ray_direction(3) = 6 维 (文档明确不是7维)。
"""
import torch
import torch.nn as nn

from .delta_t_encoder import DeltaTEncoder


class RayTokenEmbed(nn.Module):
    """将 (ray, conf, delta_t) 编码为 D=256 token。

    输入:
      ray:     (..., 6)   camera_T + ray_direction
      conf:    (..., 1)
      delta_t: (...)  或 (..., 1)  秒
    输出: (..., 256)
    """

    def __init__(self, d_ray=192, d_conf=32, d_dt=32, max_period=2.0):
        super().__init__()
        self.d_total = d_ray + d_conf + d_dt
        assert self.d_total == 256, f"Total dim should be 256, got {self.d_total}"
        self.ray_proj = nn.Sequential(
            nn.Linear(6, d_ray), nn.GELU(), nn.Linear(d_ray, d_ray)
        )
        self.conf_proj = nn.Sequential(nn.Linear(1, d_conf), nn.GELU())
        self.dt_encoder = DeltaTEncoder(d_model=d_dt, max_period=max_period)

    def forward(self, ray, conf, delta_t):
        ray_emb = self.ray_proj(ray)        # (..., 192)
        conf_emb = self.conf_proj(conf)     # (..., 32)
        dt_emb = self.dt_encoder(delta_t)   # (..., 32)
        return torch.cat([ray_emb, conf_emb, dt_emb], dim=-1)  # (..., 256)


def test_ray_token_embed():
    emb = RayTokenEmbed()
    B, J, V, L = 2, 17, 5, 5
    ray = torch.randn(B, J, V, L, 6)
    conf = torch.randn(B, J, V, L, 1)
    dt = torch.randn(B, J, V, L)
    out = emb(ray, conf, dt)
    assert out.shape == (B, J, V, L, 256), out.shape
    print("RayTokenEmbed PASS")


if __name__ == "__main__":
    test_ray_token_embed()
