"""ST-VFT Phase 1 — Δt 连续正弦编码 (设计文档 v1 §2.2)。

单位用秒, max_period=2.0s, 连续正弦 + 可学习线性投影。
正弦编码天然连续可外推到训练未见过的 Δt 值 (Phase 2 稀疏交错用)。
"""
import math
import torch
import torch.nn as nn


class DeltaTEncoder(nn.Module):
    """连续正弦编码 + 可学习线性投影。

    输入: delta_t (B, ...) 秒  (或 (B, ..., 1))
    输出: (B, ..., d_model)
    """

    def __init__(self, d_model=32, max_period=2.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, d_model // 2, dtype=torch.float32)
            / (d_model // 2)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, delta_t):
        # 末维若不是 1 则补一个, 用于和 freqs 广播
        if delta_t.shape[-1] != 1:
            delta_t = delta_t.unsqueeze(-1)  # (..., 1)
        args = delta_t * self.freqs  # (..., d_model//2)
        embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., d_model)
        return self.proj(embed)


def test_delta_t_encoder():
    enc = DeltaTEncoder(d_model=32, max_period=2.0)
    dt = torch.tensor([[0.0, 0.02, -0.04, 0.1]])  # (1, 4)
    out = enc(dt)
    assert out.shape == (1, 4, 32), out.shape
    # 不同 Δt 之间应有区分
    assert (out[0, 0] - out[0, 1]).abs().sum() > 1e-3
    # (B, V, L) 形式也应工作 -> (B, V, L, 32)
    dt2 = torch.randn(2, 5, 5)
    assert enc(dt2).shape == (2, 5, 5, 32), enc(dt2).shape
    print("DeltaTEncoder PASS")


if __name__ == "__main__":
    test_delta_t_encoder()
