"""ST-VFT Phase 1 — Stage 1: Temporal Fusion Transformer (设计文档 v1 §2.4)。

对每个 (视角 i, 关节 j) 独立, 跨 L_window 帧时序融合射线 token。
DSTA 证据: 时序融合是 80% 贡献来源, 必须放第一阶段。
Pre-LN, 可学习 fusion token (绑定 t_target)。key_padding_mask 接口 Phase1 传 None, Phase2 用。
"""
import torch
import torch.nn as nn


class TFT(nn.Module):
    """Temporal Fusion Transformer.

    输入 tokens: (B, J, V, L, D)
    输出:        (B, J, V, D)  在 t_target 时刻的每视角融合特征
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=4, ff_ratio=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.fusion_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN 更稳
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, key_padding_mask=None):
        """
        tokens: (B, J, V, L, D)
        key_padding_mask: (B, J, V, L) — True=mask掉(缺失帧); Phase1 传 None
        return: (B, J, V, D)
        """
        B, J, V, L, D = tokens.shape
        x = tokens.reshape(B * J * V, L, D)
        fusion = self.fusion_token.expand(B * J * V, 1, D)
        x = torch.cat([fusion, x], dim=1)  # (B*J*V, 1+L, D)

        if key_padding_mask is not None:
            mask_flat = key_padding_mask.reshape(B * J * V, L)
            fusion_mask = torch.zeros(B * J * V, 1, dtype=torch.bool, device=x.device)
            mask_flat = torch.cat([fusion_mask, mask_flat], dim=1)  # fusion 永不 mask
        else:
            mask_flat = None

        # 强制 SDPA 用 math 后端: efficient/flash 后端 backward 有 nan-grad bug
        # (前向正常但反向产 nan, 训练中 q/k/v 变大时触发)。序列长仅 1+L, math 开销可忽略。
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = self.encoder(x, src_key_padding_mask=mask_flat)
        x = self.norm(x)
        out = x[:, 0, :].reshape(B, J, V, D)  # 取 fusion token 输出
        return out


def test_tft():
    tft = TFT(d_model=256, n_heads=8, n_layers=4)
    x = torch.randn(2, 17, 5, 5, 256)  # (B,J,V,L,D)
    out = tft(x)
    assert out.shape == (2, 17, 5, 256), out.shape
    # mask 接口也应可用
    mask = torch.zeros(2, 17, 5, 5, dtype=torch.bool)
    mask[:, :, :, -1] = True  # mask 最后一帧
    out2 = tft(x, key_padding_mask=mask)
    assert out2.shape == (2, 17, 5, 256), out2.shape
    print("TFT PASS")


if __name__ == "__main__":
    test_tft()
