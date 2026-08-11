"""ST-VFT Phase 1 — Stage 2: View Fusion Transformer (设计文档 v1 §2.5)。

沿用 RUMPL VFT 结构, 仅小改输入接口。对每个关节 j 跨 V 视角融合。
接在 TFT 之后, 处理"已时序去噪的视角 token"。
"""
import torch
import torch.nn as nn


class VFT(nn.Module):
    """View Fusion Transformer.

    输入: (B, J, V, D)  来自 TFT 输出
    输出: (B, J, D)
    """

    def __init__(self, d_model=256, n_heads=8, n_layers=6, ff_ratio=4, dropout=0.1):
        super().__init__()
        self.fusion_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, view_tokens, key_padding_mask=None):
        """
        view_tokens: (B, J, V, D)
        key_padding_mask: (B, J, V) — 视角缺失时用; Phase1 由 random-views collate 保证 V 一致, 传 None
        return: (B, J, D)
        """
        B, J, V, D = view_tokens.shape
        x = view_tokens.reshape(B * J, V, D)
        fusion = self.fusion_token.expand(B * J, 1, D)
        x = torch.cat([fusion, x], dim=1)

        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(B * J, V)
            fusion_mask = torch.zeros(B * J, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([fusion_mask, mask], dim=1)
        else:
            mask = None

        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        out = x[:, 0, :].reshape(B, J, D)
        return out


def test_vft():
    vft = VFT(d_model=256, n_heads=8, n_layers=6)
    x = torch.randn(2, 17, 5, 256)  # (B,J,V,D)
    out = vft(x)
    assert out.shape == (2, 17, 256), out.shape
    print("VFT PASS")


if __name__ == "__main__":
    test_vft()
