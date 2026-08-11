"""ST-VFT Phase 1 — Stage 3: Pose Fusion Transformer (跨关节融合 + 回归头)。

对接决策 (用户 2026-06-16 拍板):
  (a) 方案A: 独立自包含实现, 复制 RUMPL 的 Block/Attention/Mlp 类 (非 import),
      便于 Phase2 论文独立呈现与未来独立演化。
  (b) PFT = 6 层 (设计文档 §1.2), 不沿用 RUMPL 的 12 层 (控制总深度/训练效率,
      深度兜底留给 root-relative warp)。
  (c) 修正 RUMPL 最后一层 block 被调用两次的可疑写法 -> 标准单 pass 循环。

PFT 内部细节完全按 RUMPL clip_full 实际行为复制:
  - Spatial_pos_embed = zeros(1, J, D); pos_drop = Dropout(0); Spatial_norm = LayerNorm
  - head = Sequential(LayerNorm(D), Linear(D, 3)) -> (B, J, 3) 绝对3D坐标 (无 root-relative)
  - mlp_ratio = 2.0: RUMPL clip_full 写了 TRANSFORMER_MLP_RATIO=3 但 get_multiview_rumpl_net
    未把它传入构造函数, 故实际生效的是 __init__ 默认 2.0 (RUMPL 的死配置, 复制其实际行为)
  - qkv_bias=True, drop_path_rate=0.1, attn_drop=0

Block/Attention/Mlp adapted from RUMPL MultiView_RUMPL (arXiv 2512.15488),
lib/models/multiview_rumpl.py.
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath


# ---- Adapted from RUMPL MultiView_RUMPL (arXiv 2512.15488) ----
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., learned_query=False, num_tokens=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.learned_query = learned_query
        if self.learned_query:
            self.Q_learned = torch.nn.Parameter(
                torch.randn(self.num_heads, num_tokens, dim // self.num_heads),
                requires_grad=learned_query)

    def forward(self, x, conf_weights=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.learned_query:
            q = self.Q_learned.expand(B, -1, -1, -1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if conf_weights is not None:
            attn = attn * conf_weights.unsqueeze(1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, learned_query=False, num_tokens=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, learned_query=learned_query, num_tokens=num_tokens)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, conf_weights=None):
        if conf_weights is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), conf_weights))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
# ---- end adapted ----


class PFT(nn.Module):
    """Pose Fusion Transformer: 跨 J 关节融合 + 回归头。

    输入: (B, J, D)  来自 VFT 输出
    输出: (B, J, 3)  绝对3D坐标
    """

    def __init__(self, num_joints=17, d_model=256, n_heads=8, n_layers=6,
                 mlp_ratio=2.0, qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, d_model))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(dim=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(n_layers)
        ])
        self.Spatial_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(self, x):
        """x: (B, J, D) -> (B, J, 3)"""
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        # 决策(c): RUMPL 原版把最后一层 block 调用两次(疑似 bug, paper §3 未述);
        # 此处用标准单 pass 循环, 与 paper 方法描述一致。
        for blk in self.blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = self.head(x)  # (B, J, 3)
        return x


def test_pft():
    pft = PFT(num_joints=17, d_model=256, n_heads=8, n_layers=6)
    x = torch.randn(2, 17, 256)  # (B, J, D)
    out = pft(x)
    assert out.shape == (2, 17, 3), out.shape
    print("PFT PASS")


if __name__ == "__main__":
    test_pft()
