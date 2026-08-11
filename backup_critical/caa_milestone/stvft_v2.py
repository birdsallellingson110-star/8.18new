"""ST-VFT v2 (路B): 时序融合放在 VFT 之后(多视角去噪后的特征空间), 学自 PoseFormer。
数据流: rays→encode→VFT(逐帧跨V, (B,J,L,768))→Temporal(per-joint跨L, 残差+zero-init, 取中心帧)→PFT→head。
无 gate(用 zero-init 残差从 baseline 起步)。encode/VFT/PFT 复刻并加载 baseline 权重(同 STVFTPretrained)。

分步实现: 本版(步2)先到 encode→VFT逐帧→取中心帧→PFT(无时序), 验 VFT batch 化 + 无时序时==baseline。
"""
import os
import sys
import torch
import torch.nn as nn

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from models.stvft.delta_t_encoder import DeltaTEncoder


class _Mlp(nn.Module):
    def __init__(self, d, ratio=3):
        super().__init__()
        self.fc1 = nn.Linear(d, d * ratio); self.act = nn.GELU(); self.fc2 = nn.Linear(d * ratio, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TemporalBlock(nn.Module):
    """手写注意力(避开SDPA nan-grad bug) + conf偏置(b1, 加在softmax前)。残差。"""
    def __init__(self, d=768, n_heads=8):
        super().__init__()
        self.n_heads = n_heads; self.scale = (d // n_heads) ** -0.5
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d * 3); self.proj = nn.Linear(d, d)
        self.mlp = _Mlp(d)
        self.capture = False; self.attn_cache = None       # 可视化用: 开后存 attn

    def forward(self, x, conf_bias):
        # x: (N, L, d); conf_bias: (N, L) 对每个 Key 帧的偏置(已含 -λ(1-conf))
        N, L, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # (N, nh, L, dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (N, nh, Lq, Lk)
        attn = attn + conf_bias[:, None, None, :]         # b1: 加在 K 维, softmax 前, broadcast nh/Lq
        attn = attn.softmax(dim=-1)
        if self.capture:
            self.attn_cache = attn.detach()               # (N, nh, L, L)
        out = (attn @ v).transpose(1, 2).reshape(N, L, d)
        x = x + self.proj(out)                            # 残差
        x = x + self.mlp(self.norm2(x))                   # 残差
        return x


class TemporalV2(nn.Module):
    """在 VFT 去噪后的 (B,L,J,768) 上 per-joint 跨 L 时序; 取中心帧; zero-init 残差从 baseline 起步。"""
    def __init__(self, d=768, n_heads=8, n_layers=2, max_period=2.0):
        super().__init__()
        self.dt_encoder = DeltaTEncoder(d_model=d, max_period=max_period)
        self.blocks = nn.ModuleList([TemporalBlock(d, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)                   # zero-init → 初始 delta=0
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        self.conf_bias_scale = nn.Parameter(torch.ones(1))  # λ 可学习, 初始 1

    def forward(self, vft, dt, conf_jl, t_target):
        # vft:(B,L,J,D) dt:(B,L) conf_jl:(B,J,L) [每帧每关节conf, 已对视角聚合]
        B, L, J, D = vft.shape
        x = vft.permute(0, 2, 1, 3).reshape(B * J, L, D)  # (B*J, L, D)
        dt_emb = self.dt_encoder(dt)                      # (B, L, D)
        x = x + dt_emb[:, None].expand(B, J, L, D).reshape(B * J, L, D)
        conf_kv = conf_jl.reshape(B * J, L).clamp(0, 1)   # (B*J, L)
        bias = -self.conf_bias_scale * (1.0 - conf_kv)    # 低conf→负偏置→降权
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm(x)
        delta = self.out_proj(x[:, t_target]).reshape(B, J, D)  # 中心帧 → zero-init投影
        return vft[:, t_target] + delta                   # 残差: 初始 delta=0 → ==baseline中心


class STVFTv2(nn.Module):
    def __init__(self, rumpl_cfg_path, rumpl_ckpt_path, freeze_backbone=True,
                 temporal_layers=2, n_heads=8, max_period=2.0):
        super().__init__()
        from core.config import config as rumpl_config, update_config
        update_config(rumpl_cfg_path)
        from models.multiview_rumpl import get_multiview_rumpl_net
        rumpl = get_multiview_rumpl_net(rumpl_config, is_train=True)
        sd = torch.load(rumpl_ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        missing, unexpected = rumpl.load_state_dict(sd, strict=False)
        assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"
        self.backbone = rumpl.features
        self.D = self.backbone.Spatial_pos_embed.shape[-1]
        assert self.D == 768, f"expected 768, got {self.D}"
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.temporal = TemporalV2(d=self.D, n_heads=n_heads, n_layers=temporal_layers, max_period=max_period)
        # CAA: 跨视角 conf-aware 融合。conf_weights=1-λ(1-conf) 乘进 VFT attention(用backbone预留接口)。
        # λ init 0 → conf_weights=1 → 精确 ==baseline; λ 学着 downweight 低conf视角。
        self.caa = False
        self.caa_scale = nn.Parameter(torch.zeros(1))

    # ---- 编码: 同 STVFTPretrained, 通道 [dir,inter,conf]=768 ----
    def encode(self, rays, confs):
        bb = self.backbone
        dir_emb = bb.encoding_to_embedding(rays[..., 0:3])
        inter_emb = bb.encoding_to_embedding(rays[..., 3:6])
        conf_emb = bb.confidence_to_embedding(confs)
        return torch.cat([dir_emb, inter_emb, conf_emb], dim=-1)   # (...,768)

    # ---- VFT: 复刻 baseline (单帧 (B',J,V,D)→(B',J,D)); conf!=None且self.caa→CAA conf加权 ----
    def vft_forward(self, x, conf=None):
        bb = self.backbone
        B, J, V, D = x.shape
        x = x.reshape(B * J, V, D)
        fusion = bb.fusion_token.expand(B * J, -1, -1)
        x = torch.cat([fusion, x], dim=1)                         # (B*J, V+1, D)
        if bb.add_view_enc:
            x = x + bb.View_enc_learned[:, :V + 1]
        x = bb.pos_drop(x)
        cw = None
        if self.caa and conf is not None:
            N = V + 1
            w_view = 1.0 - self.caa_scale * (1.0 - conf.reshape(B * J, V).clamp(0, 1))  # (B*J,V)
            w = torch.cat([torch.ones(B * J, 1, device=x.device, dtype=x.dtype), w_view], dim=1)  # (B*J,N)
            cw = w[:, None, :].expand(B * J, N, N)                # (B*J,N,N) 乘进attention(列=被attend的token)
        for blk in bb.blocks_view_fusion:
            x = blk(x, cw)
        x = bb.View_norm(x)
        x = x[:, 0, :]
        return x.reshape(B, J, D)

    # ---- 步2: VFT 逐帧(batch化) (B,J,V,L,D)→(B,L,J,D); conf:(B,J,V,L) ----
    def vft_all_frames(self, token, conf=None):
        B, J, V, L, D = token.shape
        x = token.permute(0, 3, 1, 2, 4).reshape(B * L, J, V, D)   # (B*L,J,V,D)
        cf = None
        if conf is not None:
            cf = conf.permute(0, 3, 1, 2).reshape(B * L, J, V)     # (B*L,J,V)
        x = self.vft_forward(x, cf)                                # (B*L,J,D)
        return x.reshape(B, L, J, D)                               # (B,L,J,D)

    # ---- PFT+head: 原样复刻 baseline (含末层双调用) ----
    def pft_forward(self, x):
        bb = self.backbone
        b, J, D = x.shape
        x = x + bb.Spatial_pos_embed
        x = bb.pos_drop(x)
        for ix, blk in enumerate(bb.blocks):
            if ix == len(bb.blocks) - 1:
                x = blk(x)
            x = blk(x)
        x = bb.Spatial_norm(x)
        x = x.reshape(b, J, -1)
        x = bb.head(x)
        return x.reshape(b, -1, 3)

    def forward(self, rays, confs, delta_ts, t_target=None, no_temporal=False):
        """路B: encode→VFT逐帧→Temporal(取中心帧,zero-init残差)→PFT。
        no_temporal=True: 跳过时序直接取中心帧 (=baseline, 调试用)。"""
        B, J, V, L, _ = rays.shape
        if t_target is None:
            t_target = L // 2
        token = self.encode(rays, confs)            # (B,J,V,L,768)
        conf_vft = confs.squeeze(-1) if self.caa else None   # (B,J,V,L) CAA 用的逐视角逐关节conf
        vft = self.vft_all_frames(token, conf_vft)  # (B,L,J,768)
        if no_temporal:
            center = vft[:, t_target]               # baseline 路径
        else:
            conf_jl = confs.mean(dim=2).squeeze(-1)  # (B,J,L) 对视角聚合的每帧每关节conf
            dt = delta_ts[:, 0, :]                   # (B,L) 时间(各视角相同)
            center = self.temporal(vft, dt, conf_jl, t_target)  # (B,J,768) = vft[:,ct]+delta
        return self.pft_forward(center)             # (B,J,3)
