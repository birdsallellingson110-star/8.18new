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
    """贡献1: 质量感知时序。质量分=MLP([conf, 几何一致性])调制时序attention(哪帧可信就attend哪帧)。
    conf 检测 mode1(低conf坏帧), 几何残差检测 mode2(conf正常但几何错)。zero-init → 初始==baseline。"""
    def __init__(self, d=768, n_heads=8, n_layers=2, max_period=2.0):
        super().__init__()
        self.dt_encoder = DeltaTEncoder(d_model=d, max_period=max_period)
        self.blocks = nn.ModuleList([TemporalBlock(d, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)                   # zero-init → 初始 delta=0
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        # 质量打分头: [conf, geom_rel] → 每帧质量bias(加进时序attention, 高质量→被attend多)
        self.quality_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.quality_mlp[-1].weight); nn.init.zeros_(self.quality_mlp[-1].bias)  # 初始bias=0=均匀

    def forward(self, vft, dt, conf_jl, geom_jl, t_target):
        # vft:(B,L,J,D) dt:(B,L) conf_jl/geom_jl:(B,J,L) [每帧每关节, 已对视角聚合]
        B, L, J, D = vft.shape
        x = vft.permute(0, 2, 1, 3).reshape(B * J, L, D)  # (B*J, L, D)
        dt_emb = self.dt_encoder(dt)                      # (B, L, D)
        x = x + dt_emb[:, None].expand(B, J, L, D).reshape(B * J, L, D)
        conf_k = conf_jl.reshape(B * J, L, 1).clamp(0, 1)
        geom_k = geom_jl.reshape(B * J, L, 1)
        geom_k = geom_k / (geom_k.mean(1, keepdim=True) + 1e-6)   # 窗口内归一化(>1=该帧几何更不一致)
        bias = self.quality_mlp(torch.cat([conf_k, geom_k], -1)).squeeze(-1)  # (B*J,L) 质量bias, init=0
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm(x)
        delta = self.out_proj(x[:, t_target]).reshape(B, J, D)  # 中心帧 → zero-init投影
        return vft[:, t_target] + delta                   # 残差: 初始 delta=0 → ==baseline中心


def ray_pairwise_residual(rays):
    """每视角的"几何不一致度": 该视角射线与其他视角射线的平均最近距离。
    rays (...,V,6) = [direction(3), intersection=射线上一点(3)] → (...,V)。
    坏视角(2D偏了)的射线和别人对不上 → 残差大。检测 mode2(conf正常但几何坏)。"""
    d = rays[..., 0:3]; p = rays[..., 3:6]                # 方向, 射线上一点
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
    V = rays.shape[-2]
    di = d.unsqueeze(-2).expand(*d.shape[:-2], V, V, 3)   # [...,i,j]=d_i
    dj = d.unsqueeze(-3).expand(*d.shape[:-2], V, V, 3)   # [...,i,j]=d_j
    pi = p.unsqueeze(-2).expand(*p.shape[:-2], V, V, 3)
    pj = p.unsqueeze(-3).expand(*p.shape[:-2], V, V, 3)
    cross = torch.cross(di, dj, dim=-1)                  # (...,V,V,3)
    num = ((pj - pi) * cross).sum(-1).abs()              # (...,V,V) 两线最近距离分子
    dist = num / (cross.norm(dim=-1) + 1e-8)
    eye = torch.eye(V, device=rays.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, 0.0)                    # 排除自己
    return dist.sum(-1) / max(V - 1, 1)                  # (...,V) 对其他视角平均


def point_ray_residual(rays, points3d):
    """每视角射线到当前3D预测点的距离。用于 DeProPose 2-pass, V=2 时也能区分视角。
    rays (B,J,V,6), points3d (B,J,3) → (B,J,V)。"""
    d = rays[..., 0:3]
    p = rays[..., 3:6]
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
    diff = points3d[:, :, None, :] - p
    return torch.cross(diff, d, dim=-1).norm(dim=-1)


class ConfToWeight(nn.Module):
    """学习版 conf→per-view 权重(A: V=2 自适应)。每视角 5 维输入:
    [conf_v, rel_conf, V/5, conf_min_other(有没有救兵), geom_rel(几何不一致度比值, 检测mode2)] → w∈(0,1] → 归一化(mean=1)。
    末层 zero-init → 初始均匀 → ==baseline。conf 维=MTF跨视角conf 轻量版; geom 维=MTF跨视角几何 轻量版。"""
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(5, hidden); self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)   # 初始均匀 → ==baseline

    def forward(self, conf_jv, geom_jv):  # (B',J,V) → (B',V) per-view 权重
        cv = conf_jv.mean(1).clamp(0, 1)                  # (B',V) 视角平均conf
        gv = geom_jv.mean(1)                              # (B',V) 视角平均几何残差
        B, V = cv.shape
        if V < 2:
            return torch.ones_like(cv)
        rel = cv - cv.mean(-1, keepdim=True)
        M = cv.unsqueeze(1).expand(B, V, V).clone()
        di = torch.arange(V, device=cv.device)
        M[:, di, di] = 2.0
        cmo = M.min(-1).values                            # 其他视角最低conf
        Vn = torch.full_like(cv, V / 5.0)
        grel = gv / (gv.mean(-1, keepdim=True) + 1e-6)    # 几何残差比(>1=该视角更不一致)
        feat = torch.stack([cv, rel, Vn, cmo, grel], dim=-1)   # (B',V,5)
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(feat)))).squeeze(-1)
        return w / w.mean(-1, keepdim=True).clamp_min(1e-6)


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
        self.caa = False                      # 固定标量 λ 路径(sweep/回退用)
        self.caa_scale = nn.Parameter(torch.zeros(1))
        self.caa_learned = False              # 学习版路径(A)
        self.conf2w = ConfToWeight()
        # average-residual 时序: final_3d = 朴素平均(逐帧3D) + zero-init残差。init==朴素平均(保证≥平均, 稳定)。
        self.avg_residual = False
        self.res_head = nn.Linear(self.D, 3)
        nn.init.zeros_(self.res_head.weight); nn.init.zeros_(self.res_head.bias)

    # ---- 编码: 同 STVFTPretrained, 通道 [dir,inter,conf]=768 ----
    def encode(self, rays, confs):
        bb = self.backbone
        dir_emb = bb.encoding_to_embedding(rays[..., 0:3])
        inter_emb = bb.encoding_to_embedding(rays[..., 3:6])
        conf_emb = bb.confidence_to_embedding(confs)
        return torch.cat([dir_emb, inter_emb, conf_emb], dim=-1)   # (...,768)

    # ---- VFT: 复刻 baseline (单帧 (B',J,V,D)→(B',J,D)); 可选 CAA/DeProPose 视角加权 ----
    def vft_forward(self, x, conf=None, geom=None):
        bb = self.backbone
        B, J, V, D = x.shape
        x = x.reshape(B * J, V, D)
        fusion = bb.fusion_token.expand(B * J, -1, -1)
        x = torch.cat([fusion, x], dim=1)                         # (B*J, V+1, D)
        if bb.add_view_enc:
            x = x + bb.View_enc_learned[:, :V + 1]
        x = bb.pos_drop(x)
        cw = None
        if conf is not None and (self.caa or self.caa_learned):
            N = V + 1
            if self.caa_learned:                                 # 学习版: per-view w 广播到关节
                w_view = self.conf2w(conf, geom)[:, None, :].expand(B, J, V).reshape(B * J, V)  # (B*J,V)
            else:                                                # 固定标量 λ
                w_view = 1.0 - self.caa_scale * (1.0 - conf.reshape(B * J, V).clamp(0, 1))
            w = torch.cat([torch.ones(B * J, 1, device=x.device, dtype=x.dtype), w_view], dim=1)  # (B*J,N)
            cw = w[:, None, :].expand(B * J, N, N)                # (B*J,N,N) 乘进attention(列=被attend token)
        # DeProPose 评估端: 与 multiview_rumpl.forward 的训练端同语义。
        # DEPRO_LAMBDA=0 时保持 cw=None, 精确保留 baseline/STVFT 原路径。
        depro = float(os.environ.get('DEPRO_LAMBDA', 0.0))
        if depro > 0 and geom is not None and V >= 2:
            N = V + 1
            consist = geom.reshape(B * J, V)
            relia = torch.exp(-consist / (consist.mean(dim=1, keepdim=True) + 1e-6))
            relia = relia / (relia.mean(dim=1, keepdim=True) + 1e-8)
            w_view = 1.0 - depro * (1.0 - relia)
            w = torch.cat([torch.ones(B * J, 1, device=x.device, dtype=x.dtype), w_view], dim=1)
            cw = w[:, None, :].expand(B * J, N, N)
        for blk in bb.blocks_view_fusion:
            x = blk(x, cw)
        x = bb.View_norm(x)
        x = x[:, 0, :]
        return x.reshape(B, J, D)

    # ---- 步2: VFT 逐帧(batch化) (B,J,V,L,D)→(B,L,J,D); conf:(B,J,V,L) ----
    def vft_all_frames(self, token, conf=None, geom=None):
        B, J, V, L, D = token.shape
        x = token.permute(0, 3, 1, 2, 4).reshape(B * L, J, V, D)   # (B*L,J,V,D)
        cf = conf.permute(0, 3, 1, 2).reshape(B * L, J, V) if conf is not None else None
        gm = geom.permute(0, 3, 1, 2).reshape(B * L, J, V) if geom is not None else None
        x = self.vft_forward(x, cf, gm)                            # (B*L,J,D)
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
        depro = float(os.environ.get('DEPRO_LAMBDA', 0.0))
        depro_2pass = float(os.environ.get('DEPRO_2PASS', 0.0))
        if no_temporal and depro > 0 and depro_2pass > 0:
            # DeProPose 2-pass: 初次预测3D → 点到各视角射线残差 → 可靠度加权重跑VFT。
            # 这是 V=2 可用的非对称信号; pairwise ray距离在V=2天然对称。
            center_token = token[:, :, :, t_target]
            first = self.pft_forward(self.vft_forward(center_token))
            center_rays = rays[:, :, :, t_target]
            geom = point_ray_residual(center_rays, first)
            second = self.vft_forward(center_token, geom=geom)
            return self.pft_forward(second)
        conf_vft = confs.squeeze(-1) if (self.caa or self.caa_learned) else None   # (B,J,V,L) CAA 用
        geom_vft = None
        if self.caa_learned or depro > 0:            # 几何残差(B,J,V,L): rays(B,J,V,L,6)→对V算
            geom_vft = ray_pairwise_residual(rays.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2)
        vft = self.vft_all_frames(token, conf_vft, geom_vft)  # (B,L,J,768)
        if no_temporal:
            return self.pft_forward(vft[:, t_target])   # baseline 路径
        conf_jl = confs.mean(dim=2).squeeze(-1)      # (B,J,L) 对视角聚合的每帧每关节conf
        geom_full = ray_pairwise_residual(rays.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2)  # (B,J,V,L)
        geom_jl = geom_full.mean(dim=2)              # (B,J,L) 每帧几何不一致度(视角间), 检测mode2
        dt = delta_ts[:, 0, :]                       # (B,L) 时间(各视角相同)
        center = self.temporal(vft, dt, conf_jl, geom_jl, t_target)  # (B,J,768)
        if self.avg_residual:
            # final = 朴素平均(逐帧3D) + zero-init残差。init==朴素平均(稳定, 保证≥平均)
            avg3d = torch.stack([self.pft_forward(vft[:, l]) for l in range(L)], 1).mean(1)  # (B,J,3)
            return avg3d + self.res_head(center)     # res_head zero-init → init=avg3d
        return self.pft_forward(center)              # (B,J,3) 普通残差路径
