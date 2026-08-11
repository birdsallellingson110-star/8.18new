# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Modified by Yanjie Li (leeyegy@gmail.com)
# TokenPose + Sparse for 2D single person PE
# Multi-view
# cross-view Fusion
# ------------------------------------------------------------------------------
## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Written by Ce Zheng (cezheng@knights.ucf.edu)
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)
# ------------------------------------------------------------------------------

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import os
from torchinfo import summary

logger = logging.getLogger(__name__)


def pairwise_ray_distance(direction, point, eps=1e-7):
    """Compute the Plucker distance between every pair of 3D rays."""
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    direction_i = direction[:, :, None, :]
    direction_j = direction[:, None, :, :]
    point_diff = point[:, None, :, :] - point[:, :, None, :]
    cross = torch.cross(direction_i, direction_j, dim=-1)
    cross_norm = cross.norm(dim=-1)
    skew_distance = (point_diff * cross).sum(-1).abs() / cross_norm.clamp_min(eps)
    parallel_distance = torch.cross(
        point_diff, direction_i.expand_as(point_diff), dim=-1
    ).norm(dim=-1)
    distance = torch.where(cross_norm > eps, skew_distance, parallel_distance)
    eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)
    return distance.masked_fill(eye[None], 0.0)


def normalize_pairwise_distance(distance, eps=1e-7):
    """Make metric ray distances dimensionless with a per-sample robust scale."""
    flat = distance.flatten(1)
    positive = flat > eps
    masked = flat.masked_fill(~positive, float("nan"))
    scale = torch.nanmedian(masked, dim=1).values
    fallback = flat.sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    scale = torch.where(torch.isfinite(scale), scale, fallback)
    return distance / scale.clamp_min(eps).view(-1, 1, 1)


def harmonic_plucker_encoding(direction, point, num_frequencies=15):
    """Encode Plücker rays [d, o×d] with NeRF-style harmonic features."""
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    moment = torch.cross(point, direction, dim=-1)
    ray = torch.cat([direction, moment], dim=-1)
    frequencies = (
        2.0 ** torch.arange(num_frequencies, device=ray.device, dtype=ray.dtype)
    ) * math.pi
    angles = ray.unsqueeze(-1) * frequencies
    harmonic = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return torch.cat([ray, harmonic], dim=-1)


def geometry_distance_with_fusion_token(
    direction, point, direct_fusion=False, num_fusion=1
):
    """Add fusion-token row(s)/column(s) to the view-ray distance matrix."""
    view_distance = pairwise_ray_distance(direction, point)
    batch_size, num_views, _ = view_distance.shape
    nf = max(int(num_fusion), 1)
    distance = view_distance.new_zeros(batch_size, num_views + nf, num_views + nf)
    distance[:, nf:, nf:] = view_distance
    if direct_fusion and num_views > 1:
        # broadcast same view-mean distance to each fusion query
        view_mean = view_distance.sum(dim=-1) / (num_views - 1)
        distance[:, :nf, nf:] = view_mean.unsqueeze(1).expand(-1, nf, -1)
    return distance


def point_to_ray_distance(direction, point, target, eps=1e-7):
    """Distance from each target joint to each corresponding observation ray."""
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_offset = target.unsqueeze(2) - point
    return torch.cross(target_offset, direction, dim=-1).norm(dim=-1)


def oracle_reliability_with_fusion_token(
    direction, point, target, temperature, num_fusion=1
):
    """Build a centered per-view oracle penalty for fusion-token queries."""
    distance = point_to_ray_distance(direction, point, target)
    centered_distance = distance - distance.mean(dim=-1, keepdim=True)
    batch_size, num_joints, num_views = centered_distance.shape
    penalty = centered_distance.reshape(batch_size * num_joints, num_views)
    penalty = (penalty / max(temperature, 1e-7)).clamp(-20.0, 20.0)
    nf = max(int(num_fusion), 1)
    matrix = penalty.new_zeros(batch_size * num_joints, num_views + nf, num_views + nf)
    matrix[:, :nf, nf:] = penalty.unsqueeze(1).expand(-1, nf, -1)
    return matrix


def reliability_with_fusion_token(reliability, temperature, gate=1.0, num_fusion=1):
    """Convert predicted ray errors into a centered fusion-query penalty."""
    centered = reliability - reliability.mean(dim=-1, keepdim=True)
    if torch.is_tensor(gate) and gate.dim() == 2:
        gate = gate.unsqueeze(-1)
    centered = gate * centered
    batch_size, num_joints, num_views = centered.shape
    penalty = centered.reshape(batch_size * num_joints, num_views)
    penalty = (penalty / max(temperature, 1e-7)).clamp(-20.0, 20.0)
    nf = max(int(num_fusion), 1)
    matrix = penalty.new_zeros(batch_size * num_joints, num_views + nf, num_views + nf)
    matrix[:, :nf, nf:] = penalty.unsqueeze(1).expand(-1, nf, -1)
    return matrix


class AdaFuseViewWeightNet(nn.Module):
    """AdaFuse ViewWeightNet (IJCV'21), adapted to RUMPL ray tokens.

    Original AdaFuse: heatmap appearance + Sampson distance + conf → view weight.
    RUMPL lifting has no heatmaps, so appearance is a confidence embedding;
    geometry uses pairwise Plucker ray distance (calibrated-ray analog of Sampson).

    Returns per-(joint, view) sigmoid weights; occluded views (conf≈0) → 0.
    """

    def __init__(self, nchan_dist=128, nchan_conf=64):
        super().__init__()
        self.dist_feature_net = nn.Sequential(
            nn.Conv1d(2, nchan_dist, 1),
            nn.BatchNorm1d(nchan_dist),
            nn.ReLU(inplace=True),
            nn.Conv1d(nchan_dist, nchan_dist, 1),
            nn.BatchNorm1d(nchan_dist),
            nn.ReLU(inplace=True),
        )
        self.conf_embed = nn.Sequential(
            nn.Linear(1, nchan_conf),
            nn.ReLU(inplace=True),
        )
        self.conf_out = nn.Sequential(
            nn.Linear(nchan_dist + nchan_conf, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        # Near-identity at init (sigmoid(4)≈0.982) so early training is stable.
        nn.init.zeros_(self.conf_out[-1].weight)
        nn.init.constant_(self.conf_out[-1].bias, 4.0)

    def forward(self, direction, point, conf):
        """
        direction/point: (B, J, V, 3); conf: (B, J, V, 1) or (B, J, V)
        returns weights (B, J, V) in [0, 1]
        """
        if conf.dim() == 4:
            conf = conf.squeeze(-1)
        b, jn, vn, _ = direction.shape
        bj = b * jn
        dist = pairwise_ray_distance(
            direction.reshape(bj, vn, 3),
            point.reshape(bj, vn, 3),
        )
        dist = torch.exp(-dist)
        conf_flat = conf.reshape(bj, vn).clamp(0, 1)

        pair_feats = []
        for i in range(vn):
            others = [k for k in range(vn) if k != i]
            if not others:
                feat = torch.stack(
                    [conf_flat.new_zeros(bj, 1), conf_flat[:, i : i + 1]],
                    dim=1,
                )
            else:
                feat = torch.stack(
                    [dist[:, i, others], conf_flat[:, others]],
                    dim=1,
                )
            pair_feats.append(feat)
        pair_feats = torch.stack(pair_feats, dim=1).reshape(bj * vn, 2, -1)
        dist_feat = self.dist_feature_net(pair_feats).mean(dim=2)
        conf_feat = self.conf_embed(conf_flat.reshape(bj * vn, 1))
        weights = torch.sigmoid(
            self.conf_out(torch.cat([dist_feat, conf_feat], dim=-1))
        ).view(b, jn, vn)
        # AdaFuse: drop weight when detection score ≈ 0
        weights = torch.where(conf > 0.01, weights, torch.zeros_like(weights))
        return weights


class PoseCodebookDCSA(nn.Module):
    """PCT-faithful discrete pose prior (UniCodebook proxy).

    Codebook is a buffer (no grad), EMA-updated from hard VQ on GT encodings,
    optionally frozen after N steps. Continuous joint tokens: soft DCSA inject
    + CE to GT hard indices (discrete pressure).
    """

    def __init__(
        self,
        dim,
        num_codes=512,
        num_heads=4,
        ema_decay=0.99,
        freeze_after_steps=8000,
        commit_scale=1.0,
        ce_scale=0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.ema_decay = ema_decay
        self.freeze_after_steps = int(freeze_after_steps)
        self.commit_scale = float(commit_scale)
        self.ce_scale = float(ce_scale)

        # PCT: codebook is buffer, not Parameter
        self.register_buffer("codebook", torch.randn(num_codes, dim) * 0.02)
        self.register_buffer("ema_cluster_size", torch.ones(num_codes))
        self.register_buffer("ema_w", torch.zeros(num_codes, dim))
        self.ema_w.data.copy_(self.codebook.data)
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

        self.pose_enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # CE head: continuous tokens → code logits
        self.ce_head = nn.Linear(dim, num_codes)
        # sigmoid(-2)≈0.12 — soft prior at start
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    @torch.no_grad()
    def _ema_update(self, encodings, indices):
        ones = torch.ones_like(indices, dtype=self.ema_cluster_size.dtype)
        cluster_size = torch.zeros_like(self.ema_cluster_size).scatter_add_(
            0, indices, ones
        )
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1.0 - self.ema_decay
        )
        dw = torch.zeros_like(self.ema_w).index_add_(0, indices, encodings)
        self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.num_codes * 1e-5)
            * n
        )
        self.codebook.copy_(self.ema_w / cluster_size.unsqueeze(1))

    def soft_inject(self, x):
        """x: (B, J, D) → residual-injected features."""
        b, jn, d = x.shape
        codes = self.codebook  # (N, D), buffer
        q = self.q_proj(x).view(b * jn, self.num_heads, self.head_dim)
        k = self.k_proj(codes).view(self.num_codes, self.num_heads, self.head_dim)
        v = self.v_proj(codes).view(self.num_codes, self.num_heads, self.head_dim)
        attn = torch.einsum("bhd,nhd->bhn", q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhn,nhd->bhd", attn, v).reshape(b, jn, d)
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * self.out_proj(out)

    def commitment_loss(self, target_3d, joint_feats=None):
        """Hard VQ on GT + optional CE from continuous joint tokens."""
        root = target_3d[:, :1, :]
        rel = target_3d - root
        z = self.pose_enc(rel)  # (B, J, D)
        flat = z.reshape(-1, self.dim)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=1)
        quantized = self.codebook[indices]
        commit = F.mse_loss(flat, quantized.detach())
        total = self.commit_scale * commit

        if joint_feats is not None:
            logits = self.ce_head(joint_feats.reshape(-1, self.dim))
            ce = F.cross_entropy(logits, indices.detach())
            total = total + self.ce_scale * ce

        if self.training and (not bool(self.frozen.item())):
            self._ema_update(flat.detach(), indices)
            self.n_updates += 1
            if (
                self.freeze_after_steps > 0
                and int(self.n_updates.item()) >= self.freeze_after_steps
            ):
                self.frozen.fill_(True)
        return total


def build_h36m17_adjacency(num_joints=17):
    """KTPFormer-faithful adj: undirected edges → row-normalize → force diag=1.

    Parents (VideoPose3D/H36M-17):
    [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]
    """
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    adj = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for i, p in enumerate(parents[:num_joints]):
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0
    # row-normalize (no self yet)
    row = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    adj = adj / row
    # force diagonal = 1 (official graph_utils.py)
    eye = torch.eye(num_joints, dtype=torch.float32)
    adj = adj * (1.0 - eye) + eye
    return adj


class LearnableGraphConv(nn.Module):
    """KTPFormer SemGCN layer (verbatim semantics; no softmax on adj)."""

    def __init__(self, in_features, out_features, adj, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(2, in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.M = nn.Parameter(torch.ones(adj.size(0), out_features))
        self.register_buffer("adj", adj.clone())
        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            stdv = 1.0 / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B, J, Cin)
        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])
        adj = self.adj + self.adj2
        adj = 0.5 * (adj.transpose(0, 1) + adj)
        eye = torch.eye(adj.size(0), device=x.device, dtype=x.dtype)
        out = torch.matmul(adj * eye, self.M * h0) + torch.matmul(
            adj * (1.0 - eye), self.M * h1
        )
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out


class FaithfulKPA(nn.Module):
    """KTPFormer KPA: gconv → BN → ReLU, once before PFT."""

    def __init__(self, dim, num_joints=17):
        super().__init__()
        adj = build_h36m17_adjacency(num_joints)
        self.gconv = LearnableGraphConv(dim, dim, adj)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, J, D)
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        return self.relu(x)


class ConfFiLM(nn.Module):
    """Per-view confidence FiLM on view tokens (audit M3)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, conf):
        # x: (B, J, V, D); conf: (B, J, V, 1) or (B, J, V)
        if conf.dim() == 3:
            conf = conf.unsqueeze(-1)
        gb = self.net(conf.clamp(0, 1))
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., learned_query=False, num_tokens=2,
                 learnable_conf_bias=False, learnable_geom_bias=False, conf_bias_init=0.1, geom_bias_init=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.learned_query = learned_query
        self.learnable_conf_bias = learnable_conf_bias
        self.learnable_geom_bias = learnable_geom_bias
        if learnable_conf_bias:
            self.gbt_conf_scale = nn.Parameter(torch.tensor(math.sqrt(max(conf_bias_init, 1e-8))))
        if learnable_geom_bias:
            self.gbt_geom_scale = nn.Parameter(torch.tensor(math.sqrt(max(geom_bias_init, 1e-8))))
        if self.learned_query:
            self.Q_learned = torch.nn.Parameter(torch.randn(self.num_heads, num_tokens, dim // self.num_heads), requires_grad=learned_query)

    def forward(self, x, conf_weights=None, conf_bias=None, geom_distance=None, reliability_penalty=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.learned_query:
            q = self.Q_learned.expand(B, -1, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if conf_bias is not None and self.learnable_conf_bias:
            if conf_bias.dim() == 3:
                conf_bias = conf_bias.unsqueeze(1)
            attn = attn + self.gbt_conf_scale.square() * conf_bias.to(dtype=attn.dtype)
        if geom_distance is not None and self.learnable_geom_bias:
            if geom_distance.dim() == 3:
                geom_distance = geom_distance.unsqueeze(1)
            attn = attn - self.gbt_geom_scale.square() * geom_distance.to(dtype=attn.dtype)
        if reliability_penalty is not None:
            if reliability_penalty.dim() == 3:
                reliability_penalty = reliability_penalty.unsqueeze(1)
            attn = attn - reliability_penalty.to(dtype=attn.dtype)
        attn = attn.softmax(dim=-1)
        if os.environ.get("GBT_SAVE_ATTN", "0") == "1":
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        if conf_weights is not None:
            attn = attn * conf_weights.unsqueeze(1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, learned_query=False, num_tokens=2,
                 learnable_conf_bias=False, learnable_geom_bias=False, conf_bias_init=0.1, geom_bias_init=0.1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            learned_query=learned_query, num_tokens=num_tokens, learnable_conf_bias=learnable_conf_bias,
            learnable_geom_bias=learnable_geom_bias, conf_bias_init=conf_bias_init, geom_bias_init=geom_bias_init)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, conf_weights=None, conf_bias=None, geom_distance=None, reliability_penalty=None):
        
        if conf_weights is not None or conf_bias is not None or geom_distance is not None or reliability_penalty is not None:
            attn_output = self.attn(
                self.norm1(x), conf_weights, conf_bias, geom_distance, reliability_penalty
            )
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QueryDecoderBlock(nn.Module):
    """Pre-norm learned-query cross-attention followed by a Transformer MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.memory_norm = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.mlp_norm = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, query, memory, key_padding_mask=None):
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attn(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.mlp(self.mlp_norm(query))


class MultiView_RUMPL(nn.Module):
    def __init__(self, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_views=5,
                 linear_weighted_mean=False,
                 hidden_dim=1024,
                 cfg=None,):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 3D joints have 3 channels: (x,y,z)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        
            
        self.num_joints = num_joints
        self.num_views = num_views
        self.embed_dim_ratio = embed_dim_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # embed_dim = embed_dim_ratio * 2 * num_joints   # because we add ray tokens
        embed_dim_multiplier = 1
        out_dim = num_joints * 3     #### output dimension is num_joints * 3
        
        self.print_macs_summary = False
        
        self.hidden_dim = hidden_dim
        self.fix_pft_last_block = os.environ.get("RUMPL_FIX_PFT_LAST_BLOCK", "0") == "1"
        self.gbt_learnable_bias = os.environ.get("GBT_LEARNABLE_BIAS", "0") == "1"
        self.gbt_learnable_conf = self.gbt_learnable_bias and os.environ.get("GBT_USE_CONF_BIAS", "1") == "1"
        self.gbt_learnable_geom = self.gbt_learnable_bias and os.environ.get("GBT_USE_GEOM_BIAS", "1") == "1"
        self.gbt_conf_init = float(os.environ.get("GBT_CONF_INIT", "0.1"))
        self.gbt_geom_init = float(os.environ.get("GBT_GEOM_INIT", "0.1"))
        self.gbt_fusion_geom = os.environ.get("GBT_FUSION_GEOM", "0") == "1"
        self.gbt_oracle_reliability = os.environ.get("GBT_ORACLE_RELIABILITY", "0") == "1"
        self.gbt_oracle_temperature = float(os.environ.get("GBT_ORACLE_TEMPERATURE", "0.02"))
        self.gbt_learned_reliability = os.environ.get("GBT_LEARNED_RELIABILITY", "0") == "1"
        self.gbt_reliability_temperature = float(
            os.environ.get("GBT_RELIABILITY_TEMPERATURE", "0.2")
        )
        self.gbt_reliability_gate_init = float(
            os.environ.get("GBT_RELIABILITY_GATE_INIT", "0.05")
        )
        self.gbt_reliability_pair_gate = (
            os.environ.get("GBT_RELIABILITY_PAIR_GATE", "0") == "1"
        )
        self.gbt_reliability_rank_weight = float(
            os.environ.get("GBT_RELIABILITY_RANK_WEIGHT", "0.0")
        )
        # Triangulation-anchored residual decoding (cf. Learnable Triangulation, ICCV'19):
        # confidence-weighted least-squares intersection of rays as anchor, head predicts residual.
        self.tri_anchor = os.environ.get("RUMPL_TRI_ANCHOR", "0") == "1"
        self.tri_anchor_reg = float(os.environ.get("RUMPL_TRI_ANCHOR_REG", "1e-4"))
        self.tri_anchor_conf_eps = float(
            os.environ.get("RUMPL_TRI_ANCHOR_CONF_EPS", "0.05")
        )
        # Per-view ray-depth auxiliary supervision (cf. PlaneSweepPose, CVPR'21):
        # each view token predicts the joint depth along its own ray.
        self.ray_depth_aux = os.environ.get("RUMPL_RAY_DEPTH_AUX", "0") == "1"
        self.global_joint_view_fusion = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_FUSION", "0") == "1"
        )
        self.global_joint_view_depth = int(
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_DEPTH", "2")
        )
        self.global_joint_view_conf_bias = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS", "1") == "1"
        )
        self.global_joint_view_geom_bias = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS", "1") == "1"
        )
        self.global_joint_view_geom_norm = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM", "0") == "1"
        )
        self.global_joint_view_gate_init = float(
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT", "0.1")
        )
        self.global_joint_view_count_gate = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE", "0") == "1"
        )
        self.global_joint_view_gate_max_init = float(
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT", "0.12")
        )
        # Exact-identity residual variants make it possible to initialize the
        # global branch from a trained RUMPL model and measure only the added
        # module.  The optional Plucker path follows the GBT paper's ray-token
        # encoder while still leaving the RUMPL backbone untouched at init.
        self.global_joint_view_residual = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_RESIDUAL", "0") == "1"
        )
        self.global_joint_view_plucker = (
            os.environ.get("RUMPL_GLOBAL_JOINT_VIEW_PLUCKER", "0") == "1"
        )
        # AdaFuse-style learned view reliability (IJCV'21), ray-space adaptation.
        self.adafuse_view_weight = os.environ.get("RUMPL_ADAFUSE_VW", "0") == "1"
        # PCT/UniCodebook discrete pose prior + soft DCSA inject.
        self.pose_codebook = os.environ.get("RUMPL_POSE_CODEBOOK", "0") == "1"
        self.pose_codebook_n = int(os.environ.get("RUMPL_POSE_CODEBOOK_N", "512"))
        self.pose_codebook_heads = int(os.environ.get("RUMPL_POSE_CODEBOOK_HEADS", "4"))
        # KTPFormer-faithful SemGCN KPA (once before PFT).
        self.use_kpa = os.environ.get("RUMPL_KPA", "0") == "1"
        # MHFormer-style multi fusion tokens in VFT (collapse → 1 before PFT).
        self.multi_hyp = max(int(os.environ.get("RUMPL_MULTI_HYP", "1")), 1)
        # Per-view confidence FiLM on view tokens (cheap control).
        self.use_conf_film = os.environ.get("RUMPL_CONF_FILM", "0") == "1"
        # VGGT-inspired alternating attention over the two axes of ray tokens:
        # view attention is applied independently per joint, then joint attention
        # independently per view. Unlike the original VFT->PFT pipeline, view
        # tokens are retained while pose context is injected.
        self.alt_joint_view = os.environ.get("RUMPL_ALT_JOINT_VIEW", "0") == "1"
        self.gated_joint_adapter = (
            os.environ.get("RUMPL_GATED_JOINT_ADAPTER", "0") == "1"
        )
        self.singleframe_gbt = os.environ.get("RUMPL_SINGLEFRAME_GBT", "0") == "1"
        self.sf_gbt_encoder_depth = int(
            os.environ.get("RUMPL_SF_GBT_ENCODER_DEPTH", "3")
        )
        self.sf_gbt_decoder_depth = int(
            os.environ.get("RUMPL_SF_GBT_DECODER_DEPTH", "2")
        )
        self.sf_gbt_pft_depth = int(
            os.environ.get("RUMPL_SF_GBT_PFT_DEPTH", "4")
        )
        self.sf_gbt_conf_bias = (
            os.environ.get("RUMPL_SF_GBT_CONF_BIAS", "1") == "1"
        )
        self.sf_gbt_geom_bias = (
            os.environ.get("RUMPL_SF_GBT_GEOM_BIAS", "1") == "1"
        )
        self.sf_gbt_geom_norm = (
            os.environ.get("RUMPL_SF_GBT_GEOM_NORM", "1") == "1"
        )
        self.joint_adapter_indices = tuple(
            int(value)
            for value in os.environ.get(
                "RUMPL_JOINT_ADAPTER_INDICES", "2,5,8"
            ).split(",")
            if value.strip()
        )
        self.joint_adapter_scale_init = float(
            os.environ.get("RUMPL_JOINT_ADAPTER_SCALE_INIT", "1.2")
        )
        self.joint_adapter_view_power = float(
            os.environ.get("RUMPL_JOINT_ADAPTER_VIEW_POWER", "1.0")
        )
        self.joint_adapter_count_lookup = (
            os.environ.get("RUMPL_JOINT_ADAPTER_COUNT_LOOKUP", "0") == "1"
        )
        self.joint_adapter_direct_readout = (
            os.environ.get("RUMPL_JOINT_ADAPTER_DIRECT_READOUT", "0") == "1"
        )
        self.joint_adapter_direct_scale_init = float(
            os.environ.get("RUMPL_JOINT_ADAPTER_DIRECT_SCALE_INIT", "0.1")
        )
        if self.joint_adapter_view_power < 0.0:
            raise ValueError("RUMPL_JOINT_ADAPTER_VIEW_POWER must be non-negative")
        self.alt_joint_view_depth = int(
            os.environ.get("RUMPL_ALT_JOINT_VIEW_DEPTH", "4")
        )
        self.vft_depth = int(os.environ.get("RUMPL_VFT_DEPTH", str(depth)))
        self.pft_depth = int(os.environ.get("RUMPL_PFT_DEPTH", str(depth)))
        if self.vft_depth < 0 or self.pft_depth < 0:
            raise ValueError("RUMPL_VFT_DEPTH and RUMPL_PFT_DEPTH must be non-negative")
        if self.alt_joint_view and self.alt_joint_view_depth < 1:
            raise ValueError("RUMPL_ALT_JOINT_VIEW_DEPTH must be positive")
        if self.alt_joint_view and self.multi_hyp != 1:
            raise ValueError(
                "RUMPL_ALT_JOINT_VIEW currently requires RUMPL_MULTI_HYP=1"
            )
        
        self.apply_view_fusion = cfg.NETWORK.APPLY_VIEW_FUSION
        
        self.add_view_enc = cfg.NETWORK.ADD_VIEW_ENCODING
        
        ## work with random number of views
        self.random_num_views = cfg.DATASET.TRAIN_RANDOM_NUM_VIEWS
        if not self.apply_view_fusion and self.random_num_views:
            raise 'This configuration is not possible!'
        if self.apply_view_fusion and self.random_num_views:
            self.max_num_views = cfg.DATASET.MAX_NUM_VIEWS
            self.min_num_views = cfg.DATASET.MIN_NUM_VIEWS
        
        
        ### spatial patch embedding
        self.point_3d_to_embedding = nn.Linear(in_chans, embed_dim_ratio)  # dummy line for legacy reasons
        
        self.apply_sine_encoding_on_points = cfg.DATASET.APPLY_SINE_ENCODING_ON_RAYS
        self.apply_sine_encoding_on_points_nerf = cfg.DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF
        if self.apply_sine_encoding_on_points and self.apply_sine_encoding_on_points_nerf:
            raise 'This configuration is not possible!'
        self.sine_d_model = cfg.DATASET.SINE_D_MODEL
        self.sine_L_nerf = cfg.DATASET.SINE_L_NERF
        
        
        self.feed_camera_calibration = cfg.NETWORK.FEED_CAMERA_CALIBRATION
        if self.feed_camera_calibration and self.apply_sine_encoding_on_points or self.feed_camera_calibration and self.apply_sine_encoding_on_points_nerf:
            raise 'This configuration is not possible!'
        self.use_only_2D = cfg.NETWORK.FEED_ONLY_2D
        if self.use_only_2D and self.feed_camera_calibration:
            raise 'This configuration is not possible!'
        if self.use_only_2D and self.apply_sine_encoding_on_points or self.use_only_2D and self.apply_sine_encoding_on_points_nerf: 
            raise 'This configuration is not possible!'
        
        #### depth information
        self.concat_depth_as_input = cfg.NETWORK.CONCAT_DEPTH_AS_INPUT
        if self.concat_depth_as_input:
            self.depth_to_embedding = nn.Linear(1, embed_dim_ratio)
            embed_dim_multiplier += 1
            
        #### no intersection features
        self.not_use_intersection_features = cfg.NETWORK.NOT_USE_INTERSECTION_FEATURES
            
        
        self.concat_direction_and_intersection_first = cfg.NETWORK.POSE_3D_FUSER_CONCAT_DIRECTION_INTERSECTION_FIRST
        if self.concat_direction_and_intersection_first and self.not_use_intersection_features:
            raise 'This configuration is not possible!'
        
        if self.apply_sine_encoding_on_points:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(self.sine_d_model * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(self.sine_d_model, embed_dim_ratio)
        elif self.apply_sine_encoding_on_points_nerf:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(self.sine_L_nerf * 2 * 3 * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(self.sine_L_nerf * 2 * 3, embed_dim_ratio)
        else:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(3 * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(3, embed_dim_ratio)
            
        self.concat_confidence = cfg.NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB
        if cfg.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS and self.concat_confidence:
            raise 'This configuration is not possible!'
        
        if self.concat_confidence:
            self.confidence_to_embedding = nn.Linear(1, embed_dim_ratio)    
            embed_dim_multiplier += 1
            
        # if self.feed_camera_calibration:
        #     self.camera_calibration_to_embedding = nn.Linear(17, embed_dim_ratio)
        #     self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
        #     if self.concat_confidence:
        #         embed_dim_ratio *= 3    # because we concatenate camera calibration features and confidence
        #     else:
        #         embed_dim_ratio *= 2    # because we concatenate camera calibration features  
        # elif self.apply_view_fusion:
        #     if self.concat_confidence:
        #         if self.concat_direction_and_intersection_first:
        #             embed_dim_ratio *= 2   # because we concatenate direction and intersection features first
        #         else:
        #             embed_dim_ratio *= 3        # because we concatenate direction and intersection features and confidence
        #     elif self.concat_direction_and_intersection_first:
        #         embed_dim_ratio *= 1        # because we concatenate direction and intersection features first
        #     else:
        #         embed_dim_ratio *= 2        # because we concatenate direction and intersection features
        # else:
        #     if self.concat_confidence:
        #         embed_dim_ratio *= 2        # because we concatenate closest points and confidence
        
        if self.use_only_2D:
            self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
        elif self.feed_camera_calibration:
            self.camera_calibration_to_embedding = nn.Linear(17, embed_dim_ratio)
            self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
            embed_dim_multiplier += 1
        elif self.apply_view_fusion:
            if not self.concat_direction_and_intersection_first and not self.not_use_intersection_features:
                embed_dim_multiplier += 1
                
        embed_dim_ratio *= embed_dim_multiplier

        if self.gbt_learned_reliability:
            rng_state = torch.get_rng_state()
            self.reliability_predictor = nn.Sequential(
                nn.LayerNorm(embed_dim_ratio * 3),
                nn.Linear(embed_dim_ratio * 3, embed_dim_ratio),
                nn.GELU(),
                nn.Linear(embed_dim_ratio, 1),
            )
            nn.init.zeros_(self.reliability_predictor[-1].weight)
            nn.init.constant_(self.reliability_predictor[-1].bias, -3.5)
            gate_init = min(max(self.gbt_reliability_gate_init, 1e-5), 1.0 - 1e-5)
            gate_logit = math.log(gate_init / (1.0 - gate_init))
            if self.gbt_reliability_pair_gate:
                self.reliability_pair_gate = nn.Sequential(
                    nn.LayerNorm(6),
                    nn.Linear(6, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
                nn.init.zeros_(self.reliability_pair_gate[-1].weight)
                nn.init.constant_(self.reliability_pair_gate[-1].bias, gate_logit)
            else:
                self.gbt_reliability_gate_logit = nn.Parameter(torch.tensor(gate_logit))
            torch.set_rng_state(rng_state)
        
        
        if self.apply_view_fusion and self.random_num_views:
            self.fusion_token = torch.nn.Parameter(
                torch.randn(1, self.multi_hyp, embed_dim_ratio), requires_grad=True
            )
            if self.multi_hyp > 1:
                # MHFormer late merge: collapse H hyp tokens → 1 (Conv1d over hyp axis).
                self.hyp_merge = nn.Conv1d(self.multi_hyp, 1, kernel_size=1)
            
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        max_transformer_depth = max(
            depth,
            self.vft_depth,
            self.pft_depth,
            self.alt_joint_view_depth if self.alt_joint_view else 0,
            1,
        )
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, max_transformer_depth)
        ]  # stochastic depth decay rule

        if self.global_joint_view_fusion:
            if self.global_joint_view_depth < 1:
                raise ValueError("RUMPL_GLOBAL_JOINT_VIEW_DEPTH must be positive")
            rng_state = torch.get_rng_state()
            self.global_joint_embedding = nn.Parameter(
                torch.zeros(1, num_joints, 1, embed_dim_ratio)
            )
            trunc_normal_(self.global_joint_embedding, std=.02)
            self.global_joint_view_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[min(i, len(dpr) - 1)],
                    norm_layer=norm_layer,
                    learnable_conf_bias=self.global_joint_view_conf_bias,
                    learnable_geom_bias=self.global_joint_view_geom_bias,
                    conf_bias_init=self.gbt_conf_init,
                    geom_bias_init=self.gbt_geom_init,
                )
                for i in range(self.global_joint_view_depth)
            ])
            self.global_joint_view_norm = norm_layer(embed_dim_ratio)
            if self.global_joint_view_plucker:
                plucker_dim = 6 * (1 + 2 * 15)
                self.global_joint_view_ray_embed = nn.Linear(
                    plucker_dim, embed_dim_ratio
                )
            if self.global_joint_view_residual:
                self.global_joint_view_out_proj = nn.Linear(
                    embed_dim_ratio, embed_dim_ratio
                )
                nn.init.zeros_(self.global_joint_view_out_proj.weight)
                nn.init.zeros_(self.global_joint_view_out_proj.bias)
            gate_init = min(max(self.global_joint_view_gate_init, 1e-5), 1.0 - 1e-5)
            self.global_joint_view_gate_logit = nn.Parameter(
                torch.tensor(math.log(gate_init / (1.0 - gate_init)))
            )
            if self.global_joint_view_count_gate:
                gate_max_init = min(
                    max(self.global_joint_view_gate_max_init, 1e-5), 1.0 - 1e-5
                )
                gate_max_logit = math.log(
                    gate_max_init / (1.0 - gate_max_init)
                )
                self.global_joint_view_gate_slope = nn.Parameter(
                    torch.tensor(
                        gate_max_logit
                        - math.log(gate_init / (1.0 - gate_init))
                    )
                )
            torch.set_rng_state(rng_state)

        if self.singleframe_gbt:
            if self.sf_gbt_encoder_depth < 1 or self.sf_gbt_decoder_depth < 1:
                raise ValueError(
                    "single-frame GBT encoder/decoder depth must be positive"
                )
            plucker_dim = 6 * (1 + 2 * 15)
            self.sf_gbt_ray_embed = nn.Linear(plucker_dim, embed_dim_ratio)
            self.sf_gbt_joint_embedding = nn.Parameter(
                torch.zeros(1, num_joints, 1, embed_dim_ratio)
            )
            self.sf_gbt_queries = nn.Parameter(
                torch.zeros(1, num_joints, embed_dim_ratio)
            )
            trunc_normal_(self.sf_gbt_joint_embedding, std=.02)
            trunc_normal_(self.sf_gbt_queries, std=.02)
            self.sf_gbt_encoder = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[min(i, len(dpr) - 1)],
                    norm_layer=norm_layer,
                    learnable_conf_bias=self.sf_gbt_conf_bias,
                    learnable_geom_bias=self.sf_gbt_geom_bias,
                    conf_bias_init=self.gbt_conf_init,
                    geom_bias_init=self.gbt_geom_init,
                )
                for i in range(self.sf_gbt_encoder_depth)
            ])
            self.sf_gbt_encoder_norm = norm_layer(embed_dim_ratio)
            self.sf_gbt_decoder = nn.ModuleList([
                QueryDecoderBlock(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(self.sf_gbt_decoder_depth)
            ])
            self.sf_gbt_pft = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[min(i, len(dpr) - 1)],
                    norm_layer=norm_layer,
                )
                for i in range(self.sf_gbt_pft_depth)
            ])
            self.sf_gbt_output_norm = norm_layer(embed_dim_ratio)
        
        
        ### view fusion
        if self.apply_view_fusion:
            if not self.random_num_views:
                self.View_enc_learned = nn.Parameter(torch.zeros(1, num_views, embed_dim_ratio))
            else:
                self.View_enc_learned = nn.Parameter(
                    torch.zeros(1, self.max_num_views + self.multi_hyp, embed_dim_ratio)
                )
            if self.alt_joint_view:
                self.blocks_view_fusion = nn.ModuleList()
                self.alt_view_blocks = nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    )
                    for i in range(self.alt_joint_view_depth)
                ])
                self.alt_joint_blocks = nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    )
                    for i in range(self.alt_joint_view_depth)
                ])
                self.alt_joint_embedding = nn.Parameter(
                    torch.zeros(1, num_joints, 1, embed_dim_ratio)
                )
                trunc_normal_(self.alt_joint_embedding, std=.02)
                # A final set-attention block lets the learned fusion query read
                # the context-enriched view tokens without fixing the view count.
                self.alt_view_readout = Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                )
            else:
                self.blocks_view_fusion = nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        learnable_conf_bias=self.gbt_learnable_conf,
                        learnable_geom_bias=self.gbt_learnable_geom,
                        conf_bias_init=self.gbt_conf_init, geom_bias_init=self.gbt_geom_init)
                    for i in range(self.vft_depth)])
                if self.gated_joint_adapter:
                    if any(
                        index < 0 or index >= self.vft_depth
                        for index in self.joint_adapter_indices
                    ):
                        raise ValueError(
                            "RUMPL_JOINT_ADAPTER_INDICES must index VFT blocks"
                        )
                    self.joint_adapter_pos_embed = nn.Parameter(
                        torch.zeros(1, num_joints, embed_dim_ratio)
                    )
                    trunc_normal_(self.joint_adapter_pos_embed, std=.02)
                    self.joint_adapters = nn.ModuleList([
                        Block(
                            dim=embed_dim_ratio, num_heads=num_heads,
                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale, drop=drop_rate,
                            attn_drop=attn_drop_rate, drop_path=0.0,
                            norm_layer=norm_layer,
                        )
                        for _ in self.joint_adapter_indices
                    ])
                    # Each adapter starts as an exact identity residual, so
                    # loading R5 produces bitwise-equivalent predictions.
                    for adapter in self.joint_adapters:
                        nn.init.zeros_(adapter.attn.proj.weight)
                        nn.init.zeros_(adapter.attn.proj.bias)
                        nn.init.zeros_(adapter.mlp.fc2.weight)
                        nn.init.zeros_(adapter.mlp.fc2.bias)
                    scale_init = max(self.joint_adapter_scale_init, 1e-6)
                    if self.joint_adapter_count_lookup:
                        # One independently learned residual strength for every
                        # (view count, insertion depth).  T3--T7 showed that a
                        # single power law cannot serve V2/V3 and V4/V5 at the
                        # same time.  Initial values retain the conservative
                        # power-law prior, but training can depart from it.
                        lookup = torch.empty(
                            self.max_num_views + 1,
                            len(self.joint_adapter_indices),
                        )
                        for view_count in range(self.max_num_views + 1):
                            effective_views = max(
                                view_count, self.min_num_views
                            )
                            initial_gate = (
                                scale_init
                                / float(effective_views)
                                ** self.joint_adapter_view_power
                            )
                            lookup[view_count].fill_(
                                math.log(math.expm1(max(initial_gate, 1e-6)))
                            )
                        self.joint_adapter_view_scale_raw = nn.Parameter(lookup)
                    else:
                        self.joint_adapter_scale_raw = nn.Parameter(
                            torch.tensor(math.log(math.expm1(scale_init)))
                        )
                    if self.joint_adapter_direct_readout:
                        direct_scale_init = max(
                            self.joint_adapter_direct_scale_init, 1e-6
                        )
                        self.joint_adapter_direct_scale_raw = nn.Parameter(
                            torch.full(
                                (len(self.joint_adapter_indices),),
                                math.log(math.expm1(direct_scale_init)),
                            )
                        )
        
        ##### create FPT blocks
        num_tokens = num_joints
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_tokens=num_tokens)
            for i in range(self.pft_depth)])
        
        

        self.Spatial_norm = norm_layer(embed_dim_ratio)

        self.View_norm = norm_layer(embed_dim_ratio)
        
        ####### A easy way to implement weighted mean
        self.linear_weighted_mean = linear_weighted_mean
        if self.linear_weighted_mean:
            self.weighted_mean = nn.Linear(num_views * embed_dim_ratio, embed_dim_ratio)
        else:
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_views, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio , 3),
        )

        if self.tri_anchor or self.ray_depth_aux:
            rng_state = torch.get_rng_state()
            if self.tri_anchor:
                self.tri_anchor_gate = nn.Parameter(torch.tensor(1.0))
            if self.ray_depth_aux:
                self.ray_depth_head = nn.Sequential(
                    norm_layer(embed_dim_ratio),
                    nn.Linear(embed_dim_ratio, 1),
                )
            torch.set_rng_state(rng_state)

        if self.adafuse_view_weight:
            rng_state = torch.get_rng_state()
            self.adafuse_vw_net = AdaFuseViewWeightNet()
            # Soft residual: x <- x * (mix + (1-mix)*w); mix=0 → pure AdaFuse weights.
            mix_init = float(os.environ.get("RUMPL_ADAFUSE_VW_MIX", "0.0"))
            self.adafuse_vw_mix = nn.Parameter(torch.tensor(mix_init))
            torch.set_rng_state(rng_state)

        if self.pose_codebook:
            rng_state = torch.get_rng_state()
            self.pose_codebook_dcsa = PoseCodebookDCSA(
                dim=embed_dim_ratio,
                num_codes=self.pose_codebook_n,
                num_heads=self.pose_codebook_heads,
                ema_decay=float(os.environ.get("RUMPL_POSE_CODEBOOK_EMA", "0.99")),
                freeze_after_steps=int(
                    os.environ.get("RUMPL_POSE_CODEBOOK_FREEZE_STEPS", "8000")
                ),
                commit_scale=float(
                    os.environ.get("RUMPL_POSE_CODEBOOK_COMMIT_SCALE", "1.0")
                ),
                ce_scale=float(os.environ.get("RUMPL_POSE_CODEBOOK_CE_SCALE", "0.5")),
            )
            torch.set_rng_state(rng_state)

        if self.use_kpa:
            rng_state = torch.get_rng_state()
            self.kpa = FaithfulKPA(dim=embed_dim_ratio, num_joints=num_joints)
            torch.set_rng_state(rng_state)

        if self.use_conf_film:
            rng_state = torch.get_rng_state()
            self.conf_film = ConfFiLM(dim=embed_dim_ratio)
            torch.set_rng_state(rng_state)
            
        
    
    def compute_sine_cosine_encoding(self, coords, d_model):
        """
        Compute sine-cosine positional encoding for a batch of coordinates.

        Args:
        coords (torch.Tensor): The coordinates of shape (batch, num_rays, 3).
        d_model (int): The dimension of the model.

        Returns:
        torch.Tensor: A positional encoding tensor of shape (batch, num_rays, d_model).
        """
        assert d_model % 6 == 0, "d_model should be a multiple of 6"

        batch_size, num_rays, _ = coords.shape
        device = coords.device

        div_term = torch.exp(torch.arange(0, d_model // 3, 2, dtype=torch.float, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / (d_model // 3)))

        # Compute positional encodings for each coordinate dimension
        pe_x = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)
        pe_y = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)
        pe_z = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)

        pe_x[:, :, 0::2] = torch.sin(coords[:, :, 0:1] * div_term)
        pe_x[:, :, 1::2] = torch.cos(coords[:, :, 0:1] * div_term)
        pe_y[:, :, 0::2] = torch.sin(coords[:, :, 1:2] * div_term)
        pe_y[:, :, 1::2] = torch.cos(coords[:, :, 1:2] * div_term)
        pe_z[:, :, 0::2] = torch.sin(coords[:, :, 2:3] * div_term)
        pe_z[:, :, 1::2] = torch.cos(coords[:, :, 2:3] * div_term)

        # Concatenate the positional encodings
        positional_encoding = torch.cat((pe_x, pe_y, pe_z), dim=-1)

        return positional_encoding
    
    
    def compute_sine_cosine_encoding_nerf(self, coords, L=4):
        """
        Compute sine-cosine positional encoding for a batch of coordinates based on NeRF.

        Args:
            coords (torch.Tensor): (batch, num_rays, 3)
            L (int, optional): Defaults to 4.
        """
        
        batch_size, num_rays, _ = coords.shape
        device = coords.device
        
        mult_terms = 2.0 ** torch.arange(0, L, device=device) * math.pi
        
        e_x = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        e_y = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        e_z = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        
        e_x[:, :, ::2] = torch.sin(coords[:, :, 0:1] * mult_terms)
        e_x[:, :, 1::2] = torch.cos(coords[:, :, 0:1] * mult_terms)
        e_y[:, :, ::2] = torch.sin(coords[:, :, 1:2] * mult_terms)
        e_y[:, :, 1::2] = torch.cos(coords[:, :, 1:2] * mult_terms)
        e_z[:, :, ::2] = torch.sin(coords[:, :, 2:] * mult_terms)
        e_z[:, :, 1::2] = torch.cos(coords[:, :, 2:] * mult_terms)
        
        encoded_coords = torch.cat((e_x, e_y, e_z), dim=-1)
        
        return encoded_coords
        

    def forward(self, x, is_training=True, **kwargs):
        # x: (b, num_joints, 1 or num_views, 4) or or (b, num_joints, num_views, 7) if apply_view_fusion
        b, num_joints, num_points, d = x.shape
        tri_anchor_point = None
        ray_depth_aux_loss = None
        pose_codebook_aux_loss = None
        # depths = kwargs['depths'] if 'depths' in kwargs else None
        if self.apply_view_fusion:
            # assert d == 6, 'Input shape should be (b, num_joints, num_views, 6) if apply_view_fusion'
            if self.random_num_views and is_training:
                num_points = torch.randint(self.min_num_views, self.max_num_views + 1, (1,)).item()
                x = x[:, :, :num_points, :]
                
            raw_direction = None
            raw_point = None
            if self.use_only_2D:
                joints_2d = x[:, :, :, :2]
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            elif self.feed_camera_calibration:
                joints_2d = x[:, :, :, :2]
                camera_calibration = x[:, :, :, 2:19]
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            else:
                direction_features = x[:, :, :, :3]
                intersection_features = x[:, :, :, 3:6]
                conf = x[:, :, :, 6:7] if self.concat_confidence else None
                needs_raw_rays = (
                    self.gbt_learnable_geom
                    or self.gbt_oracle_reliability
                    or self.gbt_learned_reliability
                    or self.global_joint_view_fusion
                    or self.tri_anchor
                    or self.ray_depth_aux
                    or self.adafuse_view_weight
                    or self.singleframe_gbt
                )
                raw_direction = direction_features if needs_raw_rays else None
                raw_point = intersection_features if needs_raw_rays else None
                if self.concat_depth_as_input:
                    depths = x[:, :, :, 7:8]
            
            if self.feed_camera_calibration or self.use_only_2D:
                x = self.encoding_to_embedding(joints_2d.view(b*num_joints, -1, 2))
                if self.feed_camera_calibration:
                    camera_calibration = self.camera_calibration_to_embedding(camera_calibration.view(b*num_joints, -1, 17))
                    x = torch.cat((x, camera_calibration), dim=-1)
                if self.concat_confidence:
                    conf_emb = self.confidence_to_embedding(conf.view(b*num_joints, -1, 1))
                    x = torch.cat((x, conf_emb), dim=-1)
            else:
                if self.apply_sine_encoding_on_points:
                    direction_features = self.compute_sine_cosine_encoding(direction_features.view(b*num_joints, -1, 3), self.sine_d_model)
                    intersection_features = self.compute_sine_cosine_encoding(intersection_features.view(b*num_joints, -1, 3), self.sine_d_model)
                elif self.apply_sine_encoding_on_points_nerf:
                    direction_features = self.compute_sine_cosine_encoding_nerf(direction_features.view(b*num_joints, -1, 3), self.sine_L_nerf)
                    intersection_features = self.compute_sine_cosine_encoding_nerf(intersection_features.view(b*num_joints, -1, 3), self.sine_L_nerf)
                else:
                    direction_features = direction_features.view(b*num_joints, -1, 3)
                    intersection_features = intersection_features.view(b*num_joints, -1, 3)
                
                # x = x.view(b*num_joints, num_points, -1)
                
                if self.not_use_intersection_features:
                    x = direction_features
                    x = self.encoding_to_embedding(x)
                elif self.concat_direction_and_intersection_first:
                    x = torch.cat((direction_features, intersection_features), dim=-1)
                    x = self.encoding_to_embedding(x)
                else:
                    direction_features = self.encoding_to_embedding(direction_features)
                    intersection_features = self.encoding_to_embedding(intersection_features)
                    x = torch.cat((direction_features, intersection_features), dim=-1)
                if self.concat_confidence:
                    conf_emb = self.confidence_to_embedding(conf.view(b*num_joints, -1, 1))
                    x = torch.cat((x, conf_emb), dim=-1)
                    
            if self.concat_depth_as_input:
                depth_emb = self.depth_to_embedding(depths.view(b*num_joints, -1, 1))
                x = torch.cat((x, depth_emb), dim=-1)

            if self.singleframe_gbt:
                plucker = harmonic_plucker_encoding(
                    raw_direction, raw_point, num_frequencies=15
                )
                memory = self.sf_gbt_ray_embed(plucker)
                memory = (
                    memory + self.sf_gbt_joint_embedding
                ).reshape(b, num_joints * num_points, -1)
                num_global_tokens = num_joints * num_points
                confidence_bias = None
                geometry_distance = None
                if self.sf_gbt_conf_bias and conf is not None:
                    key_confidence = conf.reshape(
                        b, num_global_tokens
                    ).clamp(0, 1)
                    confidence_bias = key_confidence[:, None, :].expand(
                        b, num_global_tokens, num_global_tokens
                    )
                if self.sf_gbt_geom_bias:
                    geometry_distance = pairwise_ray_distance(
                        raw_direction.reshape(b, num_global_tokens, 3),
                        raw_point.reshape(b, num_global_tokens, 3),
                    )
                    if self.sf_gbt_geom_norm:
                        geometry_distance = normalize_pairwise_distance(
                            geometry_distance
                        )
                for block in self.sf_gbt_encoder:
                    memory = block(
                        memory,
                        conf_bias=confidence_bias,
                        geom_distance=geometry_distance,
                    )
                memory = self.sf_gbt_encoder_norm(memory)
                query = self.sf_gbt_queries.expand(b, -1, -1)
                for block in self.sf_gbt_decoder:
                    query = block(query, memory)
                for block in self.sf_gbt_pft:
                    query = block(query)
                query = self.sf_gbt_output_norm(query)
                return self.head(query).view(b, num_joints, 3)

            if self.tri_anchor and raw_direction is not None and num_points >= 2:
                # Confidence-weighted least-squares intersection of the view rays:
                # argmin_p sum_i w_i ||(I - d_i d_i^T)(p - o_i)||^2, solved as a 3x3 system.
                unit_dir = raw_direction / raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
                if conf is not None:
                    anchor_weight = conf.clamp(0, 1) + self.tri_anchor_conf_eps
                else:
                    anchor_weight = torch.ones_like(unit_dir[..., :1])
                eye3 = torch.eye(3, device=x.device, dtype=unit_dir.dtype)
                ray_proj = eye3 - unit_dir.unsqueeze(-1) * unit_dir.unsqueeze(-2)
                weighted_proj = anchor_weight.unsqueeze(-1) * ray_proj
                anchor_lhs = weighted_proj.sum(dim=2) + self.tri_anchor_reg * eye3
                anchor_rhs = (weighted_proj @ raw_point.unsqueeze(-1)).sum(dim=2)
                tri_anchor_point = torch.linalg.solve(anchor_lhs, anchor_rhs).squeeze(-1)

            if self.global_joint_view_fusion:
                view_tokens = x.view(b, num_joints, num_points, -1)
                if self.global_joint_view_plucker:
                    global_tokens = self.global_joint_view_ray_embed(
                        harmonic_plucker_encoding(
                            raw_direction, raw_point, num_frequencies=15
                        )
                    )
                else:
                    global_tokens = view_tokens
                global_tokens = (
                    global_tokens + self.global_joint_embedding
                ).reshape(b, num_joints * num_points, -1)
                confidence = conf.reshape(b, num_joints * num_points).clamp(0, 1)
                confidence_bias = confidence[:, None, :].expand(
                    b, num_joints * num_points, num_joints * num_points
                )
                geometry_distance = pairwise_ray_distance(
                    raw_direction.reshape(b, num_joints * num_points, 3),
                    raw_point.reshape(b, num_joints * num_points, 3),
                )
                if self.global_joint_view_geom_norm:
                    geometry_distance = normalize_pairwise_distance(
                        geometry_distance
                    )
                for block in self.global_joint_view_blocks:
                    global_tokens = block(
                        global_tokens,
                        conf_bias=confidence_bias,
                        geom_distance=geometry_distance,
                    )
                global_tokens = self.global_joint_view_norm(global_tokens).view(
                    b, num_joints, num_points, -1
                )
                gate_logit = self.global_joint_view_gate_logit
                if self.global_joint_view_count_gate:
                    view_range = max(self.max_num_views - self.min_num_views, 1)
                    relative_view_count = (
                        float(num_points - self.min_num_views) / view_range
                    )
                    gate_logit = (
                        gate_logit
                        + self.global_joint_view_gate_slope * relative_view_count
                    )
                gate = torch.sigmoid(gate_logit)
                if self.global_joint_view_residual:
                    global_delta = self.global_joint_view_out_proj(global_tokens)
                    view_tokens = view_tokens + gate * global_delta
                else:
                    view_tokens = view_tokens + gate * (
                        global_tokens - view_tokens
                    )
                x = view_tokens.reshape(b * num_joints, num_points, -1)

            reliability_prediction = None
            reliability_aux_loss = None
            if self.gbt_learned_reliability:
                view_tokens = x.view(b, num_joints, num_points, -1)
                view_context = view_tokens.mean(dim=1, keepdim=True).expand_as(view_tokens)
                joint_context = view_tokens.mean(dim=2, keepdim=True).expand_as(view_tokens)
                reliability_features = torch.cat(
                    [view_tokens, view_context, joint_context], dim=-1
                )
                reliability_prediction = F.softplus(
                    self.reliability_predictor(reliability_features).squeeze(-1)
                )
                reliability_target = kwargs.get("reliability_target")
                if reliability_target is not None:
                    true_distance = point_to_ray_distance(
                        raw_direction,
                        raw_point,
                        reliability_target.to(device=x.device, dtype=raw_direction.dtype),
                    )
                    reliability_aux_loss = F.smooth_l1_loss(
                        reliability_prediction,
                        true_distance,
                        beta=0.02,
                    )
                    if self.gbt_reliability_rank_weight > 0 and num_points > 1:
                        pred_diff = reliability_prediction.unsqueeze(-1) - reliability_prediction.unsqueeze(-2)
                        true_diff = true_distance.unsqueeze(-1) - true_distance.unsqueeze(-2)
                        upper = torch.triu(
                            torch.ones(
                                num_points,
                                num_points,
                                device=x.device,
                                dtype=torch.bool,
                            ),
                            diagonal=1,
                        )
                        valid = upper.view(1, 1, num_points, num_points).expand_as(true_diff)
                        valid = valid & (true_diff.abs() > 0.005)
                        if valid.any():
                            rank_loss = F.binary_cross_entropy_with_logits(
                                pred_diff[valid] / 0.02,
                                (true_diff[valid] > 0).to(dtype=pred_diff.dtype),
                            )
                            reliability_aux_loss = (
                                reliability_aux_loss
                                + self.gbt_reliability_rank_weight * rank_loss
                            )
            
            if not self.random_num_views:
                if self.add_view_enc:
                    x += self.View_enc_learned
                
                x = x.view(b*num_joints, num_points, -1)

            # AdaFuse view weights: scale view tokens before fusion (occluded → ~0).
            if (
                self.adafuse_view_weight
                and raw_direction is not None
                and conf is not None
                and num_points >= 2
            ):
                vw = self.adafuse_vw_net(
                    raw_direction,
                    raw_point,
                    conf,
                )  # (B, J, V)
                mix = torch.sigmoid(self.adafuse_vw_mix)
                scale = mix + (1.0 - mix) * vw
                x = x.view(b, num_joints, num_points, -1) * scale.unsqueeze(-1)
                x = x.view(b * num_joints, num_points, -1)

            # Conf-FiLM: modulate view tokens by detection confidence.
            if self.use_conf_film and conf is not None:
                x = x.view(b, num_joints, num_points, -1)
                x = self.conf_film(x, conf)
                x = x.view(b * num_joints, num_points, -1)
            
            if self.random_num_views and not self.alt_joint_view:
                # Append H fusion token(s) (MHFormer-style when H>1)
                fusion_token = self.fusion_token.expand(b * num_joints, -1, -1)
                x = x.view(b * num_joints, num_points, -1)
                x = torch.cat([fusion_token, x], dim=1)
                
                if self.add_view_enc:
                    x += self.View_enc_learned[:, : num_points + self.multi_hyp]
            
            x = self.pos_drop(x)
            conf_bias = None
            geom_distance = None
            reliability_penalty = None
            if self.random_num_views and self.gbt_learnable_bias:
                batch_joints = b * num_joints
                num_tokens = num_points + self.multi_hyp
                if self.gbt_learnable_conf and conf is not None:
                    key_conf = conf.reshape(batch_joints, num_points).clamp(0, 1).to(dtype=x.dtype)
                    key_conf = torch.cat(
                        [
                            torch.zeros(
                                batch_joints,
                                self.multi_hyp,
                                device=x.device,
                                dtype=x.dtype,
                            ),
                            key_conf,
                        ],
                        dim=1,
                    )
                    conf_bias = key_conf[:, None, :].expand(batch_joints, num_tokens, num_tokens)
                if self.gbt_learnable_geom and num_points >= 2:
                    if raw_direction is None or raw_point is None:
                        raise ValueError("Geometry bias requires ray direction and point inputs")
                    geom_distance = geometry_distance_with_fusion_token(
                        raw_direction.reshape(batch_joints, num_points, 3),
                        raw_point.reshape(batch_joints, num_points, 3),
                        direct_fusion=self.gbt_fusion_geom,
                        num_fusion=self.multi_hyp,
                    ).to(dtype=x.dtype)
            if (
                self.random_num_views
                and self.gbt_oracle_reliability
                and not kwargs.get("disable_reliability", False)
            ):
                oracle_target = kwargs.get("oracle_target")
                if oracle_target is None:
                    raise ValueError("GBT_ORACLE_RELIABILITY requires oracle_target")
                if raw_direction is None or raw_point is None:
                    raise ValueError("Oracle reliability requires ray direction and point inputs")
                reliability_penalty = oracle_reliability_with_fusion_token(
                    raw_direction,
                    raw_point,
                    oracle_target.to(device=x.device, dtype=raw_direction.dtype),
                    self.gbt_oracle_temperature,
                    num_fusion=self.multi_hyp,
                ).to(dtype=x.dtype)
            elif self.random_num_views and self.gbt_learned_reliability:
                if self.gbt_reliability_pair_gate:
                    unit_direction = raw_direction / raw_direction.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-7)
                    direction_i = unit_direction.unsqueeze(3)
                    direction_j = unit_direction.unsqueeze(2)
                    angle = torch.cross(
                        direction_i.expand(-1, -1, -1, num_points, -1),
                        direction_j.expand(-1, -1, num_points, -1, -1),
                        dim=-1,
                    ).norm(dim=-1)
                    baseline = (
                        raw_point.unsqueeze(3) - raw_point.unsqueeze(2)
                    ).norm(dim=-1)
                    ray_distance = pairwise_ray_distance(
                        raw_direction.reshape(b * num_joints, num_points, 3),
                        raw_point.reshape(b * num_joints, num_points, 3),
                    ).reshape(b, num_joints, num_points, num_points)
                    off_diagonal = (
                        ~torch.eye(num_points, device=x.device, dtype=torch.bool)
                    ).view(1, 1, num_points, num_points)
                    pair_count = max(num_points * (num_points - 1), 1)
                    angle_mean = (angle * off_diagonal).sum(dim=(-1, -2)) / pair_count
                    baseline_mean = (baseline * off_diagonal).sum(dim=(-1, -2)) / pair_count
                    ray_distance_mean = (
                        ray_distance * off_diagonal
                    ).sum(dim=(-1, -2)) / pair_count
                    conf_values = conf.squeeze(-1)
                    gate_features = torch.stack(
                        [
                            angle_mean,
                            baseline_mean,
                            ray_distance_mean,
                            conf_values.mean(dim=-1),
                            conf_values.std(dim=-1, unbiased=False),
                            reliability_prediction.std(dim=-1, unbiased=False),
                        ],
                        dim=-1,
                    )
                    reliability_gate = torch.sigmoid(
                        self.reliability_pair_gate(gate_features).squeeze(-1)
                    )
                else:
                    reliability_gate = torch.sigmoid(
                        self.gbt_reliability_gate_logit
                    )
                reliability_penalty = reliability_with_fusion_token(
                    reliability_prediction,
                    self.gbt_reliability_temperature,
                    gate=reliability_gate,
                    num_fusion=self.multi_hyp,
                ).to(dtype=x.dtype)
            if self.alt_joint_view:
                # (B*J,V,D) -> alternating attention over V and J. No camera
                # identity encoding is used, preserving view permutation
                # equivariance and arbitrary-camera generalization.
                x = x.view(b, num_joints, num_points, -1)
                x = x + self.alt_joint_embedding
                for view_block, joint_block in zip(
                    self.alt_view_blocks, self.alt_joint_blocks
                ):
                    view_tokens = x.reshape(b * num_joints, num_points, -1)
                    view_tokens = view_block(view_tokens)
                    x = view_tokens.view(b, num_joints, num_points, -1)

                    joint_tokens = x.permute(0, 2, 1, 3).reshape(
                        b * num_points, num_joints, -1
                    )
                    joint_tokens = joint_block(joint_tokens)
                    x = joint_tokens.view(
                        b, num_points, num_joints, -1
                    ).permute(0, 2, 1, 3).contiguous()

                view_tokens = x.reshape(b * num_joints, num_points, -1)
                fusion_token = self.fusion_token[:, :1].expand(
                    b * num_joints, -1, -1
                )
                x = torch.cat([fusion_token, view_tokens], dim=1)
                x = self.alt_view_readout(x)
            else:
                adapter_lookup = (
                    {
                        block_index: adapter_index
                        for adapter_index, block_index in enumerate(
                            self.joint_adapter_indices
                        )
                    }
                    if self.gated_joint_adapter
                    else {}
                )
                for block_index, blk in enumerate(self.blocks_view_fusion):
                    x = blk(
                        x,
                        conf_bias=conf_bias,
                        geom_distance=geom_distance,
                        reliability_penalty=reliability_penalty,
                    )
                    if block_index in adapter_lookup:
                        adapter = self.joint_adapters[
                            adapter_lookup[block_index]
                        ]
                        fusion_tokens = x[:, : self.multi_hyp, :]
                        view_tokens = x[:, self.multi_hyp :, :].view(
                            b, num_joints, num_points, -1
                        )
                        joint_tokens = view_tokens.permute(
                            0, 2, 1, 3
                        ).reshape(b * num_points, num_joints, -1)
                        adapter_input = (
                            joint_tokens + self.joint_adapter_pos_embed
                        )
                        joint_delta = adapter(adapter_input) - adapter_input
                        if self.joint_adapter_direct_readout:
                            # The original adapter only changed view tokens.  A
                            # layer placed after the final VFT block was thus a
                            # dead branch because prediction reads the fusion
                            # token.  Direct readout sends the joint correction
                            # to that token explicitly, confidence-averaged
                            # across views, while remaining exact identity at
                            # initialization because joint_delta is zero.
                            delta_by_view = joint_delta.view(
                                b, num_points, num_joints, -1
                            ).permute(0, 2, 1, 3).contiguous()
                            if conf is not None:
                                readout_weight = conf.clamp(0, 1)
                                readout_weight = readout_weight / (
                                    readout_weight.sum(
                                        dim=2, keepdim=True
                                    ).clamp_min(1e-6)
                                )
                            else:
                                readout_weight = delta_by_view.new_full(
                                    (b, num_joints, num_points, 1),
                                    1.0 / float(num_points),
                                )
                            fusion_delta = (
                                delta_by_view * readout_weight
                            ).sum(dim=2)
                            direct_gate = F.softplus(
                                self.joint_adapter_direct_scale_raw[
                                    adapter_lookup[block_index]
                                ]
                            ).clamp(max=1.0)
                            fusion_tokens = fusion_tokens + direct_gate * (
                                fusion_delta.view(
                                    b * num_joints, 1, -1
                                )
                            )
                            x = torch.cat(
                                [
                                    fusion_tokens,
                                    view_tokens.view(
                                        b * num_joints, num_points, -1
                                    ),
                                ],
                                dim=1,
                            )
                            continue
                        if self.joint_adapter_count_lookup:
                            count_gate = F.softplus(
                                self.joint_adapter_view_scale_raw[
                                    num_points,
                                    adapter_lookup[block_index],
                                ]
                            ).clamp(max=1.0)
                        else:
                            count_gate = (
                                F.softplus(self.joint_adapter_scale_raw)
                                / float(num_points)
                                ** self.joint_adapter_view_power
                            ).clamp(max=1.0)
                        joint_tokens = joint_tokens + count_gate * joint_delta
                        view_tokens = joint_tokens.view(
                            b, num_points, num_joints, -1
                        ).permute(0, 2, 1, 3).contiguous()
                        x = torch.cat(
                            [
                                fusion_tokens,
                                view_tokens.view(
                                    b * num_joints, num_points, -1
                                ),
                            ],
                            dim=1,
                        )
            
            x = self.View_norm(x)

            if (
                self.ray_depth_aux
                and self.random_num_views
                and raw_direction is not None
                and is_training
            ):
                depth_target = kwargs.get("depth_target")
                if depth_target is not None:
                    view_tokens_final = x[:, self.multi_hyp :, :]
                    depth_prediction = self.ray_depth_head(view_tokens_final).squeeze(-1)
                    unit_dir = raw_direction / raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-7)
                    target_points = depth_target.to(device=x.device, dtype=unit_dir.dtype)
                    depth_gt = ((target_points.unsqueeze(2) - raw_point) * unit_dir).sum(dim=-1)
                    depth_gt = depth_gt.reshape(b * num_joints, num_points)
                    if conf is not None:
                        depth_valid = conf.reshape(b * num_joints, num_points) > 0.01
                    else:
                        depth_valid = torch.ones_like(depth_gt, dtype=torch.bool)
                    if depth_valid.any():
                        ray_depth_aux_loss = F.smooth_l1_loss(
                            depth_prediction[depth_valid],
                            depth_gt[depth_valid],
                            beta=0.05,
                        )
                    else:
                        ray_depth_aux_loss = depth_prediction.new_zeros(())

            # x = x.view(b, num_joints, num_points, -1)
            if not self.random_num_views:
                x = self.weighted_mean(x).squeeze(1)
            else:
                if self.multi_hyp == 1:
                    x = x[:, 0, :]
                else:
                    # (BJ, H, D) → Conv1d over H → (BJ, D); loss only on final pose.
                    hyp = x[:, : self.multi_hyp, :]
                    x = self.hyp_merge(hyp).squeeze(1)
                if ray_depth_aux_loss is not None:
                    # later in-place ops on the readout would otherwise clash with
                    # the depth-head backward through the shared token storage
                    x = x.clone()
            x = x.view(b, num_joints, -1)
            
        else:
            points = x[:, :, :, :3]
            conf = x[:, :, :, 3:4] if self.concat_confidence else None
            if self.apply_sine_encoding_on_points:
                points = self.compute_sine_cosine_encoding(points.view(b, -1, 3), self.sine_d_model)
            elif self.apply_sine_encoding_on_points_nerf:
                points = self.compute_sine_cosine_encoding_nerf(points.view(b, -1, 3), self.sine_L_nerf)
            
            x = self.encoding_to_embedding(points)
            if self.concat_confidence:
                conf_emb = self.confidence_to_embedding(conf.view(b, -1, 1))
                x = torch.cat((x, conf_emb), dim=-1)
            x = x.view(b, num_joints, num_points, -1)
            x = x.sum(dim=2)
        x += self.Spatial_pos_embed

        # KTPFormer KPA: skeleton graph prior once before PFT.
        if self.use_kpa:
            x = self.kpa(x)

        # PCT/UniCodebook: soft discrete prior inject before PFT.
        if self.pose_codebook:
            x = self.pose_codebook_dcsa.soft_inject(x)
            if is_training:
                cb_target = kwargs.get("codebook_target")
                if cb_target is not None:
                    pose_codebook_aux_loss = self.pose_codebook_dcsa.commitment_loss(
                        cb_target.to(device=x.device, dtype=x.dtype),
                        joint_feats=x,
                    )
        
        
        x = self.pos_drop(x)
        for ix, blk in enumerate(self.blocks):
            x = blk(x)
            if ix == len(self.blocks) - 1 and not self.fix_pft_last_block:
                x = blk(x)

        x = self.Spatial_norm(x)
        
        x = x.view(b, num_joints, -1)
            
        x = self.head(x)

        x = x.view(b, -1, 3)

        if tri_anchor_point is not None:
            x = x + self.tri_anchor_gate * tri_anchor_point

        if is_training and self.gbt_learned_reliability:
            if reliability_aux_loss is None:
                raise ValueError("Learned reliability training requires reliability_target")
            return x, reliability_aux_loss
        if is_training and self.ray_depth_aux:
            if ray_depth_aux_loss is None:
                ray_depth_aux_loss = x.new_zeros(())
            return x, ray_depth_aux_loss
        if is_training and self.pose_codebook:
            if pose_codebook_aux_loss is None:
                pose_codebook_aux_loss = x.new_zeros(())
            return x, pose_codebook_aux_loss
        return x


class MultiView_RUMPL_G(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MultiView_RUMPL_G, self).__init__()

        print(cfg.NETWORK)
        # num_views = 5 if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else 4
        if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') or cfg.DATASET.TEST_DATASET.startswith('multiview_amass_cmu_panoptic_pose_former'):
            num_views = 5
        else:
            num_views = 4
            
        if cfg.DATASET.TRAIN_VIEWS is not None:
            num_views = len(cfg.DATASET.TRAIN_VIEWS)
            if cfg.DATASET.USE_HELPER_CAMERAS:
                assert cfg.DATASET.TRAIN_VIEWS_HELPER is not None
                num_views += len(cfg.DATASET.TRAIN_VIEWS_HELPER)
                
        if cfg.DATASET.TRAIN_ON_ALL_CAMERAS and cfg.DATASET.TEST_ON_ALL_CAMERAS:
            num_views = cfg.DATASET.N_VIEWS_TRAIN_TEST_ALL
            
        if 'master_cam' in cfg.DATASET.TEST_DATASET:
            num_views = cfg.DATASET.N_MASTER_CAMERAS
                
            
        self.init_weights_from = cfg.NETWORK.INIT_WEIGHTS_FROM

        ##################################################
        self.features = MultiView_RUMPL(
                                 num_joints = cfg.NETWORK.NUM_JOINTS,
                                 embed_dim_ratio=cfg.NETWORK.DIM,
                                 depth=cfg.NETWORK.TRANSFORMER_DEPTH,
                                 num_heads=cfg.NETWORK.TRANSFORMER_HEADS,
                                 drop_rate=cfg.NETWORK.POSEFORMER_DROP_RATE,
                                 attn_drop_rate=cfg.NETWORK.POSEFORMER_ATTN_DROP_RATE,
                                 drop_path_rate=cfg.NETWORK.POSEFORMER_DROP_PATH_RATE,
                                 num_views=num_views,
                                 linear_weighted_mean=cfg.NETWORK.POSEFORMER_LINEAR_WEIGHTED_MEAN,
                                 hidden_dim=cfg.NETWORK.POSEFORMER_OUTPUT_HEAD_HIDDEN_DIM,
                                 cfg=cfg,
                                 )
        ###################################################3

    def forward(self, x, **kwargs):
        x = self.features(x, **kwargs)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            if 'multiview_h36m' in pretrained or 'multiview_amass_h36m' in pretrained or 'multiview_cmu_panoptic' in pretrained or 'multiview_amass_cmu_panoptic_pose_former' in pretrained:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from H36M pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> loading Pretrained model {}'.format(pretrained))
                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                self.load_state_dict(pretrained_state_dict, strict=False)
            else:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from COCO pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> init final MLP head from normal distribution')
                for m in self.features.mlp_head.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                logger.info('=> loading COCO Pretrained model {}'.format(pretrained))
                existing_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if name in self.state_dict():
                        #if 'mlp_head' in name or 'pos_embedding' in name or 'keypoint_token' in name or 'patch_to_embedding' in name:       # 2D Pos Embeddings
                        #    continue
                        if 'keypoint_token' in name:
                            new_m = torch.zeros(1, 17, 192)
                            # Human 36M -> MPII
                            # map_idx = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, 13, 14, 15, 12, 11, 10]
                            # Human 36M -> COCO
                            map_idx = [12, 12, 14, 16, 11, 13, 15, 11, 1, 0, 2, 5, 7, 9, 6, 8, 10]
                            new_m[0] = m[0][map_idx]
                            m = new_m
                            print('Shift Token ...')

                        existing_state_dict[name] = m
                        logger.info(":: {} is loaded from {}".format(name, pretrained))
                        print('Size: ', m.shape)

                self.load_state_dict(existing_state_dict, strict=False)

        elif self.init_weights_from == 'xavier_uniform':
            logger.info('=> init weights from xavier uniform distribution')
            for m in self.modules():
                if not isinstance(m, MultiView_RUMPL_G) or not isinstance(m, MultiView_RUMPL):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> from scratch >>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_multiview_rumpl_net(cfg, is_train, **kwargs):
    model = MultiView_RUMPL_G(cfg, **kwargs)
    if is_train and cfg.NETWORK.INIT_WEIGHTS:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model
