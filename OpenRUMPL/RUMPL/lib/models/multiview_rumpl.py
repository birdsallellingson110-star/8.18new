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

from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state
from models.semantic_graph_encoder import SemanticGraphPreVFTEncoder
from models.graformer_pft import GraFormerPFTEncoder


def sample_view_subset(x, num_views):
    """Uniformly sample a camera subset for every item in a batch.

    ``x`` is expected to be ``(B, J, V, D)``.  Selected camera indices are
    sorted so that the original camera order is preserved.  This is enabled
    only by the experiment flag in ``forward``; the default RUMPL path keeps
    its historical prefix slicing exactly.
    """
    batch_size, num_joints, total_views, feature_dim = x.shape
    if not 1 <= num_views <= total_views:
        raise ValueError(
            f'num_views must be within [1, {total_views}], got {num_views}'
        )
    random_keys = torch.rand(batch_size, total_views, device=x.device)
    view_indices = random_keys.topk(
        num_views, dim=1, largest=False, sorted=False
    ).indices.sort(dim=1).values
    gather_indices = view_indices[:, None, :, None].expand(
        batch_size, num_joints, num_views, feature_dim
    )
    return torch.gather(x, 2, gather_indices), view_indices


def build_view_attention_mask(
    batch_joints,
    num_views,
    mask_rate,
    device,
    mask_diagonal=True,
):
    """Sample the training-only view-attention mask used by controlled ablations.

    Token zero is RUMPL's fusion token and tokens ``1..V`` are camera views.
    View-to-view edges are Bernoulli masked.  ``mask_diagonal=False`` matches
    the non-diagonal random mask in the public MTF implementation, whereas
    ``mask_diagonal=True`` matches Masked Gifformer's fully-random definition.
    The fusion query always retains at least one observed view and the fusion
    token remains an unmasked key, preventing an all-``-inf`` attention row.
    """
    if not 0.0 <= mask_rate <= 1.0:
        raise ValueError('mask_rate must be in [0, 1]')
    if num_views < 1:
        raise ValueError('num_views must be positive')

    num_tokens = num_views + 1
    random_mask = torch.zeros(
        batch_joints,
        num_tokens,
        num_tokens,
        device=device,
        dtype=torch.bool,
    )
    view_mask = torch.rand(
        batch_joints, num_views, num_views, device=device
    ) < mask_rate
    if not mask_diagonal:
        diagonal = torch.arange(num_views, device=device)
        view_mask[:, diagonal, diagonal] = False
    random_mask[:, 1:, 1:] = view_mask

    fusion_view_mask = (
        torch.rand(batch_joints, num_views, device=device) < mask_rate
    )
    all_views_masked = fusion_view_mask.all(dim=1)
    if all_views_masked.any():
        rows = all_views_masked.nonzero(as_tuple=False).flatten()
        retained = torch.randint(num_views, (rows.numel(),), device=device)
        fusion_view_mask[rows, retained] = False
    random_mask[:, 0, 1:] = fusion_view_mask
    return random_mask


def apply_pose_fusion_blocks(x, blocks, repeat_last=True):
    """Apply PFT blocks while preserving the public RUMPL quirk by default.

    The released RUMPL loop executes every PFT block once and then executes
    the final block a second time.  ``repeat_last=False`` is an opt-in
    controlled ablation; the default remains bit-for-bit equivalent to the
    public path.
    """
    for block in blocks:
        x = block(x)
    if repeat_last and len(blocks) > 0:
        x = blocks[-1](x)
    return x


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


def pairwise_ray_distance(direction, point, eps=1e-7):
    """Pairwise distance between 3D lines, including the parallel-line case."""
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    direction_i = direction[:, :, None, :]
    direction_j = direction[:, None, :, :]
    point_diff = point[:, None, :, :] - point[:, :, None, :]
    cross = torch.cross(direction_i, direction_j, dim=-1)
    cross_norm = cross.norm(dim=-1)
    skew_distance = (point_diff * cross).sum(-1).abs() / cross_norm.clamp_min(eps)
    parallel_distance = torch.cross(point_diff, direction_i.expand_as(point_diff), dim=-1).norm(dim=-1)
    distance = torch.where(cross_norm > eps, skew_distance, parallel_distance)
    eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)
    return distance.masked_fill(eye[None], 0.0)


def geometry_distance_with_fusion_token(direction, point, direct_fusion=False):
    """Build VFT geometry bias and optionally expose per-view inconsistency to [FUS]."""
    view_distance = pairwise_ray_distance(direction, point)
    batch_size, num_views, _ = view_distance.shape
    distance = view_distance.new_zeros(batch_size, num_views + 1, num_views + 1)
    distance[:, 1:, 1:] = view_distance
    if direct_fusion and num_views > 1:
        distance[:, 0, 1:] = view_distance.sum(dim=-1) / (num_views - 1)
    return distance


def geometry_joint_view_reliability_logits(direction, point, eps=1e-7):
    """Return centered per-joint/per-view logits from ray inconsistency.

    ``direction`` and ``point`` are ``(B,J,V,3)`` raw camera rays.  A view is
    considered less reliable when its lines disagree with the other views
    across the skeleton.  The per-joint normalization makes this statistic
    independent of scene scale while preserving local detector failures.
    """
    if direction.ndim != 4 or point.ndim != 4 or direction.shape != point.shape:
        raise ValueError(
            'direction and point must have matching shape (B,J,V,3)'
        )
    batch_size, num_joints, num_views, dim = direction.shape
    if dim != 3 or num_views < 2:
        raise ValueError('geometry reliability requires (B,J,V,3) with V >= 2')
    pairwise = pairwise_ray_distance(
        direction.reshape(batch_size * num_joints, num_views, 3),
        point.reshape(batch_size * num_joints, num_views, 3),
        eps=eps,
    ).reshape(batch_size, num_joints, num_views, num_views)
    inconsistency = pairwise.sum(dim=-1) / max(num_views - 1, 1)
    normalized = inconsistency / inconsistency.mean(dim=-1, keepdim=True).clamp_min(eps)
    logits = -normalized
    return logits - logits.mean(dim=-1, keepdim=True)


def geometry_view_reliability_logits(direction, point, eps=1e-7):
    """Return centered per-view logits from cross-view ray inconsistency."""
    joint_logits = geometry_joint_view_reliability_logits(direction, point, eps=eps)
    logits = joint_logits.mean(dim=1)
    return logits - logits.mean(dim=1, keepdim=True)


def ray_normal_matrix_features(direction, confidence=None, eps=1e-7):
    """Return scale-normalized analytic triangulation uncertainty features.

    For each joint, the weighted ray-intersection normal matrix is
    ``A=sum_v w_v(I-d_v d_v^T)``.  Its normalized eigenvalues expose weak
    camera geometry (especially near-parallel two-view rays) without encoding
    a camera identity.  The fourth feature is mean detector confidence.
    """
    direction = F.normalize(direction, dim=-1, eps=eps)
    eye = torch.eye(3, device=direction.device, dtype=direction.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    if confidence is None:
        weights = torch.ones_like(direction[..., :1])
    else:
        weights = confidence.clamp(0, 1)
    normal = (weights.unsqueeze(-1) * projection).sum(dim=-3)
    eigenvalues = torch.linalg.eigvalsh(normal).clamp_min(eps)
    fractions = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_fractions = fractions.clamp_min(eps).log()
    mean_confidence = weights.mean(dim=-2)
    return torch.cat((log_fractions, mean_confidence), dim=-1)


def center_ray_points_on_anchor(points, anchors, per_joint=False):
    """Express point-on-ray inputs in a triangulation-anchor frame."""
    if points.ndim != 4 or anchors.ndim != 3:
        raise ValueError('points and anchors must have shapes (B,J,V,3) and (B,J,3)')
    if (
        points.shape[:2] != anchors.shape[:2]
        or points.shape[-1] != 3
        or anchors.shape[-1] != 3
    ):
        raise ValueError('points and anchors have incompatible shapes')
    center = anchors[:, :, None, :] if per_joint else anchors[:, :1, None, :]
    return points - center


def equivariant_body_canonicalize_rays(
    rays,
    regularization=1e-4,
    confidence_epsilon=0.05,
    pelvis_prior=False,
    robust_torso=False,
):
    """Express calibrated rays in a pelvis/shoulder/torso body frame.

    The frame is estimated only from confidence-weighted ray intersections.
    The Tikhonov prior is centred on a point-on-ray centroid rather than the
    arbitrary world origin, so the complete construction is SE(3)-equivariant.
    H36M-17 indices 0/8/11/14 denote pelvis, neck, left shoulder and right
    shoulder.  With ``robust_torso=True``, both shoulder and hip pairs define
    the horizontal axis and the torso midpoints replace the detector-dependent
    virtual neck.  Metric scale is deliberately preserved.

    Returns the canonical rays together with the world-space origin and basis
    needed to map a predicted canonical pose back to world coordinates.
    """
    if rays.ndim != 4 or rays.shape[1] != 17 or rays.shape[-1] < 7:
        raise ValueError(
            'body canonicalization requires rays shaped (B,17,V,>=7)'
        )
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-7)
    point = rays[..., 3:6]
    confidence = rays[..., 6:7].clamp(0, 1) + confidence_epsilon
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = confidence.unsqueeze(-1) * projection
    lhs = weighted_projection.sum(dim=2)
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    centroid = (
        (confidence * point).sum(dim=2)
        / confidence.sum(dim=2).clamp_min(1e-7)
    )
    if pelvis_prior:
        # Near-parallel two-view rays make independently triangulated shoulder
        # and neck anchors unstable.  A pelvis-centred Tikhonov prior keeps the
        # body-frame construction SE(3)-equivariant while damping precisely
        # those ill-conditioned intersections.  The default-off branch below
        # remains numerically identical to established checkpoints.
        pelvis_lhs = lhs[:, 0] + regularization * eye
        pelvis_rhs = (
            rhs[:, 0]
            + regularization * centroid[:, 0].unsqueeze(-1)
        )
        pelvis = torch.linalg.solve(pelvis_lhs, pelvis_rhs).squeeze(-1)
        lhs = lhs + regularization * eye
        rhs = rhs + regularization * pelvis[:, None, :, None]
        anchors = torch.linalg.solve(lhs, rhs).squeeze(-1)
    else:
        lhs = lhs + regularization * eye
        rhs = rhs + regularization * centroid.unsqueeze(-1)
        anchors = torch.linalg.solve(lhs, rhs).squeeze(-1)

    origin = anchors[:, 0]
    shoulder = F.normalize(
        anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7
    )
    if robust_torso:
        # Lower-body left/right conventions have historically varied across
        # detector PKLs.  Aligning the hip vector to the shoulder vector is
        # permutation-, camera- and dataset-ID-free and avoids baking that
        # frontend convention into the canonical frame.
        hip = F.normalize(anchors[:, 4] - anchors[:, 1], dim=-1, eps=1e-7)
        hip_alignment = torch.where(
            (hip * shoulder).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(hip[..., :1]),
            torch.ones_like(hip[..., :1]),
        )
        hip = hip * hip_alignment
        mean_confidence = confidence.squeeze(-1).mean(dim=2)
        shoulder_weight = torch.sqrt(
            mean_confidence[:, 14] * mean_confidence[:, 11]
        ).unsqueeze(-1)
        hip_weight = torch.sqrt(
            mean_confidence[:, 1] * mean_confidence[:, 4]
        ).unsqueeze(-1)
        x_axis = F.normalize(
            shoulder_weight * shoulder + hip_weight * hip,
            dim=-1,
            eps=1e-7,
        )
        shoulder_mid = 0.5 * (anchors[:, 11] + anchors[:, 14])
        hip_mid = 0.5 * (anchors[:, 1] + anchors[:, 4])
        up_hint = shoulder_mid - hip_mid
    else:
        # Keep the established path bit-for-bit unchanged when the new option
        # is disabled so existing checkpoints and results remain reproducible.
        x_axis = shoulder
        up_hint = anchors[:, 8] - origin
    y_axis = up_hint - (
        up_hint * x_axis
    ).sum(dim=-1, keepdim=True) * x_axis
    y_axis = F.normalize(y_axis, dim=-1, eps=1e-7)
    z_axis = F.normalize(
        torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7
    )
    y_axis = F.normalize(
        torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7
    )
    # Columns are canonical basis vectors expressed in world coordinates.
    basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    canonical = rays.clone()
    canonical[..., :3] = torch.einsum(
        'b...i,bij->b...j', rays[..., :3], basis
    )
    centered_point = rays[..., 3:6] - origin[:, None, None, :]
    canonical[..., 3:6] = torch.einsum(
        'b...i,bij->b...j', centered_point, basis
    )
    return canonical, origin, basis


def body_canonical_pose_to_world(pose, origin, basis):
    """Invert :func:`equivariant_body_canonicalize_rays` for a 3D pose."""
    return (
        torch.einsum('b...j,bij->b...i', pose, basis)
        + origin[:, None, :]
    )


def build_h36m17_adjacency(num_joints=17):
    """Return a symmetric, row-normalized H36M-17 kinematic adjacency."""
    parents = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
    if num_joints != len(parents):
        raise ValueError('the current skeleton graph requires H36M-17 joints')
    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for child, parent in enumerate(parents):
        if parent >= 0:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
    adjacency += torch.eye(num_joints)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class ZeroInitSkeletonGraphResidual(nn.Module):
    """Identity-initialized local skeleton message after global PFT fusion."""

    def __init__(self, dim, num_joints=17):
        super().__init__()
        self.register_buffer('adjacency', build_h36m17_adjacency(num_joints))
        self.input = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        message = torch.einsum('ij,bjd->bid', self.adjacency, x)
        return x + self.output(F.gelu(self.input(message)))


class ZeroInitJointSpecificHead(nn.Module):
    """Small per-joint correction alongside RUMPL's shared 3D head."""

    def __init__(self, num_joints, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_joints, dim, 3))
        self.bias = nn.Parameter(torch.zeros(num_joints, 3))

    def forward(self, x):
        return torch.einsum('bjd,jdk->bjk', x, self.weight) + self.bias


class ZeroInitGeometryConditional3DResidual(nn.Module):
    """Zero-output 3D correction conditioned on fused features and ray geometry."""

    def __init__(self, num_joints, dim, condition_dim=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim // 4)
        self.num_joints = num_joints
        self.dim = dim
        self.condition_dim = condition_dim
        self.input = nn.Linear(dim + condition_dim, hidden_dim)
        self.joint_embed = nn.Parameter(torch.zeros(num_joints, hidden_dim))
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 3)
        # Exact H76 identity at initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, condition):
        if features.ndim != 3 or condition.ndim != 3:
            raise ValueError(
                'features and condition must be (B,J,D) and (B,J,C)'
            )
        if (
            features.shape[:2] != condition.shape[:2]
            or features.shape[-1] != self.dim
            or condition.shape[-1] != self.condition_dim
            or features.shape[1] != self.num_joints
        ):
            raise ValueError('features and condition have incompatible shapes')
        hidden = self.input(torch.cat((features, condition), dim=-1))
        hidden = self.activation(hidden + self.joint_embed[None, :, :])
        return self.output(hidden)


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

    def forward(self, x, conf_weights=None, attn_bias=None, conf_bias=None, geom_distance=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.learned_query:
            q = self.Q_learned.expand(B, -1, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            attn = attn + attn_bias.to(dtype=attn.dtype)
        if conf_bias is not None and self.learnable_conf_bias:
            if conf_bias.dim() == 3:
                conf_bias = conf_bias.unsqueeze(1)
            attn = attn + self.gbt_conf_scale.square() * conf_bias.to(dtype=attn.dtype)
        if geom_distance is not None and self.learnable_geom_bias:
            if geom_distance.dim() == 3:
                geom_distance = geom_distance.unsqueeze(1)
            attn = attn - self.gbt_geom_scale.square() * geom_distance.to(dtype=attn.dtype)
        attn = attn.softmax(dim=-1)
        if os.environ.get('GBT_SAVE_ATTN', '0') == '1':
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

    def forward(self, x, conf_weights=None, attn_bias=None, conf_bias=None, geom_distance=None):
        
        if conf_weights is not None or attn_bias is not None or conf_bias is not None or geom_distance is not None:
            attn_output = self.attn(self.norm1(x), conf_weights, attn_bias, conf_bias, geom_distance)
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativeViewFusion(nn.Module):
    """MTF-style relative fusion over a variable number of camera views.

    Each view is described by its complete skeleton of embedded ray tokens.
    Pairwise query-key differences determine channel-wise source-view weights,
    while the values retain per-joint features.  There is no learned camera-ID
    table, so the block is cardinality agnostic and can be evaluated on camera
    layouts not seen during training.
    """

    def __init__(self, num_joints, dim, hidden_ratio=2.0):
        super().__init__()
        hidden_dim = max(dim, int(dim * hidden_ratio))
        self.num_joints = num_joints
        self.dim = dim
        self.input_norm = nn.LayerNorm(dim)
        self.pose_descriptor = nn.Sequential(
            nn.Linear(num_joints * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.relative_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.output = nn.Linear(dim, dim)
        # Identity at initialization protects the established RUMPL baseline.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, J, V, D)
        batch_size, num_joints, num_views, dim = x.shape
        if num_joints != self.num_joints or dim != self.dim:
            raise ValueError(
                'RelativeViewFusion received an incompatible token shape: '
                f'{tuple(x.shape)}'
            )
        residual = x
        normalized = self.input_norm(x)
        view_tokens = normalized.permute(0, 2, 1, 3).contiguous()
        descriptor = self.pose_descriptor(
            view_tokens.reshape(batch_size, num_views, num_joints * dim)
        )
        query = self.to_q(descriptor)
        key = self.to_k(descriptor)
        relative = self.relative_mlp(
            query[:, :, None, :] - key[:, None, :, :]
        )
        weights = self.attention_mlp(relative).softmax(dim=2)
        values = self.to_v(view_tokens)
        messages = (
            weights[:, :, :, None, :]
            * (values[:, None, :, :, :] + relative[:, :, :, None, :])
        ).sum(dim=2)
        messages = self.output(messages).permute(0, 2, 1, 3).contiguous()
        return residual + self.gate * messages


class MTFSourceNormalizedFusion(nn.Module):
    """A faithful coordinate-level adaptation of MTF's pairwise view step.

    MTF constructs a relation for every target/source view pair, predicts a
    source attention weight from that relation, and aggregates the transformed
    source features.  This module keeps that source-normalized pairwise
    structure while operating on RUMPL's already calibrated ray tokens
    ``(B,J,V,D)``.  It has no camera-ID embedding and therefore remains valid
    for the variable 2/3/4-view protocol.  The default path returns one fused
    token per joint; the residual ablation exposes per-view messages and keeps
    RUMPL's fusion-token VFT as the main path.  The downstream PFT and absolute
    3-D head are unchanged.
    """

    def __init__(
        self,
        num_joints,
        dim,
        use_confidence=False,
        hidden_ratio=2.0,
        residual_gate=False,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.use_confidence = use_confidence
        self.residual_gate = residual_gate
        hidden = max(dim, int(dim * hidden_ratio))
        pair_in = 3 * dim + (2 if use_confidence else 0)
        self.input_norm = nn.LayerNorm(dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.attention = nn.Linear(hidden, 1)
        self.message = nn.Linear(hidden, dim)
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        # The residual variant is an identity at initialization.  This lets
        # the established RUMPL fusion-token Transformer remain the main path
        # while the MTF relation learns only a controlled correction.
        if self.residual_gate:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward_per_view(self, tokens, confidence=None):
        if tokens.ndim != 4:
            raise ValueError('tokens must have shape (B,J,V,D)')
        batch, joints, views, dim = tokens.shape
        if joints != self.num_joints or dim != self.dim:
            raise ValueError(
                'MTFSourceNormalizedFusion received incompatible tokens: '
                f'{tuple(tokens.shape)}'
            )
        if views < 1:
            raise ValueError('MTFSourceNormalizedFusion requires at least one view')
        if self.use_confidence:
            if confidence is None or confidence.shape[:3] != (batch, joints, views):
                raise ValueError(
                    'confidence must have shape (B,J,V,1) when source weighting is enabled'
                )
            conf = confidence[..., :1].clamp_min(1e-4)
        else:
            conf = None

        normalized = self.input_norm(tokens)
        target = normalized.unsqueeze(3).expand(-1, -1, -1, views, -1)
        source = normalized.unsqueeze(2).expand(-1, -1, views, -1, -1)
        relation = source - target
        pair_parts = [target, source, relation]
        if conf is not None:
            target_conf = conf.unsqueeze(3).expand(-1, -1, -1, views, -1)
            source_conf = conf.unsqueeze(2).expand(-1, -1, views, -1, -1)
            pair_parts.extend((target_conf, source_conf))
        pair = torch.cat(pair_parts, dim=-1)
        hidden = self.pair_encoder(pair)
        logits = self.attention(hidden).squeeze(-1)
        if conf is not None:
            # MTF uses visibility in the source relation; log-confidence is a
            # bounded, scale-stable equivalent for the HRNet confidence input.
            logits = logits + conf.squeeze(-1).unsqueeze(2).log()
        weights = logits.softmax(dim=3)
        source_values = tokens.unsqueeze(2).expand(-1, -1, views, -1, -1)
        messages = self.message(hidden)
        message = self.output(
            (weights.unsqueeze(-1) * (source_values + messages)).sum(dim=3)
        )
        refined = tokens + (self.gate * message if self.residual_gate else message)
        return refined

    def forward(self, tokens, confidence=None):
        refined = self.forward_per_view(tokens, confidence)
        if not self.use_confidence:
            return refined.mean(dim=2)
        view_weights = confidence[..., :1].clamp_min(1e-4).squeeze(-1)
        view_weights = view_weights / view_weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return (refined * view_weights.unsqueeze(-1)).sum(dim=2)


class SkeletonViewReliabilityBias(nn.Module):
    """Predict one permutation-equivariant reliability logit per camera view.

    The descriptor sees the complete embedded skeleton from a view, while the
    scorer compares that descriptor with the set consensus.  It intentionally
    uses neither a camera-ID embedding nor a fixed number of views.
    """

    def __init__(self, num_joints, dim, hidden_ratio=2.0):
        super().__init__()
        hidden_dim = max(dim, int(dim * hidden_ratio))
        self.num_joints = num_joints
        self.dim = dim
        self.input_norm = nn.LayerNorm(dim)
        self.pose_descriptor = nn.Sequential(
            nn.Linear(num_joints * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.reliability_scorer = nn.Sequential(
            nn.Linear(3 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, J, V, D)
        batch_size, num_joints, num_views, dim = x.shape
        if num_joints != self.num_joints or dim != self.dim:
            raise ValueError(
                'SkeletonViewReliabilityBias received an incompatible token '
                f'shape: {tuple(x.shape)}'
            )
        view_tokens = self.input_norm(x).permute(0, 2, 1, 3).contiguous()
        descriptor = self.pose_descriptor(
            view_tokens.reshape(batch_size, num_views, num_joints * dim)
        )
        consensus = descriptor.mean(dim=1, keepdim=True)
        delta = descriptor - consensus
        score_input = torch.cat((descriptor, delta, delta.abs()), dim=-1)
        logits = self.reliability_scorer(score_input).squeeze(-1)
        # Centering removes an attention-softmax-invisible common offset and
        # makes the learned quantity directly interpretable across V=2/3/4.
        return logits - logits.mean(dim=1, keepdim=True)


class JointViewFeatureResidual(nn.Module):
    """Zero-initialized per-joint feature correction conditioned on a view scalar."""

    def __init__(self, num_joints, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_joints, dim))

    def forward(self, tokens, scalar):
        # tokens: (B,J,V,D), scalar: (B,J,V)
        if tokens.ndim != 4 or scalar.ndim != 3:
            raise ValueError('tokens must be (B,J,V,D), scalar must be (B,J,V)')
        if tokens.shape[:3] != scalar.shape:
            raise ValueError('tokens and scalar have incompatible B/J/V shapes')
        correction = scalar.unsqueeze(-1) * self.weight[None, :, None, :]
        return tokens + correction


class JointViewConditionalResidual(nn.Module):
    """Zero-output MLP adapter conditioned on token, view scalar, and joint."""

    def __init__(self, num_joints, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim // 4)
        self.input = nn.Linear(dim + 1, hidden_dim)
        self.joint_embed = nn.Parameter(torch.zeros(num_joints, hidden_dim))
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, dim)
        # The adapter is an exact identity at initialization.  The input MLP
        # can therefore be randomly initialized without changing H76 until
        # the output projection learns a correction.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens, scalar):
        # tokens: (B,J,V,D), scalar: (B,J,V)
        if tokens.ndim != 4 or scalar.ndim != 3:
            raise ValueError('tokens must be (B,J,V,D), scalar must be (B,J,V)')
        if tokens.shape[:3] != scalar.shape:
            raise ValueError('tokens and scalar have incompatible B/J/V shapes')
        adapter_input = torch.cat((tokens, scalar.unsqueeze(-1)), dim=-1)
        hidden = self.input(adapter_input)
        hidden = hidden + self.joint_embed[None, :, None, :]
        hidden = self.activation(hidden)
        return tokens + self.output(hidden)


def fusion_token_source_attention_bias(view_logits, num_joints, gate):
    """Put view reliability only on the fusion-token query's source keys."""
    if view_logits.dim() != 2:
        raise ValueError('view_logits must have shape (B, V)')
    batch_size, num_views = view_logits.shape
    per_joint_logits = view_logits[:, None, :].expand(
        batch_size, num_joints, num_views
    ).reshape(batch_size * num_joints, num_views)
    bias = view_logits.new_zeros(
        batch_size * num_joints, num_views + 1, num_views + 1
    )
    bias[:, 0, 1:] = gate.to(dtype=view_logits.dtype) * per_joint_logits
    return bias


def joint_fusion_token_source_attention_bias(joint_view_logits, gate):
    """Build a fusion-row bias from already per-joint view logits."""
    if joint_view_logits.dim() != 3:
        raise ValueError('joint_view_logits must have shape (B, J, V)')
    batch_size, num_joints, num_views = joint_view_logits.shape
    per_joint_logits = joint_view_logits.reshape(batch_size * num_joints, num_views)
    bias = joint_view_logits.new_zeros(
        batch_size * num_joints, num_views + 1, num_views + 1
    )
    bias[:, 0, 1:] = gate.to(dtype=joint_view_logits.dtype) * per_joint_logits
    return bias


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
        
        self.apply_view_fusion = cfg.NETWORK.APPLY_VIEW_FUSION
        
        self.add_view_enc = cfg.NETWORK.ADD_VIEW_ENCODING
        
        ## work with random number of views
        self.random_num_views = cfg.DATASET.TRAIN_RANDOM_NUM_VIEWS
        self.gbt_learnable_bias = int(os.environ.get('GBT_LEARNABLE_BIAS', '0')) == 1
        self.gbt_learnable_conf = self.gbt_learnable_bias and int(os.environ.get('GBT_USE_CONF_BIAS', '1')) == 1
        self.gbt_learnable_geom = self.gbt_learnable_bias and int(os.environ.get('GBT_USE_GEOM_BIAS', '1')) == 1
        self.gbt_conf_init = float(os.environ.get('GBT_CONF_INIT', '0.1'))
        self.gbt_geom_init = float(os.environ.get('GBT_GEOM_INIT', '0.1'))
        self.global_jv_depth = int(os.environ.get('GBT_GLOBAL_JV_DEPTH', '0'))
        self.global_jv_biased = int(os.environ.get('GBT_GLOBAL_JV_BIASED', '0')) == 1
        self.global_jv_gated = int(os.environ.get('GBT_GLOBAL_JV_GATED', '0')) == 1
        self.relative_view_fusion = (
            os.environ.get('RUMPL_RELATIVE_VIEW_FUSION', '0') == '1'
        )
        # MTF's source-normalized pairwise view fusion.  This is a separate
        # opt-in replacement for VFT; the historical RelativeViewFusion flag
        # remains untouched for reproducibility of earlier experiments.
        self.mtf_source_norm_fusion = (
            os.environ.get('RUMPL_MTF_SOURCE_NORM_FUSION', '0') == '1'
        )
        self.mtf_source_norm_confidence = (
            os.environ.get('RUMPL_MTF_SOURCE_NORM_CONFIDENCE', '0') == '1'
        )
        self.mtf_source_norm_residual = (
            os.environ.get('RUMPL_MTF_SOURCE_NORM_RESIDUAL', '0') == '1'
        )
        self.skeleton_view_reliability = (
            os.environ.get('RUMPL_SKELETON_VIEW_RELIABILITY', '0') == '1'
        )
        self.confidence_view_bias = (
            os.environ.get('RUMPL_CONFIDENCE_VIEW_BIAS', '0') == '1'
        )
        # The official Learnable Triangulation confidence head emits positive
        # per-view weights, not detector probabilities.  Its algebraic model
        # normalizes them across the *currently selected* camera subset for
        # every joint.  Keep this opt-in so legacy RUMPL detector inputs remain
        # bit-for-bit unchanged.
        self.normalize_view_confidence = (
            os.environ.get('RUMPL_NORMALIZE_VIEW_CONFIDENCE', '0') == '1'
        )
        self.geometry_view_bias = (
            os.environ.get('RUMPL_GEOMETRY_VIEW_BIAS', '0') == '1'
        )
        self.joint_confidence_view_bias = (
            os.environ.get('RUMPL_JOINT_CONFIDENCE_VIEW_BIAS', '0') == '1'
        )
        self.joint_geometry_view_bias = (
            os.environ.get('RUMPL_JOINT_GEOMETRY_VIEW_BIAS', '0') == '1'
        )
        self.joint_confidence_token_residual = (
            os.environ.get('RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL', '0') == '1'
        )
        self.joint_geometry_token_residual = (
            os.environ.get('RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL', '0') == '1'
        )
        self.joint_confidence_conditional_residual = (
            os.environ.get('RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL', '0') == '1'
        )
        self.joint_geometry_conditional_residual = (
            os.environ.get('RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL', '0') == '1'
        )
        self.post_pft_geometry_conditional_residual = (
            os.environ.get('RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL', '0') == '1'
        )
        enabled_view_biases = sum(
            (self.skeleton_view_reliability, self.confidence_view_bias,
             self.geometry_view_bias, self.joint_confidence_view_bias,
             self.joint_geometry_view_bias,
             self.joint_confidence_token_residual,
             self.joint_geometry_token_residual,
             self.joint_confidence_conditional_residual,
             self.joint_geometry_conditional_residual,
             self.post_pft_geometry_conditional_residual)
        )
        if enabled_view_biases > 1:
            raise ValueError(
                'skeleton, confidence, and geometry view biases are '
                'single-variable ablations and cannot be enabled together'
            )
        self.pft_repeat_last = (
            os.environ.get('RUMPL_PFT_REPEAT_LAST', '1') == '1'
        )
        self.semantic_graph_pre_vft = os.environ.get(
            'RUMPL_SEMANTIC_GRAPH_PRE_VFT', 'off'
        ).strip().lower()
        if self.semantic_graph_pre_vft not in ('off', 'position', 'full'):
            raise ValueError(
                'RUMPL_SEMANTIC_GRAPH_PRE_VFT must be off, position, or full'
            )
        self.semantic_graph_depth = int(
            os.environ.get('RUMPL_SEMANTIC_GRAPH_DEPTH', '4')
        )
        if self.semantic_graph_depth < 1:
            raise ValueError('RUMPL_SEMANTIC_GRAPH_DEPTH must be positive')
        self.graformer_pft_mode = os.environ.get(
            'RUMPL_GRAFORMER_PFT', 'off'
        ).strip().lower()
        if self.graformer_pft_mode not in ('off', 'attention', 'full'):
            raise ValueError(
                'RUMPL_GRAFORMER_PFT must be off, attention, or full'
            )
        self.graformer_pft_depth = int(
            os.environ.get('RUMPL_GRAFORMER_DEPTH', '5')
        )
        if self.graformer_pft_depth < 1:
            raise ValueError('RUMPL_GRAFORMER_DEPTH must be positive')
        if self.graformer_pft_mode != 'off' and self.semantic_graph_pre_vft != 'off':
            raise ValueError(
                'GraFormer PFT and SGraFormer pre-VFT are separate model ablations'
            )
        # GBT-style set encoder + fixed joint-query decoder.  Unlike RUMPL's
        # per-joint fusion token, this readout is independent of the number of
        # camera tokens and can therefore be trained with V=2 and evaluated
        # with any V.  It is opt-in so the established baseline is unchanged.
        self.gbt_set_decoder = os.environ.get('RUMPL_GBT_SET_DECODER', '0') == '1'
        self.gbt_set_biased = os.environ.get('RUMPL_GBT_SET_BIASED', '0') == '1'
        self.gbt_set_plucker = os.environ.get('RUMPL_GBT_SET_PLUCKER', '0') == '1'
        self.gbt_set_harmonic_l = int(
            os.environ.get('RUMPL_GBT_SET_HARMONIC_L', '0')
        )
        # Paper-style ray inputs can be evaluated without replacing RUMPL's
        # fusion backbone by the experimental GBT set decoder.  The legacy
        # GBT flags remain aliases so all existing experiments are unchanged.
        self.input_plucker = (
            os.environ.get('RUMPL_INPUT_PLUCKER', '0') == '1'
            or (self.gbt_set_decoder and self.gbt_set_plucker)
        )
        input_harmonic_value = os.environ.get('RUMPL_INPUT_HARMONIC_L')
        self.input_harmonic_l = (
            int(input_harmonic_value)
            if input_harmonic_value is not None
            else (
                self.gbt_set_harmonic_l
                if self.gbt_set_decoder else 0
            )
        )
        if self.input_harmonic_l < 0:
            raise ValueError('RUMPL_INPUT_HARMONIC_L must be non-negative')
        self.gbt_set_no_conf_concat = (
            os.environ.get('RUMPL_GBT_SET_NO_CONF_CONCAT', '0') == '1'
        )
        self.gbt_set_depth = int(os.environ.get('RUMPL_GBT_SET_DEPTH', '3'))
        self.gbt_set_decoder_depth = int(
            os.environ.get('RUMPL_GBT_SET_DECODER_DEPTH', '2')
        )
        # GBT/MVGFormer-inspired parallel joint-query readout.  This branch
        # does not replace H76's VFT/PFT path: it reads the encoded calibrated
        # ray tokens before VFT and predicts a zero-initialized 3D residual
        # beside the established head.  The memory is camera-ID free, so the
        # same weights support arbitrary view order and 2/3/4 views.
        self.gbt_query_residual = (
            os.environ.get('RUMPL_GBT_QUERY_RESIDUAL', '0') == '1'
        )
        self.gbt_query_residual_global = (
            os.environ.get('RUMPL_GBT_QUERY_RESIDUAL_GLOBAL', '1') == '1'
        )
        self.gbt_query_residual_depth = int(
            os.environ.get('RUMPL_GBT_QUERY_RESIDUAL_DEPTH', '2')
        )
        self.gbt_query_residual_max_delta = float(
            os.environ.get('RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA', '0.5')
        )
        if self.gbt_query_residual_depth < 1:
            raise ValueError('RUMPL_GBT_QUERY_RESIDUAL_DEPTH must be positive')
        if self.gbt_query_residual_max_delta <= 0:
            raise ValueError('RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA must be positive')
        # Optional geometry-equivariant residual decoding.  The default-off
        # path is parameter- and numerics-identical to the RUMPL baseline.
        self.tri_anchor = os.environ.get('RUMPL_TRI_ANCHOR', '0') == '1'
        self.tri_anchor_reg = float(
            os.environ.get('RUMPL_TRI_ANCHOR_REG', '1e-4')
        )
        self.tri_anchor_conf_eps = float(
            os.environ.get('RUMPL_TRI_ANCHOR_CONF_EPS', '0.05')
        )
        # GHT-style pose standardization moved in front of the complete RUMPL
        # generator.  It is opt-in and parameter-free, so all established
        # checkpoints and default runs remain numerically unchanged.
        self.body_canonical_frame = (
            os.environ.get('RUMPL_BODY_CANONICAL_FRAME', '0') == '1'
        )
        self.body_canonical_reg = float(
            os.environ.get('RUMPL_BODY_CANONICAL_REG', '1e-4')
        )
        self.body_canonical_pelvis_prior = (
            os.environ.get('RUMPL_BODY_CANONICAL_PELVIS_PRIOR', '0') == '1'
        )
        self.body_canonical_robust_torso = (
            os.environ.get('RUMPL_BODY_CANONICAL_ROBUST_TORSO', '0') == '1'
        )
        self.anchor_centered_rays = (
            os.environ.get('RUMPL_ANCHOR_CENTERED_RAYS', '0') == '1'
        )
        self.anchor_center_per_joint = (
            os.environ.get('RUMPL_ANCHOR_CENTER_PER_JOINT', '0') == '1'
        )
        self.geometry_uncertainty_token = (
            os.environ.get('RUMPL_GEOMETRY_UNCERTAINTY_TOKEN', '0') == '1'
        )
        self.per_joint_residual_gate = (
            os.environ.get('RUMPL_PER_JOINT_RESIDUAL_GATE', '0') == '1'
        )
        self.post_pft_graph_residual = (
            os.environ.get('RUMPL_POST_PFT_GRAPH_RESIDUAL', '0') == '1'
        )
        self.joint_specific_head = (
            os.environ.get('RUMPL_JOINT_SPECIFIC_HEAD', '0') == '1'
        )
        self.skip_vft = os.environ.get('RUMPL_SKIP_VFT', '0') == '1'
        self.skip_pft = os.environ.get('RUMPL_SKIP_PFT', '0') == '1'
        self.vft_depth_override = int(os.environ.get('RUMPL_VFT_DEPTH', '0'))
        if self.vft_depth_override < 0:
            raise ValueError('RUMPL_VFT_DEPTH must be non-negative')
        if self.skip_vft and self.gbt_set_decoder:
            raise ValueError('RUMPL_SKIP_VFT is incompatible with RUMPL_GBT_SET_DECODER')
        if self.skip_pft and not self.apply_view_fusion:
            raise ValueError('RUMPL_SKIP_PFT requires view fusion path')
        if self.global_jv_depth < 0:
            raise ValueError('GBT_GLOBAL_JV_DEPTH must be non-negative')
        if self.global_jv_biased and self.global_jv_depth == 0:
            raise ValueError('GBT_GLOBAL_JV_BIASED requires GBT_GLOBAL_JV_DEPTH > 0')
        if self.anchor_centered_rays and not self.tri_anchor:
            raise ValueError('RUMPL_ANCHOR_CENTERED_RAYS requires RUMPL_TRI_ANCHOR=1')
        if self.body_canonical_frame and not self.tri_anchor:
            raise ValueError('RUMPL_BODY_CANONICAL_FRAME requires RUMPL_TRI_ANCHOR=1')
        if self.body_canonical_pelvis_prior and not self.body_canonical_frame:
            raise ValueError(
                'RUMPL_BODY_CANONICAL_PELVIS_PRIOR requires '
                'RUMPL_BODY_CANONICAL_FRAME=1'
            )
        if self.body_canonical_robust_torso and not self.body_canonical_frame:
            raise ValueError(
                'RUMPL_BODY_CANONICAL_ROBUST_TORSO requires '
                'RUMPL_BODY_CANONICAL_FRAME=1'
            )
        if self.anchor_center_per_joint and not self.anchor_centered_rays:
            raise ValueError('RUMPL_ANCHOR_CENTER_PER_JOINT requires RUMPL_ANCHOR_CENTERED_RAYS=1')
        if self.post_pft_geometry_conditional_residual and not self.tri_anchor:
            raise ValueError(
                'RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL requires '
                'RUMPL_TRI_ANCHOR=1'
            )
        if self.geometry_uncertainty_token and not self.tri_anchor:
            raise ValueError('RUMPL_GEOMETRY_UNCERTAINTY_TOKEN requires RUMPL_TRI_ANCHOR=1')
        if self.per_joint_residual_gate and not self.tri_anchor:
            raise ValueError('RUMPL_PER_JOINT_RESIDUAL_GATE requires RUMPL_TRI_ANCHOR=1')
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
        
        if self.input_harmonic_l > 0:
            harmonic_dim = self.input_harmonic_l * 2 * 3
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                harmonic_dim *= 2
            self.encoding_to_embedding = nn.Linear(harmonic_dim, embed_dim_ratio)
        elif self.apply_sine_encoding_on_points:
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
            
        self.concat_confidence = (
            cfg.NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB
            and not (
                self.gbt_set_decoder and self.gbt_set_no_conf_concat
            )
        )
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

        if self.geometry_uncertainty_token:
            # Zero-initialized residual: at step zero the forward path is
            # exactly the established RUMPL baseline, while gradients can
            # immediately learn how geometric degeneracy should modify [FUS].
            rng_state = torch.get_rng_state()
            self.geometry_uncertainty_to_embedding = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, embed_dim_ratio),
            )
            nn.init.zeros_(self.geometry_uncertainty_to_embedding[-1].weight)
            nn.init.zeros_(self.geometry_uncertainty_to_embedding[-1].bias)
            torch.set_rng_state(rng_state)

        if self.relative_view_fusion:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_RELATIVE_VIEW_FUSION requires random-view fusion mode'
                )
            # Preserve the initialization stream of every established RUMPL
            # parameter so the zero-gated ablation differs only by this block.
            rng_state = torch.get_rng_state()
            self.Relative_view_fusion = RelativeViewFusion(
                num_joints=num_joints,
                dim=embed_dim_ratio,
            )
            torch.set_rng_state(rng_state)

        if self.mtf_source_norm_fusion:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_MTF_SOURCE_NORM_FUSION requires random-view fusion mode'
                )
            if self.mtf_source_norm_confidence and not cfg.NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB:
                raise ValueError(
                    'RUMPL_MTF_SOURCE_NORM_CONFIDENCE requires confidence input'
                )
            rng_state = torch.get_rng_state()
            self.MTF_source_norm_fusion = MTFSourceNormalizedFusion(
                num_joints=num_joints,
                dim=embed_dim_ratio,
                use_confidence=self.mtf_source_norm_confidence,
                residual_gate=self.mtf_source_norm_residual,
            )
            # Do not shift initialization of the established RUMPL/PFT/head
            # parameters so the two MTF arms share the same control stream.
            torch.set_rng_state(rng_state)

        if self.skeleton_view_reliability:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_SKELETON_VIEW_RELIABILITY requires random-view '
                    'fusion mode'
                )
            # Keep every established H76 parameter bit-identical.  The
            # per-layer zero gates additionally make the new forward path an
            # exact functional identity at initialization.
            rng_state = torch.get_rng_state()
            self.Skeleton_view_reliability = SkeletonViewReliabilityBias(
                num_joints=num_joints,
                dim=embed_dim_ratio,
            )
            self.skeleton_view_reliability_gates = nn.Parameter(
                torch.zeros(depth)
            )
            torch.set_rng_state(rng_state)

        if self.confidence_view_bias:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_CONFIDENCE_VIEW_BIAS requires random-view fusion '
                    'mode'
                )
            # The bias is a direct, calibrated confidence statistic.  Only
            # these ReZero-style per-VFT-layer scalars are trained in R2a;
            # restoring the RNG keeps all established H76 parameters exact.
            rng_state = torch.get_rng_state()
            self.confidence_view_bias_gates = nn.Parameter(
                torch.zeros(depth)
            )
            torch.set_rng_state(rng_state)

        if self.geometry_view_bias:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_GEOMETRY_VIEW_BIAS requires random-view fusion mode'
                )
            rng_state = torch.get_rng_state()
            self.geometry_view_bias_gates = nn.Parameter(torch.zeros(depth))
            torch.set_rng_state(rng_state)

        if self.joint_confidence_view_bias:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_CONFIDENCE_VIEW_BIAS requires random-view '
                    'fusion mode'
                )
            rng_state = torch.get_rng_state()
            self.joint_confidence_view_bias_gates = nn.Parameter(
                torch.zeros(depth)
            )
            torch.set_rng_state(rng_state)

        if self.joint_geometry_view_bias:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_GEOMETRY_VIEW_BIAS requires random-view '
                    'fusion mode'
                )
            rng_state = torch.get_rng_state()
            self.joint_geometry_view_bias_gates = nn.Parameter(
                torch.zeros(depth)
            )
            torch.set_rng_state(rng_state)

        if self.joint_confidence_token_residual:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL requires '
                    'random-view fusion mode'
                )
            self.joint_confidence_token_residual_module = JointViewFeatureResidual(
                num_joints=num_joints,
                dim=embed_dim_ratio,
            )

        if self.joint_geometry_token_residual:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL requires '
                    'random-view fusion mode'
                )
            self.joint_geometry_token_residual_module = JointViewFeatureResidual(
                num_joints=num_joints,
                dim=embed_dim_ratio,
            )

        if self.joint_confidence_conditional_residual:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL requires '
                    'random-view fusion mode'
                )
            rng_state = torch.get_rng_state()
            self.joint_confidence_conditional_residual_module = (
                JointViewConditionalResidual(
                    num_joints=num_joints,
                    dim=embed_dim_ratio,
                )
            )
            torch.set_rng_state(rng_state)

        if self.joint_geometry_conditional_residual:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL requires '
                    'random-view fusion mode'
                )
            rng_state = torch.get_rng_state()
            self.joint_geometry_conditional_residual_module = (
                JointViewConditionalResidual(
                    num_joints=num_joints,
                    dim=embed_dim_ratio,
                )
            )
            torch.set_rng_state(rng_state)

        if self.gbt_set_decoder:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_GBT_SET_DECODER requires random-view fusion mode'
                )
            if self.gbt_set_depth < 1 or self.gbt_set_decoder_depth < 1:
                raise ValueError('GBT set encoder/decoder depths must be positive')
            self.GBT_set_joint_embed = nn.Parameter(
                torch.zeros(1, num_joints, 1, embed_dim_ratio)
            )
            self.GBT_set_encoder = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    learnable_conf_bias=self.gbt_set_biased,
                    learnable_geom_bias=self.gbt_set_biased,
                    conf_bias_init=self.gbt_conf_init,
                    geom_bias_init=self.gbt_geom_init,
                )
                for _ in range(self.gbt_set_depth)
            ])
            self.GBT_set_encoder_norm = norm_layer(embed_dim_ratio)
            self.GBT_joint_queries = nn.Parameter(
                torch.zeros(1, num_joints, embed_dim_ratio)
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim_ratio,
                nhead=num_heads,
                dim_feedforward=int(embed_dim_ratio * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.GBT_set_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.gbt_set_decoder_depth,
                norm=norm_layer(embed_dim_ratio),
            )
            self.GBT_set_head = nn.Sequential(
                nn.LayerNorm(embed_dim_ratio),
                nn.Linear(embed_dim_ratio, 3),
            )
            trunc_normal_(self.GBT_set_joint_embed, std=.02)
            trunc_normal_(self.GBT_joint_queries, std=.02)

        if self.gbt_query_residual:
            if not self.apply_view_fusion or not self.random_num_views:
                raise ValueError(
                    'RUMPL_GBT_QUERY_RESIDUAL requires random-view fusion mode'
                )
            # Do not perturb the established H76 initialization stream.  The
            # new branch is loaded from scratch and its final projection is
            # zero, hence adding it is an exact H76 identity before training.
            rng_state = torch.get_rng_state()
            self.gbt_query_joint_queries = nn.Parameter(
                torch.zeros(1, num_joints, embed_dim_ratio)
            )
            self.gbt_query_joint_memory_embed = nn.Parameter(
                torch.zeros(1, num_joints, 1, embed_dim_ratio)
            )
            self.gbt_query_anchor_embed = nn.Linear(3, embed_dim_ratio)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim_ratio,
                nhead=num_heads,
                dim_feedforward=int(embed_dim_ratio * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.gbt_query_residual_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.gbt_query_residual_depth,
                norm=norm_layer(embed_dim_ratio),
            )
            self.gbt_query_residual_head = nn.Sequential(
                nn.LayerNorm(embed_dim_ratio),
                nn.Linear(embed_dim_ratio, 3),
            )
            trunc_normal_(self.gbt_query_joint_queries, std=.02)
            trunc_normal_(self.gbt_query_joint_memory_embed, std=.02)
            nn.init.zeros_(self.gbt_query_residual_head[-1].weight)
            nn.init.zeros_(self.gbt_query_residual_head[-1].bias)
            torch.set_rng_state(rng_state)
        
        
        if self.apply_view_fusion and self.random_num_views:
            self.fusion_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim_ratio), requires_grad=True)
            
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        if self.semantic_graph_pre_vft != 'off':
            # The new architecture must not perturb initialization of the
            # established RUMPL VFT/PFT.  This also makes SG-M0/SG-M1 share
            # the same downstream initialization under a fixed seed.
            rng_state = torch.get_rng_state()
            self.semantic_graph_encoder = SemanticGraphPreVFTEncoder(
                dim=embed_dim_ratio,
                num_joints=num_joints,
                depth=self.semantic_graph_depth,
                num_heads=num_heads,
                mlp_ratio=2.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.1,
                max_hop=4,
                mode=self.semantic_graph_pre_vft,
            )
            torch.set_rng_state(rng_state)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        
        ### view fusion
        if self.apply_view_fusion:
            if not self.random_num_views:
                self.View_enc_learned = nn.Parameter(torch.zeros(1, num_views, embed_dim_ratio))
            else:
                self.View_enc_learned = nn.Parameter(torch.zeros(1, self.max_num_views + 1, embed_dim_ratio))
            self.blocks_view_fusion = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    learnable_conf_bias=self.gbt_learnable_conf,
                    learnable_geom_bias=self.gbt_learnable_geom,
                    conf_bias_init=self.gbt_conf_init, geom_bias_init=self.gbt_geom_init)
                for i in range(depth)])
            if self.global_jv_depth > 0:
                if not self.random_num_views:
                    raise ValueError('global joint-view refinement requires random-view RUMPL')
                # The block is ReZero-gated.  Preserve baseline initialization
                # for all modules constructed after this optional branch too.
                rng_state = torch.get_rng_state()
                self.Global_joint_embed = nn.Parameter(
                    torch.zeros(1, num_joints, 1, embed_dim_ratio)
                )
                self.blocks_global_joint_view = nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer,
                        learnable_conf_bias=self.global_jv_biased,
                        learnable_geom_bias=self.global_jv_biased,
                        conf_bias_init=self.gbt_conf_init, geom_bias_init=self.gbt_geom_init,
                    )
                    for _ in range(self.global_jv_depth)
                ])
                self.Global_joint_view_norm = norm_layer(embed_dim_ratio)
                if self.global_jv_gated:
                    # ReZero-style gate preserves the established VFT input at initialization.
                    self.global_jv_gate = nn.Parameter(torch.zeros(1))
                torch.set_rng_state(rng_state)

        ##### create FPT blocks
        num_tokens = num_joints
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_tokens=num_tokens)
            for i in range(depth)])
        if self.graformer_pft_mode != 'off':
            # Construct the public PFT first to consume exactly the baseline RNG
            # stream, then replace it.  Isolating the GraFormer initialization
            # keeps the shared VFT/head initialization identical to H76.
            rng_state = torch.get_rng_state()
            self.graformer_pft = GraFormerPFTEncoder(
                dim=embed_dim_ratio,
                num_joints=num_joints,
                depth=self.graformer_pft_depth,
                num_heads=4,
                dropout=0.25,
                mode=self.graformer_pft_mode,
            )
            torch.set_rng_state(rng_state)
            self.blocks = nn.ModuleList()
        
        

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
        if self.post_pft_graph_residual:
            rng_state = torch.get_rng_state()
            self.pft_graph_residual = ZeroInitSkeletonGraphResidual(
                embed_dim_ratio, num_joints
            )
            torch.set_rng_state(rng_state)
        if self.joint_specific_head:
            self.per_joint_head = ZeroInitJointSpecificHead(
                num_joints, embed_dim_ratio
            )
        if self.post_pft_geometry_conditional_residual:
            # Preserve H76's data/view sampling RNG stream.  The zero-output
            # adapter is identity at initialization, so its random hidden
            # projection must not perturb unrelated initialization or shuffling.
            rng_state = torch.get_rng_state()
            self.post_pft_geometry_conditional_module = (
                ZeroInitGeometryConditional3DResidual(
                    num_joints=num_joints,
                    dim=embed_dim_ratio,
                    condition_dim=4,
                )
            )
            torch.set_rng_state(rng_state)
        if self.tri_anchor:
            # Preserve the baseline RNG stream: adding the scalar must not
            # change initialization of any existing RUMPL parameter.
            rng_state = torch.get_rng_state()
            self.tri_anchor_gate = nn.Parameter(torch.tensor(1.0))
            if self.per_joint_residual_gate:
                self.residual_joint_gate = nn.Parameter(
                    torch.ones(1, num_joints, 1)
                )
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
        mtf_source_norm_fused = False
        tri_anchor_point = None
        body_canonical_origin = None
        body_canonical_basis = None
        geometry_uncertainty = None
        _raw_dir = None
        _raw_int = None
        semantic_graph_coordinates = None
        # depths = kwargs['depths'] if 'depths' in kwargs else None
        if self.apply_view_fusion:
            # assert d == 6, 'Input shape should be (b, num_joints, num_views, 6) if apply_view_fusion'
            if self.random_num_views and is_training:
                fixed_train_views = os.environ.get('TRAIN_FIXED_NUM_VIEWS')
                fixed_epochs = int(os.environ.get('TRAIN_FIXED_NUM_VIEWS_EPOCHS', 0))
                current_epoch = int(kwargs.get('epoch', 0))
                use_fixed_views = fixed_train_views is not None and (
                    fixed_epochs <= 0 or current_epoch < fixed_epochs
                )
                # Optional cardinality curriculum.  The value is a semicolon
                # separated list of ``start_epoch:w2,w3,w4`` entries, e.g.
                # ``0:8,1,1;7:3,1,1;14:3,2,2``.  The last entry whose start
                # epoch is not greater than the current epoch is used.  With
                # the variable unset, the historical fixed distribution is
                # unchanged.  This is intentionally an input-sampling-only
                # control: it does not add a view-specific parameter or head.
                curriculum_spec = os.environ.get(
                    'RUMPL_CURRICULUM_VIEW_WEIGHTS', ''
                ).strip()
                raw_view_weights = os.environ.get(
                    'RUMPL_VIEW_COUNT_WEIGHTS', ''
                ).strip()
                if curriculum_spec:
                    schedule = []
                    for item in curriculum_spec.split(';'):
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            start_text, weights_text = item.split(':', 1)
                            start_epoch = int(start_text.strip())
                            weights_text = weights_text.strip()
                            if start_epoch < 0 or not weights_text:
                                raise ValueError
                            schedule.append((start_epoch, weights_text))
                        except ValueError as exc:
                            raise ValueError(
                                'RUMPL_CURRICULUM_VIEW_WEIGHTS must use '
                                'start_epoch:w2,w3,w4 entries separated by ;'
                            ) from exc
                    if not schedule:
                        raise ValueError(
                            'RUMPL_CURRICULUM_VIEW_WEIGHTS cannot be empty'
                        )
                    schedule.sort(key=lambda pair: pair[0])
                    if schedule[0][0] != 0:
                        raise ValueError(
                            'RUMPL_CURRICULUM_VIEW_WEIGHTS must start at epoch 0'
                        )
                    selected = schedule[0][1]
                    for start_epoch, weights_text in schedule:
                        if start_epoch <= current_epoch:
                            selected = weights_text
                        else:
                            break
                    raw_view_weights = selected
                if not use_fixed_views:
                    if raw_view_weights:
                        view_weights = torch.tensor(
                            [
                                float(value)
                                for value in raw_view_weights.split(',')
                            ],
                            dtype=torch.float32,
                        )
                        expected = self.max_num_views - self.min_num_views + 1
                        if (
                            view_weights.numel() != expected
                            or (view_weights < 0).any()
                            or view_weights.sum() <= 0
                        ):
                            raise ValueError(
                                'RUMPL_VIEW_COUNT_WEIGHTS must contain '
                                f'{expected} non-negative comma-separated '
                                'weights with a positive sum'
                            )
                        sampled = torch.multinomial(
                            view_weights, 1, replacement=True
                        ).item()
                        num_points = self.min_num_views + sampled
                    else:
                        num_points = torch.randint(
                            self.min_num_views,
                            self.max_num_views + 1,
                            (1,),
                        ).item()
                else:
                    num_points = int(fixed_train_views)
                    if not self.min_num_views <= num_points <= self.max_num_views:
                        raise ValueError(
                            'TRAIN_FIXED_NUM_VIEWS must be within the configured view capacity '
                            f'[{self.min_num_views}, {self.max_num_views}]'
                        )
                random_view_subset = (
                    os.environ.get('RUMPL_RANDOM_VIEW_SUBSET', '0') == '1'
                )
                if random_view_subset:
                    x, view_indices = sample_view_subset(x, num_points)
                else:
                    x = x[:, :, :num_points, :]
                if getattr(self, '_view_sampler_log_epoch', None) != current_epoch:
                    if use_fixed_views:
                        mode = f'fixed-{num_points}'
                    elif curriculum_spec:
                        mode = f'curriculum-random-{raw_view_weights}'
                    elif raw_view_weights:
                        mode = f'weighted-random-{raw_view_weights}'
                    else:
                        mode = 'uniform-random'
                    subset_mode = (
                        'uniform-camera-subset-per-sample'
                        if random_view_subset else 'historical-camera-prefix'
                    )
                    example = (
                        view_indices[0].detach().cpu().tolist()
                        if random_view_subset else list(range(num_points))
                    )
                    print(
                        f'[VIEW_SAMPLER] epoch={current_epoch} mode={mode} '
                        f'capacity={self.min_num_views}-{self.max_num_views} '
                        f'fixed_epochs={fixed_epochs} subset={subset_mode} '
                        f'example_zero_based={example}',
                        flush=True,
                    )
                    self._view_sampler_log_epoch = current_epoch

            if self.normalize_view_confidence:
                confidence_index = 19 if (
                    self.use_only_2D or self.feed_camera_calibration
                ) else 6
                if d <= confidence_index:
                    raise RuntimeError(
                        'RUMPL_NORMALIZE_VIEW_CONFIDENCE requires a confidence '
                        f'channel at index {confidence_index}, input dim={d}'
                    )
                confidence = x[..., confidence_index:confidence_index + 1]
                confidence = confidence / confidence.sum(
                    dim=2, keepdim=True
                ).clamp_min(1e-12)
                x = torch.cat(
                    (
                        x[..., :confidence_index],
                        confidence,
                        x[..., confidence_index + 1:],
                    ),
                    dim=-1,
                )
                if not hasattr(self, '_normalize_view_confidence_logged'):
                    print(
                        '[NORMALIZE_VIEW_CONFIDENCE] enabled=1 '
                        'axis=current_view_subset per_joint_sum=1',
                        flush=True,
                    )
                    self._normalize_view_confidence_logged = True

            if self.body_canonical_frame:
                if self.use_only_2D or self.feed_camera_calibration:
                    raise RuntimeError(
                        'RUMPL_BODY_CANONICAL_FRAME requires calibrated ray input'
                    )
                x, body_canonical_origin, body_canonical_basis = (
                    equivariant_body_canonicalize_rays(
                        x,
                        regularization=self.body_canonical_reg,
                        confidence_epsilon=self.tri_anchor_conf_eps,
                        pelvis_prior=self.body_canonical_pelvis_prior,
                        robust_torso=self.body_canonical_robust_torso,
                    )
                )
                if not hasattr(self, '_body_canonical_frame_logged'):
                    print(
                        '[BODY_CANONICAL_FRAME] enabled=1 '
                        'origin=triangulated_pelvis axes=shoulder_torso '
                        'pelvis_prior={} robust_torso={} '
                        'scale=metric output=inverse_world_transform'.format(
                            int(self.body_canonical_pelvis_prior),
                            int(self.body_canonical_robust_torso),
                        ),
                        flush=True,
                    )
                    self._body_canonical_frame_logged = True
                
            if self.use_only_2D:
                joints_2d = x[:, :, :, :2]
                semantic_graph_coordinates = joints_2d
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            elif self.feed_camera_calibration:
                joints_2d = x[:, :, :, :2]
                semantic_graph_coordinates = joints_2d
                camera_calibration = x[:, :, :, 2:19]
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            else:
                direction_features = x[:, :, :, :3]
                intersection_features = x[:, :, :, 3:6]
                semantic_graph_coordinates = direction_features
                # GBT uses detector confidence as an attention bias, but does
                # not concatenate it to the Pluecker ray token.  Keep access
                # to the raw score for that paper-aligned path while leaving
                # every non-GBT/default configuration unchanged.
                conf = x[:, :, :, 6:7] if (
                    self.concat_confidence
                    or (self.gbt_set_decoder and self.gbt_set_biased)
                ) else None
                if self.concat_depth_as_input:
                    depths = x[:, :, :, 7:8]
                # Keep raw rays for optional geometry-aware view fusion before positional encoding.
                _need_raw_rays = (
                    float(os.environ.get('DEPRO_LAMBDA', 0.0)) > 0
                    or float(os.environ.get('GBT_GEOM_BIAS', 0.0)) > 0
                    or self.gbt_learnable_geom
                    or self.global_jv_biased
                    or (self.gbt_set_decoder and self.gbt_set_biased)
                    or self.tri_anchor
                    or self.input_plucker
                )
                _raw_dir = direction_features if _need_raw_rays else None
                _raw_int = intersection_features if _need_raw_rays else None
                if self.tri_anchor and num_points >= 2:
                    # Confidence-weighted least-squares ray intersection:
                    # argmin_p sum_i w_i ||(I-d_i d_i^T)(p-o_i)||^2.
                    # Use the original point-on-ray representation here.  In
                    # the GBT path ``intersection_features`` has already been
                    # converted to a Pluecker moment and is not a 3D point.
                    anchor_direction = (
                        _raw_dir if _raw_dir is not None else direction_features
                    )
                    anchor_point = (
                        _raw_int if _raw_int is not None else intersection_features
                    )
                    unit_dir = anchor_direction / anchor_direction.norm(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-7)
                    if conf is not None:
                        anchor_weight = (
                            conf.clamp(0, 1) + self.tri_anchor_conf_eps
                        )
                    else:
                        anchor_weight = torch.ones_like(unit_dir[..., :1])
                    eye3 = torch.eye(
                        3, device=x.device, dtype=unit_dir.dtype
                    )
                    ray_proj = (
                        eye3
                        - unit_dir.unsqueeze(-1) * unit_dir.unsqueeze(-2)
                    )
                    weighted_proj = anchor_weight.unsqueeze(-1) * ray_proj
                    anchor_lhs = (
                        weighted_proj.sum(dim=2)
                        + self.tri_anchor_reg * eye3
                    )
                    anchor_rhs = (
                        weighted_proj @ anchor_point.unsqueeze(-1)
                    ).sum(dim=2)
                    tri_anchor_point = torch.linalg.solve(
                        anchor_lhs, anchor_rhs
                    ).squeeze(-1)
                    if self.geometry_uncertainty_token:
                        geometry_uncertainty = ray_normal_matrix_features(
                            unit_dir,
                            conf.clamp(0, 1) if conf is not None else None,
                        )
                if self.anchor_centered_rays:
                    # H76 uses one subject-root frame.  The opt-in H80 path
                    # aligns every token with the per-joint anchor restored by
                    # the residual output, without changing that output path.
                    intersection_features = center_ray_points_on_anchor(
                        intersection_features,
                        tri_anchor_point,
                        per_joint=self.anchor_center_per_joint,
                    )
                if self.input_plucker:
                    # A point anywhere on a line defines the same Pluecker
                    # moment m=o×d.  If anchor centering is active, the moment
                    # is expressed in that translated coordinate system.
                    direction_features = F.normalize(
                        direction_features, dim=-1, eps=1e-7
                    )
                    intersection_features = torch.cross(
                        intersection_features, direction_features, dim=-1
                    )
            
            if self.feed_camera_calibration or self.use_only_2D:
                x = self.encoding_to_embedding(joints_2d.view(b*num_joints, -1, 2))
                if self.feed_camera_calibration:
                    camera_calibration = self.camera_calibration_to_embedding(camera_calibration.view(b*num_joints, -1, 17))
                    x = torch.cat((x, camera_calibration), dim=-1)
                if self.concat_confidence:
                    conf_emb = self.confidence_to_embedding(conf.view(b*num_joints, -1, 1))
                    x = torch.cat((x, conf_emb), dim=-1)
            else:
                if self.input_harmonic_l > 0:
                    direction_features = self.compute_sine_cosine_encoding_nerf(
                        direction_features.view(b*num_joints, -1, 3),
                        self.input_harmonic_l,
                    )
                    intersection_features = self.compute_sine_cosine_encoding_nerf(
                        intersection_features.view(b*num_joints, -1, 3),
                        self.input_harmonic_l,
                    )
                elif self.apply_sine_encoding_on_points:
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

            if (
                self.joint_confidence_token_residual
                or self.joint_geometry_token_residual
                or self.joint_confidence_conditional_residual
                or self.joint_geometry_conditional_residual
            ):
                residual_tokens = x.view(b, num_joints, num_points, -1)
                if self.joint_confidence_token_residual:
                    if conf is None:
                        raise RuntimeError(
                            'RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL requires '
                            'detector confidence in the input'
                        )
                    confidence_scalar = conf.clamp(0, 1).squeeze(-1)
                    confidence_scalar = (
                        confidence_scalar
                        - confidence_scalar.mean(dim=-1, keepdim=True)
                    )
                    residual_tokens = (
                        self.joint_confidence_token_residual_module(
                            residual_tokens, confidence_scalar
                        )
                    )
                    if not hasattr(self, '_joint_confidence_token_residual_logged'):
                        print(
                            '[JOINT_CONFIDENCE_TOKEN_RESIDUAL] enabled=1 '
                            'injection=pre_vft per_joint_zero_init=1',
                            flush=True,
                        )
                        self._joint_confidence_token_residual_logged = True
                if self.joint_confidence_conditional_residual:
                    if conf is None:
                        raise RuntimeError(
                            'RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL '
                            'requires detector confidence in the input'
                        )
                    confidence_scalar = conf.clamp(0, 1).squeeze(-1)
                    confidence_scalar = (
                        confidence_scalar
                        - confidence_scalar.mean(dim=-1, keepdim=True)
                    )
                    residual_tokens = (
                        self.joint_confidence_conditional_residual_module(
                            residual_tokens, confidence_scalar
                        )
                    )
                    if not hasattr(self, '_joint_confidence_conditional_residual_logged'):
                        print(
                            '[JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL] enabled=1 '
                            'injection=pre_vft zero_output_mlp=1',
                            flush=True,
                        )
                        self._joint_confidence_conditional_residual_logged = True
                if self.joint_geometry_token_residual:
                    if _raw_dir is None or _raw_int is None:
                        raise RuntimeError(
                            'RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL requires '
                            'raw calibrated rays'
                        )
                    geometry_scalar = geometry_joint_view_reliability_logits(
                        _raw_dir, _raw_int
                    )
                    residual_tokens = self.joint_geometry_token_residual_module(
                        residual_tokens, geometry_scalar
                    )
                    if not hasattr(self, '_joint_geometry_token_residual_logged'):
                        print(
                            '[JOINT_GEOMETRY_TOKEN_RESIDUAL] enabled=1 '
                            'injection=pre_vft per_joint_zero_init=1',
                            flush=True,
                        )
                        self._joint_geometry_token_residual_logged = True
                if self.joint_geometry_conditional_residual:
                    if _raw_dir is None or _raw_int is None:
                        raise RuntimeError(
                            'RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL '
                            'requires raw calibrated rays'
                        )
                    geometry_scalar = geometry_joint_view_reliability_logits(
                        _raw_dir, _raw_int
                    )
                    residual_tokens = (
                        self.joint_geometry_conditional_residual_module(
                            residual_tokens, geometry_scalar
                        )
                    )
                    if not hasattr(self, '_joint_geometry_conditional_residual_logged'):
                        print(
                            '[JOINT_GEOMETRY_CONDITIONAL_RESIDUAL] enabled=1 '
                            'injection=pre_vft zero_output_mlp=1',
                            flush=True,
                        )
                        self._joint_geometry_conditional_residual_logged = True
                x = residual_tokens.reshape(b * num_joints, num_points, -1)

            if self.semantic_graph_pre_vft != 'off':
                semantic_tokens = x.view(b, num_joints, num_points, -1)
                semantic_tokens = self.semantic_graph_encoder(
                    semantic_tokens,
                    coordinates=semantic_graph_coordinates,
                )
                x = semantic_tokens.reshape(b * num_joints, num_points, -1)
                if not hasattr(self, '_semantic_graph_logged'):
                    print(
                        '[SEMANTIC_GRAPH_PRE_VFT] '
                        f'mode={self.semantic_graph_pre_vft} '
                        f'depth={self.semantic_graph_depth} '
                        'source=SGraFormer_AAAI2024 per_view_shared=1',
                        flush=True,
                    )
                    self._semantic_graph_logged = True

            # Keep the per-view encoded observations for the optional
            # parallel joint-query residual.  It is deliberately captured
            # before RUMPL's fusion-token VFT so the new decoder has a direct
            # path to every camera token instead of another compressed token.
            query_residual_memory = None
            if self.gbt_query_residual:
                query_residual_memory = x.view(
                    b, num_joints, num_points, -1
                )

            if self.gbt_set_decoder:
                set_x = x.view(b, num_joints, num_points, -1)
                set_x = set_x + self.GBT_set_joint_embed
                set_x = set_x.reshape(b, num_joints * num_points, -1)
                num_set_tokens = num_joints * num_points

                set_conf_bias = None
                set_geom_distance = None
                if self.gbt_set_biased:
                    if conf is not None:
                        flat_conf = conf.reshape(b, num_set_tokens).clamp(0, 1)
                        set_conf_bias = flat_conf[:, None, :].expand(
                            b, num_set_tokens, num_set_tokens
                        )
                    if _raw_dir is not None:
                        set_geom_distance = pairwise_ray_distance(
                            _raw_dir.reshape(b, num_set_tokens, 3),
                            _raw_int.reshape(b, num_set_tokens, 3),
                        )

                drop_rate = float(os.environ.get('RUMPL_GBT_SET_TOKEN_DROPOUT', '0'))
                if not 0.0 <= drop_rate < 1.0:
                    raise ValueError(
                        'RUMPL_GBT_SET_TOKEN_DROPOUT must be in [0, 1)'
                    )
                set_key_padding_mask = None
                set_mask_bias = None
                if is_training and drop_rate > 0:
                    set_key_padding_mask = (
                        torch.rand(b, num_set_tokens, device=x.device) < drop_rate
                    )
                    all_dropped = set_key_padding_mask.all(dim=1)
                    if all_dropped.any():
                        rows = all_dropped.nonzero(as_tuple=False).flatten()
                        retained = torch.randint(
                            num_set_tokens, (rows.numel(),), device=x.device
                        )
                        set_key_padding_mask[rows, retained] = False
                    set_mask_bias = torch.zeros(
                        b, num_set_tokens, num_set_tokens,
                        device=x.device, dtype=x.dtype,
                    )
                    set_mask_bias.masked_fill_(
                        set_key_padding_mask[:, None, :],
                        torch.finfo(x.dtype).min,
                    )

                if not hasattr(self, '_gbt_set_logged'):
                    print(
                        f'[GBT_SET] encoder_depth={self.gbt_set_depth} '
                        f'decoder_depth={self.gbt_set_decoder_depth} '
                        f'biased={int(self.gbt_set_biased)} '
                        f'plucker={int(self.gbt_set_plucker)} '
                        f'harmonic_l={self.gbt_set_harmonic_l} '
                        f'conf_concat={int(self.concat_confidence)} '
                        f'token_dropout={drop_rate} '
                        f'tokens={num_set_tokens} views={num_points}',
                        flush=True,
                    )
                    self._gbt_set_logged = True

                for block in self.GBT_set_encoder:
                    set_x = block(
                        set_x,
                        attn_bias=set_mask_bias,
                        conf_bias=set_conf_bias,
                        geom_distance=set_geom_distance,
                    )
                set_x = self.GBT_set_encoder_norm(set_x)
                queries = self.GBT_joint_queries.expand(b, -1, -1)
                decoded = self.GBT_set_decoder(
                    queries,
                    set_x,
                    memory_key_padding_mask=set_key_padding_mask,
                )
                set_output = self.GBT_set_head(decoded)
                if tri_anchor_point is not None:
                    set_output = set_output + tri_anchor_point
                if body_canonical_origin is not None:
                    set_output = body_canonical_pose_to_world(
                        set_output, body_canonical_origin,
                        body_canonical_basis,
                    )
                return set_output

            skeleton_view_logits = None
            if self.random_num_views and self.skeleton_view_reliability:
                skeleton_view_logits = self.Skeleton_view_reliability(
                    x.view(b, num_joints, num_points, -1)
                )
            confidence_view_logits = None
            if self.random_num_views and self.confidence_view_bias:
                if conf is None:
                    raise RuntimeError(
                        'RUMPL_CONFIDENCE_VIEW_BIAS requires detector '
                        'confidence in the input'
                    )
                confidence_view_logits = conf.clamp(0, 1).mean(dim=1).squeeze(-1)
                confidence_view_logits = (
                    confidence_view_logits
                    - confidence_view_logits.mean(dim=1, keepdim=True)
                )
            geometry_view_logits = None
            if self.random_num_views and self.geometry_view_bias:
                if _raw_dir is None or _raw_int is None:
                    raise RuntimeError(
                        'RUMPL_GEOMETRY_VIEW_BIAS requires raw calibrated rays'
                    )
                geometry_view_logits = geometry_view_reliability_logits(
                    _raw_dir, _raw_int
                )
            joint_confidence_view_logits = None
            if self.random_num_views and self.joint_confidence_view_bias:
                if conf is None:
                    raise RuntimeError(
                        'RUMPL_JOINT_CONFIDENCE_VIEW_BIAS requires detector '
                        'confidence in the input'
                    )
                joint_confidence_view_logits = conf.clamp(0, 1).squeeze(-1)
                joint_confidence_view_logits = (
                    joint_confidence_view_logits
                    - joint_confidence_view_logits.mean(dim=-1, keepdim=True)
                )
            joint_geometry_view_logits = None
            if self.random_num_views and self.joint_geometry_view_bias:
                if _raw_dir is None or _raw_int is None:
                    raise RuntimeError(
                        'RUMPL_JOINT_GEOMETRY_VIEW_BIAS requires raw '
                        'calibrated rays'
                    )
                joint_geometry_view_logits = geometry_joint_view_reliability_logits(
                    _raw_dir, _raw_int
                )

            if self.random_num_views and self.relative_view_fusion:
                relative_x = x.view(b, num_joints, num_points, -1)
                relative_x = self.Relative_view_fusion(relative_x)
                if not hasattr(self, '_relative_view_logged'):
                    print(
                        f'[RELATIVE_VIEW] enabled=1 gated=1 '
                        f'joints={num_joints} views={num_points}',
                        flush=True,
                    )
                    self._relative_view_logged = True
                x = relative_x.reshape(b * num_joints, num_points, -1)

            if self.random_num_views and self.global_jv_depth > 0:
                global_residual = x.view(b, num_joints, num_points, -1)
                global_x = global_residual
                global_x = global_x + self.Global_joint_embed
                global_x = global_x.reshape(b, num_joints * num_points, -1)
                global_conf_bias = None
                global_geom_distance = None
                if self.global_jv_biased:
                    if conf is not None:
                        flat_conf = conf.reshape(b, num_joints * num_points).clamp(0, 1)
                        global_conf_bias = flat_conf[:, None, :].expand(
                            b, num_joints * num_points, num_joints * num_points
                        )
                    if _raw_dir is not None:
                        flat_direction = _raw_dir.reshape(b, num_joints * num_points, 3)
                        flat_point = _raw_int.reshape(b, num_joints * num_points, 3)
                        global_geom_distance = pairwise_ray_distance(flat_direction, flat_point)
                if not hasattr(self, '_global_jv_logged'):
                    print(
                        f'[GLOBAL_JV] depth={self.global_jv_depth} biased={int(self.global_jv_biased)} '
                        f'gated={int(self.global_jv_gated)} tokens={num_joints * num_points} '
                        f'joints={num_joints} views={num_points}',
                        flush=True,
                    )
                    self._global_jv_logged = True
                for block in self.blocks_global_joint_view:
                    global_x = block(
                        global_x,
                        conf_bias=global_conf_bias,
                        geom_distance=global_geom_distance,
                    )
                global_x = self.Global_joint_view_norm(global_x).reshape(
                    b, num_joints, num_points, -1
                )
                if self.global_jv_gated:
                    global_x = global_residual + self.global_jv_gate * (
                        global_x - global_residual
                    )
                x = global_x.reshape(b * num_joints, num_points, -1)

            if self.random_num_views and self.mtf_source_norm_fusion:
                view_tokens = x.view(b, num_joints, num_points, -1)
                mtf_conf = conf if self.mtf_source_norm_confidence else None
                if self.mtf_source_norm_residual:
                    # Keep the established fusion-token VFT as the main path;
                    # MTF contributes a zero-initialized per-view residual.
                    x = self.MTF_source_norm_fusion.forward_per_view(
                        view_tokens, mtf_conf
                    ).reshape(b * num_joints, num_points, -1)
                else:
                    x = self.MTF_source_norm_fusion(view_tokens, mtf_conf)
                    mtf_source_norm_fused = True
                if not hasattr(self, '_mtf_source_norm_logged'):
                    print(
                        '[MTF_SOURCE_NORM] enabled=1 '
                        f'mode={"residual_before_vft" if self.mtf_source_norm_residual else "replacement"} '
                        f'confidence={int(self.mtf_source_norm_confidence)} '
                        'camera_id_embedding=0 pairwise_relation=source_normalized',
                        flush=True,
                    )
                    self._mtf_source_norm_logged = True
            
            if not self.random_num_views:
                if self.add_view_enc:
                    x += self.View_enc_learned
                x = x.view(b * num_joints, num_points, -1)
            
            if self.random_num_views and self.skip_vft:
                bj = b * num_joints
                x_views = x.view(bj, num_points, -1)
                if conf is not None:
                    w = conf.reshape(bj, num_points, 1).clamp(min=1e-6)
                    w = w / w.sum(dim=1, keepdim=True)
                    fused = (x_views * w).sum(dim=1)
                else:
                    fused = x_views.mean(dim=1)
                x = fused.view(b, num_joints, -1)
                if not hasattr(self, '_skip_vft_logged'):
                    print(
                        '[RUMPL_SKIP_VFT] mode=conf_weighted_mean '
                        f'views={num_points}',
                        flush=True,
                    )
                    self._skip_vft_logged = True
            elif self.random_num_views and not mtf_source_norm_fused:
                # Append the fusion token to the input
                fusion_token = self.fusion_token.expand(b*num_joints, -1, -1)  # Shape: [batch, 1, embed_dim]
                if self.geometry_uncertainty_token:
                    if geometry_uncertainty is None:
                        raise RuntimeError('geometry uncertainty was not computed')
                    uncertainty_embedding = self.geometry_uncertainty_to_embedding(
                        geometry_uncertainty.reshape(b * num_joints, 4)
                    ).unsqueeze(1)
                    fusion_token = fusion_token + uncertainty_embedding
                    if not hasattr(self, '_geometry_uncertainty_logged'):
                        print(
                            '[GEOMETRY_UNCERTAINTY_TOKEN] enabled=1 '
                            'features=log_normalized_eigenvalues+mean_confidence '
                            'injection=fusion_token zero_init=1',
                            flush=True,
                        )
                        self._geometry_uncertainty_logged = True
                x = x.view(b*num_joints, num_points, -1)
                x = torch.cat([fusion_token, x], dim=1)  # Shape: [batch, num_tokens + 1, embed_dim]
                _token_drop = float(os.environ.get('GBT_TOKEN_DROPOUT', 0.0))
                if not 0.0 <= _token_drop < 1.0:
                    raise ValueError('GBT_TOKEN_DROPOUT must be in [0, 1)')
                token_drop_epochs = int(os.environ.get('GBT_TOKEN_DROPOUT_EPOCHS', 0))
                if token_drop_epochs < 0:
                    raise ValueError('GBT_TOKEN_DROPOUT_EPOCHS must be non-negative')
                current_epoch = int(kwargs.get('epoch', 0))
                token_drop_active = token_drop_epochs == 0 or current_epoch < token_drop_epochs
                token_keep = None
                if is_training and token_drop_active and _token_drop > 0 and num_points > 1:
                    bj = b * num_joints
                    token_keep = torch.rand(bj, num_points, device=x.device) >= _token_drop
                    all_dropped = token_keep.sum(dim=1) == 0
                    if all_dropped.any():
                        rows = all_dropped.nonzero(as_tuple=False).flatten()
                        retained = torch.randint(num_points, (rows.numel(),), device=x.device)
                        token_keep[rows, retained] = True
                    keep = token_keep.unsqueeze(-1).to(dtype=x.dtype)
                    x[:, 1:, :] = x[:, 1:, :] * keep
                    if conf is not None:
                        conf = conf.reshape(bj, num_points, 1) * keep
                    if not hasattr(self, '_gbt_token_dropout_logged'):
                        print(
                            f'[GBT_TOKEN_DROPOUT] rate={_token_drop} key_removal=1 '
                            f'keep_at_least_one_per_joint=1 views={num_points} '
                            f'active_epochs={token_drop_epochs or "all"}',
                            flush=True,
                        )
                        self._gbt_token_dropout_logged = True
                
                if self.add_view_enc:
                    x += self.View_enc_learned[:, :num_points + 1]
            
                x = self.pos_drop(x)
                cw = None
                attn_bias = None
                learned_conf_bias = None
                learned_geom_distance = None
                _caa = float(os.environ.get('CAA_LAMBDA', 0.0))
                if _caa > 0 and conf is not None:
                    bj = b * num_joints
                    cf = conf.reshape(bj, num_points).clamp(0, 1)
                    w_view = 1.0 - _caa * (1.0 - cf)
                    w = torch.cat([torch.ones(bj, 1, device=x.device, dtype=x.dtype), w_view], dim=1)
                    N = num_points + 1
                    cw = w[:, None, :].expand(bj, N, N)
                _dp = float(os.environ.get('DEPRO_LAMBDA', 0.0))
                if _dp > 0 and _raw_dir is not None and num_points >= 2:
                    bj = b * num_joints
                    d = _raw_dir.reshape(bj, num_points, 3)
                    o = _raw_int.reshape(bj, num_points, 3)
                    dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
                    cr = torch.cross(
                        dn[:, :, None, :].expand(bj, num_points, num_points, 3),
                        dn[:, None, :, :].expand(bj, num_points, num_points, 3),
                        dim=-1,
                    )
                    diff = o[:, None, :, :] - o[:, :, None, :]
                    dist = (diff * cr).sum(-1).abs() / (cr.norm(dim=-1) + 1e-8)
                    eye = torch.eye(num_points, device=x.device, dtype=torch.bool)
                    dist = dist.masked_fill(eye[None], 0.0)
                    consist = dist.sum(-1) / max(num_points - 1, 1)
                    relia = torch.exp(-consist / (consist.mean(dim=1, keepdim=True) + 1e-6))
                    relia = relia / (relia.mean(dim=1, keepdim=True) + 1e-8)
                    w_view = 1.0 - _dp * (1.0 - relia)
                    w = torch.cat([torch.ones(bj, 1, device=x.device, dtype=x.dtype), w_view], dim=1)
                    N = num_points + 1
                    cw = w[:, None, :].expand(bj, N, N)
                _gbt_conf = float(os.environ.get('GBT_CONF_BIAS', 0.0))
                _gbt_geom = float(os.environ.get('GBT_GEOM_BIAS', 0.0))
                if self.gbt_learnable_bias:
                    bj = b * num_joints
                    N = num_points + 1
                    if self.gbt_learnable_conf and conf is not None:
                        cf = conf.reshape(bj, num_points).clamp(0, 1).to(dtype=x.dtype)
                        key_conf = torch.cat(
                            [torch.zeros(bj, 1, device=x.device, dtype=x.dtype), cf], dim=1
                        )
                        learned_conf_bias = key_conf[:, None, :].expand(bj, N, N)
                    if self.gbt_learnable_geom and _raw_dir is not None and num_points >= 2:
                        direction = _raw_dir.reshape(bj, num_points, 3)
                        point = _raw_int.reshape(bj, num_points, 3)
                        learned_geom_distance = geometry_distance_with_fusion_token(
                            direction,
                            point,
                            direct_fusion=os.environ.get('GBT_FUSION_GEOM', '0') == '1',
                        ).to(dtype=x.dtype)
                elif _gbt_conf > 0 or _gbt_geom > 0:
                    bj = b * num_joints
                    N = num_points + 1
                    attn_bias = torch.zeros(bj, N, N, device=x.device, dtype=x.dtype)
                    if _gbt_conf > 0 and conf is not None:
                        cf = conf.reshape(bj, num_points).clamp(0, 1).to(dtype=x.dtype)
                        key_conf = torch.cat(
                            [torch.zeros(bj, 1, device=x.device, dtype=x.dtype), cf], dim=1
                        )
                        attn_bias = attn_bias + _gbt_conf * key_conf[:, None, :]
                    if _gbt_geom > 0 and _raw_dir is not None and num_points >= 2:
                        d = _raw_dir.reshape(bj, num_points, 3)
                        o = _raw_int.reshape(bj, num_points, 3)
                        dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
                        cr = torch.cross(
                            dn[:, :, None, :].expand(bj, num_points, num_points, 3),
                            dn[:, None, :, :].expand(bj, num_points, num_points, 3),
                            dim=-1,
                        )
                        diff = o[:, None, :, :] - o[:, :, None, :]
                        dist = (diff * cr).sum(-1).abs() / (cr.norm(dim=-1) + 1e-8)
                        eye = torch.eye(num_points, device=x.device, dtype=torch.bool)
                        dist = dist.masked_fill(eye[None], 0.0)
                        consist = dist.sum(-1) / max(num_points - 1, 1)
                        penalty = consist / (consist.mean(dim=1, keepdim=True) + 1e-6)
                        pair_penalty = dist / (dist.mean(dim=(1, 2), keepdim=True) + 1e-6)
                        geom_bias = torch.zeros_like(attn_bias)
                        geom_bias[:, :, 1:] = geom_bias[:, :, 1:] - _gbt_geom * penalty[:, None, :]
                        geom_bias[:, 1:, 1:] = geom_bias[:, 1:, 1:] - _gbt_geom * pair_penalty
                        attn_bias = attn_bias + geom_bias
                if token_keep is not None:
                    bj = b * num_joints
                    N = num_points + 1
                    dropped_keys = torch.zeros(bj, N, N, device=x.device, dtype=torch.bool)
                    dropped_keys[:, :, 1:] = (~token_keep)[:, None, :]
                    token_drop_bias = torch.zeros(bj, N, N, device=x.device, dtype=x.dtype)
                    token_drop_bias.masked_fill_(dropped_keys, torch.finfo(x.dtype).min)
                    attn_bias = token_drop_bias if attn_bias is None else attn_bias + token_drop_bias
                _mask_rate = float(os.environ.get('VFT_FULL_RANDOM_MASK', 0.0))
                _mask_min_views = int(os.environ.get('VFT_MASK_MIN_VIEWS', 2))
                _mask_diagonal = int(os.environ.get('VFT_MASK_DIAGONAL', '1')) == 1
                _apply_mask = kwargs.get('apply_vft_mask', is_training)
                if _apply_mask and _mask_rate > 0 and num_points >= _mask_min_views:
                    bj = b * num_joints
                    N = num_points + 1
                    random_mask = build_view_attention_mask(
                        batch_joints=bj,
                        num_views=num_points,
                        mask_rate=_mask_rate,
                        device=x.device,
                        mask_diagonal=_mask_diagonal,
                    )
                    mask_bias = torch.zeros(bj, N, N, device=x.device, dtype=x.dtype)
                    mask_bias.masked_fill_(random_mask, torch.finfo(x.dtype).min)
                    attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias
                vft_blocks = self.blocks_view_fusion
                if self.vft_depth_override > 0:
                    vft_blocks = self.blocks_view_fusion[: self.vft_depth_override]
                    if not hasattr(self, '_vft_depth_logged'):
                        print(
                            f'[RUMPL_VFT_DEPTH] layers={self.vft_depth_override} '
                            f'of {len(self.blocks_view_fusion)}',
                            flush=True,
                        )
                        self._vft_depth_logged = True
                for layer_index, blk in enumerate(vft_blocks):
                    layer_attn_bias = attn_bias
                    if skeleton_view_logits is not None:
                        reliability_bias = fusion_token_source_attention_bias(
                            skeleton_view_logits,
                            num_joints,
                            self.skeleton_view_reliability_gates[layer_index],
                        )
                        layer_attn_bias = (
                            reliability_bias if layer_attn_bias is None
                            else layer_attn_bias + reliability_bias
                        )
                    if confidence_view_logits is not None:
                        confidence_bias = fusion_token_source_attention_bias(
                            confidence_view_logits,
                            num_joints,
                            self.confidence_view_bias_gates[layer_index],
                        )
                        layer_attn_bias = (
                            confidence_bias if layer_attn_bias is None
                            else layer_attn_bias + confidence_bias
                        )
                    if geometry_view_logits is not None:
                        geometry_bias = fusion_token_source_attention_bias(
                            geometry_view_logits,
                            num_joints,
                            self.geometry_view_bias_gates[layer_index],
                        )
                        layer_attn_bias = (
                            geometry_bias if layer_attn_bias is None
                            else layer_attn_bias + geometry_bias
                        )
                    if joint_confidence_view_logits is not None:
                        joint_confidence_bias = (
                            joint_fusion_token_source_attention_bias(
                                joint_confidence_view_logits,
                                self.joint_confidence_view_bias_gates[layer_index],
                            )
                        )
                        layer_attn_bias = (
                            joint_confidence_bias if layer_attn_bias is None
                            else layer_attn_bias + joint_confidence_bias
                        )
                    if joint_geometry_view_logits is not None:
                        joint_geometry_bias = (
                            joint_fusion_token_source_attention_bias(
                                joint_geometry_view_logits,
                                self.joint_geometry_view_bias_gates[layer_index],
                            )
                        )
                        layer_attn_bias = (
                            joint_geometry_bias if layer_attn_bias is None
                            else layer_attn_bias + joint_geometry_bias
                        )
                    x = blk(
                        x, cw, layer_attn_bias,
                        learned_conf_bias, learned_geom_distance,
                    )
                if skeleton_view_logits is not None and not hasattr(
                    self, '_skeleton_view_reliability_logged'
                ):
                    print(
                        '[SKELETON_VIEW_RELIABILITY] enabled=1 '
                        'target=fusion_query_to_view_keys '
                        f'layers={len(vft_blocks)} views={num_points} '
                        'zero_init_gates=1',
                        flush=True,
                    )
                    self._skeleton_view_reliability_logged = True
                if confidence_view_logits is not None and not hasattr(
                    self, '_confidence_view_bias_logged'
                ):
                    print(
                        '[CONFIDENCE_VIEW_BIAS] enabled=1 '
                        'target=fusion_query_to_view_keys '
                        f'layers={len(vft_blocks)} views={num_points} '
                        'zero_init_gates=1 statistic=mean_joint_confidence',
                        flush=True,
                    )
                    self._confidence_view_bias_logged = True
                if geometry_view_logits is not None and not hasattr(
                    self, '_geometry_view_bias_logged'
                ):
                    print(
                        '[GEOMETRY_VIEW_BIAS] enabled=1 '
                        'target=fusion_query_to_view_keys '
                        f'layers={len(vft_blocks)} views={num_points} '
                        'zero_init_gates=1 statistic=normalized_line_distance',
                        flush=True,
                    )
                    self._geometry_view_bias_logged = True
                if joint_confidence_view_logits is not None and not hasattr(
                    self, '_joint_confidence_view_bias_logged'
                ):
                    print(
                        '[JOINT_CONFIDENCE_VIEW_BIAS] enabled=1 '
                        'target=fusion_query_to_joint_view_keys '
                        f'layers={len(vft_blocks)} views={num_points} '
                        'zero_init_gates=1 statistic=per_joint_confidence',
                        flush=True,
                    )
                    self._joint_confidence_view_bias_logged = True
                if joint_geometry_view_logits is not None and not hasattr(
                    self, '_joint_geometry_view_bias_logged'
                ):
                    print(
                        '[JOINT_GEOMETRY_VIEW_BIAS] enabled=1 '
                        'target=fusion_query_to_joint_view_keys '
                        f'layers={len(vft_blocks)} views={num_points} '
                        'zero_init_gates=1 statistic=per_joint_normalized_line_distance',
                        flush=True,
                    )
                    self._joint_geometry_view_bias_logged = True
                x = self.View_norm(x)
                x = x[:, 0, :]
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
        
        
        x = self.pos_drop(x)
        if not hasattr(self, '_pft_repeat_logged'):
            print(
                f'[PFT_REPEAT_LAST] enabled={int(self.pft_repeat_last)} '
                f'skip_pft={int(self.skip_pft)} '
                f'graformer={self.graformer_pft_mode}',
                flush=True,
            )
            self._pft_repeat_logged = True
        if self.skip_pft:
            if not hasattr(self, '_skip_pft_logged'):
                print('[RUMPL_SKIP_PFT] head_direct_from_joint_tokens=1', flush=True)
                self._skip_pft_logged = True
            x = self.Spatial_norm(x)
        elif self.graformer_pft_mode != 'off':
            if not hasattr(self, '_graformer_pft_logged'):
                print(
                    '[GRAFORMER_PFT] enabled=1 '
                    f'mode={self.graformer_pft_mode} '
                    f'depth={self.graformer_pft_depth} heads=4 dropout=0.25 '
                    'position=after_VFT replacement=RUMPL_PFT',
                    flush=True,
                )
                self._graformer_pft_logged = True
            x = self.graformer_pft(x)
            x = self.Spatial_norm(x)
        else:
            x = apply_pose_fusion_blocks(
                x,
                self.blocks,
                repeat_last=self.pft_repeat_last,
            )
            x = self.Spatial_norm(x)
        
        x = x.view(b, num_joints, -1)

        if self.post_pft_graph_residual:
            x = self.pft_graph_residual(x)
        pft_features = x
        geometry_output_condition = None
        if self.post_pft_geometry_conditional_residual:
            if _raw_dir is None or _raw_int is None:
                raise RuntimeError(
                    'RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL requires '
                    'raw calibrated rays'
                )
            geometry_output_condition = ray_normal_matrix_features(
                _raw_dir,
                conf.clamp(0, 1) if conf is not None else None,
            )
        x = self.head(pft_features)
        if self.joint_specific_head:
            x = x + self.per_joint_head(pft_features)
        if self.post_pft_geometry_conditional_residual:
            x = x + self.post_pft_geometry_conditional_module(
                pft_features, geometry_output_condition
            )

        x = x.view(b, -1, 3)
        if tri_anchor_point is not None:
            if self.per_joint_residual_gate:
                x = x * self.residual_joint_gate
            x = x + self.tri_anchor_gate * tri_anchor_point

        if self.gbt_query_residual:
            if query_residual_memory is None:
                raise RuntimeError(
                    'GBT query residual memory was not constructed'
                )
            if tri_anchor_point is None:
                anchor_for_query = x.new_zeros(b, num_joints, 3)
            else:
                anchor_for_query = tri_anchor_point
            memory = (
                query_residual_memory
                + self.gbt_query_joint_memory_embed
            )
            query = self.gbt_query_joint_queries.expand(b, -1, -1)
            query = query + self.gbt_query_anchor_embed(anchor_for_query)
            if self.gbt_query_residual_global:
                # Global memory follows GBT/MVGFormer's joint-query decoder:
                # every learned joint query can inspect all joint/view tokens.
                memory = memory.reshape(b, num_joints * num_points, -1)
                decoded = self.gbt_query_residual_decoder(query, memory)
            else:
                # Controlled local ablation: each joint query only sees its
                # own views.  This separates cross-joint context from the
                # direct query-to-view path.
                local_memory = memory.reshape(
                    b * num_joints, num_points, -1
                )
                local_query = query.reshape(b * num_joints, 1, -1)
                decoded = self.gbt_query_residual_decoder(
                    local_query, local_memory
                ).reshape(b, num_joints, -1)
            raw_residual = self.gbt_query_residual_head(decoded)
            residual = self.gbt_query_residual_max_delta * torch.tanh(
                raw_residual
            )
            x = x + residual
            if not hasattr(self, '_gbt_query_residual_logged'):
                print(
                    '[GBT_QUERY_RESIDUAL] enabled=1 '
                    f'global={int(self.gbt_query_residual_global)} '
                    f'depth={self.gbt_query_residual_depth} '
                    f'max_delta={self.gbt_query_residual_max_delta} '
                    'position=parallel_pre_vft_memory_zero_init=1',
                    flush=True,
                )
                self._gbt_query_residual_logged = True

        if body_canonical_origin is not None:
            x = body_canonical_pose_to_world(
                x, body_canonical_origin, body_canonical_basis
            )
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
                if isinstance(pretrained_state_dict, dict) and 'state_dict' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['state_dict']
                pretrained_state_dict = {
                    (k[7:] if k.startswith('module.') else k): v
                    for k, v in pretrained_state_dict.items()
                }
                merged, skipped = merge_pretrained_into_model_state(
                    self.state_dict(), pretrained_state_dict, strict_shapes=False
                )
                if skipped:
                    logger.info(
                        '=> init_weights skipped non-adaptable keys: %s', skipped
                    )
                self.load_state_dict(merged, strict=False)
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
