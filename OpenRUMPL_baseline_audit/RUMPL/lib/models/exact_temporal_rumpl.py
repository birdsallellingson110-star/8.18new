"""Post-VFT temporal refinement for the audited RUMPL baseline."""

import math

import torch
import torch.nn as nn


class DeltaTEncoder(nn.Module):
    def __init__(self, dim, max_period=2.0):
        super().__init__()
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(0, dim // 2, dtype=torch.float32)
            / (dim // 2)
        )
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(dim, dim)

    def forward(self, delta_t):
        values = delta_t.unsqueeze(-1) * self.frequencies
        return self.projection(torch.cat([values.sin(), values.cos()], dim=-1))


class TemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class TemporalRefiner(nn.Module):
    """Refine the center-frame VFT feature while starting as an exact identity."""

    def __init__(
        self, dim, num_heads=8, depth=2, max_period=2.0, motion_only=False
    ):
        super().__init__()
        self.motion_only = motion_only
        self.time_embedding = DeltaTEncoder(dim, max_period=max_period)
        self.blocks = nn.ModuleList(
            [TemporalBlock(dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if motion_only:
            self.motion_gate_log_scale = nn.Parameter(
                torch.tensor(math.log(math.e - 1.0))
            )

    def forward(self, features, delta_t, center_index):
        # features: (B, L, J, D), delta_t: (B, L)
        batch_size, num_frames, num_joints, dim = features.shape
        center_feature = features[:, center_index]
        temporal_input = features
        motion_gate = None
        if self.motion_only:
            temporal_input = features - center_feature[:, None]
            motion_energy = temporal_input.square().mean(dim=(1, 3)).sqrt()
            motion_scale = torch.nn.functional.softplus(
                self.motion_gate_log_scale
            )
            motion_gate = torch.tanh(motion_scale * motion_energy)
        x = temporal_input.permute(0, 2, 1, 3).reshape(
            batch_size * num_joints, num_frames, dim
        )
        time_embedding = self.time_embedding(delta_t)
        x = x + time_embedding[:, None].expand(
            batch_size, num_joints, num_frames, dim
        ).reshape(batch_size * num_joints, num_frames, dim)
        for block in self.blocks:
            x = block(x)
        delta = self.output(self.norm(x[:, center_index]))
        delta = delta.reshape(batch_size, num_joints, dim)
        if motion_gate is not None:
            delta = delta * motion_gate.unsqueeze(-1)
        return center_feature + delta


class ExactTemporalRUMPL(nn.Module):
    def __init__(
        self,
        config_path,
        checkpoint_path,
        temporal_depth=2,
        temporal_heads=8,
        freeze_backbone=True,
        motion_only=False,
    ):
        super().__init__()
        from core.config import config, update_config
        from models.multiview_rumpl import get_multiview_rumpl_net

        update_config(config_path)
        rumpl = get_multiview_rumpl_net(config, is_train=False)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        rumpl.load_state_dict(state, strict=True)
        self.backbone = rumpl.features
        self.dim = self.backbone.Spatial_pos_embed.shape[-1]
        self.temporal = TemporalRefiner(
            self.dim,
            num_heads=temporal_heads,
            depth=temporal_depth,
            motion_only=motion_only,
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode_rays(self, rays, confidence):
        backbone = self.backbone
        direction = rays[..., :3]
        point = rays[..., 3:6]
        if backbone.concat_direction_and_intersection_first:
            token = backbone.encoding_to_embedding(torch.cat([direction, point], dim=-1))
        else:
            direction = backbone.encoding_to_embedding(direction)
            if backbone.not_use_intersection_features:
                token = direction
            else:
                point = backbone.encoding_to_embedding(point)
                token = torch.cat([direction, point], dim=-1)
        if backbone.concat_confidence:
            token = torch.cat(
                [token, backbone.confidence_to_embedding(confidence)], dim=-1
            )
        return token

    def view_fusion(self, token):
        backbone = self.backbone
        batch_size, num_joints, num_views, dim = token.shape
        x = token.reshape(batch_size * num_joints, num_views, dim)
        fusion_token = backbone.fusion_token.expand(batch_size * num_joints, -1, -1)
        x = torch.cat([fusion_token, x], dim=1)
        if backbone.add_view_enc:
            x = x + backbone.View_enc_learned[:, : num_views + 1]
        x = backbone.pos_drop(x)
        for block in backbone.blocks_view_fusion:
            x = block(x)
        x = backbone.View_norm(x)
        return x[:, 0].reshape(batch_size, num_joints, dim)

    def pose_fusion(self, feature):
        backbone = self.backbone
        batch_size, num_joints, _ = feature.shape
        x = backbone.pos_drop(feature + backbone.Spatial_pos_embed)
        for index, block in enumerate(backbone.blocks):
            x = block(x)
            if index == len(backbone.blocks) - 1 and not backbone.fix_pft_last_block:
                x = block(x)
        x = backbone.Spatial_norm(x)
        return backbone.head(x).reshape(batch_size, num_joints, 3)

    def frame_features(self, rays, confidence):
        # rays: (B, J, V, L, 6), confidence: (B, J, V, L, 1)
        batch_size, num_joints, num_views, num_frames, _ = rays.shape
        token = self.encode_rays(rays, confidence)
        token = token.permute(0, 3, 1, 2, 4).reshape(
            batch_size * num_frames, num_joints, num_views, self.dim
        )
        fused = self.view_fusion(token)
        return fused.reshape(batch_size, num_frames, num_joints, self.dim)

    def forward(
        self,
        rays,
        confidence,
        delta_t,
        center_index=None,
        no_temporal=False,
        return_baseline=False,
    ):
        num_frames = rays.shape[3]
        if center_index is None:
            center_index = num_frames // 2
        features = self.frame_features(rays, confidence)
        if no_temporal:
            center_feature = features[:, center_index]
        else:
            center_feature = self.temporal(features, delta_t, center_index)
        prediction = self.pose_fusion(center_feature)
        if return_baseline:
            baseline = self.pose_fusion(features[:, center_index])
            return prediction, baseline
        return prediction
