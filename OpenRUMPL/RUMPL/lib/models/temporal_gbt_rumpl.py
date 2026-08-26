"""Joint-view-time Transformer inserted before RUMPL view fusion.

This module is the controlled bridge between Geometry-Biased Transformer
(GBT) and RUMPL:

* GBT contribution: global attention over every time/joint/view observation,
  harmonic relative-time encoding, optional confidence and ray-distance
  biases, and token dropout.
* retained RUMPL contribution: ray/confidence embedding, per-joint VFT fusion,
  PFT body-joint transformer, 3-D head, and optional triangulation anchor.

The unbiased and biased variants have the same architecture.  Consequently,
their paired comparison isolates the two attention biases instead of
confounding them with a backbone replacement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiview_rumpl import Block, pairwise_ray_distance
from models.stvft.delta_t_encoder import DeltaTEncoder


class PreVFTTemporalAdapter(nn.Module):
    """Temporal attention on each joint/camera ray track before VFT.

    The old ``mixste-*`` modes apply temporal attention after RUMPL has
    already pooled the camera axis.  This adapter deliberately keeps the
    ``(joint, view)`` identity and only mixes the time axis.  It is therefore
    permutation equivariant in camera order and can be used with variable K.
    Its output projection is zero initialized, so construction is an exact
    identity around a pretrained RUMPL checkpoint.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        temporal_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("pre-vft temporal depth must be positive")
        if dim % num_heads:
            raise ValueError(
                f"temporal token dimension {dim} is not divisible by {num_heads} heads"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, temporal_length, dim)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=0.0,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _position(self, time: int) -> torch.Tensor:
        position = self.temporal_pos_embed
        if position.shape[1] != time:
            position = F.interpolate(
                position.transpose(1, 2),
                size=time,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return position

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        if token.ndim != 5:
            raise ValueError(
                f"pre-vft temporal adapter expects (B,T,J,V,D), got {tuple(token.shape)}"
            )
        batch, time, joints, views, dim = token.shape
        track = token.permute(0, 2, 3, 1, 4).reshape(
            batch * joints * views, time, dim
        )
        track = track + self._position(time)
        for block in self.blocks:
            track = block(track)
        delta = self.output(self.norm(track))
        delta = delta.reshape(batch, joints, views, time, dim).permute(
            0, 3, 1, 2, 4
        )
        return token + delta


def sample_sequence_views(
    x: torch.Tensor,
    num_views: int,
    generator: Optional[torch.Generator] = None,
):
    """Sample one camera subset per sequence and reuse it for every frame.

    Args:
        x: ``(B, T, J, V, C)`` tensor.
        num_views: number of cameras to retain.
    """

    batch, time, joints, total_views, channels = x.shape
    if not 1 <= num_views <= total_views:
        raise ValueError(
            f"num_views must be in [1, {total_views}], received {num_views}"
        )
    keys = torch.rand(
        batch, total_views, device=x.device, generator=generator
    )
    indices = keys.topk(num_views, dim=1, largest=False).indices.sort(dim=1).values
    gather = indices[:, None, None, :, None].expand(
        batch, time, joints, num_views, channels
    )
    return torch.gather(x, 3, gather), indices


def confidence_weighted_triangulation_anchor(
    direction: torch.Tensor,
    point: torch.Tensor,
    confidence: torch.Tensor,
    regularization: float,
    confidence_epsilon: float,
) -> torch.Tensor:
    """RUMPL H0 confidence-weighted least-squares ray anchor.

    Inputs are ``(B,T,J,V,3/1)`` and output is ``(B,T,J,3)``.
    """

    unit = F.normalize(direction, dim=-1, eps=1e-7)
    eye = torch.eye(3, device=direction.device, dtype=direction.dtype)
    projector = eye - unit.unsqueeze(-1) * unit.unsqueeze(-2)
    weights = confidence.clamp(0, 1) + confidence_epsilon
    weighted = weights.unsqueeze(-1) * projector
    lhs = weighted.sum(dim=3) + regularization * eye
    rhs = (weighted @ point.unsqueeze(-1)).sum(dim=3)
    return torch.linalg.solve(lhs, rhs).squeeze(-1)


class TemporalJointViewRUMPL(nn.Module):
    """Global joint-view-time refinement followed by the original RUMPL path."""

    def __init__(
        self,
        rumpl_model: nn.Module,
        depth: int = 3,
        num_heads: int = 8,
        biased: bool = False,
        token_dropout: float = 0.2,
        max_period_seconds: float = 2.0,
        conf_bias_init: float = 0.1,
        geom_bias_init: float = 0.1,
        residual_gate: bool = True,
        residual_scale: float = 0.1,
        fusion_mode: str = "global-residual",
        temporal_length: int = 9,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= token_dropout < 1.0:
            raise ValueError("token_dropout must be in [0, 1)")
        self.backbone = getattr(rumpl_model, "features", rumpl_model)
        self.num_joints = int(self.backbone.Spatial_pos_embed.shape[1])
        self.dim = int(self.backbone.Spatial_pos_embed.shape[-1])
        if self.dim % num_heads:
            raise ValueError(
                f"RUMPL token dimension {self.dim} is not divisible by {num_heads} heads"
            )
        self.biased = bool(biased)
        self.token_dropout = float(token_dropout)
        self.residual_gate = bool(residual_gate)
        if fusion_mode not in (
            "global-residual",
            "query-residual",
            "pre-vft-temporal",
            "mixste-ttb",
            "mixste-ttb-residual",
            "mixste-alternating",
            "mixste-pose-residual",
        ):
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.temporal_length = int(temporal_length)
        if self.temporal_length < 1:
            raise ValueError("temporal_length must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = float(residual_scale)
        if self.fusion_mode == "pre-vft-temporal":
            self.pre_vft_temporal = PreVFTTemporalAdapter(
                dim=self.dim,
                num_heads=num_heads,
                depth=depth,
                temporal_length=self.temporal_length,
                dropout=min(self.token_dropout, 0.10),
            )
        self.joint_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_joints, 1, self.dim)
        )
        self.time_encoder = DeltaTEncoder(
            d_model=self.dim, max_period=max_period_seconds
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dim,
                    num_heads=num_heads,
                    mlp_ratio=3.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    learnable_conf_bias=self.biased,
                    learnable_geom_bias=self.biased,
                    conf_bias_init=conf_bias_init,
                    geom_bias_init=geom_bias_init,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.dim)
        if self.residual_gate and self.fusion_mode == "global-residual":
            # Warm-started retained-RUMPL experiments must begin at the H0
            # function.  The global branch then earns its contribution rather
            # than destroying the pretrained VFT/PFT representation at step 0.
            self.global_gate = nn.Parameter(torch.zeros(1))
        if self.fusion_mode == "query-residual":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.dim,
                nhead=num_heads,
                dim_feedforward=3 * self.dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.query_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=max(1, depth // 2),
                norm=nn.LayerNorm(self.dim),
            )
            self.query_embedding = nn.Parameter(
                torch.zeros(1, self.num_joints, self.dim)
            )
            self.query_residual_head = nn.Linear(self.dim, 3)
            # Exact H76 identity at initialization without a scalar gate that
            # the optimizer can permanently close.  The head learns first;
            # gradients reach the decoder/encoder as soon as it becomes nonzero.
            nn.init.zeros_(self.query_residual_head.weight)
            nn.init.zeros_(self.query_residual_head.bias)
            nn.init.trunc_normal_(self.query_embedding, std=0.02)
        if self.fusion_mode in (
            "mixste-ttb", "mixste-ttb-residual", "mixste-alternating",
            "mixste-pose-residual",
        ):
            # MixSTE's TTB receives one token per joint trajectory:
            # (B,T,J,D) -> (B*J,T,D).  The learned temporal position table is
            # initialized to zero in the official implementation.  We keep
            # it here but zero-init every TTB output projection so the new
            # branch is an exact identity around the pretrained H76 model.
            self.mixste_temporal_pos_embed = nn.Parameter(
                torch.zeros(1, self.temporal_length, self.dim)
            )
            self.mixste_ttb = nn.ModuleList(
                [
                    Block(
                        dim=self.dim,
                        num_heads=num_heads,
                        # MixSTE2 uses an MLP expansion ratio of 2.0.  The
                        # retained RUMPL JVT branch above intentionally keeps
                        # its own ratio; this block follows the source TTB.
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                    )
                    for _ in range(depth)
                ]
            )
            # The pre-head TTB variants need identity blocks because their
            # output is consumed directly by the calibrated RUMPL PFT/head.
            # The pose-residual variant instead has a zero-initialized output
            # head, so its STB/TTB features should retain MixSTE's ordinary
            # random initialization and can learn immediately after the head's
            # first update.
            if self.fusion_mode != "mixste-pose-residual":
                for block in self.mixste_ttb:
                    nn.init.zeros_(block.attn.proj.weight)
                    if block.attn.proj.bias is not None:
                        nn.init.zeros_(block.attn.proj.bias)
                    nn.init.zeros_(block.mlp.fc2.weight)
                    if block.mlp.fc2.bias is not None:
                        nn.init.zeros_(block.mlp.fc2.bias)
        if self.fusion_mode == "mixste-pose-residual":
            # MixSTE is trained on root-relative 3-D poses. Applying its TTB to
            # RUMPL's absolute ray/VFT representation moved the pelvis and
            # damaged the main benefit of calibrated multi-view geometry. This
            # adapter runs the official STB->TTB alternation on RUMPL's decoded
            # root-relative pose and predicts only a residual; the absolute
            # pelvis is copied exactly from RUMPL.
            self.mixste_pose_embedding = nn.Linear(3, self.dim)
            self.mixste_pose_spatial_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_joints, self.dim)
            )
            self.mixste_pose_stb = nn.ModuleList(
                [
                    Block(
                        dim=self.dim,
                        num_heads=num_heads,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                    )
                    for _ in range(depth)
                ]
            )
            self.mixste_pose_spatial_norm = nn.LayerNorm(self.dim, eps=1e-6)
            self.mixste_pose_temporal_norm = nn.LayerNorm(self.dim, eps=1e-6)
            self.mixste_pose_head = nn.Sequential(
                nn.LayerNorm(self.dim, eps=1e-6), nn.Linear(self.dim, 3)
            )
            nn.init.zeros_(self.mixste_pose_head[-1].weight)
            nn.init.zeros_(self.mixste_pose_head[-1].bias)
        # Residual-safe warm start.  A randomly initialized attention/MLP
        # branch can have a large downstream 3-D effect even when the outer
        # gate is only a few 1e-3.  Zeroing only the residual projections makes
        # the new JVT branch start as an identity while retaining gradients for
        # the projections themselves; the frozen H76 path is untouched.
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.attn.proj.bias is not None:
                nn.init.zeros_(block.attn.proj.bias)
            nn.init.zeros_(block.mlp.fc2.weight)
            if block.mlp.fc2.bias is not None:
                nn.init.zeros_(block.mlp.fc2.bias)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def _encode_observations(
        self,
        rays: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ):
        bb = self.backbone
        raw_direction = rays[..., :3]
        raw_point = rays[..., 3:6]
        confidence = rays[..., 6:7]
        if getattr(bb, "input_harmonic_l", 0) > 0:
            raise NotImplementedError(
                "H40 first isolates temporal/bias training with the H0 ray encoder; "
                "harmonic ray input is a later controlled ablation"
            )
        direction = raw_direction
        point = raw_point
        if getattr(bb, "anchor_centered_rays", False):
            if anchor is None:
                raise ValueError("anchor-centered temporal rays require a triangulation anchor")
            if getattr(bb, "anchor_center_per_joint", False):
                center = anchor[:, :, :, None, :]
            else:
                # H76 expresses every joint in the same subject-root frame.
                center = anchor[:, :, :1, None, :]
            point = point - center
        if getattr(bb, "input_plucker", False):
            direction = F.normalize(direction, dim=-1, eps=1e-7)
            point = torch.cross(point, direction, dim=-1)

        if bb.concat_direction_and_intersection_first:
            token = bb.encoding_to_embedding(torch.cat([direction, point], dim=-1))
        elif bb.not_use_intersection_features:
            token = bb.encoding_to_embedding(direction)
        else:
            token = torch.cat(
                [bb.encoding_to_embedding(direction), bb.encoding_to_embedding(point)],
                dim=-1,
            )
        if bb.concat_confidence:
            token = torch.cat([token, bb.confidence_to_embedding(confidence)], dim=-1)
        if token.shape[-1] != self.dim:
            raise ValueError(
                f"encoded token dimension {token.shape[-1]} != RUMPL dimension {self.dim}"
            )
        # Geometry bias must use actual rays, not centered Pluecker moments.
        return token, raw_direction, raw_point, confidence

    def _global_refine(
        self,
        token: torch.Tensor,
        direction: torch.Tensor,
        point: torch.Tensor,
        confidence: torch.Tensor,
        delta_t: torch.Tensor,
    ):
        batch, time, joints, views, dim = token.shape
        if delta_t.shape != (batch, time):
            raise ValueError(
                f"delta_t must have shape {(batch, time)}, got {tuple(delta_t.shape)}"
            )
        observation_token = token
        temporal = self.time_encoder(delta_t)[:, :, None, None, :]
        token = token + self.joint_embedding + temporal
        token = token.reshape(batch, time * joints * views, dim)
        num_tokens = token.shape[1]

        padding_mask = None
        attention_mask = None
        if self.training and self.token_dropout > 0:
            padding_mask = torch.rand(batch, num_tokens, device=token.device) < self.token_dropout
            all_removed = padding_mask.all(dim=1)
            if all_removed.any():
                rows = all_removed.nonzero(as_tuple=False).flatten()
                keep = torch.randint(num_tokens, (rows.numel(),), device=token.device)
                padding_mask[rows, keep] = False
            token = token.masked_fill(padding_mask[..., None], 0.0)
            attention_mask = token.new_zeros(batch, num_tokens, num_tokens)
            attention_mask.masked_fill_(
                padding_mask[:, None, :], torch.finfo(token.dtype).min
            )

        confidence_bias = None
        geometry_distance = None
        if self.biased:
            flat_confidence = confidence.reshape(batch, num_tokens).clamp(0, 1)
            confidence_bias = flat_confidence[:, None, :].expand(
                batch, num_tokens, num_tokens
            )
            geometry_distance = pairwise_ray_distance(
                direction.reshape(batch, num_tokens, 3),
                point.reshape(batch, num_tokens, 3),
            )

        for block in self.blocks:
            token = block(
                token,
                attn_bias=attention_mask,
                conf_bias=confidence_bias,
                geom_distance=geometry_distance,
            )
        # Keep the residual branch in the same coordinate scale as the
        # pretrained observation token.  Applying a fresh LayerNorm to the
        # complete token here creates a large jump even when all new block
        # projections are zero; the retained RUMPL VFT/PFT already performs
        # its own normalization downstream.
        if padding_mask is not None:
            # Approximate physical token removal while keeping a dense tensor
            # for the retained RUMPL VFT implementation.
            token = token.masked_fill(padding_mask[..., None], 0.0)
        token = token.reshape(batch, time, joints, views, dim)
        if self.residual_gate:
            delta = token - observation_token
            # The downstream frozen PFT/head can amplify an unconstrained
            # random-token delta by orders of magnitude.  Normalize each
            # token's detached magnitude before applying the learned gate so
            # the first updates remain in the same scale as H76.  This does
            # not change the gate=0 identity and is differentiable through the
            # direction of the learned temporal correction.
            delta_norm = delta.detach().norm(dim=-1, keepdim=True).clamp_min(1.0)
            delta = self.residual_scale * delta / delta_norm
            token = observation_token + self.global_gate * delta
        return token

    def _encode_query_memory(
        self,
        token: torch.Tensor,
        direction: torch.Tensor,
        point: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode each frame's complete joint-view observation set.

        This retains all camera observations until a fixed set of joint
        queries reads the memory.  Unlike H17, it does not replace RUMPL's
        prediction; the decoded output is only a residual on the H76 path.
        """

        batch, time, joints, views, dim = token.shape
        memory = token + self.joint_embedding
        memory = memory.reshape(batch * time, joints * views, dim)
        num_tokens = joints * views

        padding_mask = None
        attention_mask = None
        if self.training and self.token_dropout > 0:
            padding_mask = (
                torch.rand(batch * time, num_tokens, device=token.device)
                < self.token_dropout
            )
            all_removed = padding_mask.all(dim=1)
            if all_removed.any():
                rows = all_removed.nonzero(as_tuple=False).flatten()
                keep = torch.randint(num_tokens, (rows.numel(),), device=token.device)
                padding_mask[rows, keep] = False
            memory = memory.masked_fill(padding_mask[..., None], 0.0)
            attention_mask = memory.new_zeros(
                batch * time, num_tokens, num_tokens
            )
            attention_mask.masked_fill_(
                padding_mask[:, None, :], torch.finfo(memory.dtype).min
            )

        confidence_bias = None
        geometry_distance = None
        if self.biased:
            flat_confidence = confidence.reshape(batch * time, num_tokens).clamp(0, 1)
            confidence_bias = flat_confidence[:, None, :].expand(
                batch * time, num_tokens, num_tokens
            )
            geometry_distance = pairwise_ray_distance(
                direction.reshape(batch * time, num_tokens, 3),
                point.reshape(batch * time, num_tokens, 3),
            )

        for block in self.blocks:
            memory = block(
                memory,
                attn_bias=attention_mask,
                conf_bias=confidence_bias,
                geom_distance=geometry_distance,
            )
        memory = self.norm(memory)
        return memory, padding_mask

    def _rumpl_vft(self, token: torch.Tensor):
        bb = self.backbone
        batch, time, joints, views, dim = token.shape
        x = token.reshape(batch * time * joints, views, dim)
        fusion = bb.fusion_token.expand(batch * time * joints, -1, -1)
        x = torch.cat([fusion, x], dim=1)
        if bb.add_view_enc:
            x = x + bb.View_enc_learned[:, : views + 1]
        x = bb.pos_drop(x)
        for block in bb.blocks_view_fusion:
            x = block(x)
        x = bb.View_norm(x)[:, 0]
        return x.reshape(batch, time, joints, dim)

    def _mixste_temporal_position(self, time: int):
        position = self.mixste_temporal_pos_embed
        if position.shape[1] != time:
            # T=9 is the primary protocol; interpolation lets the same
            # checkpoint be inspected at T=27 without silently slicing time.
            position = F.interpolate(
                position.transpose(1, 2),
                size=time,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return position

    def _mixste_ttb_block(
        self, token: torch.Tensor, block_index: int, add_position: bool = True
    ):
        """Apply one official-style TTB to ``(B,T,J,D)`` tokens."""

        batch, time, joints, dim = token.shape
        temporal = token.permute(0, 2, 1, 3).reshape(
            batch * joints, time, dim
        )
        if add_position:
            temporal = temporal + self._mixste_temporal_position(time)
        temporal = self.mixste_ttb[block_index](temporal)
        return temporal.reshape(batch, joints, time, dim).permute(0, 2, 1, 3)

    def _mixste_ttb(self, token: torch.Tensor):
        """MixSTE-style per-joint temporal transformer.

        The official MixSTE code explicitly separates each joint trajectory
        before temporal attention.  RUMPL's view fusion has already removed
        the camera-token axis, so this is the closest compatible insertion:
        VFT -> (B*J,T,D) TTB -> PFT.  No camera identity is introduced.
        """

        batch, time, joints, dim = token.shape
        temporal = token.permute(0, 2, 1, 3).reshape(
            batch * joints, time, dim
        )
        temporal = temporal + self._mixste_temporal_position(time)
        for block in self.mixste_ttb:
            # Block is pre-norm, matching MixSTE's Transformer block while
            # avoiding an external post-norm that would break H76 identity.
            temporal = block(temporal)
        return temporal.reshape(batch, joints, time, dim).permute(0, 2, 1, 3)

    def _mixste_alternating_features(self, token: torch.Tensor):
        """Retain RUMPL PFT weights while alternating PFT/STB and TTB.

        MixSTE's central claim is not only a TTB, but alternating spatial and
        temporal blocks.  This adapter keeps the pretrained RUMPL PFT blocks
        (including its public final-block repeat) and inserts one per-joint TTB
        after each spatial block.  With zero TTB output projections it reduces
        exactly to the original RUMPL PFT path.
        """

        bb = self.backbone
        batch, time, joints, dim = token.shape
        x = token.reshape(batch * time, joints, dim)
        x = bb.pos_drop(x + bb.Spatial_pos_embed)
        if len(self.mixste_ttb) < len(bb.blocks):
            raise ValueError(
                "MixSTE alternating mode needs at least as many TTB blocks "
                f"as retained PFT blocks ({len(self.mixste_ttb)} < {len(bb.blocks)})"
            )
        for index, spatial_block in enumerate(bb.blocks):
            x = spatial_block(x)
            temporal = x.reshape(batch, time, joints, dim)
            temporal = self._mixste_ttb_block(
                temporal, index, add_position=index == 0
            )
            x = temporal.reshape(batch * time, joints, dim)
        # RUMPL's released PFT executes the final block twice.  Keep that
        # quirk in this branch so the only new operation is the TTB insertion.
        if len(bb.blocks) > 0:
            x = bb.blocks[-1](x)
        x = bb.Spatial_norm(x)
        return x.reshape(batch, time, joints, dim)

    def _rumpl_pft_features(self, token: torch.Tensor):
        bb = self.backbone
        batch, time, joints, dim = token.shape
        x = token.reshape(batch * time, joints, dim)
        x = bb.pos_drop(x + bb.Spatial_pos_embed)
        # Preserve the published RUMPL implementation, including its duplicate
        # call of the final PFT block.  Changing this here would confound the
        # temporal/bias ablation with a baseline correction.
        for index, block in enumerate(bb.blocks):
            if index == len(bb.blocks) - 1:
                x = block(x)
            x = block(x)
        x = bb.Spatial_norm(x)
        return x.reshape(batch, time, joints, dim)

    def _rumpl_pft_and_head(self, token: torch.Tensor):
        features = self._rumpl_pft_features(token)
        return self.backbone.head(features)

    def _mixste_pose_correction(self, pose: torch.Tensor):
        """Predict a root-relative correction with MixSTE STB/TTB ordering.

        ``pose`` is RUMPL's absolute world-coordinate output ``(B,T,J,3)``.
        The official MixSTE target removes global translation, so the temporal
        branch receives ``pose - pelvis``. The returned correction has an
        exactly zero pelvis component, preventing temporal learning from
        changing RUMPL's calibrated absolute localization.
        """

        batch, time, joints, _ = pose.shape
        root_relative = pose - pose[:, :, :1]
        x = self.mixste_pose_embedding(root_relative)
        x = x + self.mixste_pose_spatial_pos_embed[:, None]
        for index, (spatial_block, temporal_block) in enumerate(
            zip(self.mixste_pose_stb, self.mixste_ttb)
        ):
            spatial = x.reshape(batch * time, joints, self.dim)
            spatial = self.mixste_pose_spatial_norm(spatial_block(spatial))
            temporal = spatial.reshape(batch, time, joints, self.dim)
            temporal = temporal.permute(0, 2, 1, 3).reshape(
                batch * joints, time, self.dim
            )
            if index == 0:
                temporal = temporal + self._mixste_temporal_position(time)
            temporal = self.mixste_pose_temporal_norm(
                temporal_block(temporal)
            )
            x = temporal.reshape(batch, joints, time, self.dim).permute(
                0, 2, 1, 3
            )
        correction = self.mixste_pose_head(x)
        root_mask = correction.new_ones(1, 1, joints, 1)
        root_mask[:, :, 0] = 0.0
        return correction * root_mask

    def _drop_joint_view_tokens(self, token: torch.Tensor) -> torch.Tensor:
        """GBT-style token dropout on (joint, view) tracks before VFT.

        Each dropped token is zeroed, matching GBT's padding mask.  At least
        one view is kept for every joint-frame so VFT still has a valid set.
        Evaluation is unchanged: dropout is training-only.
        """

        if not (self.training and self.token_dropout > 0):
            return token
        batch, time, joints, views, _ = token.shape
        drop = torch.rand(
            batch, time, joints, views, device=token.device
        ) < self.token_dropout
        if views > 1:
            all_dropped = drop.all(dim=-1)
            if bool(all_dropped.any()):
                idx = all_dropped.nonzero(as_tuple=False)
                keep = torch.randint(views, (len(idx),), device=token.device)
                drop[idx[:, 0], idx[:, 1], idx[:, 2], keep] = False
        return token.masked_fill(drop[..., None], 0.0)

    def forward(
        self,
        rays: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        num_views: Optional[int] = None,
        view_indices: Optional[torch.Tensor] = None,
        view_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all frame predictions and the camera indices used.

        ``rays`` must have layout ``(B,T,J,V,7)``.  During training, pass
        ``num_views=2`` to reproduce GBT's random two-view sequence protocol.
        During evaluation omit it (or pre-slice with ``view_indices``).
        """

        if rays.ndim != 5 or rays.shape[-1] < 7:
            raise ValueError(f"expected rays (B,T,J,V,>=7), got {tuple(rays.shape)}")
        batch, time, joints, total_views, _ = rays.shape
        if joints != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joints, got {joints}")
        if view_indices is not None:
            if view_indices.shape[0] != batch:
                raise ValueError("view_indices batch dimension mismatch")
            selected = view_indices.shape[1]
            gather = view_indices[:, None, None, :, None].expand(
                batch, time, joints, selected, rays.shape[-1]
            )
            rays = torch.gather(rays, 3, gather)
        elif num_views is not None:
            rays, view_indices = sample_sequence_views(
                rays, num_views, generator=view_generator
            )
        else:
            view_indices = torch.arange(total_views, device=rays.device)[None].expand(
                batch, -1
            )

        if delta_t is None:
            # H36M bundle: 50 Hz raw video, one processed observation per five
            # frames => 0.1 s between adjacent available temporal samples.
            relative = (torch.arange(time, device=rays.device, dtype=rays.dtype) - (time - 1)) * 0.1
            delta_t = relative[None].expand(batch, -1)
        else:
            delta_t = delta_t.to(device=rays.device, dtype=rays.dtype)

        direction = rays[..., :3]
        point = rays[..., 3:6]
        confidence = rays[..., 6:7]
        anchor = None
        if getattr(self.backbone, "tri_anchor", False) and rays.shape[3] >= 2:
            anchor = confidence_weighted_triangulation_anchor(
                direction,
                point,
                confidence,
                regularization=self.backbone.tri_anchor_reg,
                confidence_epsilon=self.backbone.tri_anchor_conf_eps,
            )
        token, direction, point, confidence = self._encode_observations(
            rays, anchor=anchor
        )
        if self.fusion_mode == "global-residual":
            token = self._global_refine(
                token, direction, point, confidence, delta_t
            )
            token = self._rumpl_vft(token)
            output = self._rumpl_pft_and_head(token)
        elif self.fusion_mode == "pre-vft-temporal":
            # Keep the camera axis intact until the original RUMPL VFT.  The
            # adapter only sees each (joint, view) trajectory through time.
            # GBT trains with 20% token dropout; H8 left that on unused JVT
            # blocks.  Dropping (joint, view) tokens here is the occlusion
            # analogue: a missing observation in one frame can still be
            # recovered from the same ray track at other times.
            token = self._drop_joint_view_tokens(token)
            token = self.pre_vft_temporal(token)
            token = self._rumpl_vft(token)
            output = self._rumpl_pft_and_head(token)
        elif self.fusion_mode == "query-residual":
            memory, memory_padding_mask = self._encode_query_memory(
                token, direction, point, confidence
            )
            fused = self._rumpl_vft(token)
            pft_features = self._rumpl_pft_features(fused)
            queries = pft_features.reshape(
                batch * time, joints, self.dim
            ) + self.query_embedding
            decoded = self.query_decoder(
                queries,
                memory,
                memory_key_padding_mask=memory_padding_mask,
            )
            correction = self.query_residual_head(decoded).reshape(
                batch, time, joints, 3
            )
            output = self.backbone.head(pft_features) + self.residual_scale * correction
        elif self.fusion_mode == "mixste-ttb":
            # Faithful compatible MixSTE insertion: all RUMPL view fusion is
            # completed before per-joint temporal attention, and the original
            # RUMPL PFT/head remain the spatial/body decoder.
            token = self._rumpl_vft(token)
            token = self._mixste_ttb(token)
            output = self._rumpl_pft_and_head(token)
        elif self.fusion_mode == "mixste-ttb-residual":
            # Stable retained-backbone variant.  The official MixSTE replaces
            # its monocular token stream outright; here a scaled residual
            # keeps RUMPL's calibrated geometry dominant while the TTB learns
            # only the temporal correction.
            token = self._rumpl_vft(token)
            refined = self._mixste_ttb(token)
            token = token + self.residual_scale * (refined - token)
            output = self._rumpl_pft_and_head(token)
        elif self.fusion_mode == "mixste-alternating":
            # Full alternating variant: retain RUMPL's PFT blocks as the STB
            # stages and insert one MixSTE TTB after each stage.
            token = self._rumpl_vft(token)
            features = self._mixste_alternating_features(token)
            output = self.backbone.head(features)
        else:
            # Pose-space MixSTE starts from the complete untouched RUMPL
            # prediction. Its residual is applied after the shared anchor add.
            token = self._rumpl_vft(token)
            output = self._rumpl_pft_and_head(token)
        if anchor is not None:
            output = output + self.backbone.tri_anchor_gate * anchor
        if self.fusion_mode == "mixste-pose-residual":
            # Decode the untouched RUMPL path first, including its absolute
            # triangulation anchor, then refine only root-relative articulation.
            output = output + self.residual_scale * self._mixste_pose_correction(
                output
            )
        return output, view_indices
