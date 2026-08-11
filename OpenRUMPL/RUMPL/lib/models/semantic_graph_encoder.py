# -----------------------------------------------------------------------------
# Semantic graph encoder adapted for RUMPL ray tokens.
#
# Architecture source:
#   L. Zhang et al., "Deep Semantic Graph Transformer for Multi-View 3D
#   Human Pose Estimation", AAAI 2024.
# Official implementation: https://github.com/z0911k/SGraFormer (MIT).
#
# The original SGraFormer applies this spatial encoder independently to every
# camera before cross-view fusion.  This module preserves that placement and
# accepts RUMPL's embedded ray tokens instead of replacing RUMPL's VFT.
# -----------------------------------------------------------------------------

import math

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_


H36M17_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def build_exact_hop_adjacency(num_joints=17, max_hop=4):
    """Build the exact 1..K-hop H36M skeleton matrices used by SGraFormer."""
    if num_joints != len(H36M17_PARENTS):
        raise ValueError('semantic graph encoder currently requires H36M-17')
    if max_hop < 1:
        raise ValueError('max_hop must be positive')

    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for child, parent in enumerate(H36M17_PARENTS):
        if parent >= 0:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0

    # Floyd-Warshall is tiny for J=17 and makes "exact hop" unambiguous.
    distance = torch.full((num_joints, num_joints), num_joints + 1, dtype=torch.long)
    distance.fill_diagonal_(0)
    distance[adjacency.bool()] = 1
    for k in range(num_joints):
        distance = torch.minimum(
            distance,
            distance[:, k:k + 1] + distance[k:k + 1, :],
        )
    return torch.stack(
        [(distance == hop).to(torch.float32) for hop in range(1, max_hop + 1)],
        dim=0,
    )


class SemanticGraphMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class SemanticGraphAttention(nn.Module):
    """Joint self-attention with SGraFormer's learned skeletal-edge bias."""

    def __init__(
        self,
        dim,
        num_joints=17,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_edge_bias=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.num_joints = num_joints
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_edge_bias = use_edge_bias
        self.edge_embedding = (
            nn.Linear(num_joints * num_joints, num_joints * num_joints)
            if use_edge_bias else None
        )

    def forward(self, x, edge_embedding=None):
        batch_size, num_joints, dim = x.shape
        if num_joints != self.num_joints:
            raise ValueError(
                f'expected {self.num_joints} joints, got {num_joints}'
            )
        head_dim = dim // self.num_heads
        qkv = self.qkv(x).reshape(
            batch_size, num_joints, 3, self.num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        attention = (query @ key.transpose(-2, -1)) * self.scale

        if self.use_edge_bias:
            if edge_embedding is None:
                raise ValueError('full semantic graph attention requires edge embedding')
            edge_bias = self.edge_embedding(edge_embedding.reshape(1, -1))
            edge_bias = edge_bias.reshape(1, 1, num_joints, num_joints)
            attention = attention + edge_bias.to(dtype=attention.dtype)

        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(
            batch_size, num_joints, dim
        )
        return self.proj_drop(self.proj(x))


class SemanticGraphBlock(nn.Module):
    """SGraFormer spatial block, including the coupled multi-hop graph stream."""

    def __init__(
        self,
        dim,
        num_joints=17,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        full_semantic_graph=True,
    ):
        super().__init__()
        self.full_semantic_graph = full_semantic_graph
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = SemanticGraphAttention(
            dim=dim,
            num_joints=num_joints,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_edge_bias=full_semantic_graph,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SemanticGraphMlp(dim, mlp_ratio=mlp_ratio, drop=drop)

        if full_semantic_graph:
            self.graph_norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.graph_norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.graph_mlp = SemanticGraphMlp(
                dim, mlp_ratio=mlp_ratio, drop=drop
            )

    def forward(self, x, graph_features=None, edge_embedding=None):
        message = self.drop_path(
            self.attention(self.norm1(x), edge_embedding=edge_embedding)
        )
        if self.full_semantic_graph:
            if graph_features is None:
                raise ValueError('full semantic graph block requires graph features')
            message = self.graph_norm1(graph_features) * message
        x = x + message
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.full_semantic_graph:
            graph_features = graph_features + message
            graph_features = graph_features + self.drop_path(
                self.graph_mlp(self.graph_norm2(graph_features))
            )
        return x, graph_features


class SemanticGraphPreVFTEncoder(nn.Module):
    """Apply an SGraFormer spatial encoder independently to every camera.

    Args:
        mode: ``position`` is the paper's graph-free position-only control;
            ``full`` enables position, 1--4 hop spatial graph, and edge bias.
    Inputs:
        tokens: ``(B, J, V, D)`` embedded RUMPL ray tokens.
        coordinates: ``(B, J, V, C)`` ray directions (or 2D coordinates).
    """

    def __init__(
        self,
        dim,
        num_joints=17,
        depth=4,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        max_hop=4,
        mode='full',
    ):
        super().__init__()
        if mode not in ('position', 'full'):
            raise ValueError("semantic graph mode must be 'position' or 'full'")
        if depth < 1:
            raise ValueError('semantic graph depth must be positive')
        self.mode = mode
        self.num_joints = num_joints
        self.dim = dim
        self.max_hop = max_hop
        self.full_semantic_graph = mode == 'full'

        self.joint_position_embedding = nn.Parameter(
            torch.zeros(1, num_joints, dim)
        )
        self.input_drop = nn.Dropout(drop)

        if self.full_semantic_graph:
            hop_adjacency = build_exact_hop_adjacency(
                num_joints=num_joints, max_hop=max_hop
            )
            self.register_buffer('hop_adjacency', hop_adjacency)
            self.graph_to_embedding = nn.Linear(max_hop * num_joints, dim)
            self.graph_position_embedding = nn.Parameter(
                torch.zeros(1, num_joints, dim)
            )
            # This mirrors SGraFormer's 4*17*17 -> 17*17 edge embedding.
            self.edge_to_embedding = nn.Linear(
                max_hop * num_joints * num_joints,
                num_joints * num_joints,
            )

        path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            SemanticGraphBlock(
                dim=dim,
                num_joints=num_joints,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=path_rates[index],
                full_semantic_graph=self.full_semantic_graph,
            )
            for index in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        trunc_normal_(self.joint_position_embedding, std=0.02)
        if self.full_semantic_graph:
            trunc_normal_(self.graph_position_embedding, std=0.02)

    def _build_graph_features(self, coordinates):
        # coordinates: (B*V, J, C). Pairwise chord distance between unit rays
        # is the ray-space counterpart of SGraFormer's 2D joint distance.
        difference = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        pair_distance = difference.square().sum(dim=-1)
        pair_distance = pair_distance / pair_distance.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        graph = (
            pair_distance[:, None, :, :] * self.hop_adjacency[None, :, :, :]
        )
        graph = graph.permute(0, 2, 1, 3).reshape(
            coordinates.shape[0], self.num_joints,
            self.max_hop * self.num_joints,
        )
        return self.input_drop(
            self.graph_to_embedding(graph) + self.graph_position_embedding
        )

    def forward(self, tokens, coordinates=None):
        if tokens.ndim != 4:
            raise ValueError('tokens must have shape (B,J,V,D)')
        batch_size, num_joints, num_views, dim = tokens.shape
        if num_joints != self.num_joints or dim != self.dim:
            raise ValueError(
                f'incompatible semantic graph token shape: {tuple(tokens.shape)}'
            )

        # Sharing this encoder over V retains camera permutation equivariance.
        x = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * num_views, num_joints, dim
        )
        x = self.input_drop(x + self.joint_position_embedding)

        graph_features = None
        edge_embedding = None
        if self.full_semantic_graph:
            if coordinates is None or coordinates.ndim != 4:
                raise ValueError(
                    'full semantic graph encoder requires coordinates (B,J,V,C)'
                )
            if coordinates.shape[:3] != tokens.shape[:3]:
                raise ValueError('coordinates and tokens have incompatible B/J/V')
            coordinates = coordinates.permute(0, 2, 1, 3).reshape(
                batch_size * num_views, num_joints, coordinates.shape[-1]
            )
            # RUMPL supplies 3D ray directions in the target experiment;
            # ordinary 2D coordinates are kept in their detector-normalized
            # space when this module is reused with a 2D-input configuration.
            if coordinates.shape[-1] == 3:
                coordinates = torch.nn.functional.normalize(
                    coordinates, dim=-1, eps=1e-7
                )
            graph_features = self._build_graph_features(coordinates)
            edge_embedding = self.edge_to_embedding(
                self.hop_adjacency.reshape(1, -1)
            ).reshape(num_joints, num_joints)

        for block in self.blocks:
            x, graph_features = block(
                x,
                graph_features=graph_features,
                edge_embedding=edge_embedding,
            )
        x = self.output_norm(x)
        return x.reshape(batch_size, num_views, num_joints, dim).permute(
            0, 2, 1, 3
        ).contiguous()
