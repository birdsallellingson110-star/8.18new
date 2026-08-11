# -----------------------------------------------------------------------------
# GraFormer feature encoder used as an opt-in RUMPL PFT replacement.
#
# Architecture source:
#   W. Zhao et al., "GraFormer: Graph Convolution Transformer for 3D Pose
#   Estimation", CVPR 2022.
# Official implementation: https://github.com/Graformer/GraFormer
# License of the source implementation: Apache-2.0.
#
# The official model maps 2-D coordinates to 3-D coordinates with an input
# ChebGConv, five alternating GraAttention/ChebGConv blocks, and an output
# ChebGConv.  Here the input/output channel count is the RUMPL token dimension:
# the encoder consumes the per-joint features after RUMPL's VFT and leaves the
# established RUMPL 3-D head and triangulation anchor unchanged.
# -----------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


H36M17_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def build_h36m17_normalized_adjacency():
    """Return GraFormer's row-normalized H36M-17 adjacency with self-loops."""
    num_joints = len(H36M17_PARENTS)
    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for child, parent in enumerate(H36M17_PARENTS):
        if parent >= 0:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
    adjacency = adjacency + torch.eye(num_joints, dtype=adjacency.dtype)
    return adjacency / adjacency.sum(dim=1, keepdim=True)


class ChebConv(nn.Module):
    """Official GraFormer Chebyshev graph convolution, adapted without SciPy."""

    def __init__(self, in_channels, out_channels, order=2, bias=True):
        super().__init__()
        if order < 0:
            raise ValueError('Chebyshev order must be non-negative')
        self.num_polynomials = order + 1
        self.weight = nn.Parameter(
            torch.empty(self.num_polynomials, 1, in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def normalized_laplacian(adjacency):
        degree = adjacency.sum(dim=-1).clamp_min(1e-12).pow(-0.5)
        normalized = degree[:, None] * adjacency * degree[None, :]
        return torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        ) - normalized

    def chebyshev_polynomials(self, laplacian):
        polynomials = [torch.eye(
            laplacian.shape[0], device=laplacian.device, dtype=laplacian.dtype
        )]
        if self.num_polynomials >= 2:
            polynomials.append(laplacian)
        for _ in range(2, self.num_polynomials):
            polynomials.append(
                2.0 * laplacian @ polynomials[-1] - polynomials[-2]
            )
        return torch.stack(polynomials, dim=0)

    def forward(self, inputs, adjacency):
        laplacian = self.normalized_laplacian(adjacency)
        polynomials = self.chebyshev_polynomials(laplacian).unsqueeze(1)
        output = torch.matmul(polynomials, inputs)
        output = torch.matmul(output, self.weight).sum(dim=0)
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.gconv = ChebConv(input_dim, output_dim, order=2)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = nn.ReLU()

    def forward(self, x, adjacency):
        x = self.gconv(x, adjacency)
        if self.dropout is not None:
            x = self.dropout(self.activation(x))
        return self.activation(x)


class ResidualChebGC(nn.Module):
    def __init__(self, adjacency, dim, dropout=0.1):
        super().__init__()
        self.register_buffer('adjacency', adjacency.clone())
        self.gconv1 = GraphConv(dim, dim, dropout)
        self.gconv2 = GraphConv(dim, dim, dropout)

    def forward(self, x):
        adjacency = self.adjacency.to(dtype=x.dtype)
        return x + self.gconv2(self.gconv1(x, adjacency), adjacency)


class GraLayerNorm(nn.Module):
    """LayerNorm definition used by the official GraFormer implementation."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = GraLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('GraFormer dimension must be divisible by heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.projections = nn.ModuleList([
            copy.deepcopy(nn.Linear(dim, dim)) for _ in range(4)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query, key, value = [
            projection(x).view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            for projection, x in zip(self.projections, (query, key, value))
        ]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attention = self.dropout(F.softmax(scores, dim=-1))
        x = (attention @ value).transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        return self.projections[-1](x)


class LearnableAdjacencyGraphNet(nn.Module):
    """LAM-Gconv feed-forward replacement from GraAttention."""

    def __init__(self, dim, num_joints):
        super().__init__()
        self.adjacency = nn.Parameter(torch.eye(num_joints, dtype=torch.float32))
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def normalize_adjacency(adjacency):
        degree = (adjacency.sum(dim=0) + 1e-5).pow(-0.5)
        return degree[:, None] * adjacency * degree[None, :]

    def forward(self, x):
        normalized = self.normalize_adjacency(self.adjacency).to(dtype=x.dtype)
        x = self.activation(self.fc1(normalized @ x))
        return self.fc2(normalized @ x)


class GraAttentionLayer(nn.Module):
    def __init__(self, dim, num_joints, num_heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadedAttention(
            num_heads=num_heads, dim=dim, dropout=0.1
        )
        self.graph_feed_forward = LearnableAdjacencyGraphNet(dim, num_joints)
        self.sublayers = nn.ModuleList([
            SublayerConnection(dim, dropout),
            SublayerConnection(dim, dropout),
        ])

    def forward(self, x, mask):
        x = self.sublayers[0](
            x, lambda normalized: self.self_attention(
                normalized, normalized, normalized, mask
            )
        )
        return self.sublayers[1](x, self.graph_feed_forward)


class GraFormerPFTEncoder(nn.Module):
    """Feature-space GraFormer replacing RUMPL's post-VFT PFT stack.

    ``attention`` corresponds to the paper's model-AT ablation (the repeated
    ChebGConv blocks are removed). ``full`` uses the complete alternating
    GraAttention + ChebGConv core. Input/output ChebGConv layers are retained
    in both modes, as in the paper's common model shell.
    """

    def __init__(
        self,
        dim,
        num_joints=17,
        depth=5,
        num_heads=4,
        dropout=0.25,
        mode='full',
    ):
        super().__init__()
        if num_joints != len(H36M17_PARENTS):
            raise ValueError('GraFormer PFT currently requires H36M-17')
        if depth < 1:
            raise ValueError('GraFormer depth must be positive')
        if mode not in ('attention', 'full'):
            raise ValueError('GraFormer PFT mode must be attention or full')

        self.mode = mode
        self.depth = depth
        adjacency = build_h36m17_normalized_adjacency()
        self.register_buffer('adjacency', adjacency)
        self.register_buffer(
            'attention_mask',
            torch.ones(1, 1, num_joints, dtype=torch.bool),
        )

        # Construct common modules first so the paper ablation and full model
        # share bit-identical common initialization under the same seed.
        self.input_gconv = ChebConv(dim, dim, order=2)
        attention_template = GraAttentionLayer(
            dim, num_joints, num_heads, dropout
        )
        self.attention_layers = nn.ModuleList([
            copy.deepcopy(attention_template)
            for _ in range(depth)
        ])
        self.output_gconv = ChebConv(dim, dim, order=2)
        self.cheb_blocks = nn.ModuleList()
        if mode == 'full':
            self.cheb_blocks = nn.ModuleList([
                ResidualChebGC(adjacency, dim, dropout=0.1)
                for _ in range(depth)
            ])

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.adjacency.shape[0]:
            raise ValueError(
                f'expected (B,{self.adjacency.shape[0]},D), got {tuple(x.shape)}'
            )
        adjacency = self.adjacency.to(dtype=x.dtype)
        x = self.input_gconv(x, adjacency)
        for layer_index, attention_layer in enumerate(self.attention_layers):
            x = attention_layer(x, self.attention_mask)
            if self.mode == 'full':
                x = self.cheb_blocks[layer_index](x)
        return self.output_gconv(x, adjacency)
