import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class GeoEncoderConfig:
    max_z: int = 128
    embedding_dim: int = 256
    ffn_embedding_dim: int = 512
    num_layers: int = 3
    num_attention_heads: int = 4
    cutoff: float = 5.0
    num_rbf: int = 32
    rbf_trainable: bool = False
    norm_type: str = "none"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "silu"
    pad_token_id: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return cutoffs * (distances < self.cutoff).float()


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff: float, num_rbf: int, trainable: bool):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff
        means, betas = self._initial_params()
        if trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1.0, self.num_rbf)
        betas = torch.tensor(
            [(2.0 / self.num_rbf * (1.0 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels: int, trainable: bool, norm_type: str = "none"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        weight = torch.ones(hidden_channels)
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)
        self.eps = 1e-6

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "none":
            return vec * self.weight.view(1, 1, 1, -1)

        dist = torch.norm(vec, dim=-2, keepdim=True).clamp(min=self.eps)
        direct = vec / dist
        max_val, _ = torch.max(dist, dim=-1, keepdim=True)
        min_val, _ = torch.min(dist, dim=-1, keepdim=True)
        denom = torch.where((max_val - min_val) == 0, torch.ones_like(max_val), max_val - min_val)
        scaled = (dist - min_val) / denom
        return scaled * direct * self.weight.view(1, 1, 1, -1)


class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config: GeoEncoderConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embedding_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embedding_dim:
            raise ValueError("embedding_dim must be divisible by num_attention_heads")

        self.act = build_activation(config.activation_function)
        self.cutoff = CosineCutoff(config.cutoff)
        self.dropout_module = nn.Dropout(config.attention_dropout)

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=config.norm_type
        )
        self.dihedral_proj = nn.Linear(self.embedding_dim, 2 * self.embedding_dim, bias=False)
        self.edge_attr_update = nn.Linear(self.embedding_dim, self.embedding_dim)

    def _reshape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels = x.shape
        x = x.view(batch, nodes, self.num_heads, channels // self.num_heads)
        return x.permute(0, 2, 1, 3).reshape(batch * self.num_heads, nodes, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        dist: torch.Tensor,
        edge_attr: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = x.shape
        q = self._reshape_qkv(self.q_proj(x))
        k = self._reshape_qkv(self.k_proj(x))
        v = self._reshape_qkv(self.v_proj(x))

        dk = self.act(self.dk_proj(edge_attr))
        dk = dk.view(batch, nodes, nodes, self.num_heads, self.head_dim)
        dk = dk.permute(0, 3, 1, 2, 4).reshape(batch * self.num_heads, nodes, nodes, self.head_dim)

        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(dim=-1)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch, self.num_heads, nodes, nodes)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            attn_weights = attn_weights.view(batch * self.num_heads, nodes, nodes)

        attn_scale = self.cutoff(dist).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn_scale = attn_scale.view(batch * self.num_heads, nodes, nodes)
        attn_probs = self.dropout_module(self.act(attn_weights) * attn_scale)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(-3)
        attn_per_nodes = attn_per_nodes.view(batch, self.num_heads, nodes, nodes, self.head_dim)
        attn_per_nodes = attn_per_nodes.permute(0, 2, 3, 1, 4).reshape(batch, nodes, nodes, self.embedding_dim)
        attn = attn_per_nodes.sum(dim=2)

        if key_padding_mask is not None:
            pair_mask = key_padding_mask.unsqueeze(-1)
        else:
            pair_mask = None

        du = self.du_update_proj(attn_per_nodes)
        if pair_mask is not None:
            du = du.masked_fill(pair_mask, 0.0)
        du = (du.unsqueeze(-2) * vec.unsqueeze(-1)).sum(dim=-3)
        du = self.du_norm(du)

        ws, wt = torch.split(self.dihedral_proj(du), self.embedding_dim, dim=-1)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe
        return attn, ipe


class GeoformerBlock(nn.Module):
    def __init__(self, config: GeoEncoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.act = build_activation(config.activation_function)
        self.self_attn = GeoformerMultiHeadAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(config.activation_dropout),
            nn.Linear(config.ffn_embedding_dim, config.embedding_dim),
        )
        self.attn_norm = nn.LayerNorm(config.embedding_dim)
        self.final_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        dist: torch.Tensor,
        edge_attr: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_x = x
        residual_edge = edge_attr
        x, edge_attr = self.self_attn(x, vec, dist, edge_attr, key_padding_mask)
        x = self.attn_norm(residual_x + self.dropout(x))
        edge_attr = edge_attr + residual_edge
        x = self.final_norm(x + self.dropout(self.ffn(x)))
        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config: GeoEncoderConfig):
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.cutoff = config.cutoff
        self.embedding = nn.Embedding(config.max_z, config.embedding_dim, padding_idx=config.pad_token_id)
        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, config.embedding_dim)
        self.act = build_activation(config.activation_function)
        self.in_norm = nn.LayerNorm(config.embedding_dim)
        self.layers = nn.ModuleList([GeoformerBlock(config) for _ in range(config.num_layers)])

    def forward(self, z: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, nodes = z.shape
        padding_mask = z.eq(self.pad_token_id)
        pos_mask = ~(padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2))

        dist = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1)
        loop_mask = torch.eye(nodes, dtype=torch.bool, device=dist.device).unsqueeze(0).expand(batch, -1, -1)
        dist = dist.masked_fill(loop_mask, 0.0)
        adj_mask = (dist < self.cutoff) & pos_mask
        loop_adj_mask = (~loop_mask) & adj_mask

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (dist.unsqueeze(-1) + 1e-8)
        vec = vec.masked_fill(~loop_adj_mask.unsqueeze(-1), 0.0)

        key_padding_mask = (~adj_mask)
        key_padding_mask = key_padding_mask.masked_fill(padding_mask.unsqueeze(-1), False)
        key_padding_mask = key_padding_mask.masked_fill(padding_mask.unsqueeze(-2), True)

        x = self.in_norm(self.embedding(z))
        edge_attr = self.distance_expansion(dist)
        edge_attr = self.act(self.dist_proj(edge_attr))
        edge_attr = edge_attr.masked_fill(~adj_mask.unsqueeze(-1), 0.0)

        for layer in self.layers:
            x, edge_attr = layer(x, vec, dist, edge_attr, key_padding_mask)

        return x, edge_attr, padding_mask
