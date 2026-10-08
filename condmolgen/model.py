from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .geometry import GeoEncoderConfig, GeoformerEncoder


class AtomTypeEncoder(nn.Module):
    """Atom-type encoder without coordinate features."""
    def __init__(self, config: "ConditionalGeneratorConfig"):
        super().__init__()
        self.pad_token_id = config.encoder_pad_token_id
        self.embedding = nn.Embedding(config.max_z, config.embedding_dim, padding_idx=self.pad_token_id)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, z: torch.Tensor, pos: torch.Tensor = None):
        padding_mask = z.eq(self.pad_token_id)
        x = self.norm(self.embedding(z))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x, None, padding_mask


@dataclass
class ConditionalGeneratorConfig:
    vocab_size: int
    max_z: int = 128
    embedding_dim: int = 256
    encoder_ffn_dim: int = 512
    encoder_layers: int = 3
    encoder_heads: int = 4
    decoder_ffn_dim: int = 512
    decoder_layers: int = 4
    decoder_heads: int = 4
    block_size: int = 256
    cutoff: float = 5.0
    num_rbf: int = 32
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "silu"
    encoder_pad_token_id: int = 0
    decoder_pad_token_id: int = 0
    scaffold_aux_weight: float = 0.0
    anchor_aux_weight: float = 0.0
    encoder_type: str = "geo"

    def to_dict(self) -> Dict:
        return asdict(self)


class DecoderBlock(nn.Module):
    def __init__(self, config: ConditionalGeneratorConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.decoder_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.decoder_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.ln3 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_ffn_dim, config.embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = x
        x_norm = self.ln1(x)
        self_attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,
            key_padding_mask=tgt_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(self_attn_out)

        residual = x
        x_norm = self.ln2(x)
        cross_attn_out, _ = self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(cross_attn_out)
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


class SmilesDecoder(nn.Module):
    def __init__(self, config: ConditionalGeneratorConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.embedding_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds block_size {self.config.block_size}."
            )

        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, memory, causal_mask, tgt_padding_mask, memory_padding_mask)

        return self.final_norm(x)


class ConditionalMolGenerator(nn.Module):
    def __init__(self, config: ConditionalGeneratorConfig):
        super().__init__()
        self.config = config
        encoder_config = GeoEncoderConfig(
            max_z=config.max_z,
            embedding_dim=config.embedding_dim,
            ffn_embedding_dim=config.encoder_ffn_dim,
            num_layers=config.encoder_layers,
            num_attention_heads=config.encoder_heads,
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_function=config.activation_function,
            pad_token_id=config.encoder_pad_token_id,
        )
        if config.encoder_type not in {"geo", "atom"}:
            raise ValueError("encoder_type must be 'geo' or 'atom'")
        self.encoder = GeoformerEncoder(encoder_config) if config.encoder_type == "geo" else AtomTypeEncoder(config)
        self.decoder = SmilesDecoder(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.scaffold_aux_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.anchor_head = nn.Linear(config.embedding_dim, config.max_z)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        scaffold_token_mask: Optional[torch.Tensor] = None,
        scaffold_target_ids: Optional[torch.Tensor] = None,
        anchor_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        memory, _, encoder_padding_mask = self.encoder(z, pos)
        hidden = self.decoder(
            input_ids=input_ids,
            memory=memory,
            tgt_padding_mask=decoder_padding_mask,
            memory_padding_mask=encoder_padding_mask,
        )
        logits = self.lm_head(hidden)
        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.transpose(1, 2),
                shift_labels,
            )
            loss = lm_loss
            output["lm_loss"] = lm_loss

            if (
                self.config.scaffold_aux_weight > 0
                and scaffold_token_mask is not None
                and scaffold_target_ids is not None
            ):
                pooled = memory.mean(dim=1)
                aux_logits = self.scaffold_aux_head(pooled)
                aux_target = torch.zeros_like(aux_logits)
                vocab_size = aux_logits.size(-1)
                for batch_idx in range(scaffold_target_ids.size(0)):
                    ids = scaffold_target_ids[batch_idx][scaffold_token_mask[batch_idx]]
                    for token_id in ids.tolist():
                        if 0 <= token_id < vocab_size:
                            aux_target[batch_idx, token_id] = 1.0
                aux_loss = nn.BCEWithLogitsLoss()(aux_logits, aux_target)
                output["scaffold_aux_loss"] = aux_loss
                loss = loss + self.config.scaffold_aux_weight * aux_loss

            if self.config.anchor_aux_weight > 0 and anchor_z is not None:
                pooled = memory.mean(dim=1)
                anchor_logits = self.anchor_head(pooled)
                anchor_loss = nn.CrossEntropyLoss()(anchor_logits, anchor_z)
                output["anchor_aux_loss"] = anchor_loss
                loss = loss + self.config.anchor_aux_weight * anchor_loss

            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_length: int,
        temperature: float = 1.0,
        greedy: bool = False,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = z.device
        if prefix_ids is None:
            generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        else:
            generated = prefix_ids.to(device)
            if generated.ndim != 2:
                raise ValueError("prefix_ids must have shape [B, T].")

        for _ in range(max_length - 1):
            decoder_padding_mask = generated.eq(pad_token_id)
            output = self.forward(
                z=z,
                pos=pos,
                input_ids=generated,
                decoder_padding_mask=decoder_padding_mask,
                labels=None,
            )
            next_logits = output["logits"][:, -1, :] / max(temperature, 1e-6)
            if greedy:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if int(next_token.item()) == eos_token_id:
                break

        return generated
