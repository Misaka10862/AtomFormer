from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .model import ConditionalGeneratorConfig, ConditionalMolGenerator


class DecorationGenerator(ConditionalMolGenerator):
    def supervised_loss(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        anchor_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(
            z=z,
            pos=pos,
            input_ids=input_ids,
            decoder_padding_mask=decoder_padding_mask,
            labels=labels,
            anchor_z=anchor_z,
        )

    @torch.no_grad()
    def generate_decorations(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        tokenizer,
        scaffold_smiles: List[str],
        max_length: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[List[str], torch.Tensor]:
        generated, lengths = build_prefix_batch(tokenizer, scaffold_smiles, z.device)
        prefix_lengths = lengths.clone()
        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z.device)

        memory, _, encoder_padding_mask = self.encoder(z, pos)
        while bool(((~finished) & lengths.lt(max_length)).any()):
            decoder_padding_mask = generated.eq(tokenizer.pad_token_id)
            hidden = self.decoder(
                input_ids=generated,
                memory=memory,
                tgt_padding_mask=decoder_padding_mask,
                memory_padding_mask=encoder_padding_mask,
            )
            last_hidden = gather_last_hidden(hidden, lengths)
            logits = self.lm_head(last_hidden) / max(temperature, 1e-6)
            if greedy:
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            active = (~finished) & lengths.lt(max_length)
            generated = append_tokens_at_lengths(
                generated,
                lengths=lengths,
                next_token=next_token,
                active=active,
                pad_token_id=tokenizer.pad_token_id,
            )
            lengths = lengths + active.long()
            finished = finished | (active & next_token.eq(tokenizer.eos_token_id)) | lengths.ge(max_length)

        decorations = decode_generated_decorations(tokenizer, generated, prefix_lengths)
        return decorations, generated

    def sample_decorations_with_logprobs(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        tokenizer,
        scaffold_smiles: List[str],
        max_length: int,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        generated, lengths = build_prefix_batch(tokenizer, scaffold_smiles, z.device)
        prefix_lengths = lengths.clone()
        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z.device)
        token_logprobs: List[torch.Tensor] = []
        action_mask: List[torch.Tensor] = []

        memory, _, encoder_padding_mask = self.encoder(z, pos)
        while bool(((~finished) & lengths.lt(max_length)).any()):
            decoder_padding_mask = generated.eq(tokenizer.pad_token_id)
            hidden = self.decoder(
                input_ids=generated,
                memory=memory,
                tgt_padding_mask=decoder_padding_mask,
                memory_padding_mask=encoder_padding_mask,
            )
            last_hidden = gather_last_hidden(hidden, lengths)
            logits = self.lm_head(last_hidden) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_logprobs = log_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
            active = (~finished) & lengths.lt(max_length)
            token_logprobs.append(selected_logprobs)
            action_mask.append(active.float())

            generated = append_tokens_at_lengths(
                generated,
                lengths=lengths,
                next_token=sampled,
                active=active,
                pad_token_id=tokenizer.pad_token_id,
            )
            lengths = lengths + active.long()
            finished = finished | (active & sampled.eq(tokenizer.eos_token_id)) | lengths.ge(max_length)

        if token_logprobs:
            logprob_tensor = torch.stack(token_logprobs, dim=1)
            mask_tensor = torch.stack(action_mask, dim=1)
        else:
            logprob_tensor = torch.empty(batch_size, 0, device=z.device)
            mask_tensor = torch.empty(batch_size, 0, device=z.device)

        decorations = decode_generated_decorations(tokenizer, generated, prefix_lengths)
        return {
            "decorations": decorations,
            "generated_ids": generated,
            "token_logprobs": logprob_tensor,
            "action_mask": mask_tensor,
            "prefix_lengths": prefix_lengths,
        }


def build_decoration_model(config: ConditionalGeneratorConfig) -> DecorationGenerator:
    return DecorationGenerator(config)


def build_prefix_ids(tokenizer, scaffold_smiles: List[str], device: torch.device) -> torch.Tensor:
    generated, _ = build_prefix_batch(tokenizer, scaffold_smiles, device)
    return generated


def build_prefix_batch(tokenizer, scaffold_smiles: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    prefix_ids = [tokenizer.encode_scaffold_target(scaffold, "")[:-1] for scaffold in scaffold_smiles]
    max_prefix = max(len(ids) for ids in prefix_ids)
    generated = torch.full(
        (len(prefix_ids), max_prefix),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.zeros(len(prefix_ids), dtype=torch.long, device=device)
    for i, ids in enumerate(prefix_ids):
        generated[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        lengths[i] = len(ids)
    return generated, lengths


def append_pad_column(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    pad_col = torch.full(
        (input_ids.size(0), 1),
        fill_value=pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, pad_col], dim=1)


def append_tokens_at_lengths(
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
    next_token: torch.Tensor,
    active: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if bool((active & lengths.ge(input_ids.size(1))).any()):
        input_ids = append_pad_column(input_ids, pad_token_id)
    updated = input_ids.clone()
    row_idx = torch.arange(input_ids.size(0), device=input_ids.device)
    updated[row_idx[active], lengths[active]] = next_token[active]
    return updated


def gather_last_hidden(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    gather_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather_idx).squeeze(1)


def decode_generated_decorations(tokenizer, generated_ids: torch.Tensor, prefix_lengths: torch.Tensor) -> List[str]:
    decorations: List[str] = []
    prefix_lengths_list = prefix_lengths.detach().cpu().tolist()
    for row, prefix_len in zip(generated_ids.detach().cpu().tolist(), prefix_lengths_list):
        ids = []
        for token_id in row[prefix_len:]:
            if token_id == tokenizer.eos_token_id:
                break
            if token_id == tokenizer.pad_token_id:
                continue
            ids.append(token_id)
        decorations.append(tokenizer.decode(ids, skip_special_tokens=True))
    return decorations


def token_supervised_loss_on_batch(
    model: nn.Module,
    batch: Dict,
    device: torch.device,
) -> torch.Tensor:
    out = model.supervised_loss(
        z=batch["z"].to(device),
        pos=batch["pos"].to(device),
        input_ids=batch["input_ids"].to(device),
        labels=batch["labels"].to(device),
        decoder_padding_mask=batch["decoder_padding_mask"].to(device),
        anchor_z=batch.get("anchor_z").to(device) if batch.get("anchor_z") is not None else None,
    )
    return out["loss"]
