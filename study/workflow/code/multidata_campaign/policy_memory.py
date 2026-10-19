import torch
from torch.utils.checkpoint import checkpoint
from condmolgen.v4_model import build_prefix_batch, append_tokens_at_lengths, gather_last_hidden, decode_generated_decorations
STRATEGY = 'decoder-step-logits-reentrant-rng-v1'

def sample_policy(model, z, pos, tokenizer, scaffold_smiles, max_length, temperature=1.0):
    if not hasattr(model, 'decoder') or not hasattr(model, 'lm_head'):
        return model.sample_decorations_with_logprobs(z=z, pos=pos, tokenizer=tokenizer, scaffold_smiles=scaffold_smiles, max_length=max_length, temperature=temperature)
    generated, lengths = build_prefix_batch(tokenizer, scaffold_smiles, z.device)
    prefix_lengths = lengths.clone()
    batch = generated.size(0)
    finished = torch.zeros(batch, dtype=torch.bool, device=z.device)
    token_logprobs = []
    action_mask = []
    memory, _, encoder_padding_mask = model.encoder(z, pos)

    def step_logits(ids, mem, lens, mem_mask):
        hidden = model.decoder(input_ids=ids, memory=mem, tgt_padding_mask=ids.eq(tokenizer.pad_token_id), memory_padding_mask=mem_mask)
        return model.lm_head(gather_last_hidden(hidden, lens)) / max(temperature, 1e-06)
    while bool((~finished & lengths.lt(max_length)).any()):
        if torch.is_grad_enabled() and memory.requires_grad:
            logits = checkpoint(step_logits, generated, memory, lengths, encoder_padding_mask, use_reentrant=True, preserve_rng_state=True)
        else:
            logits = step_logits(generated, memory, lengths, encoder_padding_mask)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        selected = torch.log_softmax(logits, dim=-1).gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        active = ~finished & lengths.lt(max_length)
        token_logprobs.append(selected)
        action_mask.append(active.float())
        generated = append_tokens_at_lengths(generated, lengths, sampled, active, tokenizer.pad_token_id)
        lengths = lengths + active.long()
        finished = finished | active & sampled.eq(tokenizer.eos_token_id) | lengths.ge(max_length)
    logprobs = torch.stack(token_logprobs, dim=1) if token_logprobs else torch.empty(batch, 0, device=z.device)
    masks = torch.stack(action_mask, dim=1) if action_mask else torch.empty(batch, 0, device=z.device)
    return {'decorations': decode_generated_decorations(tokenizer, generated, prefix_lengths), 'generated_ids': generated, 'token_logprobs': logprobs, 'action_mask': masks, 'prefix_lengths': prefix_lengths}
