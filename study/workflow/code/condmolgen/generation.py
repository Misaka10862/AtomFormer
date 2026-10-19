from typing import Dict, List
import torch

def generate_batch(model, tokenizer, z: torch.Tensor, pos: torch.Tensor, max_length: int, temperature: float=1.0, greedy: bool=False, prefix_texts: List[str]=None) -> List[str]:
    device = z.device
    batch_size = z.size(0)
    if prefix_texts:
        prefix_ids = [tokenizer.encode_scaffold_target(prefix_text, '')[:-1] for prefix_text in prefix_texts]
        max_prefix = max((len(ids) for ids in prefix_ids))
        generated = torch.full((batch_size, max_prefix), fill_value=tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, ids in enumerate(prefix_ids):
            generated[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    else:
        generated = torch.full((batch_size, 1), fill_value=tokenizer.bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    with torch.no_grad():
        memory, _, encoder_padding_mask = model.encoder(z, pos)
        for _ in range(max_length - 1):
            decoder_padding_mask = generated.eq(tokenizer.pad_token_id)
            hidden = model.decoder(input_ids=generated, memory=memory, tgt_padding_mask=decoder_padding_mask, memory_padding_mask=encoder_padding_mask)
            logits = model.lm_head(hidden)[:, -1, :] / max(temperature, 1e-06)
            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, tokenizer.eos_token_id), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(-1).eq(tokenizer.eos_token_id)
            if bool(finished.all()):
                break
        outputs = []
        for row in generated:
            row_list = row.tolist()
            decoded = tokenizer.decode(row_list, skip_special_tokens=False)
            if prefix_texts and '[SEP]' in decoded:
                decoded = decoded.split('[SEP]', 1)[1]
            decoded = decoded.replace('[BOS]', '').replace('[EOS]', '').replace('[PAD]', '')
            outputs.append(decoded)
        return outputs

def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
