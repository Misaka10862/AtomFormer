import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_records(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    if val_ratio <= 0:
        return (records, [])
    split_idx = max(1, int(len(records) * (1 - val_ratio)))
    return (records[:split_idx], records[split_idx:])

def run_epoch(model, dataloader, optimizer, device: torch.device, train: bool) -> Dict[str, float]:
    total_loss = 0.0
    total_steps = 0
    model.train(train)
    for batch in dataloader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        model_inputs = {'z': batch['z'], 'pos': batch['pos'], 'input_ids': batch['input_ids'], 'labels': batch['labels'], 'decoder_padding_mask': batch['decoder_padding_mask'], 'scaffold_token_mask': batch.get('scaffold_token_mask'), 'scaffold_target_ids': batch.get('scaffold_target_ids'), 'anchor_z': batch.get('anchor_z')}
        with torch.set_grad_enabled(train):
            output = model(**model_inputs)
            loss = output['loss']
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        total_steps += 1
    mean_loss = total_loss / max(total_steps, 1)
    return {'loss': mean_loss}

def write_json(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
