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
        return records, []
    split_idx = max(1, int(len(records) * (1 - val_ratio)))
    return records[:split_idx], records[split_idx:]




def write_json(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
