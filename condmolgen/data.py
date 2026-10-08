from typing import Dict, List

import torch

from .tokenizer import CharTokenizer


def load_records(path: str) -> List[Dict]:
    records = torch.load(path, map_location="cpu")
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}.")
    return records
