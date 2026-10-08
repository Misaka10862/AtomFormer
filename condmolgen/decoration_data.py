from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class DecorationSample:
    z: torch.Tensor
    pos: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    anchor_z: torch.Tensor
    canonical_smiles: str
    scaffold_smiles: str
    target_decoration_smiles: str
    source_smiles: str

    @property
    def rooted_decoration_smiles(self) -> str:
        """Rooted-decoration target accessor."""

        return self.target_decoration_smiles


def build_decoration_tokenizer(
    records: List[Dict],
    tokenizer_path: Optional[str] = None,
    target_key: str = "rooted_decoration_smiles",
) -> CharTokenizer:
    texts: List[str] = []
    for record in records:
        scaffold = record.get("scaffold_smiles")
        decoration = record.get(target_key)
        if scaffold and decoration:
            texts.append(scaffold)
            texts.append(decoration)
    if not texts:
        raise ValueError(f"No scaffold/{target_key} texts are available for tokenizer construction.")
    return CharTokenizer(tokenizer_path=tokenizer_path, texts=texts)


def encode_decoration_sequence(tokenizer: CharTokenizer, scaffold_smiles: str, target_decoration_smiles: str) -> Dict[str, List[int]]:
    input_ids = tokenizer.encode_scaffold_target(scaffold_smiles, target_decoration_smiles)
    prefix_len = 1 + len(scaffold_smiles) + 1
    labels = [-100] * len(input_ids)
    labels[prefix_len:] = input_ids[prefix_len:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prefix_len": prefix_len,
    }


class DecorationDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tokenizer: CharTokenizer,
        max_length: int = 0,
        target_key: str = "rooted_decoration_smiles",
        prefix_key: str = "scaffold_smiles",
    ) -> None:
        self.records: List[Dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_key = target_key
        self.prefix_key = prefix_key

        for record in records:
            scaffold = record.get(prefix_key)
            decoration = record.get(target_key)
            if not scaffold or not decoration:
                continue
            if "cond_z" not in record or "cond_pos" not in record:
                continue
            encoded = encode_decoration_sequence(tokenizer, scaffold, decoration)
            if max_length and len(encoded["input_ids"]) > max_length:
                continue
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DecorationSample:
        record = self.records[idx]
        scaffold = record[self.prefix_key]
        decoration = record[self.target_key]
        encoded = encode_decoration_sequence(self.tokenizer, scaffold, decoration)
        return DecorationSample(
            z=torch.tensor(record["cond_z"], dtype=torch.long),
            pos=torch.tensor(record["cond_pos"], dtype=torch.float32),
            input_ids=torch.tensor(encoded["input_ids"], dtype=torch.long),
            labels=torch.tensor(encoded["labels"], dtype=torch.long),
            anchor_z=torch.tensor(int(record.get("anchor_z", 0)), dtype=torch.long),
            canonical_smiles=record.get("canonical_smiles", record.get("source_smiles", "")),
            scaffold_smiles=scaffold,
            target_decoration_smiles=decoration,
            source_smiles=record.get("source_smiles", record.get("canonical_smiles", "")),
        )


class DecorationBatchCollator:
    def __init__(self, tokenizer: CharTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[DecorationSample]) -> Dict:
        batch_size = len(batch)
        max_nodes = max(sample.z.size(0) for sample in batch)
        max_seq = max(sample.input_ids.size(0) for sample in batch)

        z = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        pos = torch.zeros(batch_size, max_nodes, 3, dtype=torch.float32)
        input_ids = torch.full(
            (batch_size, max_seq),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full((batch_size, max_seq), fill_value=-100, dtype=torch.long)
        decoder_padding_mask = torch.ones(batch_size, max_seq, dtype=torch.bool)
        anchor_z = torch.zeros(batch_size, dtype=torch.long)
        canonical_smiles: List[str] = []
        scaffold_smiles: List[str] = []
        target_decoration_smiles: List[str] = []
        source_smiles: List[str] = []

        for i, sample in enumerate(batch):
            n = sample.z.size(0)
            t = sample.input_ids.size(0)
            z[i, :n] = sample.z
            pos[i, :n] = sample.pos
            input_ids[i, :t] = sample.input_ids
            labels[i, :t] = sample.labels
            decoder_padding_mask[i, :t] = False
            anchor_z[i] = sample.anchor_z
            canonical_smiles.append(sample.canonical_smiles)
            scaffold_smiles.append(sample.scaffold_smiles)
            target_decoration_smiles.append(sample.target_decoration_smiles)
            source_smiles.append(sample.source_smiles)

        return {
            "z": z,
            "pos": pos,
            "input_ids": input_ids,
            "labels": labels,
            "decoder_padding_mask": decoder_padding_mask,
            "anchor_z": anchor_z,
            "canonical_smiles": canonical_smiles,
            "scaffold_smiles": scaffold_smiles,
            "target_decoration_smiles": target_decoration_smiles,
            # Rooted-decoration target field.
            "rooted_decoration_smiles": target_decoration_smiles,
            "source_smiles": source_smiles,
        }
