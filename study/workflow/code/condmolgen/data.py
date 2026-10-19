from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from .tokenizer import CharTokenizer

def load_records(path: str) -> List[Dict]:
    records = torch.load(path, map_location='cpu')
    if not isinstance(records, list):
        raise ValueError(f'Expected a list of records in {path}.')
    return records

@dataclass
class Sample:
    z: torch.Tensor
    pos: torch.Tensor
    input_ids: torch.Tensor
    labels: torch.Tensor
    canonical_smiles: str
    cond_smiles: str
    source_smiles: str
    scaffold_smiles: str
    anchor_z: int

class ConditionalDataset(Dataset):

    def __init__(self, records: List[Dict], tokenizer: CharTokenizer, use_scaffold_prefix: bool=False, target_key: str='canonical_smiles', prefix_key: str='scaffold_smiles') -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.use_scaffold_prefix = use_scaffold_prefix
        self.target_key = target_key
        self.prefix_key = prefix_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Sample:
        record = self.records[idx]
        smiles = record[self.target_key]
        scaffold_smiles = record.get(self.prefix_key, record.get('scaffold_smiles', record.get('cond_smiles', '')))
        if self.use_scaffold_prefix:
            token_ids = self.tokenizer.encode_scaffold_target(scaffold_smiles, smiles)
        else:
            token_ids = self.tokenizer.encode_smiles(smiles)
        return Sample(z=torch.tensor(record['cond_z'], dtype=torch.long), pos=torch.tensor(record['cond_pos'], dtype=torch.float32), input_ids=torch.tensor(token_ids, dtype=torch.long), labels=torch.tensor(token_ids, dtype=torch.long), canonical_smiles=record.get('canonical_smiles', smiles), cond_smiles=record.get('cond_smiles', ''), source_smiles=record.get('source_smiles', smiles), scaffold_smiles=scaffold_smiles, anchor_z=int(record.get('anchor_z', 0)))

class ConditionalBatchCollator:

    def __init__(self, tokenizer: CharTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_nodes = max((sample.z.size(0) for sample in batch))
        max_seq = max((sample.input_ids.size(0) for sample in batch))
        z = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        pos = torch.zeros(batch_size, max_nodes, 3, dtype=torch.float32)
        input_ids = torch.full((batch_size, max_seq), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq), fill_value=-100, dtype=torch.long)
        decoder_padding_mask = torch.ones(batch_size, max_seq, dtype=torch.bool)
        scaffold_target_ids = torch.full((batch_size, max_seq), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        scaffold_token_mask = torch.zeros(batch_size, max_seq, dtype=torch.bool)
        anchor_z = torch.zeros(batch_size, dtype=torch.long)
        canonical_smiles: List[str] = []
        source_smiles: List[str] = []
        scaffold_smiles: List[str] = []
        for i, sample in enumerate(batch):
            n = sample.z.size(0)
            t = sample.input_ids.size(0)
            z[i, :n] = sample.z
            pos[i, :n] = sample.pos
            input_ids[i, :t] = sample.input_ids
            labels[i, :t] = sample.labels
            decoder_padding_mask[i, :t] = False
            scaffold_ids = self.tokenizer.encode_smiles(sample.scaffold_smiles)
            s = min(len(scaffold_ids), max_seq)
            scaffold_target_ids[i, :s] = torch.tensor(scaffold_ids[:s], dtype=torch.long)
            scaffold_token_mask[i, :s] = True
            anchor_z[i] = sample.anchor_z
            canonical_smiles.append(sample.canonical_smiles)
            source_smiles.append(sample.source_smiles)
            scaffold_smiles.append(sample.scaffold_smiles)
        return {'z': z, 'pos': pos, 'input_ids': input_ids, 'labels': labels, 'decoder_padding_mask': decoder_padding_mask, 'scaffold_target_ids': scaffold_target_ids, 'scaffold_token_mask': scaffold_token_mask, 'anchor_z': anchor_z, 'canonical_smiles': canonical_smiles, 'source_smiles': source_smiles, 'scaffold_smiles': scaffold_smiles}
