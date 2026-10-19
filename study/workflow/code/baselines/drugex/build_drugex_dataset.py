from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from drugex.data.corpus.vocabulary import VocGraph
from drugex_baseline_utils import load_records, prepare_record, save_json, sha256_file, vocabulary_words

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-records', type=Path, required=True)
    parser.add_argument('--validation-records', type=Path, required=True)
    parser.add_argument('--test-records', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-len', type=int, default=80)
    parser.add_argument('--n-frags', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--max-records-per-split', type=int, default=0)
    parser.add_argument('--without-default-vocab', action='store_true')
    return parser.parse_args()

def prepare_split(path: Path, split: str, max_records: int) -> Tuple[List[Dict], Counter]:
    rows = load_records(path)
    if max_records:
        rows = rows[:max_records]
    prepared: List[Dict] = []
    stats = Counter(input=len(rows))
    for row_index, record in enumerate(rows):
        item, reason = prepare_record(record, split, row_index)
        stats[reason] += 1
        if item is not None:
            prepared.append(item)
    return (prepared, stats)

def encode_split(records: List[Dict], vocab: VocGraph, progress_every: int) -> Tuple[np.ndarray, List[Dict], Counter]:
    encoded: List[np.ndarray] = []
    kept: List[Dict] = []
    stats = Counter(input=len(records))
    for index, record in enumerate(records, start=1):
        try:
            array = vocab.encode([record['canonical_smiles']], [record['scaffold_core_smiles']])[0]
            if array.shape != (vocab.max_len, 5):
                stats['wrong_shape'] += 1
                continue
            _, decoded = vocab.decode(np.asarray([array]))
            if not decoded or not decoded[0]:
                stats['decode_empty'] += 1
                continue
            encoded.append(np.asarray(array, dtype=np.int64))
            kept.append(record)
            stats['encoded'] += 1
        except Exception:
            stats['encode_error'] += 1
        if progress_every and index % progress_every == 0:
            print(f'encoded={index} kept={len(kept)} skipped={index - len(kept)}', flush=True)
    if not encoded:
        raise RuntimeError('No DrugEx records were encodable for this split.')
    return (np.stack(encoded, axis=0), kept, stats)

def write_metadata(path: Path, records: List[Dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for index, record in enumerate(records):
            payload = dict(record)
            payload['encoded_index'] = index
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_splits = {'train': args.train_records, 'validation': args.validation_records, 'test': args.test_records}
    prepared_splits: Dict[str, List[Dict]] = {}
    prep_stats: Dict[str, Dict[str, int]] = {}
    for split, path in raw_splits.items():
        prepared, stats = prepare_split(path, split, args.max_records_per_split)
        prepared_splits[split] = prepared
        prep_stats[split] = dict(stats)
        print(f'prepared {split}: {dict(stats)}', flush=True)
    words = vocabulary_words(prepared_splits['train'], include_defaults=not args.without_default_vocab)
    vocab = VocGraph(words=words, max_len=args.max_len, n_frags=args.n_frags)
    vocab_payload = {'words': words, 'max_len': args.max_len, 'n_frags': args.n_frags, 'vocab_size': vocab.size, 'include_default_vocab': not args.without_default_vocab, 'source': 'DrugEx VocGraph'}
    save_json(args.output_dir / 'vocab.json', vocab_payload)
    encoded_stats: Dict[str, Dict[str, int]] = {}
    kept_counts: Dict[str, int] = {}
    for split, records in prepared_splits.items():
        print(f'encoding {split}: {len(records)} records', flush=True)
        data, kept, stats = encode_split(records, vocab, args.progress_every)
        torch.save(torch.from_numpy(data).long(), args.output_dir / f'{split}.pt')
        write_metadata(args.output_dir / f'{split}.jsonl', kept)
        encoded_stats[split] = dict(stats)
        kept_counts[split] = len(kept)
        print(f'saved {split}: shape={tuple(data.shape)} stats={dict(stats)}', flush=True)
    manifest = {'task': 'drugex_graph_transformer_single_attachment', 'source_repository': 'https://github.com/CDDLeiden/DrugEx', 'source_commit': 'bf61ce71db96bdfe25cc2832ec0d4cf0b1142db4', 'max_len': args.max_len, 'n_frags': args.n_frags, 'input_fragment': 'AtomFormer scaffold core with attachment dummies removed', 'coordinates_used': False, 'raw_splits': {split: {'path': str(path), 'sha256': sha256_file(path)} for split, path in raw_splits.items()}, 'prepared_stats': prep_stats, 'encoded_stats': encoded_stats, 'encoded_records': kept_counts, 'files': {split: {'tensor': str((args.output_dir / f'{split}.pt').resolve()), 'metadata': str((args.output_dir / f'{split}.jsonl').resolve())} for split in raw_splits}, 'vocab': vocab_payload}
    save_json(args.output_dir / 'dataset_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
