from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from drugex.training.generators import GraphTransformer
from drugex_baseline_utils import canonicalize_smiles, load_sa_scorer, load_vocab, make_vocab, molecule_metrics, pairwise_diversity, save_json, scaffold_checks, protocol_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--train-metadata', type=Path, default=None, help='Baseline train.jsonl used for novelty; defaults to dataset-dir/train.jsonl.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples-per-record', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=int(os.environ.get('DRUGEX_EVAL_BATCH_SIZE', '4096')))
    parser.add_argument('--workers', type=int, default=0, help='CPU worker processes for molecule/scaffold post-processing; 0 selects up to 16 cores.')
    parser.add_argument('--progress-every', type=int, default=5, help='Print generation progress every N batches; 0 disables progress output.')
    parser.add_argument('--max-records', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    return parser.parse_args()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(checkpoint_path: Path, dataset_dir: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    vocab = load_vocab(dataset_dir / 'vocab.json')
    model_args = dict(payload['model_args'])
    model = GraphTransformer(vocab, device=device, use_gpus=(0,) if device.type == 'cuda' else (-1,), **model_args)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return (model, vocab, payload)

def load_metadata(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def resolve_workers(requested: int) -> int:
    if requested < 0:
        raise ValueError('--workers must be non-negative')
    if requested:
        return requested
    return max(1, min(16, os.cpu_count() or 1))

def _score_rows_chunk(chunk: List[Dict]) -> List[Dict]:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    sa_scorer = load_sa_scorer()
    scored: List[Dict] = []
    for source_row in chunk:
        row = dict(source_row)
        values = molecule_metrics(row.get('generated_smiles', ''), sa_scorer=sa_scorer)
        row['canonical_generated_smiles'] = values['canonical_smiles']
        row['qed'] = values['qed']
        row['sas'] = values['sas']
        if values['canonical_smiles'] is not None:
            checks = scaffold_checks(values['canonical_smiles'], row.get('scaffold_core_smiles', ''))
            if not checks['eligible']:
                raise ValueError('Valid DrugEx output has an invalid target core')
            row['_scaffold_core_match'] = checks['core_match']
            row['_scaffold_core_retained'] = checks['core_retention']
        else:
            row['_scaffold_core_match'] = False
            row['_scaffold_core_retained'] = False
        scored.append(row)
    return scored

def _record_stats_chunk(groups: List[List[str]]) -> List[Dict[str, float]]:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    return [{'unique': float(len(set(group))), 'diversity': float(pairwise_diversity(group))} for group in groups]
_DECODE_VOCAB = None

def _init_decode_worker(vocab_payload: Dict) -> None:
    global _DECODE_VOCAB
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _DECODE_VOCAB = make_vocab(vocab_payload)

def _decode_matrix(matrix: np.ndarray) -> List[str]:
    if _DECODE_VOCAB is None:
        raise RuntimeError('Decode worker vocabulary was not initialized')
    _, smiles = _DECODE_VOCAB.decode(matrix)
    return smiles

def _mp_context():
    methods = mp.get_all_start_methods()
    return mp.get_context('spawn' if 'spawn' in methods else methods[0])

def _parallel_map_chunks(function, items: List, workers: int, chunk_size: int) -> List:
    chunks = [items[start:start + chunk_size] for start in range(0, len(items), chunk_size)]
    if workers <= 1 or len(chunks) <= 1:
        return [result for chunk in chunks for result in function(chunk)]
    context = _mp_context()
    results: List = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for chunk_result in pool.map(function, chunks):
            results.extend(chunk_result)
    return results

def generate_rows(args: argparse.Namespace, model: GraphTransformer, vocab, vocab_payload: Dict, workers: int) -> List[Dict]:
    data = torch.load(args.dataset_dir / 'test.pt', map_location='cpu').long()
    metadata = load_metadata(args.dataset_dir / 'test.jsonl')
    if len(data) != len(metadata):
        raise RuntimeError(f'Test tensor/metadata mismatch: {len(data)} != {len(metadata)}')
    if args.max_records:
        data = data[:args.max_records]
        metadata = metadata[:args.max_records]
    if args.samples_per_record < 1:
        raise ValueError('--samples-per-record must be positive')
    repeated_data = data.repeat_interleave(args.samples_per_record, dim=0)
    repeated_metadata = [item for item in metadata for _ in range(args.samples_per_record)]
    loader = DataLoader(repeated_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    rows: List[Dict] = []
    offset = 0
    pending: Deque[Tuple[int, object]] = deque()

    def append_decoded(start: int, smiles: List[str]) -> None:
        batch_metadata = repeated_metadata[start:start + len(smiles)]
        for item, generated_smiles in zip(batch_metadata, smiles):
            row = dict(item)
            row['generated_smiles'] = generated_smiles
            rows.append(row)
    decode_pool = None
    if workers > 1:
        decode_pool = ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context(), initializer=_init_decode_worker, initargs=(vocab_payload,))
    try:
        for step, batch in enumerate(loader, start=1):
            with torch.no_grad():
                generated = model(batch.to(model.device), is_train=False)
            matrix = generated.detach().cpu().numpy()
            batch_start = offset
            n_batch = len(matrix)
            if decode_pool is None:
                _, smiles = vocab.decode(matrix)
                append_decoded(batch_start, smiles)
            else:
                decode_chunk_size = max(128, min(512, args.batch_size // max(workers, 1)))
                for chunk_start in range(0, n_batch, decode_chunk_size):
                    chunk_end = min(chunk_start + decode_chunk_size, n_batch)
                    pending.append((batch_start + chunk_start, decode_pool.submit(_decode_matrix, matrix[chunk_start:chunk_end])))
                if len(pending) >= workers * 2:
                    start, future = pending.popleft()
                    append_decoded(start, future.result())
            offset += n_batch
            if args.progress_every and step % args.progress_every == 0:
                print(f'generated batches={step} rows_submitted={offset} rows_decoded={len(rows)}', flush=True)
        while pending:
            start, future = pending.popleft()
            append_decoded(start, future.result())
    finally:
        if decode_pool is not None:
            decode_pool.shutdown(wait=True)
    if len(rows) != len(repeated_metadata):
        raise RuntimeError(f'Generation count mismatch: {len(rows)} != {len(repeated_metadata)}')
    return rows

def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0

def summarize(rows: List[Dict], train_records: List[Dict], intended_records: int, workers: int) -> Dict:
    train_canonical = {canonicalize_smiles(record.get('canonical_smiles', '')) for record in train_records}
    train_canonical.discard(None)
    if workers > 1:
        score_chunk_size = max(1000, len(rows) // (workers * 4) or 1)
        rows[:] = _parallel_map_chunks(_score_rows_chunk, rows, workers, score_chunk_size)
    else:
        rows[:] = _score_rows_chunk(rows)
    qed: List[float] = []
    sas: List[float] = []
    valid_rows: List[Dict] = []
    for row in rows:
        if row['canonical_generated_smiles'] is not None:
            valid_rows.append(row)
            if row['qed'] is not None:
                qed.append(row['qed'])
            if row['sas'] is not None:
                sas.append(row['sas'])
    valid_smiles = [row['canonical_generated_smiles'] for row in valid_rows]
    unique_smiles = set(valid_smiles)
    match_values = [row['_scaffold_core_match'] for row in valid_rows]
    retention_values = [row['_scaffold_core_retained'] for row in valid_rows]
    by_record: Dict[int, List[Dict]] = defaultdict(list)
    for row in valid_rows:
        by_record[int(row['row_index'])].append(row)
    groups = [[row['canonical_generated_smiles'] for row in values] for values in by_record.values()]
    record_stats = _parallel_map_chunks(_record_stats_chunk, groups, workers, max(100, len(groups) // (workers * 4) or 1))
    unique_per_record = [int(item['unique']) for item in record_stats]
    within_record_diversity = [float(item['diversity']) for item in record_stats]
    metrics = {'intended_input_records': int(intended_records), 'encoded_input_records': int(len({int(row['row_index']) for row in rows})), 'num_generated': int(len(rows)), 'num_valid': int(len(valid_rows)), 'validity': len(valid_rows) / max(len(rows), 1), 'unique_rate': len(unique_smiles) / max(len(valid_rows), 1), 'novelty': sum((smiles not in train_canonical for smiles in unique_smiles)) / max(len(unique_smiles), 1), 'qed_mean': mean_or_zero(qed), 'qed_std': float(np.std(qed)) if qed else 0.0, 'sas_mean': mean_or_zero(sas), 'sas_std': float(np.std(sas)) if sas else 0.0, 'diversity': pairwise_diversity(valid_smiles), 'scaffold_core_match_rate': mean_or_zero([float(value) for value in match_values]), 'scaffold_core_retention_rate': mean_or_zero([float(value) for value in retention_values]), 'scaffold_core_metric_n': len(valid_rows), 'mean_unique_per_record': mean_or_zero(unique_per_record), 'per_record_unique_rate': sum((value > 1 for value in unique_per_record)) / max(len(unique_per_record), 1), 'mean_within_record_diversity': mean_or_zero(within_record_diversity)}
    for row in rows:
        row.pop('_scaffold_core_match', None)
        row.pop('_scaffold_core_retained', None)
    return metrics

def main() -> None:
    args = parse_args()
    if any(((args.output_dir / name).exists() for name in ('metrics.json', 'samples.jsonl'))):
        raise RuntimeError('Use a new output directory for the corrected scaffold protocol')
    seed_everything(args.seed)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(args.workers)
    print(f'CPU post-processing workers={workers}', flush=True)
    model, vocab, checkpoint = load_model(args.checkpoint, args.dataset_dir, device)
    vocab_payload = json.loads((args.dataset_dir / 'vocab.json').read_text(encoding='utf-8'))
    rows = generate_rows(args, model, vocab, vocab_payload, workers)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    train_metadata_path = args.train_metadata or args.dataset_dir / 'train.jsonl'
    train_records = load_metadata(train_metadata_path)
    intended_records = len(load_metadata(args.dataset_dir / 'test.jsonl'))
    if args.max_records:
        intended_records = min(intended_records, args.max_records)
    metrics = summarize(rows, train_records, intended_records, workers)
    payload = {**protocol_metadata(), 'task': 'drugex_graph_transformer_single_attachment', 'metrics': metrics, 'args': vars(args), 'workers': workers, 'checkpoint_epoch': checkpoint.get('epoch'), 'checkpoint_seed': checkpoint.get('seed'), 'model_args': checkpoint.get('model_args'), 'device': str(device), 'torch_version': torch.__version__, 'cuda_build': torch.version.cuda}
    save_json(args.output_dir / 'metrics.json', payload)
    with (args.output_dir / 'samples.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
if __name__ == '__main__':
    main()
