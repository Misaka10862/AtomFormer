import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures as futures
import json
import math
import multiprocessing as mp
import random
import shutil
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from common import atomic_json, sha256, read_rows, convert, score, reward, augmented_likelihood_score, validate_attempts
from models_adapter import create
from scaffold_metrics import PROTOCOL, protocol_metadata

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(r):
    random.setstate(r['python'])
    np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch'].cpu())
    if r['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in r['cuda']])

def save_checkpoint(path, model, optimizer=None, **extra):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save({'network': model.network.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None, 'rng': rng_state(), **protocol_metadata(), **extra}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer=None):
    state = torch.load(path, map_location=model.device)
    model.network.load_state_dict(state['network'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state

def append_log(path, data):
    print(json.dumps(data), flush=True)
    with path.open('a') as f:
        f.write(json.dumps(data) + '\n')
        f.flush()

def prepare(cfg, root, workers):
    cache = cfg.get('prepared_cache')
    if cache:
        cache = Path(cache)
        report = json.loads((cache / 'audit.json').read_text())
        if not report.get('passed') or not report.get('model_inputs_ready') or report['adapter_sha256'] != cfg.get('prepared_cache_adapter_sha256'):
            raise RuntimeError('prepared cache is not validated against the current adapter')
        for split, expected in cfg['expected_records'].items():
            info = report['splits'][split]
            source = Path(cfg['data_dir']) / f'{split}.pt'
            prepared = cache / f'{split}.jsonl'
            if info['prepared_count'] != expected or sha256(source) != info['source_sha256'] or sha256(prepared) != info['prepared_sha256']:
                raise RuntimeError(f'prepared cache changed: {split}')
            out = root / f'{split}.jsonl'
            if out.exists():
                if sha256(out) != info['prepared_sha256']:
                    raise RuntimeError(f'refusing to overwrite changed prepared data: {split}')
            else:
                shutil.copy2(prepared, out)
        atomic_json(root / 'dataset_manifest.json', report)
        return
    metadata = {}
    for split, expected in cfg['expected_records'].items():
        path = Path(cfg['data_dir']) / f'{split}.pt'
        out = root / f'{split}.jsonl'
        records = torch.load(path, map_location='cpu')
        if len(records) != expected:
            raise ValueError(f'{split}: {len(records)} != {expected}')
        tmp = out.with_suffix('.tmp')
        with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool, tmp.open('w') as f:
            for r in pool.map(convert, enumerate(records), chunksize=128):
                f.write(json.dumps(r) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
        metadata[split] = {'count': len(records), 'source_sha256': sha256(path), 'output_sha256': sha256(out)}
        del records
    atomic_json(root / 'dataset_manifest.json', metadata)

def train(cfg, root, model, rows):
    validation = read_rows(root / 'validation.jsonl')
    for split_rows in (rows, validation):
        if not all((model.encodable(r) for r in split_rows)):
            raise ValueError('unencodable training/validation record, must resolve without dropping records')
    opt = torch.optim.Adam(model.network.parameters(), lr=cfg['supervised_lr'])
    start, best, bad = (0, float('inf'), 0)
    checkpoint = root / 'supervised_resume.pt'
    if checkpoint.exists():
        st = load_checkpoint(checkpoint, model, opt)
        start, best, bad = (st['epoch'], st['best'], st['bad'])
    for epoch in range(start, cfg['max_epochs']):
        if bad >= cfg['patience']:
            break
        tick = time.monotonic()
        order = np.random.permutation(len(rows)).tolist()
        model.network.train()
        total = 0.0
        for start_index in range(0, len(rows), cfg['supervised_batch']):
            batch = [rows[i] for i in order[start_index:start_index + cfg['supervised_batch']]]
            opt.zero_grad(set_to_none=True)
            loss = model.nll(batch).mean()
            if not torch.isfinite(loss):
                raise ValueError('non-finite supervised loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        model.network.eval()
        val = 0.0
        with torch.no_grad():
            for i in range(0, len(validation), cfg['supervised_batch']):
                val += model.nll(validation[i:i + cfg['supervised_batch']]).sum().item()
        val /= len(validation)
        improved = val < best
        bad = 0 if improved else bad + 1
        best = min(best, val)
        if improved:
            save_checkpoint(root / 'supervised_best.pt', model, epoch=epoch + 1)
        save_checkpoint(checkpoint, model, opt, epoch=epoch + 1, best=best, bad=bad)
        append_log(root / 'train_history.jsonl', {'epoch': epoch + 1, 'train_nll': total / len(rows), 'validation_nll': val, 'seconds': time.monotonic() - tick, 'best': best, 'patience_used': bad})
    atomic_json(root / 'supervised.done.json', {'best_validation_nll': best, 'checkpoint_sha256': sha256(root / 'supervised_best.pt'), 'budget_limited': bad < cfg['patience']})

def rl(cfg, root, model, rows, method, workers):
    load_checkpoint(root / 'supervised_best.pt', model)
    prior = create(method, rows, cfg, model.device)
    prior.network.load_state_dict(model.network.state_dict())
    prior.network.eval()
    for p in prior.network.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.network.parameters(), lr=cfg['rl_lr'])
    seed_all(cfg['seed'])
    chosen = np.random.RandomState(cfg['seed']).permutation(len(rows))[:min(cfg['rl_records'], len(rows))].tolist()
    policy_rows = [rows[i] for i in chosen]
    reference = {r['source'] for r in rows}
    offset, steps = (0, 0)
    checkpoint = root / 'rl_resume.pt'
    if checkpoint.exists():
        st = load_checkpoint(checkpoint, model, opt)
        if st.get('scaffold_metric_protocol') != PROTOCOL or st.get('rdkit_version') != protocol_metadata()['rdkit_version']:
            raise RuntimeError('RL resume checkpoint uses a different scaffold protocol/environment')
        offset, steps = (st['attempts'], st['steps'])
    trace = root / 'rl_trace'
    trace.mkdir(exist_ok=True)
    model.network.train()
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        while offset < cfg['rl_attempts']:
            tick = time.monotonic()
            n = min(cfg['rl_batch'], cfg['rl_attempts'] - offset)
            batch = [policy_rows[(offset + k) % len(policy_rows)] for k in range(n)]
            outputs, decs = model.sample(batch)
            scored = list(pool.map(score, [(r, s, d, 0) for r, s, d in zip(batch, outputs, decs)], chunksize=8))
            rewards = [reward(s, reference) for s in scored]
            transformed_scores = [augmented_likelihood_score(s, reference) for s in scored]
            trace_path = trace / f'{offset:09d}.json'
            trace_rows = [{**s, 'attempt_id': offset + k, 'raw_reward': rewards[k], 'augmented_likelihood_score': transformed_scores[k]} for k, s in enumerate(scored)]
            trace_payload = {'attempt_start': offset, 'attempt_end': offset + n, **protocol_metadata(), 'rows': trace_rows}
            if trace_path.exists() and json.loads(trace_path.read_text()) != trace_payload:
                raise RuntimeError('RL trace differs from deterministic checkpoint replay')
            atomic_json(trace_path, trace_payload)
            strings = decs if method == 'libinvent' else outputs
            safe_strings = [s if s is not None else '' for s in strings]
            actor_nll = model.nll(batch, safe_strings)
            with torch.no_grad():
                prior_nll = prior.nll(batch, safe_strings)
            scores = torch.tensor(transformed_scores, device=model.device, dtype=torch.float)
            loss = (actor_nll - prior_nll + cfg['sigma'] * scores).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('non-finite augmented-likelihood loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            opt.step()
            offset += n
            steps += 1
            if steps % 50 == 0 or offset == cfg['rl_attempts']:
                save_checkpoint(checkpoint, model, opt, attempts=offset, steps=steps)
                append_log(root / 'rl.jsonl', {'attempts': offset, 'steps': steps, 'loss': loss.item(), 'raw_reward': statistics.mean(rewards), 'augmented_likelihood_score': statistics.mean(transformed_scores), 'last_batch_seconds': time.monotonic() - tick})
    save_checkpoint(root / 'rl_final.pt', model, attempts=offset)
    atomic_json(root / 'rl.done.json', {'attempts': offset, 'checkpoint_sha256': sha256(root / 'rl_final.pt'), **protocol_metadata(), 'trace_dir': str(trace)})

def evaluate(cfg, root, model, workers):
    checkpoint = load_checkpoint(root / 'rl_final.pt', model)
    if checkpoint.get('scaffold_metric_protocol') != PROTOCOL or checkpoint.get('rdkit_version') != protocol_metadata()['rdkit_version']:
        raise RuntimeError('Evaluation requires an RL checkpoint from this scaffold protocol/environment')
    model.network.eval()
    rows = read_rows(root / 'test.jsonl')
    shards = root / 'shards'
    shards.mkdir(exist_ok=True)
    total = len(rows) * cfg['samples_per_record']

    def commit(future, path):
        atomic_json(path, future.result())
    from collections import deque
    pending = deque()
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        for offset in range(0, total, cfg['eval_batch']):
            path = shards / f'{offset:09d}.json'
            if path.exists():
                chunk = json.loads(path.read_text())
                if any((r.get('scaffold_metric_protocol') != PROTOCOL or r.get('rdkit_version') != protocol_metadata()['rdkit_version'] for r in chunk)):
                    raise RuntimeError('Resume shard uses a different scaffold protocol/environment')
                if [r['record_id'] * cfg['samples_per_record'] + r['sample_id'] for r in chunk] != list(range(offset, min(total, offset + cfg['eval_batch']))):
                    raise ValueError('corrupt resume shard')
                continue
            seed_all(cfg['sampling_seed'] + offset)
            indices = list(range(offset, min(total, offset + cfg['eval_batch'])))
            batch = [rows[i // cfg['samples_per_record']] for i in indices]
            eligible = []
            for j, row in enumerate(batch):
                try:
                    if hasattr(model, 'vocab'):
                        okay = model.vocab.encode_scaffold(row['scaffold']) is not None
                    else:
                        model.model.vocabulary.encode(model.model.tokenizer.tokenize(model.prompt(row)))
                        okay = True
                except (KeyError, ValueError, TypeError, IndexError):
                    okay = False
                if okay:
                    eligible.append(j)
            outputs, decs = ([None] * len(batch), [None] * len(batch))
            if eligible:
                generated, decorations = model.sample([batch[j] for j in eligible])
                for j, s, d in zip(eligible, generated, decorations):
                    outputs[j], decs[j] = (s, d)
            items = [(r, s, d, i % cfg['samples_per_record']) for r, s, d, i in zip(batch, outputs, decs, indices)]
            tasks = [pool.submit(score_chunk, items[j::workers]) for j in range(workers)]
            pending.append((tasks, path, offset))
            if len(pending) >= 2:
                flush_chunk(pending.popleft(), cfg)
        while pending:
            flush_chunk(pending.popleft(), cfg)
    atomic_json(root / 'generation.done.json', {'attempts': total, 'shards': len(list(shards.glob('*.json'))), 'sampling_rng': 'seed 0 + global chunk offset; fixed batch boundaries'})

def score_chunk(items):
    return [score(i) for i in items]

def flush_chunk(pending, cfg):
    tasks, path, offset = pending
    result = sorted([r for t in tasks for r in t.result()], key=lambda r: (r['record_id'], r['sample_id']))
    atomic_json(path, result)
    print(json.dumps({'generated_until': offset + len(result)}), flush=True)

def diversity_partial(args):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    smiles, start, stride = args
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    total, count = (0.0, 0)
    for i in range(start, len(fps) - 1, stride):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum((1.0 - s for s in similarities))
        count += len(similarities)
    return (total, count)

def aggregate(cfg, root, workers):
    rows = [r for path in sorted((root / 'shards').glob('*.json')) for r in json.loads(path.read_text())]
    validate_attempts(rows, cfg['expected_records']['test'], cfg['samples_per_record'])
    reference = {r['source'] for r in read_rows(root / 'train.jsonl')}
    valid = [r for r in rows if r['valid']]
    if any((r.get('scaffold_metric_protocol') != PROTOCOL or r.get('rdkit_version') != protocol_metadata()['rdkit_version'] for r in rows)):
        raise RuntimeError('Cannot aggregate legacy/mixed scaffold scoring shards')
    if any((r['core_retention'] and (not r['core_match']) for r in valid)):
        raise RuntimeError('Scaffold metric implication violated')
    smiles = [r['generated'] for r in valid]
    capped = smiles[::max(1, len(smiles) // 5000)][:5000] if len(smiles) > 5000 else smiles
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        partials = list(pool.map(diversity_partial, [(capped, i, workers) for i in range(workers)]))
    metrics = {**protocol_metadata(), 'num_generated': len(rows), 'num_valid': len(valid), 'validity': len(valid) / len(rows), 'scaffold_core_match_rate': sum((r['core_match'] for r in valid)) / max(1, len(valid)), 'scaffold_core_retention_rate': sum((r['core_retention'] for r in valid)) / max(1, len(valid)), 'scaffold_core_metric_n': len(valid), 'novelty': sum((s not in reference for s in smiles)) / max(1, len(smiles)), 'diversity': sum((p[0] for p in partials)) / max(1, sum((p[1] for p in partials))), 'diversity_cap': 5000, 'training_seeds': [cfg['seed']], 'sampling_seed': cfg['sampling_seed'], 'protocol': cfg, 'record_coverage': len({r['record_id'] for r in rows})}
    for k in ('qed', 'sas'):
        values = [r[k] for r in valid if r[k] is not None and math.isfinite(r[k])]
        metrics[k + '_mean'] = statistics.mean(values) if values else None
        metrics[k + '_n'] = len(values)
    out = root / 'samples.jsonl'
    with out.with_suffix('.tmp').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(out.with_suffix('.tmp'), out)
    metrics['samples_sha256'] = sha256(out)
    atomic_json(root / 'metrics.json', metrics)
