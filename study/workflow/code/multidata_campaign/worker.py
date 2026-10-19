import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from rerun_campaign.runtime import atomic_json, atomic_torch, frozen_contract, sha256, execution_identity, signature, read_jsonl, write_jsonl

def bench_imports():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'baselines/scaffold_bench'))
    sys.path.insert(0, str(root / 'baselines/drugex'))

def copy_checked(source, target):
    source, target = (Path(source), Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError('Refusing changed existing artifact: ' + str(target))
    else:
        shutil.copy2(source, target)

def adapt(plan, dataset, family, workers):
    bench_imports()
    data = data_dir(plan, dataset)
    root = Path(plan['root']) / 'adapted' / dataset / family
    root.mkdir(parents=True, exist_ok=True)
    contract = {'data': {n: sha256(data / f'{n}.pt') for n in ('train', 'validation', 'test')}, 'policy': plan['policy_sha256'], 'family': family}
    frozen_contract(root / 'adapter_contract.json', contract)
    if family == 'drugex':
        from build_drugex_dataset import main as build
        before = sys.argv
        sys.argv = ['build_drugex_dataset.py', '--train-records', str(data / 'train.pt'), '--validation-records', str(data / 'validation.pt'), '--test-records', str(data / 'test.pt'), '--output-dir', str(root), '--max-len', '128']
        try:
            build()
        finally:
            sys.argv = before
        roundtrip_exclusions = {}
        import torch
        from drugex_baseline_utils import load_vocab
        from common import same_molecule
        vocab = load_vocab(root / 'vocab.json')
        for split in ('train', 'validation', 'test'):
            rows = list(read_jsonl(root / f'{split}.jsonl'))
            tensors = torch.load(root / f'{split}.pt', map_location='cpu')
            keep = []
            for offset in range(0, len(rows), 512):
                _, decoded = vocab.decode(tensors[offset:offset + 512].numpy())
                keep.extend((offset + j for j, (r, s) in enumerate(zip(rows[offset:offset + 512], decoded)) if same_molecule(r['canonical_smiles'], s)))
            roundtrip_exclusions[split] = {'excluded': len(rows) - len(keep), 'kept': len(keep)}
            if not keep:
                raise ValueError('Empty stereo/label-preserving DrugEx encoding split')
            rows = [rows[i] for i in keep]
            atomic_torch(root / f'{split}.pt', tensors[keep])
            source_rows = __import__('torch').load(data / f'{split}.pt', map_location='cpu')
            for i, r in enumerate(rows):
                r.update(origin_encoded_index=r['encoded_index'], encoded_index=i, record_id=r['row_index'], stable_id=source_rows[r['row_index']].get('stable_id', str(r['row_index'])))
            write_jsonl(root / f'{split}.jsonl', rows)
        manifest = json.loads((root / 'dataset_manifest.json').read_text())
        manifest['exact_roundtrip_coverage'] = roundtrip_exclusions
        atomic_json(root / 'dataset_manifest.json', manifest)
    else:
        from common import convert
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        import torch
        counts = {}
        for split in ('train', 'validation', 'test'):
            rows = torch.load(data / f'{split}.pt', map_location='cpu')
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                prepared = list(pool.map(convert, enumerate(rows), chunksize=64))
            for r, origin in zip(prepared, rows):
                r['stable_id'] = origin.get('stable_id', str(r['record_id']))
            write_jsonl(root / f'{split}.jsonl', prepared)
            counts[split] = len(prepared)
        from models_adapter import create
        cfg = {'hidden_size': 512, 'num_layers': 3, 'dropout': 0.0, 'max_length': 256, 'prompt_ring_grammar': family == 'promptsmiles' and dataset != 'gdb13'}
        model = create(family, list(read_jsonl(root / 'train.jsonl')), cfg, 'cpu')
        coverage = {}
        for split in ('train', 'validation', 'test'):
            rows = list(read_jsonl(root / f'{split}.jsonl'))
            supported = []
            for r in rows:
                if split != 'test':
                    okay = model.encodable(r)
                else:
                    try:
                        if family == 'libinvent':
                            okay = model.vocab.encode_scaffold(r['scaffold']) is not None
                        else:
                            model.model.vocabulary.encode(model.model.tokenizer.tokenize(model.prompt(r).replace('[SH]', 'S')))
                            okay = True
                    except (KeyError, ValueError, TypeError, IndexError):
                        okay = False
                    r['input_supported'] = okay
                if okay:
                    supported.append(r['record_id'])
            coverage[split] = {'records': len(rows), 'supported_records': len(supported), 'unsupported_record_ids': sorted(set((r['record_id'] for r in rows)) - set(supported))}
            if split != 'test' and len(supported) != len(rows):
                atomic_json(root / 'coverage_failure.json', coverage)
                raise ValueError('Unencodable native train/validation; do not drop records or derive vocabulary from held-out labels')
            if split == 'test':
                write_jsonl(root / f'{split}.jsonl', rows)
        atomic_json(root / 'input_support.json', coverage)
        atomic_json(root / 'dataset_manifest.json', {'contract': contract, 'counts': counts, 'input_coverage': coverage, 'vocabulary_source': 'training plus predefined ring syntax' if cfg.get('prompt_ring_grammar') else 'training only', 'predefined_ring_tokens': list('0123456789') + ['%%%02d' % i for i in range(10, 100)] if cfg.get('prompt_ring_grammar') else []})
    return root

def baseline_cfg(plan, unit, profile=False):
    r = resource_for(plan, unit, profile)
    n = __import__('torch').load(data_dir(plan, unit['dataset_id']) / 'train.pt', map_location='cpu')
    count = min(len(n), 32768)
    policy_count = int(count * (1 - 0.1))
    return {'seed': unit['seed'], 'sampling_seed': 0, 'max_length': unit['max_length'], 'hidden_size': 512, 'num_layers': 3, 'dropout': 0.0, 'supervised_lr': 0.001, 'rl_lr': 0.0001, 'supervised_batch': r['baseline_supervised_batch'], 'rl_batch': 64, 'max_epochs': 12, 'patience': 3, 'sigma': 120.0, 'rl_records': policy_count, 'rl_attempts': min(20 * 64, 1024) if profile else policy_count * 4, **({'prompt_ring_grammar': True} if unit['family'] == 'promptsmiles' and unit['dataset_id'] != 'gdb13' else {})}

def native_supervised(plan, unit, profile=False):
    bench_imports()
    import torch
    from common import read_rows
    from models_adapter import create
    from worker import seed_all, save_checkpoint
    from .training import durable_epochs
    cfg = baseline_cfg(plan, unit, profile)
    family = unit['family']
    data = Path(plan['root']) / 'adapted' / unit['dataset_id'] / family
    rows = read_rows(data / 'train.jsonl')
    val = read_rows(data / 'validation.jsonl')
    if profile:
        rows = rows[:1024]
        val = rows[:100]
    seed_all(unit['seed'])
    model = create(family, rows, cfg, 'cuda')
    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.001)
    for split_rows in (rows, val):
        if not all((model.encodable(r) for r in split_rows)):
            raise ValueError('Native train/validation coverage failure')
    root = Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'supervised'
    contract = {'unit': unit, 'config': cfg, 'policy': plan['policy_sha256'], 'profile': profile, 'data': {n: sha256(data / f'{n}.jsonl') for n in ('train', 'validation')}}

    def step(indices):
        optimizer.zero_grad(set_to_none=True)
        loss = model.nll([rows[i] for i in indices]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    def evaluate():
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(val), cfg['supervised_batch']):
                total += float(model.nll(val[i:i + cfg['supervised_batch']]).sum())
        return total / len(val)

    def save(path, epoch, history):
        save_checkpoint(path, model, epoch=epoch, history=history, contract=contract)
    result = durable_epochs(model.network, optimizer, len(rows), step, evaluate, save, root, contract, 12, cfg['supervised_batch'], patience=3, max_updates=50 if profile else None)
    atomic_json(root / 'run_manifest.json', {'contract': contract, 'result': result})
    return root / ('last.pt' if profile else 'best.pt')

def sup_path(plan, unit, profile=False):
    if profile and unit.get('profile_prior'):
        return Path(unit['profile_prior']['checkpoint'])
    if unit['supervised_reuse'] and (not profile):
        return Path(unit['supervised_checkpoint'])
    return Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'supervised' / ('best.pt' if unit['family'] != 'atomformer' and (not profile) else 'last.pt')

def supervised(plan, unit, profile=False):
    if unit['family'] == 'atomformer':
        from .training import supervised as run
        return run(plan, unit, profile)
    if unit['family'] == 'drugex':
        bench_imports()
        from .drugex import supervised as run
        return run(plan, unit, Path(plan['root']) / 'adapted' / unit['dataset_id'] / 'drugex', profile)
    return native_supervised(plan, unit, profile)

def reinforcement(plan, unit, workers, profile=False):
    prior = sup_path(plan, unit, profile)
    root = Path(plan['root']) / ('profiles' if profile else 'models') / unit['unit_id'] / 'rl'
    if unit['family'] == 'atomformer':
        from train_v4_rl import parse_args
        from .af_rl import run
        data = data_dir(plan, unit['dataset_id'])
        train = data / 'train.pt'
        if profile:
            import torch
            input_root = root.parent / 'profile_inputs'
            input_root.mkdir(parents=True, exist_ok=True)
            train = input_root / 'train.pt'
            atomic_torch(train, torch.load(data / 'train.pt', map_location='cpu')[:1024])
        before = sys.argv
        sys.argv = ['train_v4_rl.py', '--checkpoint', str(prior), '--train-path', str(train), '--output-dir', str(root), '--epochs', '1' if profile else '4', '--batch-size', '64', '--learning-rate', '2e-6', '--seed', str(unit['seed']), '--max-train-samples', '32768', '--max-length', str(unit['max_length']), '--supervised-weight', '.2', '--sas-weight', '.2', '--qed-weight', '1', '--novelty-weight', '.2', '--representation', unit['representation'], '--reward-workers', str(workers), '--campaign-resume', '--resume', '--device', 'cuda']
        try:
            args = parse_args()
            run(args)
        finally:
            sys.argv = before
        return root / 'last.pt'
    if unit['family'] == 'drugex':
        bench_imports()
        from .drugex import reinforcement as run
        return run(plan, unit, Path(plan['root']) / 'adapted' / unit['dataset_id'] / 'drugex', prior, profile)
    bench_imports()
    from common import read_rows
    from models_adapter import create
    from worker import seed_all, rl
    cfg = baseline_cfg(plan, unit, profile)
    root.mkdir(parents=True, exist_ok=True)
    contract = {'unit': unit, 'config': cfg, 'prior': sha256(prior), 'profile': profile, 'policy': plan['policy_sha256']}
    frozen_contract(root / 'training_contract.json', contract)
    data = Path(plan['root']) / 'adapted' / unit['dataset_id'] / unit['family']
    rows = read_rows(data / 'train.jsonl')
    if profile:
        rows = rows[:1024]
    copy_checked(prior, root / 'supervised_best.pt')
    seed_all(unit['seed'])
    model = create(unit['family'], rows, cfg, 'cuda')
    rl(cfg, root, model, rows, unit['family'], workers)
    copy_checked(root / 'rl_final.pt', root / 'last.pt')
    atomic_json(root / 'run_manifest.json', {'contract': contract, 'checkpoint_sha256': sha256(root / 'last.pt')})
    return root / 'last.pt'

def generate(plan, run, profile=False):
    if run['dataset_id'] == 'gdb13' and run['family'] == 'atomformer':
        from .af_generation_compat import generate as compat
        return compat(plan, run, profile)
    from .generation import generate as gen
    return gen(plan, run, profile)

def data_dir(plan, dataset):
    return Path(plan['runtime_paths']['datasets'][dataset])

def resource_for(plan, unit, profile=False):
    if profile:
        raise ValueError('Use the recorded production resources')
    return unit['resource']
