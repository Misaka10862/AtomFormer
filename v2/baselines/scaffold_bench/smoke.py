"""Tiny CPU integration test using synthetic molecules, never experiment data."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from common import convert, join, atomic_json, read_rows, validate_attempts, sha256
from models_adapter import create


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    cfg.update(run_root=str(root), hidden_size=16, num_layers=2, max_length=32, max_epochs=1,
               rl_attempts=8, rl_records=4, supervised_batch=3, rl_batch=3, eval_batch=3,
               samples_per_record=2, expected_records={'train': 4, 'validation': 4, 'test': 4})
    rows = [convert((i, {'scaffold_smiles': '*c1ccccc1', 'rooted_decoration_smiles': d,
                         'canonical_smiles': d + 'c1ccccc1'})) for i, d in enumerate(['C', 'N', 'O', 'CC'])]
    assert join(rows[0]['scaffold'], 'bad') is None
    assert join(rows[0]['scaffold'], '**') is None
    run = root / a.method
    run.mkdir()
    for split in cfg['expected_records']:
        (run / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    path = root / 'config.json'
    atomic_json(path, cfg)
    model = create(a.method, rows, cfg, 'cpu')
    assert all(model.encodable(r) for r in rows)
    nll = model.nll(rows[:1])
    assert nll.shape == (1,) and torch.isfinite(nll).all()
    if a.method == 'promptsmiles':
        # Distinct scaffolds must retain order in the actual batched call.
        import unittest.mock
        other = dict(rows[0], scaffold='*C1CCCCC1')
        batch = [rows[0], other, rows[1]]
        with unittest.mock.patch.object(model.model, '_pSMILES_sample', side_effect=lambda prompt, **kw: prompt):
            out, _ = model.sample(batch)
            assert out == [model.prompt(r) for r in batch]
            assert out[0] != out[1]
    data_hash = sha256(run / 'train.jsonl')
    held = None
    for task in ('probe', 'train', 'train', 'rl', 'rl', 'evaluate', 'evaluate', 'aggregate'):
        subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py')), task, '--method', a.method,
                        '--config', str(path), '--device', 'cpu', '--workers', '1', '--smoke'], check=True)
        if task == 'evaluate':
            shard = sorted((run / 'shards').glob('*.json'))[-1]
            if held is None:
                held = (shard, shard.read_bytes())
                shard.rename(shard.with_suffix('.test-backup'))
            else:
                assert shard.read_bytes() == held[1], 'resumed shard differs from original'
    assert sha256(run / 'train.jsonl') == data_hash, 'training data mutated'
    result = read_rows(run / 'samples.jsonl')
    assert validate_attempts(result, 4, 2) == 8
    assert json.loads((run / 'rl.done.json').read_text())['attempts'] == 8
    atomic_json(root / 'CPU_TEST_PASSED.json', {'method': a.method, 'time': time.time(), 'attempts': 8,
                                             'formal_results': False, 'gpu_used': False})


if __name__ == '__main__': main()
