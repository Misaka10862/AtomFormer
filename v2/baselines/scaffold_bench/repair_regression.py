"""Tiny CPU rehearsal using actual audited records, with no shutdown path."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from common import atomic_json, sha256, read_rows
from models_adapter import create


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    cache = Path(cfg['prepared_cache'])
    rows = []
    for split, wanted in [('train', {0, 1, 2, 123}), ('validation', {15596, 16000})]:
        found = 0
        with (cache / f'{split}.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                if r['record_id'] in wanted:
                    r['source_split'] = split
                    rows.append(r)
                    found += 1
                if found == len(wanted):
                    break
        assert found == len(wanted)
    assert len(rows) == 6
    for i, row in enumerate(rows):
        row['source_record_id'] = row['record_id']
        row['record_id'] = i
    # Tiny network and six examples for the CPU rehearsal. Complete production
    # vocabulary coverage is checked separately by seal_compatibility.py.
    cfg.update(hidden_size=16, num_layers=2, max_epochs=1, rl_attempts=6, rl_records=6,
               supervised_batch=3, rl_batch=3, eval_batch=3, samples_per_record=2,
               run_root=str(a.output), expected_records={'train': 6, 'validation': 6, 'test': 6})
    cfg.pop('prepared_cache', None)
    a.output.mkdir(parents=True, exist_ok=False)
    root = a.output / a.method
    root.mkdir()
    for split in cfg['expected_records']:
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    atomic_json(a.output / 'config.json', cfg)
    model = create(a.method, rows, cfg, 'cpu')
    # Full geometry/ring syntax in these source fragments exercises more than
    # the original simple benzene smoke. Test all singleton dimensions too.
    for row in rows:
        assert model.encodable(row), row
        assert torch.isfinite(model.nll([row])).all()
    data_hash = sha256(root / 'train.jsonl')
    for task in ('train', 'train', 'rl', 'rl', 'evaluate', 'evaluate', 'aggregate'):
        subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py')), task,
                        '--method', a.method, '--config', str(a.output / 'config.json'),
                        '--device', 'cpu', '--workers', '1', '--smoke'], check=True)
    assert sha256(root / 'train.jsonl') == data_hash
    metrics = json.loads((root / 'metrics.json').read_text())
    assert metrics['num_generated'] == 12 and metrics['record_coverage'] == 6
    atomic_json(a.output / 'PASSED.json', {'method': a.method, 'records': 6, 'attempts': 12,
                                         'formal_experiment': False, 'shutdown': False})


if __name__ == '__main__':
    main()
