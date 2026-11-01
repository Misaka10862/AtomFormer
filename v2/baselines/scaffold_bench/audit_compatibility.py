"""Full input compatibility diagnostic, with no training, CUDA, or shutdown.

Scan all records, collect ALL errors rather than stopping at the first one,
and retain a prepared cache only when every record roundtrips correctly.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import collections
import gc
import json
from pathlib import Path
import sys
import time
import torch
from rdkit import RDLogger
from common import atomic_json, sha256, convert


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    torch.set_num_threads(1)
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, '/root/autodl-tmp/baseline-upstream/libinvent')
    from models.vocabulary import SMILESTokenizer as LibTokenizer
    from smilesrnn.vocabulary import SMILESTokenizer as PromptTokenizer
    from promptsmiles import ScaffoldDecorator
    lt, pt = LibTokenizer(), PromptTokenizer()
    train_tokens = [set(), set(), set()]
    stats, errors = {}, []
    for split, expected in cfg['expected_records'].items():
        records = torch.load(Path(cfg['data_dir']) / f'{split}.pt', map_location='cpu')
        count = 0
        unseen = [collections.Counter(), collections.Counter(), collections.Counter()]
        tick = time.monotonic()
        with (a.output / f'{split}.jsonl').open('w') as out:
            for i, r in enumerate(records):
                try:
                    item = convert((i, r))
                    wrapper = ScaffoldDecorator(scaffold=item['scaffold'], batch_size=1,
                                                sample_fn=lambda **kw: None,
                                                evaluate_fn=lambda *args: None,
                                                optimize_prompts=False, shuffle=False)
                    prompt = wrapper.scaffolds[0]['variants'][0].strip_smiles
                    item['prompt'] = prompt
                    tokens = [lt.tokenize(item['scaffold'], with_begin_and_end=False),
                              lt.tokenize(item['decoration'], with_begin_and_end=False),
                              pt.tokenize(item['source'], with_begin_and_end=False)]
                    if split == 'train':
                        for vocabulary, sequence in zip(train_tokens, tokens):
                            vocabulary.update(sequence)
                    else:
                        for counter, vocabulary, sequence in zip(unseen, train_tokens, tokens):
                            counter.update(set(sequence) - vocabulary)
                    out.write(json.dumps(item) + '\n')
                    count += 1
                except Exception as exc:
                    errors.append({'split': split, 'record_id': i, 'error': str(exc)})
                if i and i % 10000 == 0:
                    atomic_json(a.output / 'progress.json', {'split': split, 'visited': i + 1, 'errors': len(errors)})
            out.flush()
            os.fsync(out.fileno())
        stats[split] = {'input_count': len(records), 'prepared_count': count, 'expected': expected,
                        'seconds': time.monotonic() - tick, 'unseen_tokens': [dict(c) for c in unseen],
                        'source_sha256': sha256(Path(cfg['data_dir']) / f'{split}.pt'),
                        'prepared_sha256': sha256(a.output / f'{split}.jsonl')}
        del records
        gc.collect()
        atomic_json(a.output / 'errors.json', errors)
        atomic_json(a.output / 'progress.json', {'completed_splits': stats, 'errors': len(errors)})
    # Prompt vocabulary comes from complete training molecules. Audit prompts
    # against it only after collecting that complete training vocabulary.
    for split in stats:
        unknown = collections.Counter()
        longest = 0
        with (a.output / f'{split}.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                sequence = pt.tokenize(row['prompt'], with_begin_and_end=False)
                longest = max(longest, len(sequence))
                unknown.update(set(sequence) - train_tokens[2])
        stats[split]['prompt_unseen_tokens'] = dict(unknown)
        stats[split]['max_prompt_tokens'] = longest
    passed = not errors and all(s['prepared_count'] == s['expected'] for s in stats.values())
    atomic_json(a.output / 'audit.json', {'passed': passed, 'formal_experiment': False,
                                        'splits': stats, 'errors': len(errors),
                                        'adapter_sha256': sha256(Path(__file__).with_name('common.py'))})
    print(json.dumps({'passed': passed, 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    main()
