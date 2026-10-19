"""Run one scientific stage with a selected recorded configuration on a GPU host."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code'
CONFIGS = {'atomformer': 'rooted_balanced_rl', 'supervised': 'rooted_sup',
           'plain_supervised': 'plain_sup', 'plain_balanced_rl': 'plain_balanced_rl',
           'pure_rl': 'rooted_pure_rl', 'geometry_extension': 'full_rl',
           'drugex': 'drugex_rl', 'libinvent': 'libinvent_rl', 'promptsmiles': 'promptsmiles_rl'}


def select(dataset, configuration, run):
    rows = json.loads((HERE / 'run_parameters.json').read_text())['runs']
    return copy.deepcopy(next(r for r in rows if (r['dataset'], r['configuration'], r['run']) == (dataset, configuration, run)))


def blocks(row, phase):
    result = {}
    for block in row[phase + '_parameter_blocks']:
        result.update(block['parameters'])
    return result


def paths(upstream):
    sys.path.insert(0, str(CODE))
    for p in [CODE / 'baselines/scaffold_bench', CODE / 'baselines/drugex']:
        sys.path.insert(0, str(p))
    for name in ['libinvent', 'smilesrnn', 'promptsmiles', 'drugex']:
        sys.path.insert(0, str(upstream / name))


def invoke(module_name, required_args, recorded):
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name] + required_args
        args = module.parse_args()
        for key, value in recorded.items():
            if hasattr(args, key):
                setattr(args, key, value)
        module.parse_args = lambda: args
        module.main()
    finally:
        sys.argv = previous


def generate_original_record_batches(a, row):
    import evaluate_v4
    import condmolgen.checkpoint as checkpoints
    from rerun_campaign import generate
    from rerun_campaign.runtime import sha256
    from scaffold_metrics import protocol_metadata
    previous = sys.argv
    output = a.work_root / 'results' / (a.dataset + '__' + a.configuration + '__run' + str(a.run))
    output.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = ['evaluate_v4', '--checkpoint', str(a.checkpoint), '--processed-path', str(a.data_dir / 'test.pt'),
                    '--output-json', str(output / 'generation.json')]
        args = evaluate_v4.parse_args()
    finally:
        sys.argv = previous
    recorded = next(item['blocks']['args'] for item in row['recorded_generation_metadata'] if 'args' in item['blocks'])
    for key, value in recorded.items():
        if hasattr(args, key): setattr(args, key, value)
    args.device = 'cuda'
    evaluate_v4.parse_args = lambda: args
    original_load = checkpoints.load_checkpoint
    known = {m['sha256'] for m in json.loads((HERE / 'public_models.json').read_text())['models']}
    def load(path, map_location='cpu'):
        model, tokenizer, extra = original_load(path, map_location)
        if sha256(path) in known:
            extra = dict(extra, scaffold_metric_protocol=protocol_metadata()['scaffold_metric_protocol'])
        return model, tokenizer, extra
    checkpoints.load_checkpoint = load
    try: generate.main()
    finally: checkpoints.load_checkpoint = original_load


def execute(a, row):
    if a.stage == 'reinforcement' and row.get('observed_training', {}).get('rl') is None:
        raise ValueError('This configuration has no reinforcement-learning stage')
    if a.configuration == 'geometry_extension':
        raise ValueError('Use the per-run extension recipe and included implementations; this convenience runner covers the coordinate-free model and baselines.')
    paths(a.upstream.resolve())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run formal stages on a GPU host. --show is a read-only configuration check.')
    from rerun_campaign.runtime import signature
    from multidata_campaign import worker
    family = a.configuration if a.configuration in ('drugex', 'libinvent', 'promptsmiles') else 'atomformer'
    resources = dict(row.get('resource', {}))
    for item in row.get('recorded_generation_metadata', []):
        resources.update(item['blocks'].get('resource', {}))
    observed = row['observed_training']['supervised']
    resources.setdefault('physical_batch', observed.get('physical'))
    resources.setdefault('baseline_supervised_batch', blocks(row, 'supervised').get('supervised_batch', observed.get('physical')))
    resources.setdefault('drugex_supervised_batch', observed.get('physical'))
    evaluation = blocks(row, 'rl').get('eval_batch')
    resources.setdefault('eval_batch', evaluation)
    uid = a.dataset + '__' + a.configuration + '__run' + str(a.run)
    unit = {'dataset_id': a.dataset, 'family': family, 'seed': row['training_seed'],
            'encoder_type': 'geo' if a.configuration == 'geometry_extension' else 'atom',
            'representation': row.get('representation', 'rooted'),
            'max_length': row.get('recorded_evaluation_settings', {}).get('max_length', row['observed_training']['architecture']['max_length']),
            'unit_id': uid, 'run_id': uid, 'config_id': CONFIGS[a.configuration],
            'supervised_reuse': a.checkpoint is not None, 'resource': resources}
    if a.checkpoint:
        unit['supervised_checkpoint'] = str(a.checkpoint.resolve())
        unit['checkpoint'] = str(a.checkpoint.resolve())
    plan = {'root': str(a.work_root.resolve()), 'runtime_paths': {'datasets': {a.dataset: str(a.data_dir.resolve())}},
            'policy': {'sampling_seed': 0}, 'policy_sha256': signature(row)}
    if a.stage == 'prepare':
        if family == 'atomformer':
            raise ValueError('Use the archived fixed inputs or the GDB13 restoration utility.')
        if a.dataset == 'gdb13' and family == 'drugex':
            invoke('build_drugex_dataset', ['--train-records', str(a.data_dir / 'train.pt'),
                   '--validation-records', str(a.data_dir / 'validation.pt'), '--test-records', str(a.data_dir / 'test.pt'),
                   '--output-dir', str(a.work_root / 'adapted' / a.dataset / family), '--max-len', '80'], {})
        elif a.dataset == 'gdb13':
            import worker as native
            cfg = blocks(row, 'supervised')
            cfg.update(data_dir=str(a.data_dir), expected_records={'train':181472,'validation':23368,'test':24441})
            root = a.work_root / 'adapted' / a.dataset / family
            root.mkdir(parents=True, exist_ok=True)
            native.prepare(cfg, root, a.workers)
        else:
            worker.adapt(plan, a.dataset, family, a.workers)
        return
    if family == 'atomformer':
        if a.stage == 'supervised':
            if row.get('supervised_engine') == 'legacy_supervised' or a.dataset == 'gdb13':
                recorded = row.get('supervised_arguments') or blocks(row, 'supervised')
                recorded['encoder_type'] = unit['encoder_type']
                invoke('train_v4_supervised', ['--train-path', str(a.data_dir / 'train.pt'),
                       '--val-path', str(a.data_dir / 'validation.pt'), '--output-dir', str(a.work_root / 'models' / uid / 'supervised'),
                       '--tokenizer-path', str(a.data_dir / 'tokenizer.json')], recorded)
            else:
                worker.supervised(plan, unit)
        elif a.stage == 'reinforcement':
            if not a.checkpoint:
                raise ValueError('--checkpoint must identify the matching supervised model')
            import train_v4_rl
            previous = sys.argv
            try:
                sys.argv = ['train_v4_rl', '--checkpoint', str(a.checkpoint), '--train-path', str(a.data_dir / 'train.pt'),
                            '--output-dir', str(a.work_root / 'models' / uid / 'rl')]
                args = train_v4_rl.parse_args()
            finally:
                sys.argv = previous
            recorded = row.get('rl_arguments') or blocks(row, 'rl')
            for key, value in recorded.items():
                if hasattr(args, key): setattr(args, key, value)
            args.resume = False
            args.campaign_resume = True
            module = 'rerun_campaign.af_training' if row.get('rl_engine') == 'legacy_rl' or a.dataset == 'gdb13' else 'multidata_campaign.af_rl'
            importlib.import_module(module).run(args)
        else:
            if not a.checkpoint: raise ValueError('--checkpoint is required for generation')
            if resources['eval_batch'] is None and a.dataset != 'gdb13': raise ValueError('Recorded evaluation batch is unavailable')
            if row.get('generation', {}).get('implementation') == 'rerun_campaign.generate':
                generate_original_record_batches(a, row)
            else:
                worker.generate(plan, unit)
        return
    if a.dataset != 'gdb13':
        if a.stage == 'supervised': worker.supervised(plan, unit)
        elif a.stage == 'reinforcement':
            if not a.checkpoint: raise ValueError('--checkpoint must identify the matching prior')
            worker.reinforcement(plan, unit, a.workers)
        else: worker.generate(plan, unit)
        return
    # The GDB13 baselines used their own recorded training and sampling entry points.
    if family == 'drugex':
        adapted = a.work_root / 'adapted' / a.dataset / family
        common = ['--dataset-dir', str(adapted), '--output-dir', str(a.work_root / 'models' / uid / a.stage)]
        if a.stage == 'supervised': invoke('train_drugex', common, blocks(row, 'supervised'))
        elif a.stage == 'reinforcement':
            if not a.checkpoint: raise ValueError('--checkpoint is required')
            invoke('rl_drugex', common + ['--checkpoint', str(a.checkpoint)], blocks(row, 'rl'))
        else:
            raise ValueError('Use code/baselines/drugex/evaluate_drugex.py with the recorded generation settings.')
    else:
        import worker as native
        from models_adapter import create
        from common import read_rows
        cfg = blocks(row, 'supervised' if a.stage == 'supervised' else 'rl')
        cfg['data_dir'] = str(a.data_dir)
        root = a.work_root / 'adapted' / a.dataset / family
        rows = read_rows(root / 'train.jsonl')
        native.seed_all(cfg['seed'])
        model = create(family, rows, cfg, 'cuda')
        if a.stage == 'supervised': native.train(cfg, root, model, rows)
        elif a.stage == 'reinforcement':
            if not a.checkpoint: raise ValueError('--checkpoint is required')
            worker.copy_checked(a.checkpoint, root / 'supervised_best.pt')
            native.rl(cfg, root, model, rows, family, a.workers)
        else:
            if not a.checkpoint: raise ValueError('--checkpoint is required')
            worker.copy_checked(a.checkpoint, root / 'rl_final.pt')
            native.evaluate(cfg, root, model, a.workers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['gdb13', 'chembl37', 'moses'], required=True)
    p.add_argument('--configuration', choices=sorted(CONFIGS), required=True)
    p.add_argument('--run', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--stage', choices=['prepare', 'supervised', 'reinforcement', 'generate'], required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--work-root', type=Path)
    p.add_argument('--upstream', type=Path, default=HERE / 'upstream')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--show', action='store_true', help='Print the selected record without computation')
    a = p.parse_args()
    row = select(a.dataset, a.configuration, a.run)
    if a.show:
        print(json.dumps(row, indent=2)); return
    if a.data_dir is None or a.work_root is None: p.error('--data-dir and --work-root are required')
    a.work_root.mkdir(parents=True, exist_ok=True)
    execute(a, row)

if __name__ == '__main__':
    main()
