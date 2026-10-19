import json
from pathlib import Path
import torch
from rerun_campaign.runtime import atomic_json, frozen_contract, sha256, signature
from .worker import data_dir, resource_for, bench_imports, sup_path

def generate(plan, run, profile=False):
    family = run['family']
    unit_id = run.get('unit_id') or f"{run['dataset_id']}__{run['config_id'].rsplit('_', 1)[0]}__s{run['seed']}"
    unit = {**run, 'unit_id': unit_id, 'supervised_reuse': run.get('supervised_reuse', False)}
    modelroot = Path(plan['root']) / ('profiles' if profile else 'models') / unit_id
    checkpoint = modelroot / 'rl/last.pt' if profile or run['config_id'].endswith('_rl') else sup_path(plan, unit)
    if run.get('checkpoint'):
        checkpoint = Path(run['checkpoint'])
    if family == 'atomformer' and run['config_id'] == 'full_sup' and unit['supervised_reuse'] and (not profile):
        checkpoint = Path(run['supervised_checkpoint'])
    root = Path(plan['root']) / ('profile_generation' if profile else 'results') / (unit_id if profile else run['run_id'])
    root.mkdir(parents=True, exist_ok=True)
    shards = root / 'raw_shards'
    shards.mkdir(exist_ok=True)
    resources = resource_for(plan, unit, profile)
    batch_size = resources['eval_batch']
    data = data_dir(plan, run['dataset_id'])
    from condmolgen.train_utils import seed_everything
    seed_everything(run['seed'])
    limit = 100 if profile else None
    if family == 'atomformer':
        from condmolgen.model import ConditionalGeneratorConfig
        from condmolgen.v4_model import V4ConditionalAutoregGenerator
        from condmolgen.tokenizer import CharTokenizer
        payload = torch.load(checkpoint, map_location='cpu')
        config = ConditionalGeneratorConfig(**payload['model_config'])
        model = V4ConditionalAutoregGenerator(config).to('cuda')
        model.load_state_dict(payload['model_state'], strict=True)
        model.eval()
        tokenizer = CharTokenizer.from_state(payload['tokenizer_state'])
        input_path = data / ('train.pt' if profile else 'test.pt')
        rows = torch.load(input_path, map_location='cpu')
    elif family == 'drugex':
        bench_imports()
        from evaluate_drugex import load_model
        from common import read_rows
        adapted = Path(plan['root']) / 'adapted' / run['dataset_id'] / 'drugex'
        model, vocab, _ = load_model(checkpoint, adapted, torch.device('cuda'))
        input_path = adapted / ('train.jsonl' if profile else 'test.jsonl')
        rows = read_rows(input_path)
        tensors = torch.load(adapted / ('train.pt' if profile else 'test.pt'), map_location='cpu')
        if len(rows) != len(tensors):
            raise ValueError('DrugEx tensors/record IDs differ')
    else:
        bench_imports()
        from common import read_rows
        from models_adapter import create
        from worker import load_checkpoint
        from .worker import baseline_cfg
        adapted = Path(plan['root']) / 'adapted' / run['dataset_id'] / family
        training = read_rows(adapted / 'train.jsonl')
        input_path = adapted / ('train.jsonl' if profile else 'test.jsonl')
        rows = read_rows(input_path)
        model = create(family, training[:1024] if profile else training, baseline_cfg(plan, unit, profile), 'cuda')
        load_checkpoint(checkpoint, model)
        model.network.eval()
    if limit:
        rows = rows[:limit]
    contract = {'checkpoint': sha256(checkpoint), 'input': sha256(input_path), 'run': run, 'resource': resources, 'policy': plan['policy_sha256'], 'profile': profile, 'input_only_eligibility': True}
    frozen_contract(root / 'generation_contract.json', contract)
    total = len(rows) * 20
    for offset in range(0, total, batch_size):
        flat = list(range(offset, min(offset + batch_size, total)))
        batch = [rows[i // 20] for i in flat]
        ids = [(int(r.get('record_id', r.get('row_index', i // 20))), i % 20) for r, i in zip(batch, flat)]
        path = shards / f'{offset:09d}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['contract'] != contract or [(r['record_id'], r['sample_id']) for r in old['rows']] != ids:
                raise RuntimeError('Generation shard drift')
            continue
        seed_everything(plan['policy']['sampling_seed'] + offset)
        if family == 'atomformer':
            n = max((len(r['cond_z']) for r in batch))
            z = torch.zeros(len(batch), n, dtype=torch.long, device='cuda')
            pos = torch.zeros(len(batch), n, 3, device='cuda')
            eligibility = []
            outs = [None] * len(batch)
            for j, r in enumerate(batch):
                z[j, :len(r['cond_z'])] = torch.tensor(r['cond_z'], device='cuda')
                pos[j, :len(r['cond_z'])] = torch.tensor(r['cond_pos'], device='cuda')
                prefix = r['scaffold_smiles']
                if len(prefix) + 3 <= run['max_length'] and all((c in tokenizer.token2id for c in prefix)):
                    eligibility.append(j)
            if eligibility:
                with torch.no_grad():
                    ds, _ = model.generate_decorations(z=z[eligibility], pos=pos[eligibility], tokenizer=tokenizer, scaffold_smiles=[batch[j]['scaffold_smiles'] for j in eligibility], max_length=run['max_length'], temperature=1.0, greedy=False)
                for j, s in zip(eligibility, ds):
                    outs[j] = s
            result = [{'record_id': rid, 'sample_id': sid, 'scaffold_smiles': r['scaffold_smiles'], 'reference_smiles': r['canonical_smiles'], 'generated_decoration': d or '', 'representation': run['representation'], 'input_supported': j in eligibility} for j, ((rid, sid), r, d) in enumerate(zip(ids, batch, outs))]
        elif family == 'drugex':
            indices = [i // 20 for i in flat]
            with torch.no_grad():
                encoded = model(tensors[indices].clone().to('cuda'), is_train=False)
            _, outputs = vocab.decode(encoded.detach().cpu().numpy())
            if len(outputs) != len(batch):
                raise ValueError('Native DrugEx changed attempt count')
            result = [{**r, 'record_id': rid, 'sample_id': sid, 'row_index': rid, 'generated_smiles': s} for r, s, (rid, sid) in zip(batch, outputs, ids)]
        else:
            eligible = []
            outputs = [None] * len(batch)
            decs = [None] * len(batch)
            for j, r in enumerate(batch):
                try:
                    if family == 'libinvent':
                        okay = model.vocab.encode_scaffold(r['scaffold']) is not None
                    else:
                        model.model.vocabulary.encode(model.model.tokenizer.tokenize(model.prompt(r).replace('[SH]', 'S')))
                        okay = True
                except (KeyError, ValueError, TypeError, IndexError):
                    okay = False
                if r.get('input_supported', okay) != okay:
                    raise RuntimeError('Input eligibility differs from frozen adaptation audit')
                if okay:
                    eligible.append(j)
            if eligible:
                generated, decorations = model.sample([batch[j] for j in eligible])
                if len(generated) != len(eligible):
                    raise ValueError('Native generator changed attempts')
                for j, s, d in zip(eligible, generated, decorations):
                    outputs[j], decs[j] = (s, d)
            result = [{'record_id': rid, 'sample_id': sid, 'scaffold': r['scaffold'], 'core': r['core'], 'reference_smiles': r['source'], 'raw_output': s, 'decoration': d, 'input_supported': j in eligible} for j, (r, s, d, (rid, sid)) in enumerate(zip(batch, outputs, decs, ids))]
        atomic_json(path, {'contract': contract, 'rows': result})
        atomic_json(root / 'progress.json', {'attempts': offset + len(result), 'total': total, 'time': __import__('time').time()})
    report = {'records': len(rows), 'attempts': total, 'checkpoint': str(checkpoint), 'contract': contract, 'shards': {p.name: sha256(p) for p in sorted(shards.glob('*.json'))}}
    atomic_json(root / 'generation.done.json', report)
    return report
