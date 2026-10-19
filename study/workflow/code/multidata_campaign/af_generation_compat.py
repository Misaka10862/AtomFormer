import json
from pathlib import Path
import torch
from condmolgen.model import ConditionalGeneratorConfig
from condmolgen.v4_model import V4ConditionalAutoregGenerator
from condmolgen.tokenizer import CharTokenizer
from rerun_campaign.runtime import atomic_json, frozen_contract, sha256
from .worker import data_dir, sup_path

def generate(plan, run, profile=False):
    uid = run.get('unit_id') or f"{run['dataset_id']}__{run['config_id'].rsplit('_', 1)[0]}__s{run['seed']}"
    unit = {**run, 'unit_id': uid}
    base = Path(plan['root']) / ('profiles' if profile else 'models') / uid
    checkpoint = base / 'rl/last.pt' if profile or run['config_id'].endswith('_rl') else sup_path(plan, unit)
    if run.get('checkpoint'):
        checkpoint = Path(run['checkpoint'])
    payload = torch.load(checkpoint, map_location='cpu')
    model = V4ConditionalAutoregGenerator(ConditionalGeneratorConfig(**payload['model_config'])).to('cuda')
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    tokenizer = CharTokenizer.from_state(payload['tokenizer_state'])
    input_path = data_dir(plan, 'gdb13') / ('train.pt' if profile else 'test.pt')
    rows = torch.load(input_path, map_location='cpu')
    if profile:
        rows = rows[:100]
    root = Path(plan['root']) / ('profile_generation' if profile else 'results') / (uid if profile else run['run_id'])
    root.mkdir(parents=True, exist_ok=True)
    shards = root / 'raw_shards'
    shards.mkdir(exist_ok=True)
    contract = {'checkpoint': sha256(checkpoint), 'input': sha256(input_path), 'run': run, 'policy': plan['policy_sha256'], 'batch_records': 128, 'sample_block': 4, 'sampling_seed': 0, 'profile': profile, 'input_only_eligibility': True}
    frozen_contract(root / 'generation_contract.json', contract)
    for offset in range(0, len(rows), 128):
        batch = rows[offset:offset + 128]
        n = max((len(r['cond_z']) for r in batch))
        z = torch.zeros(len(batch), n, dtype=torch.long, device='cuda')
        pos = torch.zeros(len(batch), n, 3, device='cuda')
        for i, r in enumerate(batch):
            z[i, :len(r['cond_z'])] = torch.tensor(r['cond_z'], device='cuda')
            pos[i, :len(r['cond_z'])] = torch.tensor(r['cond_pos'], device='cuda')
        for start in range(0, 20, 4):
            path = shards / f'{offset:09d}_{start:03d}.json'
            expected = [(offset + i, j) for i in range(len(batch)) for j in range(start, start + 4)]
            if path.exists():
                old = json.loads(path.read_text())
                if old['contract'] != contract or [(r['record_id'], r['sample_id']) for r in old['rows']] != expected:
                    raise RuntimeError('Legacy block resume differs')
                continue
            torch.manual_seed(start)
            repeated = [r for r in batch for _ in range(4)]
            eligible = [i for i, r in enumerate(repeated) if len(r['scaffold_smiles']) + 3 <= 128 and all((c in tokenizer.token2id for c in r['scaffold_smiles']))]
            outputs = [''] * len(repeated)
            if eligible:
                with torch.no_grad():
                    decorations, _ = model.generate_decorations(z=z.repeat_interleave(4, 0)[eligible], pos=pos.repeat_interleave(4, 0)[eligible], tokenizer=tokenizer, scaffold_smiles=[repeated[i]['scaffold_smiles'] for i in eligible], max_length=128, temperature=1.0, greedy=False)
                for i, s in zip(eligible, decorations):
                    outputs[i] = s
            result = [{'record_id': rid, 'sample_id': sid, 'scaffold_smiles': r['scaffold_smiles'], 'reference_smiles': r['canonical_smiles'], 'generated_decoration': s, 'representation': run['representation'], 'input_supported': i in eligible} for i, ((rid, sid), r, s) in enumerate(zip(expected, repeated, outputs))]
            atomic_json(path, {'contract': contract, 'rows': result})
            atomic_json(root / 'progress.json', {'last_record_start': offset, 'last_sample_start': start, 'time': __import__('time').time()})
    report = {'records': len(rows), 'attempts': len(rows) * 20, 'checkpoint': str(checkpoint), 'contract': contract, 'shards': {p.name: sha256(p) for p in sorted(shards.glob('*.json'))}}
    atomic_json(root / 'generation.done.json', report)
    return report
