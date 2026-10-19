import argparse
import json
from pathlib import Path
import time
from .runtime import atomic_json, frozen_contract, sha256, execution_identity

def main():
    from evaluate_v4 import parse_args
    args = parse_args()
    if args.validate_only:
        print('CLI_VALIDATED')
        return
    identity = execution_identity()
    import torch
    from torch.utils.data import DataLoader
    from condmolgen.checkpoint import load_checkpoint
    from condmolgen.data import load_records
    from condmolgen.representation import get_representation_spec
    from condmolgen.v4_data import V4DecorationAutoregDataset, V4BatchCollator
    from condmolgen.v4_model import V4ConditionalAutoregGenerator
    from scaffold_metrics import protocol_metadata
    if args.expected_rdkit != protocol_metadata()['rdkit_version']:
        raise RuntimeError('Generation environment drift')
    out = Path(args.output_json).parent
    shard_root = out / 'raw_shards'
    contract = {'args': vars(args), **protocol_metadata(), 'checkpoint_sha256': sha256(args.checkpoint), 'input_sha256': sha256(args.processed_path), 'code_sha256': identity['source_sha256'], 'campaign': identity['campaign_id']}
    frozen_contract(out / 'generation_contract.json', contract)
    shard_root.mkdir(exist_ok=True)
    loaded, tokenizer, extra = load_checkpoint(args.checkpoint, map_location=args.device)
    if extra.get('scaffold_metric_protocol') != protocol_metadata()['scaffold_metric_protocol']:
        raise RuntimeError('Cannot generate from a legacy RL checkpoint')
    model = V4ConditionalAutoregGenerator(loaded.config)
    model.load_state_dict(loaded.state_dict(), strict=True)
    model = model.to(args.device).eval()
    del loaded
    representation = get_representation_spec(args.representation)
    records = load_records(args.processed_path)
    if args.max_samples:
        raise RuntimeError('Campaign evaluation cannot truncate records')
    dataset = V4DecorationAutoregDataset(records, tokenizer, args.max_length, target_key=representation.target_key)
    if len(dataset) != len(records):
        raise RuntimeError('Evaluation input coverage changed')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=V4BatchCollator(tokenizer))
    for batch_index, batch in enumerate(loader):
        record_start = batch_index * args.batch_size
        for sample_start in range(0, args.samples_per_scaffold, 4):
            ids = list(range(sample_start, min(sample_start + 4, args.samples_per_scaffold)))
            repeat = len(ids)
            path = shard_root / f'{record_start:09d}_{sample_start:03d}.json'
            expected = [(record_start + i, j) for i in range(len(batch['scaffold_smiles'])) for j in ids]
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['contract'] != contract or [(r['record_id'], r['sample_id']) for r in saved['rows']] != expected:
                    raise RuntimeError('Resume shard contract/coverage mismatch')
                continue
            torch.manual_seed(args.sampling_seed + sample_start)
            z, pos = (batch['z'].to(args.device), batch['pos'].to(args.device))
            scaffolds = [s for s in batch['scaffold_smiles'] for _ in ids]
            if repeat > 1:
                z, pos = (z.repeat_interleave(repeat, dim=0), pos.repeat_interleave(repeat, dim=0))
            with torch.no_grad():
                decorations, _ = model.generate_decorations(z=z, pos=pos, tokenizer=tokenizer, scaffold_smiles=scaffolds, max_length=args.max_length, temperature=args.temperature, greedy=bool(args.greedy))
            rows = []
            for k, decoration in enumerate(decorations):
                i = k // repeat
                rows.append({'record_id': record_start + i, 'sample_id': ids[k % repeat], 'scaffold_smiles': batch['scaffold_smiles'][i], 'reference_smiles': batch['canonical_smiles'][i], 'target_decoration': batch['target_decoration_smiles'][i], 'generated_decoration': decoration, 'representation': representation.name})
            if [(r['record_id'], r['sample_id']) for r in rows] != expected:
                raise RuntimeError('Generation returned a different number of attempts')
            atomic_json(path, {'contract': contract, 'rows': rows})
            atomic_json(out / 'progress.json', {'last_record_start': record_start, 'last_sample_start': sample_start, 'time': time.time()})
    atomic_json(out / 'generation.done.json', {'attempts': len(records) * args.samples_per_scaffold, 'records': len(records), 'samples_per_record': args.samples_per_scaffold, 'contract': contract, 'shards': {p.name: sha256(p) for p in sorted(shard_root.glob('*.json'))}})
if __name__ == '__main__':
    main()
