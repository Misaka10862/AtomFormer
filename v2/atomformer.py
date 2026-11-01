#!/usr/bin/env python3
"""Portable entry points for the archived AtomFormer model and evaluation protocol."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    train = sub.add_parser('train')
    train.add_argument('--spec', required=True)
    train.add_argument('--data', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--initial-checkpoint')
    train.add_argument('--device', default='cuda')
    train.add_argument('--resume', action='store_true')
    generate = sub.add_parser('generate')
    generate.add_argument('--checkpoint', required=True)
    generate.add_argument('--inputs', required=True)
    generate.add_argument('--output', required=True)
    generate.add_argument('--sampling-seed', type=int, required=True)
    generate.add_argument('--device', default='cuda')
    generate.add_argument('--resume', action='store_true')
    score = sub.add_parser('score')
    score.add_argument('--events', required=True)
    score.add_argument('--inputs', required=True)
    score.add_argument('--reference', required=True)
    score.add_argument('--output', required=True)
    args = parser.parse_args()
    from re_atomformer.io import read_json, read_rows, write_rows, atomic_json, file_sha, digest
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not getattr(args, 'resume', False):
        parser.error('Use a fresh output directory; existing results are not overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'train':
        from condmolgen.tokenizer import CharTokenizer
        from re_atomformer.supplement_training import train as fit
        data = Path(args.data)
        spec = read_json(args.spec)
        if spec['method'] != 'atomformer':
            parser.error('Official reference models are inference-only.')
        if args.initial_checkpoint:
            spec['training']['source_checkpoint'] = args.initial_checkpoint
            spec['training']['source_checkpoint_sha256'] = file_sha(args.initial_checkpoint)
        tokenizer = CharTokenizer.from_state(read_json(data/'tokenizer.json'))
        policy = read_json(data/('policy_ids_'+str(spec['seed'])+'.json'))['ids']
        fit(spec, list(read_rows(data/'train.jsonl.gz')), list(read_rows(data/'v_select.jsonl.gz')),
            output, tokenizer, policy, device=args.device, resume=args.resume)
    elif args.action == 'generate':
        from re_atomformer.training import load_model
        from re_atomformer.scheme_a.baselines import AtomFormerAdapter
        from re_atomformer.scheme_a.generation import generate as sample
        model, tokenizer, saved = load_model(args.checkpoint, args.device)
        spec = saved['model_spec']
        rows = list(read_rows(args.inputs))
        adapter = AtomFormerAdapter(model, tokenizer, spec['training']['max_length'])
        context = {'generation_protocol': 'supplement_A_v1', 'dataset_id': spec['dataset_id'],
                   'split': rows[0]['split'], 'method': 'atomformer',
                   'model_variant': spec['id']+'-'+spec['training']['mode'],
                   'checkpoint_sha256': saved.get('original_checkpoint_sha256', file_sha(args.checkpoint)), 'sampling_seed': args.sampling_seed,
                   'train_seed': spec['seed'], 'tokenizer_sha256': digest(tokenizer.to_state())}
        events = sample(rows, adapter, output/'shards', context, args.resume)
        write_rows(output/'samples.jsonl.gz', events)
        atomic_json(output/'generation.json', {'context': context, 'samples_sha256': file_sha(output/'samples.jsonl.gz')})
    else:
        import re_atomformer.scheme_a.evaluation as evaluation
        from re_atomformer.scheme_a.kekule_repair import wrap_scaffold_checks
        import scaffold_metrics
        scaffold_metrics.scaffold_checks = wrap_scaffold_checks(scaffold_metrics.scaffold_checks)
        from re_atomformer.scheme_a.parallel_score import score_rows
        records = list(read_rows(args.inputs))
        by_id = {r['record_id']: r for r in records}
        events = list(read_rows(args.events))
        for event in events:
            event['scoring_target_core'] = by_id[event['record_id']]['core_smiles']
        from re_atomformer.scheme_a.events import validate_events
        support = validate_events(events, list(by_id))
        reference = read_json(args.reference)
        scored = score_rows(events, reference['canonical_smiles'], workers=1)
        write_rows(output/'scored.jsonl.gz', scored)
        atomic_json(output/'summary.json', evaluation.summarize(scored, list(by_id)))
        atomic_json(output/'native_summary.json', evaluation.summarize(scored, list(by_id),
                    [rid for rid in by_id if support[rid]], 'native_support'))


if __name__ == '__main__':
    main()
