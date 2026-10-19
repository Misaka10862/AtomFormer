"""Summarize saved attempts or calculate the recorded paired component intervals."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from rerun_campaign.scoring import canonical, dataset_rows, summarize
from multidata_campaign.evaluation import aggregate, paired_bootstrap
from scaffold_metrics import scaffold_core_from_scaffold


def read_samples(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def reference_set(path):
    values = {canonical(r.get('canonical_smiles', r.get('source', r.get('source_smiles'))))
              for r in dataset_rows(path)}
    values.discard(None)
    return values


def component_records(path, inputs, reference):
    groups = defaultdict(list)
    for row in read_samples(path):
        groups[int(row['record_id'])].append(row)
    expected = {int(r.get('record_id', r.get('row_index', i))): r for i, r in enumerate(inputs)}
    if set(groups) != set(expected):
        raise ValueError('Component samples must cover the complete fixed input set')
    result = {}
    for rid, rows in groups.items():
        if len(rows) != 20 or {r['sample_id'] for r in rows} != set(range(20)):
            raise ValueError('Component record requires exactly the original 20 attempts')
        if not all(r.get('input_supported', True) for r in rows):
            raise ValueError('Component analysis requires complete input support')
        valid = [r for r in rows if r['valid']]
        retained = {r['generated'] for r in valid if r['core_retention']}
        data = {'validity': [len(valid), len(rows)],
                'scaffold_core_match_rate': [sum(r['core_match'] for r in valid), len(valid)],
                'scaffold_core_retention_rate': [sum(r['core_retention'] for r in valid), len(valid)],
                'retained_yield': [sum(r['core_retention'] for r in valid), len(rows)],
                'novelty': [sum(r['generated'] not in reference for r in valid), len(valid)]}
        for metric, field in [('qed_mean', 'qed'), ('sas_mean', 'sas')]:
            values = [r[field] for r in valid if r[field] is not None]
            data[metric] = [math.fsum(values), len(values)]
        if all(r['assembly_success'] is not None for r in rows):
            data['assembly_success_rate'] = [sum(r['assembly_success'] for r in rows), len(rows)]
        data['unique_retained_per_intended_record'] = [len(retained), 1]
        data['zero_valid_record_fraction'] = [int(not valid), 1]
        data['zero_retained_record_fraction'] = [int(not retained), 1]
        data['decoration_validity'] = [sum(r['decoration_valid'] for r in rows), 20]
        result[str(rid)] = {'core': scaffold_core_from_scaffold(expected[rid]['scaffold_smiles']),
                            'statistics': data}
    return result


def summarize_native(path, reference, intended_records, workers, replicates):
    # Native support is filtered explicitly. The full requested budget is retained separately.
    native_records = set()
    directory = Path(tempfile.mkdtemp(prefix='atomformer-analysis-'))
    native_path = Path(directory) / 'native.jsonl'
    with native_path.open('w') as stream:
        for row in read_samples(path):
            if row.get('input_supported', True):
                native_records.add(row['record_id'])
                stream.write(json.dumps(row) + '\n')
    metrics, intervals = summarize(native_path, reference, workers, bootstrap_seed=0, replicates=replicates)
    metrics['retained_yield'] = metrics['scaffold_core_retention_rate'] * metrics['num_valid'] / (intended_records * 20)
    return {'metrics': metrics, 'record_bootstrap': intervals,
            'coverage': {'intended_records': intended_records, 'input_supported_records': len(native_records),
                         'intended_attempts': intended_records * 20}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='task', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--samples', required=True)
    s.add_argument('--intended-records', type=int, required=True)
    s.add_argument('--workers', type=int, default=1)
    c = sub.add_parser('component-ci')
    c.add_argument('--left', nargs=3, required=True, help='Run1, run2, run3 of the minuend')
    c.add_argument('--right', nargs=3, required=True, help='Matching runs of the subtrahend')
    c.add_argument('--test-input', required=True)
    for parser in (s, c):
        parser.add_argument('--train-input', required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--replicates', type=int, default=2000)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a new result file: ' + str(a.output))
    from scaffold_metrics import protocol_metadata
    if protocol_metadata()['rdkit_version'] != '2022.09.5':
        raise RuntimeError('Use the recorded common-scoring RDKit 2022.09.5 environment')
    reference = reference_set(a.train_input)
    if a.task == 'summary':
        result = summarize_native(a.samples, reference, a.intended_records, a.workers, a.replicates)
    else:
        inputs = dataset_rows(a.test_input)
        left = [component_records(f, inputs, reference) for f in a.left]
        right = [component_records(f, inputs, reference) for f in a.right]
        result = paired_bootstrap(left, right, replicates=a.replicates, seed=0, cluster=True)
        # Same within-run pooling and three-pair summary as the recorded component study.
        metrics = sorted(result['metrics'])
        delta = []
        for l, r in zip(left, right):
            current = {}
            for metric in metrics:
                totals = [(math.fsum(x['statistics'][metric][0] for x in side.values()),
                           math.fsum(x['statistics'][metric][1] for x in side.values())) for side in (l, r)]
                if all(d > 0 for _, d in totals):
                    current[metric] = totals[0][0] / totals[0][1] - totals[1][0] / totals[1][1]
            delta.append(current)
        result['paired_run_differences'] = delta
        result['paired_run_summary'] = {m: aggregate([d.get(m) for d in delta]) for m in metrics}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
