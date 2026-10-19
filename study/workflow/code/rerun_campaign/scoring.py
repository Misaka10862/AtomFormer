import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time
from .runtime import atomic_json, frozen_contract, sha256, read_jsonl, write_jsonl, execution_identity, signature
from .statistics import bootstrap_records
from scaffold_metrics import scaffold_checks, scaffold_core_from_scaffold, protocol_metadata
_SA = None

def canonical(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None or not m.GetNumAtoms() or any((a.GetAtomicNum() == 0 for a in m.GetAtoms())):
        return None
    return Chem.MolToSmiles(m, canonical=True)

def score_one(item):
    global _SA
    family, row = item
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, RDConfig
    RDLogger.DisableLog('rdApp.*')
    if _SA is None:
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / 'SA_Score'))
        import sascorer
        _SA = sascorer
    decoration = None
    if family == 'atomformer':
        scaffold = row['scaffold_smiles']
        decoration = row['generated_decoration']
        if 'assembled_smiles' in row:
            output = row['assembled_smiles']
            assembled = output is not None
            if 'assembly_success' in row and row['assembly_success'] != assembled:
                raise ValueError('Saved assembly flag disagrees with the actual assembly output')
        else:
            from condmolgen.representation import assemble_with_representation
            output = assemble_with_representation(row['representation'], scaffold, decoration)
            assembled = output is not None
        core = scaffold_core_from_scaffold(scaffold)
    elif family == 'drugex':
        scaffold = row['scaffold_smiles']
        core = row['scaffold_core_smiles']
        output = row.get('generated_smiles')
        assembled = None
    else:
        scaffold = row['scaffold']
        core = row['core']
        decoration = row.get('decoration')
        output = row.get('raw_output')
        if output == '__assemble__':
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/scaffold_bench'))
            from common import join
            output = join(scaffold, decoration)
        assembled = None
    smi = canonical(output)
    valid = smi is not None
    result = {'record_id': row['record_id'], 'sample_id': row['sample_id'], 'generated': smi, 'valid': valid, 'assembly_success': assembled, 'core_match': False, 'core_retention': False, 'qed': None, 'sas': None, 'scaffold': scaffold, 'core': core, 'decoration_valid': canonical(decoration) is not None if family == 'atomformer' else None}
    if valid:
        checks = scaffold_checks(smi, core)
        if not checks['eligible']:
            raise ValueError('Valid product with unparseable target core')
        result.update(core_match=checks['core_match'], core_retention=checks['core_retention'], kekule_rescued=checks['kekule_rescued'])
        m = Chem.MolFromSmiles(smi)
        for name, fn in [('qed', QED.qed), ('sas', _SA.calculateScore)]:
            try:
                value = float(fn(m))
                result[name] = value if math.isfinite(value) else None
            except (ValueError, RuntimeError):
                pass
    return result

def score_chunk(items):
    return [score_one(x) for x in items]

def dataset_rows(path):
    path = Path(path)
    if path.suffix == '.pt':
        import torch
        return list(torch.load(path, map_location='cpu'))
    return list(read_jsonl(path))

def normalize_inputs(spec):
    expected = dataset_rows(spec['test_input'])
    keys = {}
    by_id = {}
    expected_ids = set()
    for i, r in enumerate(expected):
        rid = int(r.get('row_index', r.get('record_id', i)))
        expected_ids.add(rid)
        if rid in by_id:
            raise ValueError('Duplicate record ID in input manifest')
        by_id[rid] = r
        if spec['family'] == 'atomformer':
            key = (r.get('canonical_smiles', r.get('source_smiles')), r['scaffold_smiles'])
            if key in keys:
                raise ValueError('Ambiguous reference/scaffold pair in input manifest')
            keys[key] = rid
    counts = Counter()
    seen = set()
    source = Path(spec['samples'])

    def source_rows():
        if source.is_dir():
            for p in sorted(source.glob('*.json')):
                data = json.loads(p.read_text())
                if not isinstance(data, dict) or 'contract' not in data:
                    raise ValueError('Legacy raw shard')
                yield from data['rows']
        else:
            yield from read_jsonl(source)
    for value in source_rows():
        r = dict(value)
        if spec['family'] == 'atomformer':
            key = (r['reference_smiles'], r['scaffold_smiles'])
            if key not in keys:
                raise ValueError('Saved sample does not correspond to an input record')
            rid = keys[key]
            if 'record_id' in r and int(r['record_id']) != rid:
                raise ValueError('Record ID disagrees with input manifest')
            sid = int(r.get('sample_id', r.get('sample_idx', -1)))
            r['representation'] = spec['representation']
        elif spec['family'] == 'drugex':
            rid = int(r['row_index'])
            sid = counts[rid]
            target = by_id.get(rid, {})
            for field in ('canonical_smiles', 'scaffold_smiles', 'scaffold_core_smiles'):
                if r.get(field) != target.get(field):
                    raise ValueError('DrugEx sample disagrees with its encoded input manifest')
        else:
            rid = int(r['record_id'])
            sid = int(r['sample_id'])
            target = by_id.get(rid, {})
            if r.get('core') != target.get('core') or r.get('scaffold') != target.get('scaffold') or r.get('reference_smiles') != target.get('source'):
                raise ValueError('Baseline sample disagrees with its prepared input manifest')
        if rid not in expected_ids or not 0 <= sid < spec['samples_per_record'] or (rid, sid) in seen:
            raise ValueError('Unknown, duplicate or invalid attempt ID')
        seen.add((rid, sid))
        counts[rid] += 1
        r.update(record_id=rid, sample_id=sid)
        yield r
    if set(counts) != expected_ids or any((n != spec['samples_per_record'] for n in counts.values())):
        raise ValueError('Missing or extra attempts per input record')
    if len(expected_ids) != spec['expected_records']:
        raise ValueError('Input manifest record coverage changed')

def diversity_part(args):
    from rdkit import Chem, DataStructs, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem import AllChem
    smiles, indices = args
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles]
    totals = []
    count = 0
    for i in indices:
        distances = [1.0 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])]
        totals.append(math.fsum(distances))
        count += len(distances)
    return (math.fsum(totals), count)

def within_record_chunk(groups):
    return [diversity_part((smiles, list(range(max(0, len(smiles) - 1))))) for smiles in groups]

def summarize(path, reference, workers, bootstrap_seed=0, replicates=2000):
    counts = Counter()
    values = {'qed': [], 'sas': []}
    valid_smiles = []
    groups = defaultdict(list)
    for r in read_jsonl(path):
        counts['attempts'] += 1
        groups[r['record_id']].append(r)
        counts['assembly_known'] += r['assembly_success'] is not None
        counts['assembled'] += r['assembly_success'] is True
        counts['decoration_known'] += r['decoration_valid'] is not None
        counts['decoration_valid'] += r['decoration_valid'] is True
        if r['core_retention'] and (not r['core_match']):
            raise ValueError('Retention implication violated')
        if not r['valid']:
            if r['core_match'] or r['core_retention']:
                raise ValueError('Invalid molecule passed structure gate')
            continue
        counts['valid'] += 1
        counts['matched'] += r['core_match']
        counts['retained'] += r['core_retention']
        counts['novel'] += r['generated'] not in reference
        valid_smiles.append(r['generated'])
        for k in values:
            if r[k] is not None:
                values[k].append(r[k])
    n = counts['attempts']
    v = counts['valid']
    capped = valid_smiles[::max(1, v // 5000)][:5000] if v > 5000 else valid_smiles
    if len(capped) > 1:
        tasks = [(capped, list(range(i, len(capped) - 1, workers))) for i in range(workers)]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                parts = list(pool.map(diversity_part, tasks))
        else:
            parts = [diversity_part(tasks[0])]
        diversity = math.fsum((p[0] for p in parts)) / sum((p[1] for p in parts))
    else:
        diversity = 0.0
    metrics = {'num_generated': n, 'num_valid': v, 'validity': v / max(1, n), 'record_coverage': len(groups), 'scaffold_core_match_rate': counts['matched'] / max(1, v), 'scaffold_core_retention_rate': counts['retained'] / max(1, v), 'scaffold_core_metric_n': v, 'novelty': counts['novel'] / max(1, v), 'diversity': diversity, 'diversity_cap': 5000, 'diversity_n': len(capped), 'unique_rate': len(set(valid_smiles)) / max(1, v), 'mean_unique_per_record': statistics.mean((len({r['generated'] for r in rows if r['valid']}) for rows in groups.values()))}
    valid_groups = [[r['generated'] for r in rows if r['valid']] for rows in groups.values()]
    valid_groups = [g for g in valid_groups if g]
    chunks = [valid_groups[i:i + 256] for i in range(0, len(valid_groups), 256)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            within = [p for block in pool.map(within_record_chunk, chunks) for p in block]
    else:
        within = [p for block in map(within_record_chunk, chunks) for p in block]
    metrics['mean_within_record_diversity'] = statistics.mean((s / max(1, n) for s, n in within)) if within else None
    metrics['within_record_n'] = len(valid_groups)
    metrics['mean_unique_per_record'] = statistics.mean((len(set(g)) for g in valid_groups)) if valid_groups else None
    if counts['assembly_known'] == n:
        metrics['assembly_success_rate'] = counts['assembled'] / max(1, n)
    if counts['decoration_known'] == n:
        metrics['decoration_validity'] = counts['decoration_valid'] / max(1, n)
    for k, val in values.items():
        metrics.update({k + '_n': len(val), k + '_mean': statistics.mean(val) if val else None, k + '_std': statistics.pstdev(val) if val else None, k + '_median': statistics.median(val) if val else None})
    records = {}
    for rid, rows in groups.items():
        valid = [r for r in rows if r['valid']]
        d = {'validity': (len(valid), len(rows)), 'scaffold_core_match_rate': (sum((r['core_match'] for r in valid)), len(valid)), 'scaffold_core_retention_rate': (sum((r['core_retention'] for r in valid)), len(valid)), 'novelty': (sum((r['generated'] not in reference for r in valid)), len(valid))}
        if counts['assembly_known'] == n:
            d['assembly_success_rate'] = (sum((r['assembly_success'] for r in rows)), len(rows))
        for k in values:
            val = [r[k] for r in valid if r[k] is not None]
            d[k + '_mean'] = (math.fsum(val), len(val))
        records[rid] = d
    return (metrics, bootstrap_records(records, replicates=replicates, seed=bootstrap_seed))
