"""Attempt-level budgets, record-equal candidate yield and paired uncertainty."""
from collections import defaultdict, Counter
import math
import random
from statistics import mean, stdev


def summarize(rows, intended_ids, attempts_per_record, thresholds):
    ids = list(intended_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Intended record IDs must be unique and nonempty')
    groups = {i: [] for i in ids}
    seen = set()
    for row in rows:
        identity = (row['record_id'], row['attempt_id'])
        if row['record_id'] not in groups or identity in seen:
            raise ValueError('Unexpected or duplicate attempt')
        if not isinstance(row['attempt_id'], int) or not 0 <= row['attempt_id'] < attempts_per_record:
            raise ValueError('Attempt index outside budget')
        seen.add(identity)
        groups[row['record_id']].append(row)
    if any(len(g) != attempts_per_record for g in groups.values()):
        raise ValueError('Incomplete attempt budget: failures must be recorded')
    all_rows = [r for g in groups.values() for r in g]
    valid = [r for r in all_rows if r['valid']]
    retained = [r for r in valid if r['core_retention']]
    def props(rs):
        return {p: mean(r[p] for r in rs if r.get(p) is not None) if any(r.get(p) is not None for r in rs) else None for p in ('qed', 'sas')}
    records = []
    for rid, group in groups.items():
        unique = {r['canonical_smiles']: r for r in group if r['valid'] and r['core_retention']}
        good = [r for r in unique.values() if r.get('qed') is not None and r.get('sas') is not None and r['qed'] >= thresholds['qed_min'] and r['sas'] <= thresholds['sas_max']]
        physics = [r for r in good if r.get('physical_valid') is True]
        valid_products = [r['canonical_smiles'] for r in group if r['valid']]
        retained_products = [r['canonical_smiles'] for r in group if r['valid'] and r['core_retention']]
        scorable=[r for r in group if r['valid'] and r['core_retention'] and r.get('qed') is not None and r.get('sas') is not None]
        records.append({'attempts':len(group),'retained_outputs':len(retained_products),
                        'retained_scored_count':len(scorable),'retained_qed_sum':sum(r['qed'] for r in scorable),
                        'retained_sas_sum':sum(r['sas'] for r in scorable),'valid_outputs':len(valid_products),
                        'within_record_diversity':local_diversity(valid_products) if valid_products else None,
                        'retained_diversity_all_records':local_diversity(retained_products),
                        'record_id': rid, 'unique_retained': len(unique), 'u_good': len(good),
                        'u_good_3d': len(physics), 'local_diversity': local_diversity(list(unique)), 'unique_properties': props(list(unique.values()))})
    n = len(all_rows)
    def record_property(name):
        values = [r['unique_properties'][name] for r in records if r['unique_properties'][name] is not None]
        return mean(values) if values else None
    return {'requested_attempts': n, 'records': len(ids), 'attempts_per_record': attempts_per_record,
            'validity': len(valid)/n, 'retained_yield': len(retained)/n,
            'core_match_yield':sum(bool(r.get('core_match')) for r in all_rows)/n,
            'mean_within_record_diversity':mean(r['within_record_diversity'] for r in records if r['valid_outputs']) if any(r['valid_outputs'] for r in records) else None,
            'within_record_n':sum(bool(r['valid_outputs']) for r in records),
            'retained_diversity_all_records':mean(r['retained_diversity_all_records'] for r in records),
            'retained_diversity_record_n':len(ids),
            'failure_counts':dict(Counter(r.get('failure_reason') for r in all_rows if r.get('failure_reason'))),
            'core_retention_given_valid': len(retained)/len(valid) if valid else None,
            'sample_weighted_properties': props(valid), 'retained_properties': props(retained),
            'record_equal_unique_properties': {p: record_property(p) for p in ('qed', 'sas')},
            'mean_unique_retained': mean(r['unique_retained'] for r in records),
            'mean_local_diversity': mean(r['local_diversity'] for r in records),
            'unique_retained_diversity_all_records':mean(r['local_diversity'] for r in records),
            'diversity_protocols':{'mean_within_record_diversity':'valid_with_duplicates_nonempty_records',
                'retained_diversity_all_records':'retained_with_duplicates_all_records_zero_if_under_two',
                'mean_local_diversity':'legacy_alias_unique_retained_all_records'},
            'u_good': mean(r['u_good'] for r in records),
            'u_good_3d': mean(r['u_good_3d'] for r in records) if any('physical_valid' in r for r in all_rows) else None,
            'physical_evaluated_attempts': sum('physical_valid' in r for r in all_rows),
            'unsupported_records': sum(all(r.get('input_supported') is False for r in g) for g in groups.values()),
            'per_record': records}


def paired_cluster_ci(left, right, record_groups, metric='u_good', replicates=2000, seed=13):
    if set(left) != set(right) or set(left) != set(record_groups):
        raise ValueError('Paired identity coverage differs')
    clusters = defaultdict(list)
    for rid in left:
        clusters[record_groups[rid]].append(left[rid][metric] - right[rid][metric])
    units = list(clusters.values())
    if not units:
        raise ValueError('No matched clusters')
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        chosen = [rng.choice(units) for _ in units]
        samples.append(mean(x for group in chosen for x in group))
    samples.sort()
    return {'difference': mean(left[r][metric]-right[r][metric] for r in left),
            'ci95': [samples[int(.025*(replicates-1))], samples[int(.975*(replicates-1))]],
            'clusters': len(units), 'replicates': replicates, 'conditional_on_fitted_models': True}


def training_variation(values):
    return {'mean': mean(values), 'sample_sd': stdev(values) if len(values) > 1 else None,
            'independent_training_runs': len(values)}


def local_diversity(smiles):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import AllChem
    if len(smiles)<2:return 0.0
    fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=2048) for s in smiles]
    values=[1.0-x for i,fp in enumerate(fps[:-1]) for x in DataStructs.BulkTanimotoSimilarity(fp,fps[i+1:])]
    return mean(values)


def common_input_summary(method_rows,intended_ids,attempts,thresholds):
    supported={}
    for method,rows in method_rows.items():
        summarize(rows,intended_ids,attempts,thresholds)
        groups=defaultdict(set)
        for row in rows:groups[row['record_id']].add(row.get('input_supported',True))
        if any(len(values)!=1 for values in groups.values()):raise ValueError('Input eligibility varies across attempts')
        supported[method]={rid for rid,values in groups.items() if values=={True}}
    common=set.intersection(*supported.values()) if supported else set()
    return {'common_records':len(common),'intended_records':len(intended_ids),
            'coverage':{m:len(ids)/len(intended_ids) for m,ids in supported.items()},
            'methods':{m:summarize([r for r in rows if r['record_id'] in common],sorted(common),attempts,thresholds) for m,rows in method_rows.items()} if common else {}}


def quality_threshold_curve(rows,intended_ids,qed_values=(.4,.6,.8),sas_values=(2.,3.,5.)):
    candidates={rid:{} for rid in intended_ids}
    for row in rows:
        if row['valid'] and row['core_retention']:
            candidates[row['record_id']][row['canonical_smiles']]=row
    return [{'qed_min':q,'sas_max':s,'u_good':sum(sum(r['qed']>=q and r['sas']<=s for r in group.values()) for group in candidates.values())/len(candidates)} for q in qed_values for s in sas_values]
