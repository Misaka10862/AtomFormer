"""Seed42 shortlist, then complete V-select three-seed equal-weight selection."""
import math
from statistics import mean
from . import PROTOCOL,SCORER_VERSION
from .contracts import LAMBDAS
from ..supplement import metric_vector,acceptable
from ..io import digest,freeze


def _validate(e, split, size=None):
    if (e['protocol_id']!=PROTOCOL or e['method']!='atomformer' or e['phase']!='1C'
        or e['split']!=split or e['scorer_version']!=SCORER_VERSION or not e['completed_nominal_budget']
        or e['checkpoint_kind']!='last' or e['train_seed'] not in (42,43,44)
        or e['sampling_seed']!=e['train_seed']+100 or e['lambda'] not in LAMBDAS):
        raise ValueError('Invalid lambda evidence role/budget/checkpoint')
    n=e['metrics']['records']
    if not n or e['metrics']['requested_attempts']!=20*n or (size is not None and n!=size):
        raise ValueError('Incomplete validation attempt budget')
    metric_vector(e['metrics'])


def _objective(metrics):
    v=metric_vector(metrics);return v['Q']-.2*v['S']


def shortlist(evidence, path):
    if len(evidence)!=len(LAMBDAS):raise ValueError('Complete seed42 grid required')
    by_value={}
    for e in evidence:
        _validate(e,'v_select_quick',2048)
        if e['train_seed']!=42 or e['lambda'] in by_value:raise ValueError('Seed42 or unique grid violation')
        by_value[e['lambda']]=e
    if set(by_value)!=set(LAMBDAS):raise ValueError('Grid differs')
    for field in ('context','input_sha256','source_checkpoint_sha256','policy_ids_sha256','reference_sha256'):
        if len({e[field] for e in evidence})!=1:raise ValueError('Grid evidence differs: '+field)
    default=by_value[.2]['metrics']
    eligible=[v for v,e in by_value.items() if v!=.2 and acceptable(e['metrics'],default) and _objective(e['metrics'])>_objective(default)+1e-12]
    best=max(eligible,key=lambda v:(_objective(by_value[v]['metrics']),v)) if eligible else None
    payload={'protocol_id':PROTOCOL,'selection_stage':'shortlist_only','context':evidence[0]['context'],
             'values':[.2]+([best] if best is not None else []),'seed':42,'evidence_sha256':digest(evidence),
             'source_by_seed':{'42':evidence[0]['source_checkpoint_sha256']},'final_lambda_frozen':False}
    return freeze(path,payload)


def freeze_lambda(evidence, shortlist_record, path):
    if shortlist_record.get('protocol_id')!=PROTOCOL or shortlist_record.get('selection_stage')!='shortlist_only':raise ValueError('Invalid shortlist')
    values=shortlist_record['values'];expected={(s,v) for s in (42,43,44) for v in values};runs={}
    for e in evidence:
        _validate(e,'v_select')
        k=(e['train_seed'],e['lambda'])
        if k not in expected or k in runs or e['context']!=shortlist_record['context']:raise ValueError('Run outside frozen shortlist')
        if not e.get('complete_v_select'):raise ValueError('Complete V-select required, not a relabelled quick set')
        runs[k]=e
    if set(runs)!=expected:raise ValueError('All three independent seeds and shortlisted lambdas are required')
    if len({e['input_sha256'] for e in evidence})!=1 or len({e['reference_sha256'] for e in evidence})!=1:raise ValueError('Validation/reference mismatch')
    for seed in (42,43,44):
        for field in ('source_checkpoint_sha256','policy_ids_sha256'):
            if len({runs[seed,v][field] for v in values})!=1:raise ValueError('Within-seed source or policy mismatch')
    if runs[42,.2]['source_checkpoint_sha256']!=shortlist_record['source_by_seed']['42']:raise ValueError('Shortlist source changed')
    if len({runs[s,.2]['source_checkpoint_sha256'] for s in (42,43,44)})!=3:
        raise ValueError('Three inference seeds on one checkpoint are not independent training runs')
    aggregates={}
    for value in values:
        vectors=[metric_vector(runs[s,value]['metrics']) for s in (42,43,44)]
        avg={k:mean(v[k] for v in vectors) for k in vectors[0]}
        aggregates[value]={'retained_yield':avg['Y'],'mean_unique_retained':avg['U'],
            'retained_diversity_all_records':avg['D'],'retained_properties':{'qed':avg['Q'],'sas':avg['S']}}
    eligible=[v for v in values if acceptable(aggregates[v],aggregates[.2])]
    highest=max(_objective(aggregates[v]) for v in eligible)
    ties=[v for v in eligible if math.isclose(_objective(aggregates[v]),highest,abs_tol=1e-12,rel_tol=0)]
    chosen=.2 if .2 in ties else max(ties)
    return freeze(path,{'protocol_id':PROTOCOL,'context':shortlist_record['context'],'value':chosen,
        'selection_seeds':[42,43,44],'validation_role':'v_select','aggregation':'equal_weight_independent_runs',
        'metrics_by_lambda':{str(k):v for k,v in aggregates.items()},'shortlist_sha256':digest(shortlist_record),
        'evidence_sha256':digest(evidence),'no_reselection_from_vcheck_or_test':True})
