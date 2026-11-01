"""Core-aggregated implementation of the frozen 2000-resample statistics.

Same Python Random(seed) core draws and equal-run aggregation as the scalar
reference, with additive sufficient statistics grouped before resampling.
"""
import math,random
from statistics import mean,stdev
import numpy as np

FIELDS=('attempts','valid_outputs','retained_outputs','unique_retained','retained_diversity_all_records',
        'qed_sum','qed_n','sas_sum','sas_n','novel_count','core_match_count',
        'retained_qed_sum','retained_qed_n','retained_sas_sum','retained_sas_n')
METRICS=('validity','valid_yield','qed','sas','novelty','core_match','core_retention','Y','U','D','retained_qed','retained_sas')


def ratios(x):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.stack((x[...,1]/x[...,16],x[...,1]/x[...,0],x[...,5]/x[...,6],x[...,7]/x[...,8],
            x[...,9]/x[...,1],x[...,10]/x[...,1],x[...,2]/x[...,1],x[...,2]/x[...,0],
            x[...,3]/x[...,15],x[...,4]/x[...,15],x[...,11]/x[...,12],x[...,13]/x[...,14]),axis=-1)


def finite(value):return float(value) if np.isfinite(value) else None


def core_bootstrap(systems,core_by_record,reference='atomformer_rl',replicates=2000,seed=0):
    if reference not in systems or any(len(v)!=3 for v in systems.values()):raise ValueError('Three repeats required')
    names=list(systems);cores=sorted(set(core_by_record.values()));lookup={c:i for i,c in enumerate(cores)};ids=set(core_by_record)
    if not ids:raise ValueError('Empty common population')
    totals=np.zeros((len(cores),len(names),3,17),dtype=np.float64)
    for si,name in enumerate(names):
        for ri,run in enumerate(systems[name]):
            rows=run['per_record']
            if len(rows)!=len(ids) or {r['record_id'] for r in rows}!=ids:raise ValueError('Common population differs')
            for row in rows:
                k=lookup[core_by_record[row['record_id']]]
                totals[k,si,ri,:15]+=[row[f] for f in FIELDS]
                totals[k,si,ri,15]+=1;totals[k,si,ri,16]+=row['attempts'] if row['input_supported'] else 0
    per_run=ratios(totals.sum(axis=0));observed=per_run.mean(axis=1);ref=names.index(reference)
    draws=[];rng=random.Random(seed);matrix=totals.reshape(len(cores),-1)
    for start in range(0,replicates,16):
        counts=np.array([np.bincount([rng.choice(range(len(cores))) for _ in cores],minlength=len(cores))
                         for _ in range(min(16,replicates-start))],dtype=np.float64)
        values=ratios((counts@matrix).reshape(len(counts),len(names),3,17)).mean(axis=2)
        draws.append(values-values[:,ref:ref+1,:])
    draws=np.concatenate(draws,axis=0);comparisons={}
    for si,name in enumerate(names):
        if name==reference:continue
        comparisons[name]={}
        for mi,metric in enumerate(METRICS):
            distribution=np.sort(draws[:,si,mi][np.isfinite(draws[:,si,mi])]);n=len(distribution)
            comparisons[name][metric]={'difference':finite(observed[si,mi]-observed[ref,mi]),
                'ci95':[float(distribution[int(.025*(n-1))]),float(distribution[int(.975*(n-1))])] if n else None,
                'defined_bootstrap_replicates':n}
    return {'reference':reference,'difference_direction':'system_minus_reference','comparisons':comparisons,
        'bootstrap_replicates':replicates,'bootstrap_seed':seed,'core_count':len(cores),
        'conditional_on_evaluated_checkpoints':True,'external_repeats_are_training_pairs':False,
        'aggregate_diversity_bootstrap_performed':False,
        'implementation':'core_sufficient_statistics_python_random_draws_v1'}


def paired_run_differences(left,right):
    if len(left)!=3 or len(right)!=3:raise ValueError('Three paired runs required')
    a={r['train_seed']:r for r in left};b={r['train_seed']:r for r in right}
    if set(a)!={42,43,44} or set(b)!=set(a):raise ValueError('Actual training-seed pairing required')
    result={}
    for field in ('qed','sas','retained_yield','mean_unique_retained','retained_diversity_all_records','aggregate_diversity_uniform'):
        values=[a[s]['summary'][field]-b[s]['summary'][field] if a[s]['summary'][field] is not None and b[s]['summary'][field] is not None else None for s in (42,43,44)]
        defined=all(v is not None for v in values)
        result[field]={'per_training_seed':dict(zip((42,43,44),values)),'mean':mean(values) if defined else None,'sample_sd':stdev(values) if defined else None}
    return {'difference_direction':'left_minus_right','metrics':result}
