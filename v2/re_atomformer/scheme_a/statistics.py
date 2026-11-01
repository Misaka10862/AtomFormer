"""Equal-run summaries and common-core bootstrap without fictitious training-seed pairing."""
from collections import Counter
from statistics import mean,stdev
import random


def aggregate(records,weights):
    def total(name):return sum(weights.get(r['record_id'],0)*r[name] for r in records)
    n=sum(weights.get(r['record_id'],0) for r in records);attempts=total('attempts');valid=total('valid_outputs')
    def ratio(a,b):return a/b if b else None
    supported=sum(weights.get(r['record_id'],0)*r['attempts'] for r in records if r['input_supported'])
    return {'validity':ratio(valid,supported),'valid_yield':ratio(valid,attempts),
        'qed':ratio(total('qed_sum'),total('qed_n')),'sas':ratio(total('sas_sum'),total('sas_n')),
        'novelty':ratio(total('novel_count'),valid),'core_match':ratio(total('core_match_count'),valid),
        'core_retention':ratio(total('retained_outputs'),valid),'Y':ratio(total('retained_outputs'),attempts),
        'U':ratio(total('unique_retained'),n),'D':ratio(total('retained_diversity_all_records'),n)}


def system_bootstrap(systems,core_by_record,reference='atomformer_rl',replicates=2000,seed=0):
    """systems[name] is three evaluated runs; resample cores, never pair external training seeds."""
    if reference not in systems or any(len(runs)!=3 for runs in systems.values()):raise ValueError('Three repeats and reference required')
    ids=set(core_by_record);cores=sorted(set(core_by_record.values()))
    if not cores:raise ValueError('Empty common core population')
    for runs in systems.values():
        for run in runs:
            rows=run['per_record']
            if len(rows)!=len(ids) or {r['record_id'] for r in rows}!=ids:raise ValueError('Common population differs')
    def evaluate(weights):
        values={}
        for name,runs in systems.items():
            individual=[aggregate(run['per_record'],weights) for run in runs]
            values[name]={metric:mean(v[metric] for v in individual) if all(v[metric] is not None for v in individual) else None for metric in individual[0]}
        return values
    observed=evaluate({rid:1 for rid in ids});samples={name:{m:[] for m in values} for name,values in observed.items() if name!=reference}
    rng=random.Random(seed)
    for _ in range(replicates):
        counts=Counter(rng.choice(cores) for _ in cores)
        values=evaluate({rid:counts[core] for rid,core in core_by_record.items()})
        for name,metrics in samples.items():
            for metric,distribution in metrics.items():
                if values[name][metric] is not None and values[reference][metric] is not None:
                    distribution.append(values[name][metric]-values[reference][metric])
    result={}
    for name,metrics in samples.items():
        result[name]={}
        for metric,distribution in metrics.items():
            distribution.sort();a,b=observed[name][metric],observed[reference][metric]
            result[name][metric]={'difference':a-b if a is not None and b is not None else None,
                'ci95':[distribution[int(.025*(len(distribution)-1))],distribution[int(.975*(len(distribution)-1))]] if distribution else None,
                'defined_bootstrap_replicates':len(distribution)}
    return {'reference':reference,'comparisons':result,'bootstrap_replicates':replicates,'core_count':len(cores),
            'conditional_on_evaluated_checkpoints':True,'external_repeats_are_training_pairs':False,
            'aggregate_diversity_bootstrap_performed':False}


def repeated_summary(runs,kind):
    if len(runs)!=3:raise ValueError('Three evaluated repeats required')
    checkpoints={r['checkpoint_sha256'] for r in runs}
    if kind=='official':
        if len(checkpoints)!=1 or any(r['train_seed'] is not None for r in runs):raise ValueError('Official repeats must share one frozen checkpoint')
    elif kind=='atomformer':
        if len(checkpoints)!=3 or {r['train_seed'] for r in runs}!={42,43,44}:raise ValueError('Independent training checkpoints required')
    else:raise ValueError('Explicit repeat kind required')
    if {r['sampling_seed'] for r in runs}!={142,143,144}:raise ValueError('Sampling repeats differ')
    result={}
    for metric in ('qed','sas','aggregate_diversity_uniform','novelty_vs_benchmark_train','validity','core_match','core_retention','retained_yield','mean_unique_retained','retained_diversity_all_records'):
        values=[r['summary'][metric] for r in runs];defined=[v for v in values if v is not None]
        result[metric]={'values':values,'mean':mean(defined) if len(defined)==3 else None,'sample_sd':stdev(defined) if len(defined)==3 else None,'defined_runs':len(defined)}
    return {'kind':kind,'n_checkpoints':len(checkpoints),'n_sampling_repeats':3,'metrics':result,
            'sd_interpretation':'fixed-checkpoint sampling variation' if kind=='official' else 'training plus sampling variation'}
