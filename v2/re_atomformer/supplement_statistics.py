"""Final statistics separate training variability from core-resampling uncertainty."""
from collections import Counter
import random
from statistics import mean,stdev


def aggregate(records, weights):
    total=sum(weights.get(r['record_id'],0) for r in records)
    if not total:raise ValueError('Empty weighted population')
    def summed(field):return sum(weights.get(r['record_id'],0)*r[field] for r in records)
    count=summed('retained_scored_count')
    return {'Y':summed('retained_outputs')/summed('attempts'),
            'U':summed('unique_retained')/total,'D':summed('retained_diversity_all_records')/total,
            'QED':summed('retained_qed_sum')/count if count else None,
            'SAS':summed('retained_sas_sum')/count if count else None}


def paired_bootstrap(left,right,core_by_record,replicates=2000,seed=0):
    """left/right: {training_seed: per_record}. No selection or ranking is performed."""
    if set(left)!=set(right) or set(left)!={42,43,44}:raise ValueError('Three independent paired training seeds required')
    ids=set(core_by_record)
    for records in list(left.values())+list(right.values()):
        if len(records)!=len(ids) or {r['record_id'] for r in records}!=ids:raise ValueError('Paired population differs')
    cores=sorted(set(core_by_record.values()));rng=random.Random(seed)
    per_run={};samples={k:[] for k in ('Y','U','D','QED','SAS')}
    weights={rid:1 for rid in ids}
    for training_seed in sorted(left):
        a,b=aggregate(left[training_seed],weights),aggregate(right[training_seed],weights)
        per_run[training_seed]={k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in a}
    for _ in range(replicates):
        draws=Counter(rng.choice(cores) for _ in cores)
        w={rid:draws[core] for rid,core in core_by_record.items()}
        diffs={k:[] for k in samples}
        for training_seed in sorted(left):
            a,b=aggregate(left[training_seed],w),aggregate(right[training_seed],w)
            for k in diffs:
                if a[k] is not None and b[k] is not None:diffs[k].append(a[k]-b[k])
        for k,values in diffs.items():
            if len(values)==3:samples[k].append(mean(values))
    metrics={}
    for k,values in samples.items():
        observed=[v[k] for v in per_run.values() if v[k] is not None];values.sort()
        metrics[k]={'mean_paired_difference':mean(observed) if len(observed)==3 else None,
            'training_sample_sd':stdev(observed) if len(observed)==3 else None,
            'conditional_ci95':[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None,
            'bootstrap_defined_replicates':len(values)}
    return {'metrics':metrics,'per_seed_differences':per_run,'bootstrap_replicates':replicates,
            'core_count':len(cores),'conditional_on_fitted_models':True,
            'warning':'Core bootstrap does not replace independent-training variability.'}
