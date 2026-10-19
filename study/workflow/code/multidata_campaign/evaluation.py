import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import sys
from rerun_campaign.runtime import atomic_json, sha256, read_jsonl, signature, write_jsonl

def aggregate(values):
    if any((v is None for v in values)):
        return {'mean': None, 'sd': None, 'n': len(values)}
    return {'mean': statistics.mean(values), 'sd': statistics.stdev(values) if len(values) > 1 else None, 'n': len(values)}

def paired_bootstrap(left, right, replicates=2000, seed=0, cluster=False):
    import numpy as np
    if len(left) != 3 or len(right) != 3:
        raise ValueError('Paired CI requires three training runs')
    keys = sorted(left[0], key=lambda k: int(k))
    if not keys or any((set(d) != set(keys) for d in left + right)):
        raise ValueError('Paired coverage differs')
    metrics = sorted(set.intersection(*(set(d[k]['statistics']) for d in left + right for k in keys)))
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        cores = {d[k]['core'] for d in left + right}
        if len(cores) != 1:
            raise ValueError('Paired core mapping differs')
        groups[next(iter(cores)) if cluster else k].append(i)
    membership = [groups[k] for k in sorted(groups)]
    arrays = []
    for side in (left, right):
        a = np.asarray([[[d[k]['statistics'][m] for m in metrics] for k in keys] for d in side], dtype=float)
        arrays.append(np.stack([a[:, indices].sum(axis=1) for indices in membership], axis=1))

    def delta(picked):
        a, b = (arr[:, picked].sum(axis=1) for arr in arrays)
        means = []
        for x in (a, b):
            with np.errstate(divide='ignore', invalid='ignore'):
                means.append(x[:, :, 0] / x[:, :, 1])
        return (means[0] - means[1]).mean(axis=0)
    point = delta(np.arange(len(membership)))
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(replicates):
        samples.append(delta(rng.randint(0, len(membership), len(membership))))
    samples = np.asarray(samples)
    return {'unit': 'scaffold_core' if cluster else 'test_record', 'groups': len(membership), 'training_runs': 3, 'replicates': replicates, 'training_seed_uncertainty': False, 'metrics': {m: {'point': float(point[i]) if np.isfinite(point[i]) else None, 'ci95': [float(x) for x in np.quantile(samples[np.isfinite(samples[:, i]), i], [0.025, 0.975])] if np.isfinite(samples[:, i]).any() else None} for i, m in enumerate(metrics)}}
