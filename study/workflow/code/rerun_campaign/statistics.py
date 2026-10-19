import numpy as np

def bootstrap_records(records, replicates=2000, seed=0):
    if not records:
        raise ValueError('No test records')
    keys = sorted(next(iter(records.values())))
    ordered = [records[k] for k in sorted(records)]
    matrix = np.asarray([[r[k] for k in keys] for r in ordered], dtype=np.float64)
    n = len(ordered)
    rng = np.random.RandomState(seed)
    total = matrix.sum(axis=0)
    samples = {k: [] for k in keys}
    for _ in range(replicates):
        picked = matrix[rng.randint(0, n, n)].sum(axis=0)
        for i, k in enumerate(keys):
            if picked[i, 1] > 0:
                samples[k].append(float(picked[i, 0] / picked[i, 1]))
    result = {}
    for i, k in enumerate(keys):
        values = samples[k]
        result[k] = {'point': float(total[i, 0] / total[i, 1]) if total[i, 1] > 0 else None, 'ci95': [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None, 'effective_replicates': len(values)}
    return {'unit': 'test_record', 'records': n, 'replicates': replicates, 'seed': seed, 'quantiles': [0.025, 0.975], 'metrics': result}
