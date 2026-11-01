"""Ordered CPU scoring; process count never changes outputs or denominators."""
_EVALUATOR=None

def _initialize(reference):
    global _EVALUATOR
    from .evaluation import Evaluator
    _EVALUATOR=Evaluator(reference)

def _one(row):
    value=_EVALUATOR.score(row.get('assembled_output'),row['scoring_target_core'])
    return {**row,**value,'failure_reason':row.get('failure_reason') or value.get('failure_reason')}

def score_rows(rows,reference,workers=1):
    if workers==1:
        _initialize(reference);return [_one(row) for row in rows]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(reference,)) as pool:
        return list(pool.map(_one,rows,chunksize=128))
