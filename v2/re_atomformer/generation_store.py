"""Durable record-sized generation shards with stable per-input sampling RNG."""
import hashlib
from pathlib import Path
from .io import freeze,read_json,atomic_json,digest
from .checkpoints import seed_all


def generate_records(records,create_record,root,contract,resume=False,base_seed=0,attempts=20):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    freeze(root/'contract.json',contract)
    expected_ids=[r['record_id'] for r in records]
    if len(expected_ids)!=len(set(expected_ids)):raise ValueError('Duplicate generation record IDs')
    outputs=[]
    for record in records:
        rid=record['record_id'];name=hashlib.sha256(rid.encode()).hexdigest()
        path=root/(name+'.json')
        if path.exists():
            if not resume:raise ValueError('Generation shards exist; explicit resume required')
            payload=read_json(path)
            if payload['condition_sha256']!=digest(record) or payload['contract_sha256']!=digest(contract):
                raise ValueError('Generation shard provenance differs')
            rows=payload['rows']
        else:
            seed=(base_seed+int(name[:12],16))%(2**31-1)
            seed_all(seed)
            rows=create_record(record)
            if len(rows)!=attempts:raise ValueError('Incomplete record output')
            freeze(path,{'record_id':rid,'condition_sha256':digest(record),'contract_sha256':digest(contract),'sampling_seed':seed,'rows':rows})
        if [r['attempt_id'] for r in rows]!=list(range(attempts)) or any(r['record_id']!=rid for r in rows):
            raise ValueError('Generation shard attempt identity differs')
        outputs.extend(rows)
        atomic_json(root/'progress.json',{'completed_records':len(outputs)//attempts,'requested_records':len(records),
                    'completed_attempts':len(outputs),'atomic_boundary':'one_record',
                    'operational_replay_boundary':'An interrupted, uncommitted record may be replayed; completed shards are never sampled again.'})
    return outputs
