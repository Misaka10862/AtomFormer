"""Immutable raw-attempt ledger; generation never invokes the benchmark scorer."""
import hashlib
from pathlib import Path
from . import PROTOCOL
from .conditions import Request
from .events import event_id,validate_events
from .inference import frozen_session,assert_plain_output
from ..io import freeze,read_json,digest,atomic_json
from ..checkpoints import seed_all


def generate(records,adapter,root,context,resume=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    required=('dataset_id','split','method','model_variant','checkpoint_sha256','sampling_seed')
    if any(k not in context for k in required):raise ValueError('Missing scientific generation context')
    if context.get('generation_protocol')!=PROTOCOL:raise ValueError('Wrong generation protocol')
    requests=[Request.from_record(r) for r in records]
    if len({r.record_id for r in requests})!=len(requests):raise ValueError('Duplicate request ID')
    # Contract binds conditions only. Swapping reference labels cannot affect support or RNG.
    contract={'context':context,'conditions_sha256':digest([r.__dict__ for r in requests]),'attempts':20}
    freeze(root/'contract.json',contract)
    outputs=[]
    with frozen_session(adapter.network) as state:
        support={r.record_id:adapter.support(r) for r in requests}
        freeze(root/'support.json',support)
        for request in requests:
            name=hashlib.sha256(request.record_id.encode()).hexdigest();path=root/(name+'.json')
            if path.exists():
                if not resume:raise ValueError('Existing event shard requires explicit resume')
                shard=read_json(path)
                if shard['contract_sha256']!=digest(contract) or shard['request_sha256']!=request.identity():raise ValueError('Shard identity differs')
                rows=shard['rows']
            else:
                seed=(context['sampling_seed']+int(name[:12],16))%(2**31-1);seed_all(seed)
                item=support[request.record_id]
                raw=adapter.generate(request,20) if item['supported'] else []
                assert_plain_output(raw)
                if item['supported'] and len(raw)!=20:raise RuntimeError('Native interface did not deliver exactly 20 raw trajectories')
                rows=[]
                for attempt in range(20):
                    payload=raw[attempt] if item['supported'] else {'raw_generated_output':None,'assembled_output':None,'failure_reason':'unsupported_input'}
                    protected=set(context)|{'record_id','scoring_target_core','attempt_id','input_supported','native_input','attempt_status','generation_event_id'}
                    if protected.intersection(payload):raise RuntimeError('Adapter attempted to override scientific identity')
                    row={**context,'record_id':request.record_id,'scaffold_smiles_with_attachment':request.scaffold_smiles_with_attachment,
                         'scoring_target_core':request.scoring_target_core,'attempt_id':attempt,'input_supported':item['supported'],
                         'native_input':item['native_input'],'unsupported_reason':item.get('reason'),
                         'attempt_status':'GENERATED' if item['supported'] else 'NOT_DELIVERED_UNSUPPORTED',**payload}
                    row['generation_event_id']=event_id(row);rows.append(row)
                freeze(path,{'request_sha256':request.identity(),'contract_sha256':digest(contract),'rows':rows})
            outputs.extend(rows)
            atomic_json(root/'progress.json',{'completed_records':len(outputs)//20,'requested_records':len(requests),
                'completed_slots':len(outputs),'recovery':'complete record shards never resample; uncommitted record can replay'})
    validate_events(outputs,[r.record_id for r in requests])
    from .inference import state_sha256
    atomic_json(root/'inference_state.json',{**state,'state_after_sha256':state_sha256(adapter.network),'passed':True})
    return outputs
