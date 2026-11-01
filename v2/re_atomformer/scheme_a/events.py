"""Immutable generation identity and order-independent uniform event subsampling."""
import random
import sys
from ..io import digest

IDENTITY_FIELDS = ('generation_protocol','dataset_id','split','record_id','method','model_variant',
                   'checkpoint_sha256','sampling_seed','attempt_id')
DIVERSITY_SEED = 20261025
DIVERSITY_CAP = 5000


def event_id(event):
    identity={name:event[name] for name in IDENTITY_FIELDS}
    if not isinstance(identity['record_id'],str) or not identity['record_id']:
        raise ValueError('Stable record identity required')
    if not isinstance(identity['attempt_id'],int) or not 0<=identity['attempt_id']<20:
        raise ValueError('Event attempt index outside the fixed budget')
    if not isinstance(identity['sampling_seed'],int):raise ValueError('Explicit sampling seed required')
    return digest(identity)


def validate_events(events, expected_ids=None, attempts=20):
    seen=set();slots=set();support={};counts={};contexts=set()
    for row in events:
        eid=event_id(row)
        contexts.add(digest({k:row[k] for k in IDENTITY_FIELDS if k not in ('record_id','attempt_id')}))
        if len(contexts)>1:raise ValueError('Mixed scientific runs in one attempt ledger')
        if row.get('generation_event_id')!=eid:raise ValueError('Generation event identity mismatch')
        if eid in seen:raise ValueError('Duplicate generation event ID')
        slot=(row['record_id'],row['attempt_id'])
        if slot in slots:raise ValueError('Duplicate request/attempt slot')
        seen.add(eid);slots.add(slot)
        rid=row['record_id'];value=row['input_supported']
        if not isinstance(value,bool):raise ValueError('Support mask must be explicit')
        if rid in support and support[rid]!=value:raise ValueError('Support mask changed after generation')
        if not value and row.get('assembled_output') is not None:raise ValueError('Unsupported input cannot contain a generated product')
        support[rid]=value;counts[rid]=counts.get(rid,0)+1
    if expected_ids is not None:
        ids=list(expected_ids)
        if len(ids)!=len(set(ids)) or not ids:raise ValueError('Nonempty unique request list required')
        if set(counts)!=set(ids) or any(counts[i]!=attempts for i in ids):
            raise ValueError('Incomplete requested attempt budget')
    return support


def uniform_events(rows, seed=DIVERSITY_SEED, cap=DIVERSITY_CAP):
    """Uniform without replacement over events; molecules are never deduplicated."""
    if seed!=DIVERSITY_SEED or cap!=DIVERSITY_CAP:raise ValueError('Diversity sampling protocol is frozen')
    by_id={}
    for row in rows:
        eid=row['generation_event_id']
        if eid in by_id:raise ValueError('Duplicate event in diversity population')
        if eid!=event_id(row):raise ValueError('Invalid event identity')
        by_id[eid]=row
    eligible=sorted(eid for eid,row in by_id.items() if row['valid'])
    chosen=sorted(random.Random(seed).sample(eligible,min(len(eligible),cap)))
    manifest={'algorithm':'sorted_event_ids_python_random_sample_v1','seed':seed,'cap':cap,
              'python':sys.version.split()[0],'candidate_events':len(eligible),'selected_events':len(chosen),
              'eligible_ids_sha256':digest(eligible),'selected_ids':chosen,'selected_ids_sha256':digest(chosen),
              'molecule_deduplication':False}
    return [by_id[eid] for eid in chosen],manifest
