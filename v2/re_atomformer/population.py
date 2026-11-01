"""Explicit paired/native condition populations; eligibility is reported, never hidden."""
from .io import digest


def training_population(training,validation,spec):
    population=spec.get('population','native')
    kind=spec.get('encoder','atom')
    needs_geometry=population=='common_geometry' or spec.get('decoder')=='diffusion' or kind in ('geo','gated_geo','pocket_geo','pocket_distance')
    if population not in ('native','common_geometry'):
        raise ValueError('Unknown population policy')
    def select(rows):
        kept=[];unsupported=[]
        for row in rows:
            reason=None
            if needs_geometry and row.get('pos') is None:reason='missing_geometry'
            if spec.get('decoder')=='diffusion' and not row.get('target_3d'):reason='missing_3d_training_target'
            if reason:unsupported.append({'record_id':row['record_id'],'reason':reason})
            else:kept.append(row)
        return kept,unsupported
    train,ut=select(training);val,uv=select(validation)
    if not train or not val:raise ValueError('Selected population has an empty train/validation split')
    return train,val,{'policy':population,'training_requested':len(training),'validation_requested':len(validation),
                     'training_supported':len(train),'validation_supported':len(val),'unsupported_training':ut,'unsupported_validation':uv,
                     'training_ids_sha256':digest(sorted(r['record_id'] for r in train)),'validation_ids_sha256':digest(sorted(r['record_id'] for r in val))}


def pocket_intervention(records,mode='correct'):
    """Fixed donor mapping, independent of minibatch composition and repeated rollouts."""
    if mode=='correct':return list(records)
    if mode=='none':return [dict(r,pocket_z=[],pocket_pos=[],input_pocket_id=None) for r in records]
    if mode!='shuffle':raise ValueError('Unknown pocket intervention')
    ordered=sorted(records,key=lambda r:r['record_id'])
    if len({r.get('pocket_id') for r in ordered})<2:raise ValueError('Pocket shuffle needs distinct pocket identities')
    donors={}
    for i,row in enumerate(ordered):
        donor=next(ordered[(i+j)%len(ordered)] for j in range(1,len(ordered)) if ordered[(i+j)%len(ordered)].get('pocket_id')!=row.get('pocket_id'))
        donors[row['record_id']]=donor
    output=[]
    for row in records:
        donor=donors[row['record_id']]
        old=donor['pos'][donor['anchor_index']];new=row['pos'][row['anchor_index']]
        points=[[point[k]-old[k]+new[k] for k in range(3)] for point in donor['pocket_pos']]
        output.append(dict(row,pocket_z=donor['pocket_z'],pocket_pos=points,input_pocket_id=donor['pocket_id'],pocket_donor_record_id=donor['record_id']))
    return output
