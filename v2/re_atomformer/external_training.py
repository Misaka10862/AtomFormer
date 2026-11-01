"""Sequence baseline training uses native likelihoods and shared scientific contracts."""
import copy
from pathlib import Path
import torch
from .external import SequenceBaseline,adapt_training_rows
from .data import Condition
from .checkpoints import seed_all,save_training,restore_training,save,load,rng_state,restore_rng
from .io import digest,freeze,atomic_json
from .protocol import RewardSpec,score,training_reference,training_thresholds
from .metrics import summarize
from .optimization import augmented_likelihood_loss
from .splits import audit_splits,stratified_order


def evaluate_sequence(adapter,rows,reference,spec,attempts=20):
    conditions=[Condition.from_record(r) for r in rows];outputs=[]
    adapter.network.eval()
    for r,c in zip(rows,conditions):
        supported=adapter.supports(c)
        for a in range(attempts):
            smiles=adapter.sample([c])[0][0] if supported else None
            outputs.append({**score(smiles,r,reference,spec),'record_id':c.record_id,'attempt_id':a,'input_supported':supported})
    return outputs


def train_sequence(config,train_rows,val_rows,root,device='cpu',resume=False):
    audit_splits(train_rows+val_rows)
    if any(r['split']!='train' for r in train_rows) or any(r['split']!='validation' for r in val_rows):
        raise ValueError('External training split mismatch')
    method=config['method'];settings=config['training'];root=Path(root);root.mkdir(parents=True,exist_ok=True)
    seed_all(config['seed'])
    adapter=SequenceBaseline(method,train_rows,config.get('architecture',{}),config['upstreams'],config['upstream_lock'],device)
    full_training=list(train_rows)
    from .population import training_population
    train_rows,val_rows,population=training_population(train_rows,val_rows,config)
    atomic_json(root/'population.json',population)
    native_train=adapt_training_rows(train_rows)
    from baselines.scaffold_bench.common import convert
    native_val=[convert((i,r)) for i,r in enumerate(val_rows)]
    if not all(adapter.backend.encodable(r) for r in native_train+native_val):
        raise ValueError('External NLL coverage is incomplete; report before altering the task')
    reference,rhash=training_reference(full_training);thresholds=training_thresholds(full_training)
    reward_settings=dict(config.get('reward',{}));reward_settings['method']=method
    reward_spec=RewardSpec(**reward_settings)
    mode=settings.get('mode','supervised');prior=None
    if mode=='rl':
        source=load(settings['source_checkpoint'],device)
        adapter.network.load_state_dict(source['model'],strict=True)
        prior=copy.deepcopy(adapter.backend)
        prior.network.eval()
        for p in prior.network.parameters():p.requires_grad_(False)
    elif mode!='supervised':raise ValueError('Unknown baseline training mode')
    optimizer=torch.optim.Adam(adapter.network.parameters(),lr=settings.get('lr',1e-4))
    contract={'config':config,'train_sha256':digest(full_training),'policy_population_sha256':digest(train_rows),'validation_sha256':digest(val_rows),'reference':rhash}
    freeze(root/'contract.json',contract)
    progress={'epoch':0,'cursor':0,'order':None,'updates':0,'attempts':0,'best':None,'bad':0,'history':[]}
    if (root/'resume.pt').exists():
        if not resume:raise ValueError('Explicit resume required')
        progress=restore_training(root/'resume.pt',adapter.network,optimizer,contract,device)
    index={r['record_id']:i for i,r in enumerate(train_rows)}
    chosen=stratified_order(train_rows,config['seed'])[:settings.get('max_train_records',len(train_rows))]
    batch_size=settings.get('batch_size',64)
    while progress['epoch']<settings.get('epochs',12) and progress['bad']<settings.get('patience',5):
        if progress['order'] is None:
            progress['order']=torch.randperm(len(chosen)).tolist();progress['cursor']=0
        adapter.network.train()
        while progress['cursor']<len(chosen):
            original=[chosen[i] for i in progress['order'][progress['cursor']:progress['cursor']+batch_size]]
            native=[native_train[index[r['record_id']]] for r in original]
            if mode=='supervised':
                loss=adapter.backend.nll(native).mean()
            else:
                conditions=[Condition.from_record(r) for r in original]
                generated,decorations=adapter.sample(conditions)
                scored=[score(s,r,reference,reward_spec) for s,r in zip(generated,original)]
                strings=decorations if method=='libinvent' else generated
                strings=[s if s is not None else '' for s in strings]
                actor=-adapter.backend.nll(native,strings)
                with torch.no_grad():prior_lp=-prior.nll(native,strings)
                rewards=torch.tensor([r['reward'] for r in scored],device=device)
                loss=augmented_likelihood_loss(actor,prior_lp,rewards,settings.get('sigma',120.))
                freeze(root/'rl_trace'/('%08d.json'%(progress['updates']+1)),{'update':progress['updates']+1,'rows':[dict(v,record_id=r['record_id']) for r,v in zip(original,scored)]})
                progress['attempts']+=len(original)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite external loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapter.network.parameters(),1.);optimizer.step()
            progress['cursor']+=len(original);progress['updates']+=1
            if progress['updates']%settings.get('checkpoint_every',25)==0 or progress['cursor']==len(chosen):
                save_training(root/'resume.pt',adapter.network,optimizer,contract,progress)
        adapter.network.eval();state=rng_state();seed_all(701)
        if mode=='supervised':
            with torch.no_grad():
                value=sum(float(adapter.backend.nll(native_val[i:i+batch_size]).sum()) for i in range(0,len(native_val),batch_size))/len(native_val)
        else:
            evaluated=evaluate_sequence(adapter,val_rows,reference,reward_spec)
            value=-summarize(evaluated,[r['record_id'] for r in val_rows],20,thresholds)['u_good']
        restore_rng(state)
        improved=progress['best'] is None or value<progress['best']
        progress['best']=value if improved else progress['best'];progress['bad']=0 if improved else progress['bad']+1
        progress['epoch']+=1;progress['cursor']=0;progress['order']=None
        progress['history'].append({'epoch':progress['epoch'],'validation':value,'updates':progress['updates'],'attempts':progress['attempts']})
        payload={'model':adapter.network.state_dict(),'config':config,'progress':progress,'contract':contract}
        if improved:save(root/'best.pt',payload)
        save(root/'last.pt',payload);save_training(root/'resume.pt',adapter.network,optimizer,contract,progress)
        atomic_json(root/'history.json',progress['history'])
    return adapter,progress
