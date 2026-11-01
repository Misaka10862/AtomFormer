"""Native DrugEx graph network, optimizer and mutation-policy sampler."""
import copy
import sys
from pathlib import Path
import numpy as np
import torch
from .external import require_upstream,DrugExAlignedEnvironment
from .data import Condition
from .protocol import score,RewardSpec,training_reference,training_thresholds
from .io import freeze,digest,atomic_json
from .checkpoints import seed_all,save_training,restore_training,save,load,rng_state,restore_rng
from .splits import audit_splits,stratified_order
from .metrics import summarize


class ScheduledState:
    def __init__(self,native):self.native=native
    def zero_grad(self,set_to_none=True):self.native._optimizer.zero_grad(set_to_none=set_to_none)
    def step(self):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in self.native._optimizer.param_groups for p in g['params']):
            raise FloatingPointError('Non-finite native optimizer gradient')
        self.native.step()
    def state_dict(self):return {'optimizer':self.native._optimizer.state_dict(),'n_steps':self.native.n_steps}
    def load_state_dict(self,state):self.native._optimizer.load_state_dict(state['optimizer']);self.native.n_steps=state['n_steps']


class NativeDrugEx:
    def __init__(self,training_rows,spec,device='cpu',vocabulary=None):
        path=require_upstream('drugex',spec['upstreams']['drugex'],spec['upstream_lock'])
        if str(path) not in sys.path:sys.path.insert(0,str(path))
        from drugex.data.corpus.vocabulary import VocGraph
        from drugex.training.generators import GraphTransformer
        from baselines.drugex.drugex_baseline_utils import vocabulary_words
        self.device=torch.device(device);self.spec=spec
        if vocabulary is None:
            if any(r['split']!='train' for r in training_rows):raise ValueError('DrugEx vocabulary uses training only')
            words=vocabulary_words([dict(r,scaffold_core_smiles=r['core_smiles']) for r in training_rows])
            vocabulary={'words':words,'max_len':spec.get('max_graph_length',128),'n_frags':1}
        self.vocabulary=vocabulary
        self.vocab=VocGraph(**vocabulary)
        model_args={'d_emb':512,'d_model':512,'d_inner':1024,'n_head':8,'n_layer':12,**spec.get('architecture',{})}
        self.network=GraphTransformer(self.vocab,device=self.device,use_gpus=(0,) if self.device.type=='cuda' else (-1,),**model_args)

    def encode(self,rows):
        arrays=[]
        for r in rows:
            result=self.vocab.encode([r['canonical_smiles']],[r['core_smiles']])[0]
            if np.asarray(result).shape!=(self.vocab.max_len,5):raise ValueError('Unsupported DrugEx graph')
            arrays.append(result)
        return torch.as_tensor(np.stack(arrays),dtype=torch.long,device=self.device)

    def condition_tensor(self,conditions):
        # Core-only template: held-out decoration labels never enter generation.
        return self.encode([{'canonical_smiles':c.core_smiles,'core_smiles':c.core_smiles} for c in conditions])

    def supports(self,condition):
        try:self.condition_tensor([condition]);return True
        except (ValueError,RuntimeError,IndexError,KeyError):return False

    @torch.no_grad()
    def sample(self,conditions):
        generated=self.network(self.condition_tensor(conditions).clone(),is_train=False)
        return self.vocab.decode(generated.detach().cpu().numpy())[1]

    def nll(self,tensor):
        return sum(-v.mean((1,2)) for v in self.network(tensor,is_train=True))


def evaluate(adapter,rows,reference,spec,attempts=20):
    result=[];adapter.network.eval()
    for r in rows:
        c=Condition.from_record(r);supported=adapter.supports(c)
        for a in range(attempts):
            smiles=adapter.sample([c])[0] if supported else None
            result.append({**score(smiles,r,reference,spec),'record_id':c.record_id,'attempt_id':a,'input_supported':supported})
    return result


def train(spec,rows,validation,root,device='cpu',resume=False):
    audit_splits(rows+validation)
    if any(r['split']!='train' for r in rows) or any(r['split']!='validation' for r in validation):raise ValueError('Split mismatch')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seed_all(spec.get('seed',11))
    adapter=NativeDrugEx(rows,spec,device);settings=spec['training'];mode=settings.get('mode','supervised')
    reference,rhash=training_reference(rows);thresholds=training_thresholds(rows)
    full_training=list(rows)
    from .population import training_population
    rows,validation,population=training_population(rows,validation,spec)
    atomic_json(root/'population.json',population)
    rsettings=dict(spec.get('reward',{}));rsettings['method']='drugex';reward_spec=RewardSpec(**rsettings)
    selected=stratified_order(rows,spec.get('seed',11))[:settings.get('max_train_records',len(rows))]
    kept=[];unsupported=[]
    for r in selected:
        try:adapter.encode([r]);kept.append(r)
        except (ValueError,RuntimeError,IndexError,KeyError) as error:unsupported.append({'record_id':r['record_id'],'reason':type(error).__name__})
    atomic_json(root/'training_coverage.json',{'requested':len(selected),'supported':len(kept),'unsupported':unsupported})
    if not kept:raise ValueError('No supported DrugEx training records')
    active=adapter.network;explorer=None
    if mode=='rl':
        from drugex.training.explorers import FragGraphExplorer
        from drugex.training.monitors import NullMonitor
        from drugex.training.environment import DrugExEnvironment
        from drugex.training.rewards import ParetoCrowdingDistance
        from drugex.training.scorers.properties import Property
        from drugex.training.scorers.modifiers import ClippedScore
        source=load(settings['source_checkpoint'],device)
        if source['vocabulary']!=adapter.vocabulary:raise ValueError('Source vocabulary differs')
        adapter.network.load_state_dict(source['model'],strict=True)
        prior=copy.deepcopy(adapter.network).eval()
        for p in prior.parameters():p.requires_grad_(False)
        if reward_spec.protocol=='native':
            env=DrugExEnvironment(scorers=[Property('QED',modifier=ClippedScore(lower_x=0.,upper_x=1.)),Property('SA',modifier=ClippedScore(lower_x=10.,upper_x=1.))],thresholds=[.5,.5],reward_scheme=ParetoCrowdingDistance())
        else:
            env=DrugExAlignedEnvironment(reference,{r['core_smiles']:r['core_smiles'] for r in rows})
        native_get_rewards=env.getRewards
        def audited_rewards(smiles,frags=None):
            values=native_get_rewards(smiles,frags=frags)
            if not np.isfinite(values).all():raise FloatingPointError('Non-finite native reward')
            env.last_reward_trace=[{'generated':s,'fragment':f,'native_reward':float(v)} for s,f,v in zip(smiles,frags,np.asarray(values).reshape(-1))]
            return values
        env.getRewards=audited_rewards
        explorer=FragGraphExplorer(adapter.network,env,mutate=prior,batch_size=settings.get('batch_size',64),epsilon=settings.get('epsilon',.1),beta=0.,n_samples=-1,device=adapter.device,use_gpus=(0,) if adapter.device.type=='cuda' else (-1,))
        explorer.monitor=NullMonitor();active=explorer
    elif mode!='supervised':raise ValueError('Unknown DrugEx mode')
    optimizer=ScheduledState(active.optim)
    if explorer is not None:explorer.optim=optimizer
    contract={'spec':spec,'train':digest(full_training),'policy_population':digest(rows),'validation':digest(validation),'reference':rhash,'vocabulary':adapter.vocabulary}
    freeze(root/'contract.json',contract)
    state={'epoch':0,'cursor':0,'order':None,'updates':0,'attempts':0,'best':None,'bad':0,'history':[]}
    if (root/'resume.pt').exists():
        if not resume:raise ValueError('Explicit resume required')
        state=restore_training(root/'resume.pt',active,optimizer,contract,device)
    batch_size=settings.get('batch_size',32)
    while state['epoch']<settings.get('epochs',12) and state['bad']<settings.get('patience',5):
        if state['order'] is None:state['order']=torch.randperm(len(kept)).tolist();state['cursor']=0
        adapter.network.train()
        while state['cursor']<len(kept):
            chunk=[kept[i] for i in state['order'][state['cursor']:state['cursor']+batch_size]]
            if mode=='supervised':
                loss=adapter.nll(adapter.encode(chunk)).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite DrugEx loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
            else:
                from torch.utils.data import DataLoader
                with torch.no_grad():generated=explorer(adapter.condition_tensor([Condition.from_record(r) for r in chunk]).clone()).detach().cpu()
                value=explorer.policy_gradient(DataLoader(generated,batch_size=len(chunk),shuffle=False))
                loss=torch.tensor(value);state['attempts']+=len(chunk)
                freeze(root/'rl_trace'/('%08d.json'%(state['updates']+1)),{'update':state['updates']+1,'rows':[dict(v,record_id=r['record_id']) for r,v in zip(chunk,env.last_reward_trace)]})
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite DrugEx loss')
            state['cursor']+=len(chunk);state['updates']+=1
            if state['updates']%settings.get('checkpoint_every',25)==0 or state['cursor']==len(kept):save_training(root/'resume.pt',active,optimizer,contract,state)
        adapter.network.eval();random_state=rng_state();seed_all(701)
        if mode=='supervised':
            with torch.no_grad():value=sum(float(adapter.nll(adapter.encode(validation[i:i+batch_size])).sum()) for i in range(0,len(validation),batch_size))/len(validation)
        else:value=-summarize(evaluate(adapter,validation,reference,reward_spec),[r['record_id'] for r in validation],20,thresholds)['u_good']
        restore_rng(random_state)
        improved=state['best'] is None or value<state['best'];state['best']=value if improved else state['best'];state['bad']=0 if improved else state['bad']+1
        state['epoch']+=1;state['order']=None;state['cursor']=0
        state['history'].append({'epoch':state['epoch'],'validation':value,'updates':state['updates'],'attempts':state['attempts']})
        artifact={'model':adapter.network.state_dict(),'vocabulary':adapter.vocabulary,'spec':spec,'state':state,'contract':contract}
        if improved:save(root/'best.pt',artifact)
        save(root/'last.pt',artifact);save_training(root/'resume.pt',active,optimizer,contract,state);atomic_json(root/'history.json',state['history'])
    return adapter,state
