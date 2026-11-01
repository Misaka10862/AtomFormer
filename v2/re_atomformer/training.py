"""Supervised and RL training with validation selection and exact-boundary resume."""
import copy
import math
import time
from pathlib import Path

import torch
from condmolgen.model import ConditionalGeneratorConfig
from condmolgen.tokenizer import CharTokenizer
from .checkpoints import seed_all,save_training,restore_training,save,load,rng_state,restore_rng
from .data import Condition,collate_conditions,collate_supervised
from .diffusion import DiffusionGenerator,collate_targets
from .smiles import ResearchSmiles
from .optimization import policy_loss,augmented_likelihood_loss,gradient_diagnostics,contrastive_geometry_loss
from .encoders import masked_mean
from .io import atomic_json,digest,freeze
from .protocol import RewardSpec,score,training_reference,training_thresholds
from .splits import stratified_order,audit_splits


def tokenizer_from_train(rows):
    if any(r.get('split')!='train' for r in rows):
        raise ValueError('Tokenizer training split violation')
    # Fixed SMILES grammar symbols are independent of validation/test labels.
    grammar='0123456789%()[]=#-+\\/@.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ*:'
    return CharTokenizer(texts=[grammar]+[r['scaffold_smiles']+r['rooted_decoration_smiles'] for r in rows])


def build_model(spec,tokenizer):
    cfg=ConditionalGeneratorConfig(vocab_size=tokenizer.vocab_size,**spec.get('architecture',{}))
    if spec.get('decoder','smiles')=='smiles':
        model=ResearchSmiles(cfg,spec.get('encoder','graph'),spec.get('gate_initial',.01))
        model.encoder.geometry_enabled=spec.get('geometry_enabled',True)
        return model
    if spec['decoder']=='diffusion':
        return DiffusionGenerator(cfg,spec.get('encoder','gated_geo'),**spec.get('diffusion',{}))
    raise ValueError('Unknown decoder')


def load_model(path,device='cpu'):
    checkpoint=load(path,device)
    tok=CharTokenizer.from_state(checkpoint['tokenizer'])
    model=build_model(checkpoint['model_spec'],tok).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model,tok,checkpoint


def supervised_loss(model,rows,tokenizer,device,settings):
    if isinstance(model,ResearchSmiles):
        b=collate_supervised(rows,tokenizer,device,pocket_mode=settings.get('pocket_mode','correct'))
        return model(b),b
    b=collate_conditions([Condition.from_record(r) for r in rows],device,pocket_mode=settings.get('pocket_mode','correct'))
    return model.loss(b,*collate_targets(rows,b['z'].shape[1],device)),b


def validate_supervised(model,rows,tokenizer,device,settings):
    lm_total=aux_total=other_total=0.;tokens=count=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),settings['micro_batch']):
            chunk=rows[start:start+settings['micro_batch']]
            result,_=supervised_loss(model,chunk,tokenizer,device,settings)
            count+=len(chunk)
            if 'lm_loss' in result:
                lm_total+=float(result['lm_loss'])*result['token_count'];tokens+=result['token_count']
                aux_total+=float(result.get('anchor_aux_loss',0))*len(chunk)
            else:
                other_total+=float(result['loss'])*len(chunk)
    if not count:
        raise ValueError('Empty validation split')
    if tokens:
        return lm_total/tokens+model.config.anchor_aux_weight*aux_total/count
    return other_total/count


def trajectory_logprob(model,batch,generated,prefix_lengths):
    memory,padding=model.encoder(batch)
    hidden=model.decoder(generated,memory,generated.eq(model.config.decoder_pad_token_id),padding)
    lp=model.lm_head(hidden[:,:-1]).log_softmax(-1).gather(-1,generated[:,1:,None]).squeeze(-1)
    positions=torch.arange(1,generated.shape[1],device=generated.device)[None,:]
    mask=(positions>=prefix_lengths[:,None])&generated[:,1:].ne(model.config.decoder_pad_token_id)
    return (lp*mask).sum(1)


def reinforcement_loss(model,prior,rows,tokenizer,device,settings,reference,repeat_counts):
    from condmolgen.v2_assembly import assemble_single_attachment_rooted
    if not isinstance(model,ResearchSmiles):
        raise ValueError('REINFORCE acts on the SMILES branch; diffusion uses supervised geometric objectives')
    conditions=[Condition.from_record(r) for r in rows]
    b=collate_conditions(conditions,device,pocket_mode=settings.get('pocket_mode','correct'))
    sampled=model.sample(b,[r['scaffold_smiles'] for r in rows],tokenizer,settings.get('max_length',256),with_grad=True)
    spec=RewardSpec(**settings.get('reward',{}))
    scored=[]
    for r,d in zip(rows,sampled['decorations']):
        smi=assemble_single_attachment_rooted(r['scaffold_smiles'],d) if d else None
        item=score(smi,r,reference,spec)
        identity=(r['record_id'],item['canonical_smiles'])
        count=repeat_counts.get(identity,0)
        if item['gate_passed'] and item['reward'] is not None:
            item['reward']-=spec.repetition_weight*count
        repeat_counts[identity]=count+1
        scored.append(item)
    if any(r['reward'] is None for r in scored):
        raise ValueError('Native DrugEx Pareto objective belongs to its external optimizer')
    rewards=torch.tensor([r['reward'] for r in scored],device=device)
    trajectory=(sampled['token_logprobs']*sampled['action_mask']).sum(1)
    if settings.get('objective','reinforce')=='augmented_likelihood':
        if prior is None:
            raise ValueError('Augmented likelihood requires a frozen source prior')
        with torch.no_grad():
            prefix=torch.tensor([len(r['scaffold_smiles'])+2 for r in rows],device=device)
            plp=trajectory_logprob(prior,b,sampled['generated'],prefix)
        rl=augmented_likelihood_loss(trajectory,plp,rewards,settings.get('sigma',120.))
    else:
        rl=policy_loss(sampled['token_logprobs'],sampled['action_mask'],rewards,
             [r['record_id'] for r in rows],settings.get('baseline','batch_mean'),
             settings.get('normalization','length_mean'),settings.get('fixed_scale',32.))
    sup,_=supervised_loss(model,rows,tokenizer,device,settings)
    weight=settings.get('supervised_weight',.2)
    return {'loss':rl+weight*sup['loss'],'rl_loss':rl,'supervised_loss':sup['loss'],
            'reward_mean':float(rewards.mean()),'novelty_mean':sum(r['novelty'] for r in scored)/len(scored),
            'token_count':len(rows),'trace':scored},b


def train(spec,train_rows,validation_rows,root,device='cpu',resume=False,stop_after_updates=None):
    """Internal API; CLI enforces remote authorization before calling it."""
    audit_splits(list(train_rows)+list(validation_rows))
    if any(r['split']!='train' for r in train_rows) or any(r['split']!='validation' for r in validation_rows):
        raise ValueError('Explicit independent training and validation splits are required')
    if not train_rows or not validation_rows:
        raise ValueError('Empty data split')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    settings={'epochs':12,'micro_batch':4,'effective_batch':64,'lr':3e-4,'weight_decay':.01,
              'patience':5,'max_grad_norm':1.,'checkpoint_every':25,**spec.get('training',{})}
    stage=settings.get('mode','supervised')
    if stage not in ('supervised','rl','contrastive'):
        raise ValueError('Unknown training mode')
    if stage=='rl' and settings.get('baseline')=='batch_mean' and settings['micro_batch']!=settings['effective_batch']:
        raise ValueError('Batch-mean RL needs one reward batch per update; use equal micro/effective batch or grouped LOO')
    original_rows=list(train_rows)
    from .population import training_population
    train_rows,validation_rows,population_report=training_population(train_rows,validation_rows,spec)
    atomic_json(root/'population.json',population_report)
    reference,reference_hash=training_reference(original_rows)
    thresholds=training_thresholds(original_rows)
    train_rows=stratified_order(train_rows,spec.get('seed',11))
    cap=settings.get('max_train_records')
    if cap:
        train_rows=train_rows[:cap]
    from .population import pocket_intervention
    intervention=settings.get('pocket_mode','correct')
    train_rows=pocket_intervention(train_rows,intervention)
    validation_rows=pocket_intervention(validation_rows,intervention)
    settings['pocket_intervention']=intervention
    settings['pocket_mode']='correct'
    tokenizer=tokenizer_from_train(original_rows)
    seed_all(spec.get('seed',11))
    model=build_model(spec,tokenizer).to(device)
    prior=None
    if settings.get('source_checkpoint'):
        source,source_tok,_=load_model(settings['source_checkpoint'],device)
        if source_tok.to_state()!=tokenizer.to_state():
            raise ValueError('Source tokenizer differs from complete training vocabulary')
        model.load_state_dict(source.state_dict(),strict=True)
    elif stage=='rl':
        raise ValueError('RL requires an explicitly selected supervised checkpoint')
    if stage=='rl' and settings.get('objective')=='augmented_likelihood':
        prior=copy.deepcopy(model).eval()
        for parameter in prior.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    amp=bool(settings.get('amp',False))
    if amp and not str(device).startswith('cuda'):
        raise ValueError('AMP profile requires CUDA; stage0 tests stay FP32')
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    contract={'model_spec':spec,'settings':settings,'train_ids':digest([r['record_id'] for r in train_rows]),
              'validation_ids':digest([r['record_id'] for r in validation_rows]),'training_content_sha256':digest(original_rows),'validation_content_sha256':digest(validation_rows),'reference_sha256':reference_hash,
              'thresholds':thresholds,'tokenizer':tokenizer.to_state()}
    freeze(root/'contract.json',contract)
    progress={'epoch':0,'cursor':0,'order':None,'updates':0,'best':None,'bad_epochs':0,'history':[],
              'total_loss':0.,'total_weight':0,'repeat_counts':{}}
    if (root/'resume.pt').exists():
        if not resume:raise ValueError('Existing checkpoint requires explicit resume')
        progress=restore_training(root/'resume.pt',model,optimizer,contract,device,scaler)
    elif resume:
        raise ValueError('No checkpoint to resume')
    def artifact(name):
        save(root/name,{'model':model.state_dict(),'model_spec':spec,'tokenizer':tokenizer.to_state(),
                        'contract':contract,'progress':progress})
    repeat_counts=progress['repeat_counts']
    repeats=settings.get('samples_per_condition',1) if stage=='rl' else 1
    if settings['effective_batch']%repeats or settings['micro_batch']%repeats:
        raise ValueError('Batch sizes must preserve whole condition sampling groups')
    started=time.monotonic()
    while progress['epoch']<settings['epochs'] and (settings['patience'] is None or progress['bad_epochs']<settings['patience']):
        if progress['order'] is None:
            progress['order']=torch.randperm(len(train_rows)).tolist();progress['cursor']=0
            progress['total_loss']=0.;progress['total_weight']=0
        order=progress['order'];unique_batch=settings['effective_batch']//repeats
        model.train()
        while progress['cursor']<len(order):
            indices=order[progress['cursor']:progress['cursor']+unique_batch]
            selected=[train_rows[i] for i in indices]
            selected=[dict(r,pos=r['conformers'][int(torch.randint(len(r['conformers']),(1,)))]) if r.get('geometry_source')=='multi' and r.get('conformers') else r for r in selected]
            effective=[r for r in selected for _ in range(repeats)]
            optimizer.zero_grad(set_to_none=True)
            # LM is token-weighted, auxiliary classification is record-weighted.
            target_tokens=sum(len(r['rooted_decoration_smiles'])+1 for r in effective)
            update_loss=0.;diagnostics={};update_trace=[]
            for start in range(0,len(effective),settings['micro_batch']):
                chunk=effective[start:start+settings['micro_batch']]
                if stage=='rl':
                    with torch.cuda.amp.autocast(enabled=amp):
                        result,batch=reinforcement_loss(model,prior,chunk,tokenizer,device,settings,reference,repeat_counts)
                    loss=result['loss']*len(chunk)/len(effective)
                    update_trace.extend(dict(v,record_id=r['record_id'],sample_in_update=start+j) for j,(r,v) in enumerate(zip(chunk,result['trace'])))
                    if progress['updates']%settings.get('diagnostics_every',50)==0:
                        diagnostics=gradient_diagnostics(result['rl_loss'],result['supervised_loss'],model.parameters(),settings.get('supervised_weight',.2))
                else:
                    with torch.cuda.amp.autocast(enabled=amp):
                        result,batch=supervised_loss(model,chunk,tokenizer,device,settings)
                    if 'lm_loss' in result:
                        loss=result['lm_loss']*result['token_count']/target_tokens
                        if 'anchor_aux_loss' in result:
                            loss=loss+model.config.anchor_aux_weight*result['anchor_aux_loss']*len(chunk)/len(effective)
                    else:loss=result['loss']*len(chunk)/len(effective)
                    if stage=='contrastive':
                        if not isinstance(model,ResearchSmiles) or model.encoder.graph is None or model.encoder.geo is None:
                            raise ValueError('Contrastive mode needs graph and geometric views')
                        graph=model.encoder.graph(batch)
                        geo=model.encoder.geo(batch['z'],batch['pos'])[0]
                        if len(chunk)<2:
                            contrast=loss.new_zeros(())
                            progress['contrastive_singletons']=progress.get('contrastive_singletons',0)+1
                        else:
                            contrast=contrastive_geometry_loss(masked_mean(graph,batch['mask']),masked_mean(geo,batch['mask']),group_ids=[r['score_core'] for r in chunk])
                        loss=loss+settings.get('contrastive_weight',.1)*contrast*len(chunk)/len(effective)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward();update_loss+=float(loss.detach())
            if update_trace:
                freeze(root/'rl_trace'/('%08d.json'%(progress['updates']+1)),{'update':progress['updates']+1,'rows':update_trace})
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['max_grad_norm'])
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
            scaler.step(optimizer);scaler.update()
            progress['cursor']+=len(indices);progress['updates']+=1
            progress['total_loss']+=update_loss*len(effective);progress['total_weight']+=len(effective)
            progress['repeat_counts']=repeat_counts
            if progress['updates']%settings['checkpoint_every']==0 or progress['cursor']==len(order) or progress['updates']==stop_after_updates:
                save_training(root/'resume.pt',model,optimizer,contract,progress,scaler)
            atomic_json(root/'progress.json',{'epoch':progress['epoch'],'cursor':progress['cursor'],'updates':progress['updates'],
                       'loss':update_loss,'gradient_norm':float(norm),'diagnostics':diagnostics,'elapsed_seconds':time.monotonic()-started})
            if progress['updates']==stop_after_updates:raise InterruptedError('Controlled interruption at optimizer boundary')
        state=rng_state();seed_all(settings.get('validation_seed',701))
        if stage=='rl':
            from .generation import generate_scored
            from .metrics import summarize
            evaluated=generate_scored(model,tokenizer,validation_rows,device,reference,RewardSpec(**settings.get('reward',{})),settings.get('validation_attempts',20),settings.get('max_length',256))
            val=summarize(evaluated,[r['record_id'] for r in validation_rows],settings.get('validation_attempts',20),thresholds)
            validation=-val['u_good']
        else:
            validation=validate_supervised(model,validation_rows,tokenizer,device,settings)
        restore_rng(state)
        improved=progress['best'] is None or validation<progress['best']-settings.get('min_delta',0.)
        progress['best']=validation if improved else progress['best']
        progress['bad_epochs']=0 if improved else progress['bad_epochs']+1
        progress['history'].append({'epoch':progress['epoch']+1,'updates':progress['updates'],
           'training_loss':progress['total_loss']/progress['total_weight'],'validation_objective':validation,
           'validation_metric':'negative_u_good' if stage=='rl' else 'loss'})
        progress['epoch']+=1;progress['order']=None;progress['cursor']=0
        if improved:artifact('best.pt')
        artifact('last.pt');save_training(root/'resume.pt',model,optimizer,contract,progress,scaler)
        atomic_json(root/'history.json',progress['history'])
    atomic_json(root/'completed.json',{'epochs':progress['epoch'],'updates':progress['updates'],'best':progress['best'],
          'selected_checkpoint':'best.pt','completed_nominal_budget':progress['epoch']==settings['epochs'],
          'early_stopped':progress['epoch']<settings['epochs'],'contract_sha256':digest(contract)})
    return model,tokenizer,progress
