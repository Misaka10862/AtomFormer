"""Fixed-budget supplement training; effective-batch RL baselines survive microbatching."""
import copy
import time
from pathlib import Path
import torch
from .checkpoints import seed_all, save, load, save_training, restore_training, rng_state, restore_rng
from .io import digest, freeze, atomic_json, file_sha
from .training import build_model, supervised_loss, validate_supervised
from .data import Condition, collate_conditions
from .protocol import RewardSpec, score, training_reference
from .optimization import augmented_likelihood_loss
from .supplement import PROTOCOL


def assembled_score(row, decoration, reference, reward, evaluator=None):
    from condmolgen.v2_assembly import assemble_single_attachment_rooted
    try:
        product=assemble_single_attachment_rooted(row['scaffold_smiles'],decoration) if decoration else None
    except (ValueError,RuntimeError):
        product=None
    return dict(evaluator.reward(product,row) if evaluator is not None else score(product,row,reference,reward), scaffold_smiles=row['scaffold_smiles'],decoration=decoration,assembled_smiles=product)


def policy_backward(model, rows, tokenizer, settings, reference, scaler, evaluator=None):
    """Sample once per input; replay identical RNG/actions for one global advantage baseline.

    Replays evaluate log probabilities, not new reward queries. Supervised LM and
    attachment terms use separate token/record denominators over the logical batch.
    """
    device=next(model.parameters()).device;micro=settings['micro_batch'];amp=settings.get('amp',False)
    reward=RewardSpec(**settings['reward']);parts=[];trace=[]
    for start in range(0,len(rows),micro):
        chunk=rows[start:start+micro]
        batch=collate_conditions([Condition.from_record(r) for r in chunk],device)
        state=rng_state()
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
            sampled=model.sample(batch,[r['scaffold_smiles'] for r in chunk],tokenizer,settings['max_length'])
        scored=[assembled_score(r,d,reference,reward,evaluator) for r,d in zip(chunk,sampled['decorations'])]
        trace.extend(dict(s,record_id=r['record_id']) for r,s in zip(chunk,scored))
        parts.append((chunk,batch,state,sampled['generated'].detach().clone(),sampled['token_logprobs'].detach().clone()))
    rewards=torch.tensor([r['reward'] for r in trace],device=device)
    advantage=(rewards-rewards.mean()).detach()
    total_tokens=sum(len(r['rooted_decoration_smiles'])+1 for r in rows)
    loss_value=0.;offset=0
    for chunk,batch,state,expected,expected_lp in parts:
        future=rng_state();restore_rng(state)
        with torch.cuda.amp.autocast(enabled=amp):
            sampled=model.sample(batch,[r['scaffold_smiles'] for r in chunk],tokenizer,settings['max_length'],with_grad=True)
            if not torch.equal(sampled['generated'],expected) or not torch.allclose(sampled['token_logprobs'].detach(),expected_lp,atol=1e-6,rtol=1e-5):
                raise RuntimeError('Policy replay changed actions/log probabilities')
            restore_rng(future)
            logmean=(sampled['token_logprobs']*sampled['action_mask']).sum(1)/sampled['action_mask'].sum(1).clamp(min=1)
            rl=-(advantage[offset:offset+len(chunk)]*logmean).sum()/len(rows)
            sup,_=supervised_loss(model,chunk,tokenizer,device,settings)
            anchor=sup['lm_loss']*sup['token_count']/total_tokens
            if 'anchor_aux_loss' in sup:
                anchor=anchor+model.config.anchor_aux_weight*sup['anchor_aux_loss']*len(chunk)/len(rows)
            loss=rl+settings['supervised_weight']*anchor
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite policy objective')
        scaler.scale(loss).backward();loss_value+=float(loss.detach());offset+=len(chunk)
    return loss_value,trace


def train(spec, train_rows, validation_rows, root, tokenizer, policy_ids=None, support_ids=None,
          device='cpu', resume=False, stop_after_updates=None):
    if any(r['split']!='train' for r in train_rows) or any(r.get('validation_role')!='v_select' for r in validation_rows):
        raise ValueError('Supervision/checkpoint selection accepts training + V-select only')
    if set(r['score_core'] for r in train_rows)&set(r['score_core'] for r in validation_rows):
        raise ValueError('Training and selection cores overlap')
    if not train_rows or not validation_rows:raise ValueError('Empty training or selection set')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    settings=dict(spec['training'],reward=spec['reward']);mode=settings['mode'];seed_all(spec['seed'])
    external=spec['method'] in ('libinvent','promptsmiles')
    protocol_id=spec.get('protocol_id',PROTOCOL)
    if protocol_id=='supplement_A_v1' and external:raise ValueError('Scheme A never trains external baselines')
    reference,reference_hash=training_reference(train_rows)
    evaluator=None
    if protocol_id=='supplement_A_v1':
        from .scheme_a.evaluation import Evaluator
        evaluator=Evaluator(reference)
    index={r['record_id']:r for r in train_rows}
    if len(index)!=len(train_rows):raise ValueError('Duplicate training identity')
    if mode=='rl':
        if not policy_ids or len(policy_ids)!=len(set(policy_ids)) or not set(policy_ids)<=set(index):
            raise ValueError('Invalid frozen policy list')
        rows=[index[i] for i in policy_ids]
    else:rows=list(train_rows)
    native=None;prior=None
    if external:
        from .external import SequenceBaseline,adapt_training_rows
        from baselines.scaffold_bench.common import convert
        native=SequenceBaseline(spec['method'],train_rows,spec['architecture'],spec['upstreams'],spec['upstream_lock'],device)
        model=native.network
        if support_ids is None:raise ValueError('Native supervision requires an explicit support ledger')
        support=set(support_ids)
        if mode=='rl' and not set(policy_ids)<=support:raise ValueError('Policy input unsupported by baseline')
        rows=[r for r in rows if r['record_id'] in support]
        native_rows={r['record_id']:n for r,n in zip(rows,adapt_training_rows(rows))}
        # Validation eligibility is recorded; generation always retains every requested input.
        native_val=[convert((i,r)) for i,r in enumerate(validation_rows)]
        native_val=[r for r in native_val if native.backend.encodable(r)]
        if not native_val:raise ValueError('No native validation support')
    else:model=build_model(spec,tokenizer).to(device)
    if not rows:raise ValueError('No supported training records')
    source_sha=None
    if mode=='rl':
        source_path=settings['source_checkpoint'];source_sha=file_sha(source_path)
        if settings.get('source_checkpoint_sha256',source_sha)!=source_sha:raise ValueError('Source checkpoint changed')
        source=load(source_path,device)
        if source['progress']['epoch']!=source['training_epochs'] or source['mode']!='supervised' or source['checkpoint_kind']!='best':
            raise ValueError('RL requires completed supplement supervision and its selected best')
        if source['tokenizer']!=tokenizer.to_state() or source['training_sha256']!=digest(train_rows):
            raise ValueError('Source supervision data/tokenizer differs')
        if source['seed']!=spec['seed'] or source['architecture']!=spec['architecture'] or source['method']!=spec['method'] or source['encoder']!=spec.get('encoder'):
            raise ValueError('Source supervised model identity differs')
        model.load_state_dict(source['model'],strict=True)
        if external:
            prior=copy.deepcopy(native.backend);prior.network.eval()
            for p in prior.network.parameters():p.requires_grad_(False)
    elif settings.get('source_checkpoint'):
        raise ValueError('Supplement supervised runs start from scratch')
    optimizer=(torch.optim.Adam(model.parameters(),lr=settings['lr']) if external else
               torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay']))
    amp=settings.get('amp',False)
    if amp and not str(device).startswith('cuda'):raise ValueError('CPU fixtures require amp=false')
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    contract={'protocol':protocol_id,'spec':spec,'training_sha256':digest(train_rows),'validation_sha256':digest(validation_rows),
        'policy_ids_sha256':digest(policy_ids),'support_ids_sha256':digest(support_ids),
        'source_checkpoint_sha256':source_sha,'reference_sha256':reference_hash,'tokenizer':tokenizer.to_state()}
    freeze(root/'contract.json',contract)
    progress={'epoch':0,'cursor':0,'order':None,'updates':0,'attempts':0,'scoring_calls':0,'processed_records':0,
        'best':None,'best_epoch':None,'history':[],'epoch_loss':0.,'epoch_records':0}
    if (root/'resume.pt').exists():
        if not resume:raise ValueError('Explicit resume required')
        progress=restore_training(root/'resume.pt',model,optimizer,contract,device,scaler)
    elif resume:raise ValueError('No resume checkpoint')
    def payload(kind):
        return {'model':model.state_dict(),'model_spec':spec,'tokenizer':tokenizer.to_state(),
            'progress':copy.deepcopy(progress),'training_epochs':settings['epochs'],'training_sha256':digest(train_rows),
            'mode':mode,'checkpoint_kind':kind,'seed':spec['seed'],'method':spec['method'],
            'encoder':spec.get('encoder'),'architecture':spec['architecture'],'contract':contract}
    best_weights=load(root/'best.pt','cpu')['model'] if (root/'best.pt').exists() else None
    batchsize=settings['effective_batch'];micro=settings['micro_batch'];started=time.monotonic();initial_updates=progress['updates']
    while progress['epoch']<settings['epochs']:
        if progress['order'] is None:
            progress['order']=torch.randperm(len(rows)).tolist();progress['cursor']=0
            progress['epoch_loss']=0.;progress['epoch_records']=0
        while progress['cursor']<len(rows):
            chosen=[rows[i] for i in progress['order'][progress['cursor']:progress['cursor']+batchsize]]
            model.train();optimizer.zero_grad(set_to_none=True);trace=[];loss_value=0.
            if mode=='rl' and not external:
                loss_value,trace=policy_backward(model,chosen,tokenizer,settings,reference,scaler,evaluator)
            else:
                tokens=sum(len(r['rooted_decoration_smiles'])+1 for r in chosen)
                for start in range(0,len(chosen),micro):
                    chunk=chosen[start:start+micro]
                    with torch.cuda.amp.autocast(enabled=amp):
                        if external:
                            nrows=[native_rows[r['record_id']] for r in chunk]
                            if mode=='supervised':loss=native.backend.nll(nrows).sum()/len(chosen)
                            else:
                                products,decorations=native.sample([Condition.from_record(r) for r in chunk])
                                scored=[score(s,r,reference,RewardSpec(**spec['reward'])) for s,r in zip(products,chunk)]
                                strings=decorations if spec['method']=='libinvent' else products
                                strings=[s if s is not None else '' for s in strings]
                                model.train();agent_lp=-native.backend.nll(nrows,strings)
                                with torch.no_grad():prior_lp=-prior.nll(nrows,strings)
                                rewards=torch.tensor([s['reward'] for s in scored],device=device)
                                loss=augmented_likelihood_loss(agent_lp,prior_lp,rewards,settings['sigma'])*len(chunk)/len(chosen)
                                trace.extend(dict(s,record_id=r['record_id'],scaffold_smiles=r['scaffold_smiles'],decoration=d,assembled_smiles=product)
                                             for s,r,d,product in zip(scored,chunk,decorations,products))
                        else:
                            out,_=supervised_loss(model,chunk,tokenizer,device,settings)
                            loss=out['lm_loss']*out['token_count']/tokens
                            if 'anchor_aux_loss' in out:loss=loss+model.config.anchor_aux_weight*out['anchor_aux_loss']*len(chunk)/len(chosen)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                    scaler.scale(loss).backward();loss_value+=float(loss.detach())
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['max_grad_norm'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update()
            if trace:
                freeze(root/'rl_trace'/('%08d.json'%(progress['updates']+1)),{'rows':trace,'reward_queries':len(trace)})
            progress['cursor']+=len(chosen);progress['updates']+=1;progress['processed_records']+=len(chosen)
            progress['attempts']+=len(trace);progress['scoring_calls']+=len(trace)
            progress['epoch_loss']+=loss_value*len(chosen);progress['epoch_records']+=len(chosen)
            if progress['updates']%settings['checkpoint_every']==0 or progress['cursor']==len(rows) or progress['updates']==stop_after_updates:
                save_training(root/'resume.pt',model,optimizer,contract,progress,scaler)
                atomic_json(root/'progress.json',{'epoch':progress['epoch'],'cursor':progress['cursor'],'updates':progress['updates'],
                    'total_updates':settings['epochs']*((len(rows)+batchsize-1)//batchsize),'initial_updates':initial_updates,
                    'seconds_this_invocation':time.monotonic()-started,'attempts':progress['attempts'],'pid':__import__('os').getpid(),
                    'updated_at_unix':time.time(),'mode':mode})
            if progress['updates']==stop_after_updates:raise InterruptedError('Synthetic recovery checkpoint')
        value=None
        if mode=='supervised':
            state=rng_state();model.eval()
            if external:
                with torch.no_grad():value=sum(float(native.backend.nll(native_val[i:i+micro]).sum()) for i in range(0,len(native_val),micro))/len(native_val)
            else:value=validate_supervised(model,validation_rows,tokenizer,device,settings)
            restore_rng(state)
            if not math_isfinite(value):raise FloatingPointError('Nonfinite validation loss')
            if progress['best'] is None or value<progress['best']:
                progress['best']=value;progress['best_epoch']=progress['epoch']+1
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        progress['epoch']+=1;progress['order']=None;progress['cursor']=0
        progress['history'].append({'epoch':progress['epoch'],'v_select_loss':value,
            'training_loss':progress['epoch_loss']/progress['epoch_records'],'updates':progress['updates'],
            'attempts':progress['attempts'],'scoring_calls':progress['scoring_calls']})
        save(root/'last.pt',payload('last'))
        if best_weights is not None:
            best=payload('best');best['model']=best_weights;save(root/'best.pt',best)
        save_training(root/'resume.pt',model,optimizer,contract,progress,scaler)
        atomic_json(root/'history.json',progress['history'])
    completed={'protocol':protocol_id,'epochs':progress['epoch'],'selected_checkpoint':'best.pt' if mode=='supervised' else 'last.pt',
        'selected_epoch':progress['best_epoch'] if mode=='supervised' else progress['epoch'],
        'completed_nominal_budget':True,'early_stopped':False,'updates':progress['updates'],
        'attempts':progress['attempts'],'scoring_calls':progress['scoring_calls'],'processed_records':progress['processed_records'],
        'training_supported':len(rows),'training_requested':len(train_rows),
        'validation_supported':len(native_val) if external else len(validation_rows),'validation_requested':len(validation_rows),
        'parameters':sum(p.numel() for p in model.parameters()),'seconds_this_invocation':time.monotonic()-started,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if amp or str(device).startswith('cuda') else 0,
        'contract_sha256':digest(contract)}
    atomic_json(root/'completed.json',completed)
    return model,progress


def math_isfinite(value):
    import math
    return value is not None and math.isfinite(value)
