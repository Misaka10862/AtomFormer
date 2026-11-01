"""Pre-registered supplement_v2 contracts. Selection is exclusively seed 42."""
import copy
import math
from pathlib import Path
from .io import digest, freeze, read_json

PROTOCOL = 'supplement_v2'
DATASETS = {'chembl37':180000, 'moses':180000, 'gdb13':181472}
GRIDS = {'lambda':[0., .02, .05, .1, .2], 'sigma':[60., 120., 240.]}
DEFAULTS = {'lambda':.2, 'sigma':120.}
REWARD = {'task':'murcko', 'protocol':'aligned', 'qed_weight':1.,
          'novelty_weight':.2, 'sas_weight':.2, 'failure':-2., 'repetition_weight':0.}
D_FIELD = 'retained_diversity_all_records'


def model_spec(dataset, method='A', seed=42, mode='supervised', value=None):
    length = 128 if dataset=='gdb13' else 256
    external = method in ('libinvent', 'promptsmiles')
    parameter = 'sigma' if external else 'lambda'
    architecture = ({'hidden_size':512, 'num_layers':3, 'dropout':0., 'max_length':length,
                     'prompt_ring_grammar':True} if external else
        {'max_z':64, 'embedding_dim':256, 'encoder_layers':4, 'encoder_heads':8,
         'encoder_ffn_dim':512, 'decoder_layers':6, 'decoder_heads':8, 'decoder_ffn_dim':1024,
         'block_size':length, 'scaffold_block_size':length, 'dropout':.1, 'attention_dropout':0.,
         'atom_attention_dropout':.1, 'encoder_norm_first':False, 'auxiliary_pooling':'masked_mean',
         'anchor_aux_weight':0. if method=='B' else .1, 'scaffold_aux_weight':0.,
         'decoder_cross_attention':method!='B'})
    settings = {'mode':mode, 'epochs':12 if mode=='supervised' else 4,
        'effective_batch':(128 if external else 256) if mode=='supervised' else 64,
        'micro_batch':(128 if external else 64) if mode=='supervised' else 16,
        'lr':(1e-3 if external else 3e-4) if mode=='supervised' else (1e-4 if external else 2e-6),
        'weight_decay':0. if external else .01, 'max_grad_norm':1., 'patience':None,
        'checkpoint_selection':'best' if mode=='supervised' else 'last',
        'max_length':length, 'temperature':1., 'amp':not external,
        'baseline':'batch_mean', 'normalization':'length_mean', 'checkpoint_every':25}
    if mode=='rl':
        settings['sigma' if external else 'supervised_weight'] = DEFAULTS[parameter] if value is None else value
    return {'protocol_id':PROTOCOL, 'dataset_id':dataset, 'id':method, 'seed':seed,
            'method':method if external else 'atomformer', 'encoder':None if external else {'A':'atom','B':'prefix_only','C':'smiles'}[method],
            'decoder':'smiles', 'architecture':architecture, 'training':settings,
            'reward':dict(REWARD, method=method if external else 'atomformer')}


def selected_parameter(spec):
    name = 'sigma' if spec['method'] in ('libinvent','promptsmiles') else 'lambda'
    return name, spec['training']['sigma' if name=='sigma' else 'supervised_weight']


def parameter_context(spec):
    # Source checkpoint / seed differ across independent confirmations; scientific settings do not.
    training = {k:v for k,v in spec['training'].items() if k not in
                ('source_checkpoint','source_checkpoint_sha256','supervised_weight','sigma','micro_batch','checkpoint_every')}
    return digest({'protocol':PROTOCOL,'dataset':spec['dataset_id'],'method':spec['method'],
                   'encoder':spec.get('encoder'),'architecture':spec['architecture'],
                   'training':training,'reward':spec['reward']})


def metric_vector(summary):
    values = {'Y':summary['retained_yield'], 'U':summary['mean_unique_retained'],
              'D':summary[D_FIELD], 'Q':summary['retained_properties']['qed'],
              'S':summary['retained_properties']['sas']}
    if any(v is None or not math.isfinite(v) for v in values.values()):
        raise ValueError('Missing/nonfinite decision metric')
    return values


def acceptable(candidate, reference, exploration=.9):
    c, r = metric_vector(candidate), metric_vector(reference)
    return (c['Y'] >= r['Y']-.002-1e-12 and c['Q'] >= r['Q']-.002-1e-12
            and c['S'] <= r['S']+.03+1e-12 and c['U'] >= exploration*r['U']-1e-12
            and c['D'] >= exploration*r['D']-1e-12)


def architecture_decision(atom, smiles):
    a,c=metric_vector(atom),metric_vector(smiles)
    gain = c['Q']>=a['Q']+.005 or c['S']<=a['S']-.05 or (a['U']>0 and c['U']>=1.1*a['U'])
    safe = acceptable(smiles,atom,.95)
    return {'decision':'expand' if safe and gain else ('stop_C' if not safe and not gain else 'paired_005_probe'),
            'safe':safe,'practical_gain':gain,'auto_promote_main_model':False}


def freeze_parameter(evidence, parameter, path):
    if parameter not in GRIDS or len(evidence)!=len(GRIDS[parameter]):
        raise ValueError('A complete preregistered grid is required')
    seen={};contexts=set();inputs=set();sources=set()
    for e in evidence:
        if (e['protocol_id']!=PROTOCOL or e['seed']!=42 or e['validation_role']!='v_select_quick'
            or e['stage']!=1 or not e['completed_nominal_budget'] or e['checkpoint_selection']!='last'):
            raise ValueError('Only completed seed42 V-select RL endpoints may select parameters')
        value=e['parameter_value']
        if e['parameter']!=parameter or value not in GRIDS[parameter] or value in seen:
            raise ValueError('Unexpected or duplicate grid value')
        if e['metrics']['requested_attempts']!=40960 or e['metrics']['records']!=2048:
            raise ValueError('Selection requires the full fixed quick-evaluation budget')
        metric_vector(e['metrics'])
        contexts.add(e['parameter_context']);inputs.add(e['input_sha256']);sources.add(e['source_checkpoint_sha256'])
        seen[value]=e
    if len(contexts)!=1 or len(inputs)!=1 or len(sources)!=1:
        raise ValueError('Grid scientific settings, validation input or supervised starting point differ')
    reference=seen[DEFAULTS[parameter]]['metrics']
    eligible=[v for v,e in seen.items() if acceptable(e['metrics'],reference)]
    def objective(v):
        m=metric_vector(seen[v]['metrics']);return m['Q']-.2*m['S']
    highest=max(objective(v) for v in eligible)
    ties=[v for v in eligible if math.isclose(objective(v),highest,rel_tol=0.,abs_tol=1e-12)]
    chosen=DEFAULTS[parameter] if DEFAULTS[parameter] in ties else (max(ties) if parameter=='lambda' else min(ties))
    payload={'protocol_id':PROTOCOL,'parameter':parameter,'value':chosen,'selection_seed':42,
        'validation_role':'v_select_quick','parameter_context':next(iter(contexts)),
        'input_sha256':next(iter(inputs)), 'source_checkpoint_sha256':next(iter(sources)),
        'evidence_sha256':digest(evidence),'eligible_values':sorted(eligible),
        'confirmation_seeds':[43,44],'may_reselect_on_confirmation':False}
    return freeze(path,payload)


def validate_confirmation(spec, selection, role='selected'):
    name,value=selected_parameter(spec)
    if (selection.get('protocol_id')!=PROTOCOL or selection.get('selection_seed')!=42
        or selection.get('parameter')!=name or selection.get('parameter_context')!=parameter_context(spec)
        or selection.get('may_reselect_on_confirmation') is not False):
        raise ValueError('Parameter freeze does not match this confirmation')
    expected=selection['value'] if role=='selected' else DEFAULTS[name] if role=='default_reference' else None
    if value!=expected:
        raise ValueError('Confirmation cannot change the frozen parameter')


def validate_config(cfg):
    if cfg.get('protocol_id')!=PROTOCOL or cfg.get('track_id')!=PROTOCOL:
        raise ValueError('Explicit supplement track is required')
    spec=cfg['model'];ds=spec['dataset_id'];method=spec['id'];mode=spec['training']['mode']
    if ds not in DATASETS or method not in ('A','B','C','libinvent','promptsmiles') or spec['seed'] not in (42,43,44):
        raise ValueError('Unregistered supplement experiment')
    if mode not in ('supervised','rl'):
        raise ValueError('Supplement supports supervised and RL only')
    parameter,value=selected_parameter(spec) if mode=='rl' else ('lambda',None)
    expected=model_spec(ds,method,spec['seed'],mode,value)
    for name in ('protocol_id','method','encoder','decoder','architecture','reward'):
        if spec.get(name)!=expected[name]:raise ValueError('Scientific configuration differs: '+name)
    for name,value0 in expected['training'].items():
        if name not in ('micro_batch','checkpoint_every','amp') and spec['training'].get(name)!=value0:
            raise ValueError('Training contract differs: '+name)
    if cfg.get('reward')!=spec['reward']:raise ValueError('Training/evaluation reward mismatch')
    if mode=='supervised' and spec['training'].get('source_checkpoint'):
        raise ValueError('Supplement supervision starts from scratch')
    if mode=='rl':
        if value not in GRIDS[parameter]:raise ValueError('Unregistered grid')
        if not spec['training'].get('source_checkpoint') or not spec['training'].get('source_checkpoint_sha256'):
            raise ValueError('RL requires an explicitly hash-bound supervised best checkpoint')
        role=cfg.get('run_role')
        if role=='search':
            if spec['seed']!=42 or cfg['stage']!=1:raise ValueError('Only seed42 selects')
        elif role in ('selected','default_reference'):
            validate_confirmation(spec,read_json(cfg['selection_file']),role)
        elif role=='architecture':
            if method not in ('A','B','C') or value!=.2:raise ValueError('Architecture comparison uses shared lambda .2')
        elif role=='probe':
            if method not in ('A','C') or spec['seed']!=42 or value!=.05 or not cfg.get('paired_probe_receipt'):
                raise ValueError('Paired probe requires its recorded trigger')
        elif role=='zero_anchor':
            if cfg['stage']!=2 or parameter!='lambda' or value!=0.:raise ValueError('Zero-anchor ablation is stage2')
        else:raise ValueError('Explicit RL run role required')
    evaluation=cfg.get('evaluation',{})
    if evaluation.get('split') not in ('v_select','v_select_quick','v_check','test'):
        raise ValueError('Explicit validation role required')
    if cfg['stage']==1 and evaluation['split'] in ('v_check','test'):
        raise ValueError('Stage1 cannot read V-check/test outputs')
    if evaluation.get('attempts')!=20 or evaluation.get('temperature')!=1.:
        raise ValueError('Evaluation budget/temperature differs')
    if cfg.get('resource_margin',.1)<.1:raise ValueError('At least 10% margin required')
    if cfg['data'].get('mode')!='historical_import':raise ValueError('No new corpus conversion or split allowed')
    return True
