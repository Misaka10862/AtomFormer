"""Phase gates and immutable settings for the approved frozen-baseline scheme."""
from . import PROTOCOL
from ..supplement import model_spec as legacy_spec, DATASETS, REWARD, metric_vector, acceptable, architecture_decision
from ..io import digest,read_json

LAMBDAS=(0.,.02,.05,.1,.2)
BASELINES=('libinvent','promptsmiles','drugex')
PHASES=('1A','1B','1C','2')


def atom_spec(dataset, variant='A', seed=42, mode='supervised', value=.2):
    if variant not in ('A','B','C'):raise ValueError('Only AtomFormer trains in scheme A')
    spec=legacy_spec(dataset,variant,seed,mode,value)
    spec['protocol_id']=PROTOCOL
    return spec


def selection_context(spec):
    training={k:v for k,v in spec['training'].items() if k not in
        ('source_checkpoint','source_checkpoint_sha256','supervised_weight','micro_batch','checkpoint_every')}
    return digest({'protocol':PROTOCOL,'dataset':spec['dataset_id'],'variant':spec['id'],
                   'architecture':spec['architecture'],'training':training,'reward':spec['reward']})


def validate(cfg, action=None):
    if cfg.get('protocol_id')!=PROTOCOL or cfg.get('track_id')!=PROTOCOL:raise ValueError('Scheme A identity required')
    phase=cfg.get('phase')
    if phase not in PHASES or cfg['stage']!=(2 if phase=='2' else 1):raise ValueError('Explicit stage/subphase required')
    spec=cfg['model'];method=spec['method'];ds=spec['dataset_id'];evaluation=cfg['evaluation']
    if ds not in DATASETS:raise ValueError('Unregistered dataset')
    if method in BASELINES:
        if action in ('train','rl','fine_tune','sigma_search','select','shortlist') or 'training' in spec:
            raise ValueError('Official baselines are frozen inference only')
        if spec.get('train_seed') is not None:raise ValueError('Official models have no benchmark training seed')
        if not spec.get('registry_id'):raise ValueError('Registered official checkpoint required')
        if phase=='1B' or phase=='1C':raise ValueError('No new external matrix in architecture/lambda confirmation')
    elif method=='atomformer':
        if spec['seed'] not in (42,43,44):raise ValueError('Unregistered independent training seed')
        mode=spec['training']['mode'];value=spec['training'].get('supervised_weight',.2)
        if mode not in ('supervised','rl') or value not in LAMBDAS:raise ValueError('Unknown training mode/grid')
        expected=atom_spec(ds,spec['id'],spec['seed'],mode,value)
        for field in ('architecture','encoder','decoder','reward'):
            if spec.get(field)!=expected[field]:raise ValueError('Frozen model settings changed: '+field)
        for key,v in expected['training'].items():
            if key not in ('micro_batch','checkpoint_every','amp') and spec['training'].get(key)!=v:
                raise ValueError('Frozen training setting changed: '+key)
        if mode=='supervised' and spec['training'].get('source_checkpoint'):
            raise ValueError('New supervision starts from scratch')
        if mode=='rl':
            if not spec['training'].get('source_checkpoint') or not spec['training'].get('source_checkpoint_sha256'):
                raise ValueError('RL requires hash-bound supervised best')
            role=cfg.get('run_role')
            if role=='architecture':
                if value!=.2:raise ValueError('Architecture uses shared lambda .2')
            elif role=='probe':
                if phase!='1B' or ds!='chembl37' or spec['seed']!=42 or spec['id'] not in ('A','C') or value!=.05 or not cfg.get('paired_probe_receipt'):
                    raise ValueError('Only the reviewed paired probe is allowed')
            elif role=='lambda_search':
                if phase!='1C' or spec['seed']!=42:raise ValueError('Initial lambda search uses seed42 in 1C')
            elif role=='lambda_confirm':
                if phase!='1C':raise ValueError('Lambda confirmation belongs to 1C')
                shortlist=read_json(cfg['shortlist_file'])
                if shortlist['context']!=selection_context(spec) or value not in shortlist['values']:
                    raise ValueError('Run is outside the frozen shortlist')
            elif role=='selected':
                selected=read_json(cfg['selection_file'])
                if selected['context']!=selection_context(spec) or value!=selected['value'] or selected['selection_seeds']!=[42,43,44]:
                    raise ValueError('Final lambda freeze differs')
            elif role=='uniform_main':
                fixed=read_json(cfg['uniform_lambda_file'])
                if (phase!='2' or fixed.get('policy')!='user_fixed_uniform' or fixed.get('value')!=.1
                    or value!=fixed['value'] or spec['id']!='A' or ds not in fixed['datasets']
                    or not fixed.get('user_message')):
                    raise ValueError('Uniform main lambda differs from explicit user decision')
            elif role=='zero_anchor':
                if phase!='2' or value!=0.:raise ValueError('Zero anchor belongs to the approved final matrix')
            else:raise ValueError('Explicit RL run role required')
    else:raise ValueError('Method outside approved scheme A matrix')
    if phase=='1A' and (ds!='chembl37' or (method=='atomformer' and spec['seed']!=42)):
        raise ValueError('1A is ChEMBL37 seed42 only')
    if phase=='1B' and method=='atomformer' and (spec['id'] not in ('A','C') or ds not in ('chembl37','moses') or (ds=='moses' and spec['seed']!=42)):
        raise ValueError('1B is the reviewed A/C confirmation matrix')
    if method=='atomformer' and spec['id']=='B' and phase!='1A':raise ValueError('B does not auto-expand')
    if evaluation.get('split') not in ('v_select_quick','v_select','v_check','test'):raise ValueError('Explicit evaluation role required')
    if phase!='2' and evaluation['split'] in ('v_check','test'):raise ValueError('V-check/test remain sealed')
    if phase=='1A' and evaluation['split']!='v_select_quick':raise ValueError('1A uses the frozen quick validation set')
    if evaluation.get('attempts')!=20 or evaluation.get('sampling_seed') not in (142,143,144):raise ValueError('Frozen inference budget/seeds differ')
    if method=='atomformer' and evaluation['sampling_seed']!=spec['seed']+100:raise ValueError('Training/inference seed pairing differs')
    if method in BASELINES and phase=='1A' and evaluation['sampling_seed']!=142:raise ValueError('1A baseline inference uses seed142')
    if evaluation.get('model_mode')!='eval' or evaluation.get('grad_enabled') is not False or evaluation.get('decoding')!='stochastic':
        raise ValueError('Formal evaluation must be eval/no_grad/stochastic')
    if evaluation.get('diversity_sampling_seed')!=20261025:raise ValueError('Frozen diversity seed differs')
    if method=='atomformer' and evaluation.get('temperature')!=1.:raise ValueError('AtomFormer evaluation temperature is 1')
    if cfg.get('resource_margin',0)<.1:raise ValueError('At least 10% resource margin required')
    if cfg['data'].get('mode')!='historical_import':raise ValueError('Only historical records are allowed')
    return {'valid':True,'protocol':PROTOCOL,'phase':phase,'config_sha256':digest(cfg)}
