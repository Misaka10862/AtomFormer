"""Seven planned comparison blocks; later rounds are explicit, never Cartesian."""
import copy


def exploration_matrix():
    base={'population':'common_geometry','decoder':'smiles','encoder':'atom','seed':11,'architecture':{'auxiliary_pooling':'masked_mean','encoder_norm_first':False,'atom_attention_dropout':.1,'anchor_aux_weight':.1},
          'training':{'epochs':12,'micro_batch':4,'effective_batch':64,'lr':3e-4,'max_train_records':50000,'mode':'supervised'}}
    units=[]
    def add(identity,group,encoder=None,decoder=None,training=None,architecture=None,**extras):
        spec=copy.deepcopy(base);spec['id']=identity;spec['group']=group
        if encoder:spec['encoder']=encoder
        if decoder:spec['decoder']=decoder
        spec['training'].update(training or {});spec['architecture'].update(architecture or {});spec.update(extras)
        units.append(spec)
    add('legacy_atom','repair',architecture={'auxiliary_pooling':'legacy_mean'})
    add('fixed_atom','repair')
    add('fixed_geo','repair',encoder='geo',geometry_source='parent')
    for kind in ('graph','gated_geo','pocket_distance','pocket_geo'):
        add(kind,'geometry' if not kind.startswith('pocket') else 'pocket',encoder=kind,
            dataset='crossdock_single' if kind.startswith('pocket') else 'chembl37',geometry_source='parent')
    add('gated_geometry_off','geometry',encoder='gated_geo',geometry_source='parent',geometry_enabled=False)
    for geometry in ('parent','independent','multi'):
        add('gated_'+geometry,'geometry',encoder='gated_geo',geometry_source=geometry)
    add('geometry_contrastive','geometry',encoder='gated_geo',geometry_source='parent',training={'mode':'contrastive'})
    for kind in ('gated_geo','pocket_geo'):
        add('diffusion_'+kind,'direct3d',encoder=kind,decoder='diffusion',geometry_source='parent' if kind=='gated_geo' else 'pocket_complex',
            dataset='chembl37' if kind=='gated_geo' else 'crossdock_single',
            diffusion={'steps':100,'width':128,'layers':6,'max_new_atoms':48,'condition_features':True},training={'max_train_records':20000})
    add('diffusion_no_graph_prior','direct3d',encoder='atom',decoder='diffusion',geometry_source='parent',
        diffusion={'steps':100,'width':128,'layers':6,'max_new_atoms':48,'condition_features':False},training={'max_train_records':20000})
    for weight in (0,.02,.05,.1,.2):
        add('rl_anchor_'+str(weight).replace('.','p'),'rl_anchor',training={'mode':'rl','epochs':4,'micro_batch':64,'effective_batch':64,'lr':2e-6,'supervised_weight':weight,'source_checkpoint':'$FIXED_ATOM_BEST'})
    return {'schema_version':1,'units':units,
        'dependent_rounds':[
            {'group':'rl_learning_rate','parents':'top_two_anchor_validation','values':[2e-6,1e-5,2e-5]},
            {'group':'rl_objective','parents':'best_anchor_lr_validation','normalization':['length_mean','fixed_scale'],'baselines':['batch_mean','leave_one_out'],'objectives':['reinforce','augmented_likelihood'],'repetition_weight':[0,.1]},
            {'group':'pocket_intervention','parents':['pocket_distance','pocket_geo','diffusion_pocket_geo'],'pocket_mode':['none','correct','shuffle']},
            {'group':'duration','parents':'top_two_per_route','supervised_epochs':[12,24,48],'rl_epochs':[4,8,16]},
            {'group':'scale','parents':'top_two_per_route','train_records':[50000,180000,'all_eligible']},
            {'group':'replication','parents':'top_two_per_route_and_controls','seeds':[11,23,37],'datasets':['chembl37','gdb13','moses']}],
        'external_baselines':['drugex','libinvent','promptsmiles','diffdec'],
        'evaluation':{'attempts_per_record':20,'conformer_comparison':[[1,20],[5,4]],'selection_split':'validation','test_sealed':True},
        'status':'PLANNED_NOT_EXECUTED'}
