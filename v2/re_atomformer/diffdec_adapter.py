"""Native DiffDec single-R-group baseline with condition-only inference templates."""
import os
import subprocess
import sys
from pathlib import Path
import torch
from .external import require_upstream,diffdec_training_config
from .data import Condition
from .io import freeze,atomic_json


def activate(directory,lock):
    path=require_upstream('diffdec',directory,lock)
    if 'src' in sys.modules:
        origin=getattr(sys.modules['src'],'__file__',None)
        if origin and not str(Path(origin).resolve()).startswith(str(path)):
            raise RuntimeError('Run DiffDec in its own process to avoid the generic src namespace collision')
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    os.environ['WANDB_MODE']='disabled'
    return path


def native_record(condition,target=None,slots=10):
    """Upstream fake-atom capacity is fixed; target size never controls inference slots."""
    from rdkit import Chem
    from src import const
    c=condition;core=len(c.z);p=len(c.pocket_z or [])
    if c.pos is None:raise ValueError('DiffDec needs scaffold coordinates')
    if target is not None and len(target['z'])>slots:
        raise ValueError('DiffDec native R-group capacity exceeded; record unsupported')
    z=c.z+(c.pocket_z or [])+([*target['z']]+[0]*(slots-len(target['z'])) if target else [0]*slots)
    anchor_pos=c.pos[c.anchor_index]
    positions=c.pos+(c.pocket_pos or [])+(target['pos']+[anchor_pos]*(slots-len(target['z'])) if target else [anchor_pos]*slots)
    one_hot=torch.zeros(len(z),len(const.ATOM2IDX))
    periodic=Chem.GetPeriodicTable()
    for i,atomic_number in enumerate(z):
        symbol=periodic.GetElementSymbol(atomic_number) if atomic_number else '#'
        if symbol not in const.ATOM2IDX:raise ValueError('DiffDec vocabulary does not cover '+symbol)
        one_hot[i,const.ATOM2IDX[symbol]]=1
    anchors=torch.zeros(len(z));anchors[c.anchor_index]=1
    scaffold_only=torch.tensor([1.]*core+[0.]*(p+slots))
    pocket=torch.tensor([0.]*core+[1.]*p+[0.]*slots)
    rgroup=torch.tensor([0.]*(core+p)+[1.]*slots)
    return {'uuid':c.record_id,'name':c.scaffold_smiles,'positions':torch.tensor(positions,dtype=torch.float32),
            'one_hot':one_hot,'charges':torch.tensor(z,dtype=torch.float32),'anchors':anchors,
            'scaffold_only_mask':scaffold_only,'pocket_mask':pocket,'scaffold_mask':scaffold_only+pocket,
            'rgroup_mask':rgroup,'num_atoms':len(z)}


def export_training(rows,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    report={}
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        if not selected:raise ValueError('DiffDec export requires independent train/validation records')
        supported=[];rejected=[]
        for r in selected:
            try:supported.append(native_record(Condition.from_record(r),r['target_3d']))
            except ValueError as error:rejected.append({'record_id':r['record_id'],'reason':str(error)})
        if not supported:raise ValueError('No supported DiffDec '+split+' records')
        torch.save(supported,destination/('re_'+split+'_full.pt'))
        report[split]={'supported':len(supported),'requested':len(selected),'unsupported':rejected}
    atomic_json(destination/'coverage.json',report)
    return report


class NativeDiffDec:
    def __init__(self,checkpoint,directory,lock,device='cpu'):
        activate(directory,lock)
        from src.model_single import DDPM
        self.device=device
        self.model=DDPM.load_from_checkpoint(str(checkpoint),map_location=device).eval().to(device)

    def supports(self,condition):
        try:native_record(condition);return True
        except (ValueError,KeyError):return False

    @torch.no_grad()
    def sample(self,conditions):
        from src.datasets import collate
        from src.molecule_builder import build_molecules
        from rdkit import Chem
        data=collate([native_record(c) for c in conditions])
        data={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        # Every template is fixed at the upstream capacity; no true sizes are passed.
        chain,mask=self.model.sample_chain(data,keep_frames=1)
        x,h=chain[-1][:,:,:3],chain[-1][:,:,3:]
        if self.model.center_of_mass=='anchors':
            center=(data['positions']*data['anchors']).sum(1)/data['anchors'].sum(1)
        else:
            center=(data['positions']*data['scaffold_only_mask']).sum(1)/data['scaffold_only_mask'].sum(1)
        x=x+center[:,None,:]
        molecules=build_molecules(h,x,mask)
        output=[]
        for mol in molecules:
            try:
                Chem.SanitizeMol(mol);Chem.AssignStereochemistryFrom3D(mol)
                output.append({'smiles':Chem.MolToSmiles(mol),'pose_molblock':Chem.MolToMolBlock(mol)})
            except (ValueError,RuntimeError):output.append({'smiles':None,'failure_reason':'diffdec_reconstruction'})
        return output


def launch_training(settings,upstream,lock,data_root,run_root,python=None,resume=False):
    """Use native DDPM/loss with independent validation and non-deleting best/last saves."""
    import yaml
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer,Callback
    upstream=activate(upstream,lock)
    from src.model_single import DDPM
    from src.const import NUMBER_OF_ATOM_TYPES
    run_root=Path(run_root).resolve();run_root.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((upstream/'configs/single.yml').read_text())
    config=diffdec_training_config(config,'re_train.full','re_validation.full')
    config.update(settings)
    if 'test' in config['val_data_prefix']:raise ValueError('Test prefix cannot be validation')
    freeze(run_root/'native_config.json',config)
    device='cuda:0' if config.get('device','gpu')=='gpu' else 'cpu'
    pl.seed_everything(config.get('seed',11),workers=True)
    model=DDPM(in_node_nf=NUMBER_OF_ATOM_TYPES,n_dims=3,context_node_nf=3,
        hidden_nf=config.get('nf',128),activation='silu',tanh=config.get('tanh',False),n_layers=config.get('n_layers',6),
        attention=config.get('attention',False),norm_constant=config.get('norm_constant',1e-6),
        inv_sublayers=config.get('inv_sublayers',2),sin_embedding=False,normalization_factor=config.get('normalization_factor',100),
        aggregation_method='sum',diffusion_steps=config.get('diffusion_steps',500),diffusion_noise_schedule='polynomial_2',
        diffusion_noise_precision=1e-5,diffusion_loss_type='l2',normalize_factors=[1,4,10],include_charges=False,
        model='egnn_dynamics',data_path=str(Path(data_root).resolve()),train_data_prefix=config['train_data_prefix'],
        val_data_prefix=config['val_data_prefix'],batch_size=config.get('batch_size',8),lr=config.get('lr',2e-4),
        torch_device=device,test_epochs=config.get('test_epochs',20),n_stability_samples=config.get('n_stability_samples',5),
        normalization='batch_norm',samples_dir=str(run_root/'validation_samples'),center_of_mass='anchors')
    class SaveSelected(Callback):
        def __init__(self):self.best=None;self.bad=0;self.history=[]
        def state_dict(self):return {'best':self.best,'bad':self.bad,'history':self.history}
        def load_state_dict(self,state):self.best=state['best'];self.bad=state['bad'];self.history=state['history']
        def on_validation_end(self,trainer,module):
            if trainer.sanity_checking:return
            if 'loss/val' not in trainer.callback_metrics:raise ValueError('Native validation metric is missing')
            value=float(trainer.callback_metrics['loss/val'])
            if not __import__('math').isfinite(value):raise FloatingPointError('Non-finite native validation')
            improved=self.best is None or value<self.best
            self.best=value if improved else self.best;self.bad=0 if improved else self.bad+1
            self.history.append({'epoch':trainer.current_epoch+1,'validation_loss':value})
            if improved:trainer.save_checkpoint(str(run_root/'best.ckpt'))
            trainer.save_checkpoint(str(run_root/'last.ckpt'))
            atomic_json(run_root/'history.json',self.history)
            if self.bad>=config.get('patience',5):trainer.should_stop=True
    checkpoint=SaveSelected()
    trainer=Trainer(max_epochs=config.get('n_epochs',48),logger=False,callbacks=[checkpoint],
        enable_checkpointing=False,accelerator='gpu' if device.startswith('cuda') else 'cpu',devices=1,
        num_sanity_val_steps=0,enable_progress_bar=False)
    previous=run_root/'last.ckpt'
    if previous.exists() and not resume:raise ValueError('Explicit resume required')
    trainer.fit(model,ckpt_path=str(previous) if resume else None)
    if not (run_root/'best.ckpt').is_file():raise RuntimeError('Native training produced no selected checkpoint')
    return {'checkpoint':str(run_root/'best.ckpt'),'history':checkpoint.history}
