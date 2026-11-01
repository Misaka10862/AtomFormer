"""Exact optimizer-boundary state, including RNG and sampler position."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from .io import freeze


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA topology changed on exact resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def load(path,device='cpu'):
    import inspect
    kwargs={'weights_only':False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path,map_location=device,**kwargs)


def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    torch.save(payload,temp)
    os.replace(temp,path)


def save_training(path,model,optimizer,contract,progress,scaler=None):
    save(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'contract':contract,
               'progress':progress,'rng':rng_state(),'scaler':scaler.state_dict() if scaler else None})


def restore_training(path,model,optimizer,contract,device='cpu',scaler=None):
    state=load(path,device)
    if state['contract']!=contract:
        raise ValueError('Checkpoint scientific contract mismatch')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    if scaler and state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state['progress']
