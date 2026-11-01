"""Pinned external methods use common output contracts and aligned reward gates."""
import importlib
import subprocess
import sys
from pathlib import Path

from .io import read_json,file_sha,freeze
from .protocol import RewardSpec,score,augmented_score


def require_upstream(name,directory,lock):
    entry=lock[name];directory=Path(directory).resolve()
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(directory),text=True).strip()
    if actual!=entry['commit']:
        raise ValueError('Upstream revision differs: '+name)
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=str(directory),text=True).strip()
    if dirty:raise ValueError('Pinned upstream working tree is not clean: '+name)
    return directory


def adapt_training_rows(rows):
    from baselines.scaffold_bench.common import convert
    if any(r.get('split')!='train' for r in rows):
        raise ValueError('External vocabulary construction requires training rows only')
    return [convert((i,r)) for i,r in enumerate(rows)]


class SequenceBaseline:
    """Native LibINVENT/PromptSMILES networks; input coverage ignores target labels."""
    def __init__(self,method,training_rows,settings,upstreams,lock,device='cpu'):
        if method not in ('libinvent','promptsmiles'):
            raise ValueError('Unsupported sequence baseline')
        names=['libinvent'] if method=='libinvent' else ['smilesrnn','promptsmiles']
        for name in names:
            path=require_upstream(name,upstreams[name],lock)
            if str(path) not in sys.path:sys.path.insert(0,str(path))
        from baselines.scaffold_bench import models_adapter
        cfg={'num_layers':3,'hidden_size':512,'dropout':0.,'max_length':256,'prompt_ring_grammar':True,**settings}
        rows=adapt_training_rows(training_rows)
        self.backend=models_adapter.create(method,rows,cfg,device)
        self.network=self.backend.network;self.method=method

    def condition_row(self,condition):
        from baselines.scaffold_bench.common import normalize_dummy
        return {'scaffold':normalize_dummy(condition.scaffold_smiles)}

    def supports(self,condition):
        r=self.condition_row(condition)
        try:
            if self.method=='libinvent':
                return self.backend.vocab.encode_scaffold(r['scaffold']) is not None
            prompt=self.backend.prompt(r).replace('[SH]','S')
            self.backend.model.vocabulary.encode(self.backend.model.tokenizer.tokenize(prompt))
            return True
        except (KeyError,ValueError,IndexError):return False

    def sample(self,conditions):
        from baselines.scaffold_bench.common import join
        native=[self.condition_row(c) for c in conditions]
        outputs,decorations=self.backend.sample(native)
        return [join(r['scaffold'],d) if out=='__assemble__' else out for r,out,d in zip(native,outputs,decorations)],decorations

    def reward(self,smiles,condition,reference,protocol='aligned',task='murcko'):
        return score(smiles,condition.__dict__,reference,RewardSpec(task=task,protocol=protocol,method=self.method))


def scored_attempts(samples,conditions,reference,method,task='murcko',protocol='aligned',attempts=20):
    """Adapters must supply explicit unsupported slots, never silently shorten inputs."""
    conditions={c.record_id:c for c in conditions};seen=set();rows=[]
    for sample in samples:
        rid=sample['record_id'];a=sample['attempt_id']
        if rid not in conditions or (rid,a) in seen or not 0<=a<attempts:
            raise ValueError('External sample identity/budget mismatch')
        seen.add((rid,a))
        row=score(sample.get('smiles'),conditions[rid].__dict__,reference,RewardSpec(task=task,method=method,protocol=protocol))
        rows.append({**row,**{k:sample[k] for k in ('record_id','attempt_id')},'input_supported':sample.get('input_supported',True)})
    if len(seen)!=len(conditions)*attempts:
        raise ValueError('External output must account for the complete requested budget')
    return rows


class DrugExAlignedEnvironment:
    """DrugEx explorer-compatible reward interface, paired with input fragment cores.

    The native Pareto environment remains in the imported legacy implementation.
    This adapter intentionally uses the shared scalar aligned task reward.
    """
    def __init__(self,reference,core_by_fragment):
        self.reference=reference;self.core_by_fragment=core_by_fragment

    def getRewards(self,smiles,frags=None):
        import numpy as np
        if frags is None or len(frags)!=len(smiles):
            raise ValueError('DrugEx rewards require the corresponding input fragments')
        values=[]
        from .protocol import canonical
        for smi,frag in zip(smiles,frags):
            frag=canonical(frag) or frag
            if frag not in self.core_by_fragment:
                raise ValueError('Unknown DrugEx condition; cannot guess a scaffold')
            r=score(smi,{'core_smiles':self.core_by_fragment[frag]},self.reference,RewardSpec(method='drugex'))
            values.append([augmented_score(r['reward'])])
        return np.asarray(values,dtype=float)

    def getScores(self,smiles,frags=None):
        import pandas as pd
        rewards=self.getRewards(smiles,frags).reshape(-1)
        return pd.DataFrame({'AlignedReward':rewards,'Valid':[r>0 for r in rewards],'Desired':[r>.5 for r in rewards]})

    def getScorerKeys(self):
        return ['AlignedReward']


def diffdec_training_config(upstream_config,train_prefix,validation_prefix):
    """Never inherit the upstream configuration's test-as-validation alias."""
    if not train_prefix or not validation_prefix or train_prefix==validation_prefix or 'test' in validation_prefix.lower():
        raise ValueError('DiffDec needs an independent training-side validation split')
    result=dict(upstream_config)
    result.update(train_data_prefix=train_prefix,val_data_prefix=validation_prefix)
    return result
