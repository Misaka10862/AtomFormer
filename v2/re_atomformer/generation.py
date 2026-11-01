"""One output row for every requested input/attempt, including failures."""
from dataclasses import replace
from pathlib import Path
import torch
from .data import Condition,collate_conditions
from .smiles import ResearchSmiles
from .protocol import score
from .chemistry import diffusion_molecule


def generate_scored(model,tokenizer,records,device,reference,reward_spec,attempts=20,max_length=256,multi_conformer=False,pocket_mode='correct'):
    from rdkit import Chem
    from condmolgen.v2_assembly import assemble_single_attachment_rooted
    if attempts!=20 and not all(r.get('dataset_id')=='synthetic' for r in records):
        raise ValueError('Formal evaluation uses 20 attempts per input')
    from .population import pocket_intervention
    records=pocket_intervention(records,pocket_mode)
    model.eval();result=[]
    for row in records:
        c=Condition.from_record(row)
        needs_geometry=not isinstance(model,ResearchSmiles) or model.encoder_kind not in ('atom','graph','smiles','prefix_only')
        supported=not needs_geometry or c.pos is not None
        if isinstance(model,ResearchSmiles):
            tokens=tokenizer.encode_smiles(c.scaffold_smiles)
            supported=supported and tokenizer.unk_token_id not in tokens and len(tokens)+2<max_length
            if model.encoder_kind=='smiles':supported=supported and len(tokens)<=model.config.scaffold_block_size
        if multi_conformer:
            if attempts!=20:raise ValueError('Multi-conformer protocol requires 20 attempts')
            supported=supported and bool(c.conformers) and len(c.conformers)==5
        for a in range(attempts):
            condition=replace(c,pos=c.conformers[a//4]) if multi_conformer and supported else c
            failure=None;smi=None;extra={};decoration=None
            try:
                if not supported:raise ValueError('unsupported_input')
                b=collate_conditions([condition],device)
                if isinstance(model,ResearchSmiles):
                    sampled=model.sample(b,[condition.scaffold_smiles],tokenizer,max_length)
                    decoration=sampled['decorations'][0]
                    smi=assemble_single_attachment_rooted(condition.scaffold_smiles,decoration) if decoration else None
                    failure='assembly_or_eos_failure' if smi is None else None
                else:
                    sampled=model.sample(b)
                    mol=diffusion_molecule(condition,sampled)
                    smi=Chem.MolToSmiles(mol,canonical=True)
                    extra['pose_molblock']=Chem.MolToMolBlock(mol)
            except (ValueError,RuntimeError) as error:
                # Resource/numerical failures abort the run; only chemical failures consume attempts.
                if isinstance(error,RuntimeError):raise
                failure=str(error)
            evaluated=score(smi,row,reference,reward_spec)
            result.append({**evaluated,'record_id':row['record_id'],'attempt_id':a,
                           'scaffold_smiles':row['scaffold_smiles'],'decoration':decoration,'assembled_smiles':smi,
                           'input_supported':supported,'failure_reason':failure or evaluated.get('failure_reason'),
                           'conformer_index':a//4 if multi_conformer else 0,**extra})
    return result
