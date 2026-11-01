"""Geometry assessment separates native poses from a shared conformer workflow."""
import json
import subprocess
from pathlib import Path
from .data import Condition
from .chemistry import constrained_poses,geometry_diagnostics
from .io import atomic_json,file_sha,freeze


def posebusters_result(mol,protein=None):
    from posebusters import PoseBusters
    table=PoseBusters(config='gen' if protein else 'mol',max_workers=0).bust(mol_pred=mol,mol_cond=protein,full_report=False)
    if len(table)!=1 or table.empty:
        raise ValueError('PoseBusters did not return exactly one pose')
    checks={str(k):bool(v) if v==v else None for k,v in table.iloc[0].items()}
    return {'physical_valid':bool(checks) and all(v is True for v in checks.values()),'posebusters_checks':checks}


def assess_attempt(row,condition,out,conformer_budget=5,seed=17,protein=None):
    from rdkit import Chem
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result={'record_id':row['record_id'],'attempt_id':row['attempt_id'],'raw_pose':None,'shared_poses':[],
            'physical_valid':False,'requested_conformers':conformer_budget}
    if not row['valid']:
        result['failure_reason']='invalid_graph';return result
    if row.get('pose_molblock'):
        raw=Chem.MolFromMolBlock(row['pose_molblock'],removeHs=False)
        if raw is not None:
            result['raw_pose']={**geometry_diagnostics(raw,condition),**posebusters_result(raw,protein)}
            writer=Chem.SDWriter(str(out/'raw.sdf'));writer.write(raw);writer.close()
    poses,budget=constrained_poses(row['canonical_smiles'],condition,seed,conformer_budget)
    result.update(budget)
    for i,mol in enumerate(poses):
        details={**geometry_diagnostics(mol,condition),**posebusters_result(mol,protein)}
        details['core_constraint_passed']=details['core_rmsd'] is not None and details['core_rmsd']<=.5
        details['physical_valid']=details['physical_valid'] and details['core_constraint_passed']
        result['shared_poses'].append(details)
        writer=Chem.SDWriter(str(out/('shared_%03d.sdf'%i)));writer.write(mol);writer.close()
    result['physical_valid']=any(p['physical_valid'] for p in result['shared_poses'])
    atomic_json(out/'geometry.json',result)
    return result


def vina_score(receptor_pdbqt,ligand_pdbqt,center,box_size,seed=17,exhaustiveness=8):
    """A single fixed-budget docking call; never search parameters per candidate."""
    from vina import Vina
    if len(center)!=3 or len(box_size)!=3 or min(box_size)<=0:
        raise ValueError('Explicit task-defined docking box is required')
    engine=Vina(sf_name='vina',cpu=1,seed=seed)
    engine.set_receptor(str(receptor_pdbqt));engine.set_ligand_from_file(str(ligand_pdbqt))
    engine.compute_vina_maps(center=list(center),box_size=list(box_size))
    initial=float(engine.score()[0])
    engine.dock(exhaustiveness=exhaustiveness,n_poses=1)
    energy=float(engine.energies(n_poses=1)[0,0])
    return {'initial_vina_score':initial,'docking_score':energy,'docking_calls':1,
            'exhaustiveness':exhaustiveness,'seed':seed,'center':list(center),'box_size':list(box_size),
            'receptor_sha256':file_sha(receptor_pdbqt),'ligand_sha256':file_sha(ligand_pdbqt)}
