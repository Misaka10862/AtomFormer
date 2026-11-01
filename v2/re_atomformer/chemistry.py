"""Shared assembly and fixed-budget poses; failed chemistry remains an attempt."""
from .data import BOND_TYPES


def core_molecule(condition):
    from rdkit import Chem
    mapped=getattr(condition,'mapped_core_smiles',None)
    mol=Chem.MolFromSmiles(mapped or condition.core_smiles)
    if mapped:
        order=sorted(range(mol.GetNumAtoms()),key=lambda i:mol.GetAtomWithIdx(i).GetAtomMapNum())
        mol=Chem.RenumberAtoms(mol,order)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    if [a.GetAtomicNum() for a in mol.GetAtoms()]!=condition.z:
        raise ValueError('Core atom mapping differs')
    return mol


def diffusion_molecule(condition,sample,index=0):
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    core=core_molecule(condition)
    rw=Chem.RWMol(core)
    real_core=len(condition.z); padded_core=sample['scaffold_pos'].shape[1]
    size=int(sample['sizes'][index])
    for z in sample['new_z'][index,:size].tolist():
        rw.AddAtom(Chem.Atom(int(z)))
    edges=sample['bonds'][index].cpu()
    cross=[]
    types={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
    for i in range(real_core+size):
        for j in range(max(i+1,real_core),real_core+size):
            ii=i if i<real_core else padded_core+i-real_core
            jj=padded_core+j-real_core
            kind=int(edges[ii,jj])
            if kind:
                if i<real_core:
                    cross.append((i,j,kind))
                rw.AddBond(i,j,types[kind])
    if len(cross)!=1 or cross[0][0]!=condition.anchor_index or cross[0][2]!=1:
        raise ValueError('Single attachment constraint failed')
    mol=rw.GetMol()
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol))!=1:
        raise ValueError('Disconnected decoration')
    conf=Chem.Conformer(mol.GetNumAtoms())
    coords=sample['scaffold_pos'][index,:real_core].tolist()+sample['new_pos'][index,:size].tolist()
    for i,p in enumerate(coords):
        conf.SetAtomPosition(i,Point3D(*p))
    mol.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(mol)
    from .protocol import retained_subgraph
    if not retained_subgraph(Chem.MolToSmiles(mol),condition.core_smiles):
        raise ValueError('Core stereochemistry changed')
    return mol


def constrained_poses(smiles,condition,seed=17,budget=5):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    if budget<1 or condition.pos is None:
        raise ValueError('Pose budget and mapped core coordinates are required')
    core=core_molecule(condition)
    conf=Chem.Conformer(core.GetNumAtoms())
    for i,p in enumerate(condition.pos):
        conf.SetAtomPosition(i,Point3D(*p))
    core.AddConformer(conf)
    poses,failures=[],[]
    for j in range(budget):
        try:
            mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
            mol=AllChem.ConstrainedEmbed(mol,core,useTethers=True,randomseed=seed+j)
            poses.append(Chem.RemoveHs(mol))
        except (ValueError,RuntimeError) as error:
            failures.append(type(error).__name__)
    return poses,{'requested_conformers':budget,'successful_conformers':len(poses),'failures':failures}


def geometry_diagnostics(mol,condition):
    """These checks are diagnostics, not a substitute for a PoseBusters pass."""
    import numpy as np
    from rdkit.Chem import AllChem
    from rdkit import Chem
    core=core_molecule(condition)
    matches=mol.GetSubstructMatches(core,useChirality=True,uniquify=False)
    if not matches or mol.GetNumConformers()==0:
        return {'core_rmsd':None,'finite_coordinates':False,'severe_clashes':None,'uff_energy':None}
    pos=mol.GetConformer().GetPositions()
    rmsd=min(float(np.sqrt(((pos[list(m)]-np.array(condition.pos))**2).sum(1).mean())) for m in matches)
    distances=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
    bond=Chem.GetAdjacencyMatrix(mol)
    clashes=sum(distances[i,j]<.7 and not bond[i,j] for i in range(len(pos)) for j in range(i+1,len(pos)))
    energy=None
    try:
        h=Chem.AddHs(mol,addCoords=True)
        if AllChem.UFFHasAllMoleculeParams(h):
            energy=float(AllChem.UFFGetMoleculeForceField(h).CalcEnergy())
    except (ValueError,RuntimeError):
        pass
    return {'core_rmsd':rmsd,'finite_coordinates':bool(np.isfinite(pos).all()),'severe_clashes':int(clashes),'uff_energy':energy}
