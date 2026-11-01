"""Restore misidentified GDB13 scaffold fragments from the immutable parent molecule.

No conformer generation, new molecules, target edits or split reassignment. The old
extractor selected a fragment by size alone; here original parent atom identity wins.
"""
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from condmolgen.v2_assembly import assemble_single_attachment_rooted, assemble_single_attachment, strip_single_attachment_dummy_with_root, _renumber_scaffold_dummy_atoms
from ..protocol import canonical


def capped_composition(scaffold):
    mol=Chem.MolFromSmiles(scaffold)
    if mol is None:raise ValueError('Unparseable scaffold')
    dummies=[a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(dummies)!=1 or dummies[0].GetDegree()!=1:raise ValueError('Not a single attachment')
    a=dummies[0];a.SetAtomicNum(1);a.SetIsotope(0);a.SetAtomMapNum(0)
    mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
    return sorted(a.GetAtomicNum() for a in mol.GetAtoms())


def restore_record(row):
    parent=canonical(row.get('canonical_smiles') or row['source_smiles'])
    if not parent or ('source_smiles' in row and canonical(row['source_smiles'])!=parent):
        raise ValueError('Parent identity ambiguous')
    z=sorted(int(x) for x in row['cond_z'])
    target=row['rooted_decoration_smiles'];old=row['scaffold_smiles']
    target_mol=Chem.MolFromSmiles(target)
    if target_mol is None or target_mol.GetAtomWithIdx(0).GetAtomicNum()!=int(row['anchor_z']):
        raise ValueError('Historical target/anchor label differs')
    composition_ok=capped_composition(old)==z
    assembly_ok=canonical(assemble_single_attachment_rooted(old,target))==parent
    # The original dummy-bearing upstream target can prove atom correspondence even
    # where the frozen rooted assembler mishandles an explicit H on sulfur. Keep
    # that existing target/protocol limitation; this restoration does not fix it.
    single=row.get('single_decoration_smiles')
    root_atom=target_mol.GetAtomWithIdx(0)
    known_sulfur=(root_atom.GetAtomicNum()==16 and root_atom.GetNumExplicitHs()==1 and single
                 and strip_single_attachment_dummy_with_root(single)==(target,int(row['anchor_z'])))
    def upstream_matches(scaffold):
        return bool(known_sulfur and canonical(assemble_single_attachment(scaffold,single))==parent)
    if composition_ok and assembly_ok:return row,None
    if composition_ok and upstream_matches(old):
        return row,dict(changes={},reason='unchanged_historical_rooted_sulfur_assembly_limitation',
                        known_rooted_assembly_failure=True,upstream_dummy_assembly_matches=True)
    mol=Chem.MolFromSmiles(parent);core=MurckoScaffold.GetScaffoldForMol(mol)
    if not core.GetNumAtoms():raise ValueError('No parent Murcko core')
    for a in mol.GetAtoms():a.SetIntProp('_repair_parent_index',a.GetIdx())
    candidates=set()
    for match in mol.GetSubstructMatches(core,uniquify=True,useChirality=True,maxMatches=10000):
        atomset=set(match)
        if sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in match)!=z:continue
        cuts=[b.GetIdx() for b in mol.GetBonds() if (b.GetBeginAtomIdx() in atomset)!=(b.GetEndAtomIdx() in atomset)]
        if len(cuts)!=1:continue
        fragmented=Chem.FragmentOnBonds(mol,cuts,addDummies=True)
        for frag in Chem.GetMolFrags(fragmented,asMols=True,sanitizeFrags=True):
            ids={a.GetIntProp('_repair_parent_index') for a in frag.GetAtoms() if a.GetAtomicNum()>0}
            if ids!=atomset:continue
            frag,_=_renumber_scaffold_dummy_atoms(frag)
            candidate=Chem.MolToSmiles(frag,canonical=True)
            if capped_composition(candidate)==z and (canonical(assemble_single_attachment_rooted(candidate,target))==parent or upstream_matches(candidate)):
                candidates.add(candidate)
    if len(candidates)!=1:raise ValueError('No unique source-consistent restoration: %d candidates'%len(candidates))
    new=next(iter(candidates));result=dict(row,scaffold_smiles=new)
    changes={'scaffold_smiles':{'old':old,'new':new}}
    if 'cond_smiles' in row:
        if row['cond_smiles']!=old:raise ValueError('Historical condition alias differs')
        result['cond_smiles']=new;changes['cond_smiles']={'old':old,'new':new}
    return result,dict(changes=changes,reason='parent_atom_identity_restores_size_only_fragment_selection',
                       old_composition_matches=composition_ok,old_assembly_matches=assembly_ok,
                       restored_assembly_matches=canonical(assemble_single_attachment_rooted(new,target))==parent,
                       upstream_dummy_assembly_matches=upstream_matches(new),
                       known_rooted_assembly_failure=canonical(assemble_single_attachment_rooted(new,target))!=parent,unique_candidates=1)
