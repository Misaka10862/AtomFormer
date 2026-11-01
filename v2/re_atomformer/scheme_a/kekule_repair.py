"""Exception-only repair of RDKit 2022.09.5 Murcko aromatic carbanion H capping.

Successful legacy calls are returned unchanged. Repair uses only the generated
parent's cut bonds, never a target, method name, record ID or desired result.
"""
from unittest.mock import patch

PATCH_ID='murcko-aromatic-carbanion-cut-H-v1'


def capped_framework(parent, original_get_scaffold):
    from rdkit import Chem
    tagged=Chem.Mol(parent)
    for atom in tagged.GetAtoms():atom.SetIntProp('_re_atomformer_parent_idx',atom.GetIdx())
    framework=original_get_scaffold(tagged)
    retained={a.GetIntProp('_re_atomformer_parent_idx') for a in framework.GetAtoms()}
    capped=[]
    original_atoms=[(a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),int(a.GetChiralTag())) for a in framework.GetAtoms()]
    original_edges=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),int(b.GetStereo())) for b in framework.GetBonds()]
    for atom in framework.GetAtoms():
        source=tagged.GetAtomWithIdx(atom.GetIntProp('_re_atomformer_parent_idx'))
        removed=[b for b in source.GetBonds() if b.GetOtherAtomIdx(source.GetIdx()) not in retained]
        # Narrow, causal condition: one removed single side-chain bond at an
        # aromatic C(-1), and RDKit supplied neither an explicit nor implicit H.
        if (atom.GetAtomicNum()==6 and atom.GetFormalCharge()==-1 and atom.GetIsAromatic()
            and atom.GetTotalNumHs()==0 and len(removed)==1
            and removed[0].GetBondType()==Chem.BondType.SINGLE):
            atom.SetNumExplicitHs(1);capped.append(atom.GetIdx())
    if not capped:raise ValueError('No proven aromatic carbanion boundary-H defect')
    Chem.SanitizeMol(framework)
    if original_atoms!=[(a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),int(a.GetChiralTag())) for a in framework.GetAtoms()]:
        raise ValueError('Repair changed atom identity, charge or stereochemistry')
    if original_edges!=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),int(b.GetStereo())) for b in framework.GetBonds()]:
        raise ValueError('Repair changed heavy-atom connectivity or bond stereo')
    return framework


def wrap_scaffold_checks(legacy):
    def checked(generated,target_core):
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        try:return legacy(generated,target_core)
        except Chem.rdchem.KekulizeException as original_error:
            original_get=MurckoScaffold.GetScaffoldForMol
            def fixed(parent):return capped_framework(parent,original_get)
            try:
                with patch.object(MurckoScaffold,'GetScaffoldForMol',fixed):
                    return legacy(generated,target_core)
            except (ValueError,RuntimeError):
                # Never convert an unexplained error into invalid/non-retained.
                raise original_error
    return checked
