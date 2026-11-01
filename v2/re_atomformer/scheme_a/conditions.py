"""Evaluation conditions are independent of reference molecules, labels and conformers."""
from dataclasses import dataclass
from ..io import digest


@dataclass(frozen=True)
class Request:
    record_id: str
    scaffold_smiles_with_attachment: str
    scoring_target_core: str

    @classmethod
    def from_record(cls,row):
        from scaffold_metrics import scaffold_core_from_scaffold
        scaffold=row.get('scaffold_smiles_with_attachment',row.get('scaffold_smiles'))
        if not scaffold:raise ValueError('Missing input scaffold')
        # Derive from the input scaffold, never from canonical/source/target molecule fields.
        core=scaffold_core_from_scaffold(scaffold)
        declared=row.get('scoring_target_core',row.get('core_smiles'))
        if declared is not None and declared!=core:
            from ..protocol import canonical
            if canonical(declared)!=canonical(core):raise ValueError('Declared core differs from input scaffold')
        if not core:raise ValueError('Invalid input core')
        return cls(str(row['record_id']),scaffold,core)

    def identity(self):
        return digest(self.__dict__)


def atomformer_condition(request):
    """Build condition-only atom features; no target-dependent preprocessing is used."""
    from rdkit import Chem
    from ..data import Condition,molecule_features
    mol=Chem.MolFromSmiles(request.scaffold_smiles_with_attachment)
    if mol is None:raise ValueError('Invalid scaffold')
    dummy=[a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(dummy)!=1 or dummy[0].GetDegree()!=1:raise ValueError('Expected one attachment')
    anchor=dummy[0].GetNeighbors()[0].GetIdx()
    for atom in mol.GetAtoms():atom.SetIntProp('_source_idx',atom.GetIdx())
    dummy[0].SetAtomicNum(1);dummy[0].SetIsotope(0);dummy[0].SetAtomMapNum(0)
    mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
    position=next(a.GetIdx() for a in mol.GetAtoms() if a.GetIntProp('_source_idx')==anchor)
    return Condition(record_id=request.record_id,scaffold_smiles=request.scaffold_smiles_with_attachment,
                     core_smiles=request.scoring_target_core,anchor_index=position,**molecule_features(mol))


def label_independence(adapter, record):
    """Poison and remove reference fields; retain immutable request identity/condition."""
    import copy
    forbidden=('canonical_smiles','source_smiles','rooted_decoration_smiles','plain_decoration_smiles',
               'decoration_smiles','single_decoration_smiles','anchor_z','target_3d','pos','cond_pos',
               'conformers','target_size','decoration_size','legacy_atom_z','cond_z','z')
    request=Request.from_record(record);expected=adapter.support(request)
    poisoned=copy.deepcopy(record)
    for name in forbidden:poisoned[name]={'poison':'must not be read'}
    removed={k:v for k,v in record.items() if k not in forbidden}
    for other in (poisoned,removed):
        observed=adapter.support(Request.from_record(other))
        if observed!=expected:raise AssertionError('Support/native condition depends on reference fields')
    return {'passed':True,'request_sha256':request.identity(),'fields_tested':list(forbidden)}


def shuffled_label_independence(adapter, records):
    import copy
    if len(records)<2:raise ValueError('Shuffle test needs at least two conditions')
    fields=('canonical_smiles','source_smiles','rooted_decoration_smiles','plain_decoration_smiles',
            'decoration_smiles','single_decoration_smiles','anchor_z','target_3d','pos','cond_pos','conformers')
    before=[adapter.support(Request.from_record(row)) for row in records]
    for i,row in enumerate(records):
        other=copy.deepcopy(row);donor=records[(i+1)%len(records)]
        for name in fields:
            if name in donor:other[name]=copy.deepcopy(donor[name])
            else:other.pop(name,None)
        if adapter.support(Request.from_record(other))!=before[i]:raise AssertionError('Shuffled reference labels changed support')
    return {'passed':True,'records':len(records),'fields_tested':list(fields)}
