from functools import lru_cache
PROTOCOL = 'scaffold-v2-direct-or-kekule-nested'

def protocol_metadata():
    import rdkit
    return {'scaffold_metric_protocol': PROTOCOL, 'rdkit_version': rdkit.__version__, 'match_chirality': False, 'retention_isomeric': True, 'retention_requires_match': True}

@lru_cache(maxsize=50000)
def scaffold_core_from_scaffold(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetDegree() != 1:
                    return None
                rw.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))
        core = rw.GetMol()
        Chem.SanitizeMol(core)
        return Chem.MolToSmiles(Chem.RemoveHs(core), canonical=True)
    except (ValueError, RuntimeError):
        return None

@lru_cache(maxsize=50000)
def _target(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not mol.GetNumAtoms() or any((a.GetAtomicNum() == 0 for a in mol.GetAtoms())):
        return None
    return mol

def _kekule(mol):
    from rdkit import Chem
    result = Chem.Mol(mol)
    Chem.Kekulize(result, clearAromaticFlags=True)
    return result

def scaffold_checks(generated, target_core):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    result = {'eligible': False, 'core_match': False, 'core_retention': False, 'direct_match': False, 'kekule_rescued': False, 'murcko_equal': False}
    mol = Chem.MolFromSmiles(generated) if generated else None
    core = _target(target_core)
    if mol is None or core is None or (not mol.GetNumAtoms()) or any((a.GetAtomicNum() == 0 for a in mol.GetAtoms())):
        return result
    result['eligible'] = True
    direct = mol.HasSubstructMatch(core, useChirality=False)
    kmol = kcore = None
    matched = direct
    if not direct:
        kmol, kcore = (_kekule(mol), _kekule(core))
        matched = kmol.HasSubstructMatch(kcore, useChirality=False)
    framework = MurckoScaffold.GetScaffoldForMol(mol)
    equal = False
    if framework.GetNumAtoms() == core.GetNumAtoms() and framework.GetNumBonds() == core.GetNumBonds():
        equal = Chem.MolToSmiles(framework, canonical=True, isomericSmiles=True) == Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
        if not equal:
            kcore = kcore if kcore is not None else _kekule(core)
            equal = Chem.MolToSmiles(_kekule(framework), canonical=True, isomericSmiles=True) == Chem.MolToSmiles(kcore, canonical=True, isomericSmiles=True)
    result.update(core_match=bool(matched), core_retention=bool(matched and equal), direct_match=bool(direct), kekule_rescued=bool(matched and (not direct)), murcko_equal=bool(equal))
    return result
