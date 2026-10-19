from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scaffold_metrics import scaffold_checks, protocol_metadata
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def strip_dummy_atoms(scaffold_smiles: str) -> Optional[str]:
    if not scaffold_smiles:
        return None
    mol = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    if mol is None:
        return None
    try:
        dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if not dummy_indices:
            return Chem.MolToSmiles(mol, canonical=True)
        rw = Chem.RWMol(mol)
        for idx in dummy_indices:
            rw.ReplaceAtom(idx, Chem.Atom(1))
        hydrogenated = rw.GetMol()
        Chem.SanitizeMol(hydrogenated)
        core = Chem.RemoveHs(hydrogenated)
        Chem.SanitizeMol(core)
        if core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return None

def atom_token(atom) -> str:
    charge = {0: '', 1: '+', -1: '-'}.get(atom.GetFormalCharge())
    if charge is None:
        raise ValueError(f'Unsupported formal charge: {atom.GetFormalCharge()}')
    valence = atom.GetExplicitValence() + atom.GetImplicitValence()
    return f'{valence}{atom.GetSymbol()}{charge}'

def vocabulary_words(records: Sequence[Dict], include_defaults: bool=True) -> List[str]:
    words = set()
    if include_defaults:
        from drugex.data.corpus.vocabulary import VocGraph
        words.update(VocGraph.defaultWords)
    for record in records:
        for key in ('canonical_smiles', 'scaffold_core_smiles'):
            mol = Chem.MolFromSmiles(record.get(key, ''), sanitize=True)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    continue
                try:
                    words.add(atom_token(atom))
                except ValueError:
                    continue
    return sorted(words)

def make_vocab(payload: Dict):
    from drugex.data.corpus.vocabulary import VocGraph
    return VocGraph(words=payload['words'], max_len=int(payload['max_len']), n_frags=int(payload['n_frags']))

def load_vocab(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return make_vocab(json.load(handle))

def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def load_records(path: Path) -> List[Dict]:
    import torch
    return list(torch.load(path, map_location='cpu'))

def prepare_record(record: Dict, split: str, row_index: int) -> Tuple[Optional[Dict], str]:
    source = canonicalize_smiles(record.get('canonical_smiles') or record.get('source_smiles', ''))
    if source is None:
        return (None, 'invalid_source')
    core = strip_dummy_atoms(record.get('scaffold_smiles', ''))
    if core is None:
        return (None, 'invalid_scaffold_core')
    source_mol = Chem.MolFromSmiles(source, sanitize=True)
    core_mol = Chem.MolFromSmiles(core, sanitize=True)
    if source_mol is None or core_mol is None:
        return (None, 'invalid_molecule')
    if not source_mol.HasSubstructMatch(core_mol):
        return (None, 'core_not_substructure')
    prepared = {'split': split, 'row_index': int(row_index), 'source_smiles': record.get('source_smiles', source), 'canonical_smiles': source, 'scaffold_smiles': record.get('scaffold_smiles', ''), 'scaffold_core_smiles': core, 'rooted_decoration_smiles': record.get('rooted_decoration_smiles', ''), 'anchor_z': int(record.get('anchor_z', 0))}
    return (prepared, 'ok')

def load_sa_scorer():
    import sys
    contrib_dir = getattr(RDConfig, 'RDContribDir', None)
    if contrib_dir:
        sa_dir = str(Path(contrib_dir) / 'SA_Score')
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer
    return sascorer

def molecule_metrics(smiles: str, sa_scorer=None) -> Dict[str, Optional[float]]:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {'canonical_smiles': None, 'qed': None, 'sas': None}
    mol = Chem.MolFromSmiles(canonical)
    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = None
    if sa_scorer is None:
        sa_scorer = load_sa_scorer()
    try:
        sas = float(sa_scorer.calculateScore(mol))
    except Exception:
        sas = None
    return {'canonical_smiles': canonical, 'qed': qed, 'sas': sas}

def pairwise_diversity(smiles: Sequence[str], radius: int=2, n_bits: int=2048, max_items: int=5000) -> float:
    if max_items and len(smiles) > max_items:
        step = max(1, len(smiles) // max_items)
        smiles = list(smiles)[::step][:max_items]
    fps = []
    for smiles_item in smiles:
        mol = Chem.MolFromSmiles(smiles_item)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))
    if len(fps) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for idx, fp in enumerate(fps):
        for other in fps[idx + 1:]:
            total += 1.0 - DataStructs.TanimotoSimilarity(fp, other)
            count += 1
    return total / max(count, 1)

def scaffold_match(generated: str, scaffold_core: str) -> bool:
    return scaffold_checks(generated, scaffold_core)['core_match']

def scaffold_retained(generated: str, scaffold_core: str) -> bool:
    return scaffold_checks(generated, scaffold_core)['core_retention']

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
