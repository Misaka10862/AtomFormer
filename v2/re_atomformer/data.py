"""Condition-only inference records and chemically audited single-site extraction."""
from dataclasses import dataclass, fields
import hashlib
from typing import List, Optional

import torch

from .protocol import canonical
from scaffold_metrics import scaffold_core_from_scaffold

BOND_TYPES = ('NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')


@dataclass
class Condition:
    record_id: str
    scaffold_smiles: str
    core_smiles: str
    z: list
    bonds: list
    charges: list
    aromatic: list
    anchor_index: int
    legacy_atom_z: Optional[list] = None
    pos: Optional[list] = None
    conformers: Optional[list] = None
    geometry_source: str = 'none'
    pocket_z: Optional[list] = None
    pocket_pos: Optional[list] = None
    pocket_id: Optional[str] = None
    mapped_core_smiles: Optional[str] = None

    def __post_init__(self):
        n = len(self.z)
        if not n or not 0 <= self.anchor_index < n:
            raise ValueError('Invalid scaffold/attachment atom')
        if len(self.bonds) != n or any(len(r) != n for r in self.bonds):
            raise ValueError('Bond matrix shape differs from scaffold')
        if len(self.charges) != n or len(self.aromatic) != n:
            raise ValueError('Atom feature shape mismatch')
        if self.pos is not None and (len(self.pos) != n or any(len(p) != 3 for p in self.pos)):
            raise ValueError('Coordinate shape mismatch')
        if len(self.pocket_z or []) != len(self.pocket_pos or []):
            raise ValueError('Pocket coordinate/type mismatch')

    @classmethod
    def from_record(cls, row):
        # Whitelisting makes target leakage impossible at the public inference boundary.
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in names})


def molecule_features(mol):
    from rdkit import Chem
    n = mol.GetNumAtoms()
    bonds = [[0]*n for _ in range(n)]
    for b in mol.GetBonds():
        kind = BOND_TYPES.index(str(b.GetBondType()))
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds[i][j] = bonds[j][i] = kind
    return {'z': [a.GetAtomicNum() for a in mol.GetAtoms()], 'bonds': bonds,
            'charges': [a.GetFormalCharge() for a in mol.GetAtoms()],
            'aromatic': [int(a.GetIsAromatic()) for a in mol.GetAtoms()]}


def embed(mol, seed, count=1):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=count, params=params))
    conformers, statuses = [], []
    for cid in ids:
        try:
            status = int(AllChem.UFFOptimizeMolecule(m, confId=cid))
        except (ValueError, RuntimeError):
            status = 'exception_retained'
        conformers.append(m.GetConformer(cid).GetPositions()[:mol.GetNumAtoms()].tolist())
        statuses.append(status)
    return conformers, statuses


def extract_record(smiles, dataset='chembl37', seed=1729, geometry='none', conformer_count=1):
    """2D acceptance does not depend on ETKDG; optional geometry is separately audited."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from condmolgen.v2_assembly import strip_single_attachment_dummy_with_root, assemble_single_attachment_rooted
    canon = canonical(smiles)
    if not canon:
        raise ValueError('invalid_molecule')
    mol = Chem.MolFromSmiles(canon)
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('disconnected_or_radical')
    if any(a.GetAtomicNum() not in (6,7,8,9,15,16,17,35,53) for a in mol.GetAtoms()):
        raise ValueError('unsupported_element')
    if mol.GetNumHeavyAtoms() > 60:
        raise ValueError('heavy_atom_limit')
    framework = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(framework, useChirality=True)
    if not match:
        raise ValueError('empty_scaffold')
    inside = set(match)
    cuts = [b for b in mol.GetBonds() if (b.GetBeginAtomIdx() in inside) != (b.GetEndAtomIdx() in inside)]
    if len(cuts) != 1 or cuts[0].GetBondType() != Chem.BondType.SINGLE:
        raise ValueError('not_single_attachment')
    cut = cuts[0]
    core_parent_anchor = cut.GetBeginAtomIdx() if cut.GetBeginAtomIdx() in inside else cut.GetEndAtomIdx()
    deco_parent_anchor = cut.GetOtherAtomIdx(core_parent_anchor)
    fragmented = Chem.FragmentOnBonds(mol, [cut.GetIdx()], addDummies=True, dummyLabels=[(1,1)])
    mappings = []
    fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mappings)
    ci = next(i for i, mapping in enumerate(mappings) if core_parent_anchor in mapping)
    di = 1-ci
    scaf_smi = Chem.MolToSmiles(fragments[ci], canonical=True)
    deco_smi = Chem.MolToSmiles(fragments[di], canonical=True)
    rooted = strip_single_attachment_dummy_with_root(deco_smi)
    if rooted is None:
        raise ValueError('decoration_root_failure')
    target, anchor_z = rooted
    rebuilt = assemble_single_attachment_rooted(scaf_smi, target)
    if canonical(rebuilt) != canon:
        raise ValueError('stereo_roundtrip_failure')
    # Preserve atom order explicitly while replacing dummy with H and removing H.
    core = Chem.Mol(fragments[ci])
    for a in core.GetAtoms():
        a.SetIntProp('_orig_index', mappings[ci][a.GetIdx()])
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1); a.SetIsotope(0)
    core = Chem.RemoveHs(core)
    Chem.SanitizeMol(core)
    parent_indices = [a.GetIntProp('_orig_index') for a in core.GetAtoms()]
    anchor = parent_indices.index(core_parent_anchor)
    core_smiles = Chem.MolToSmiles(core, canonical=True)
    mapped = Chem.Mol(core)
    for i, atom in enumerate(mapped.GetAtoms()):
        atom.SetAtomMapNum(i+1)
    mapped_core_smiles = Chem.MolToSmiles(mapped, canonical=True)
    record_id = hashlib.sha256((dataset+'\0'+canon+'\0'+scaf_smi).encode()).hexdigest()
    row = {'schema_version':1, 'record_id':record_id, 'dataset_id':dataset,
           'canonical_smiles':canon, 'scaffold_smiles':scaf_smi, 'core_smiles':core_smiles,
           'scaffold_group':Chem.MolToSmiles(framework, isomericSmiles=False),
           'score_core':scaffold_core_from_scaffold(scaf_smi),
           'mapped_core_smiles':mapped_core_smiles, 'anchor_index':anchor, 'attachment_element':core.GetAtomWithIdx(anchor).GetAtomicNum(),
           'anchor_z':anchor_z, 'rooted_decoration_smiles':target,
           'decoration_size':mol.GetNumHeavyAtoms()-core.GetNumHeavyAtoms(),
           'geometry_source':geometry, 'geometry_status':'not_requested',
           'parent_atom_indices':parent_indices, **molecule_features(core)}
    if geometry != 'none':
        source_mol = mol if geometry == 'parent' else core
        if geometry not in ('parent','independent','multi'):
            raise ValueError('Unknown geometry source')
        confs, statuses = embed(source_mol, seed, conformer_count)
        row['geometry_status'] = 'success' if confs else 'embedding_failed'
        row['uff_status'] = statuses
        if confs:
            row['conformers'] = [[[p[j][k] for k in range(3)] for j in parent_indices] for p in confs] if geometry == 'parent' else confs
            row['pos'] = row['conformers'][0]
            if geometry == 'parent':
                outside = [i for i in range(mol.GetNumAtoms()) if i not in inside]
                # Attachment atom first, followed by original order; never exposed to inference.
                outside.remove(deco_parent_anchor); outside.insert(0,deco_parent_anchor)
                order = parent_indices + outside
                full = molecule_features(mol)
                row['target_3d'] = {'z':[full['z'][i] for i in outside],
                                    'pos':[confs[0][i] for i in outside],
                                    'bonds':[[full['bonds'][i][j] for j in order] for i in order]}
    Condition.from_record(row)
    return row


def collate_conditions(conditions: List[Condition], device='cpu', pocket_mode='correct', max_pocket_atoms=256):
    if not conditions:
        raise ValueError('Empty condition batch')
    b, n = len(conditions), max(len(c.z) for c in conditions)
    p = min(max((len(c.pocket_z or []) for c in conditions), default=0), max_pocket_atoms)
    z = torch.zeros(b,n,dtype=torch.long)
    bonds = torch.zeros(b,n,n,dtype=torch.long)
    charges = torch.zeros(b,n,dtype=torch.long)
    aromatic = torch.zeros(b,n,dtype=torch.long)
    anchor = torch.zeros(b,n,dtype=torch.bool)
    pos = torch.zeros(b,n,3)
    geom = torch.zeros(b,dtype=torch.bool)
    pz, pp = torch.zeros(b,p,dtype=torch.long), torch.zeros(b,p,3)
    for i,c in enumerate(conditions):
        size = len(c.z)
        z[i,:size], bonds[i,:size,:size] = torch.tensor(c.z),torch.tensor(c.bonds)
        charges[i,:size], aromatic[i,:size] = torch.tensor(c.charges),torch.tensor(c.aromatic)
        anchor[i,c.anchor_index] = True
        if c.pos is not None:
            pos[i,:size] = torch.tensor(c.pos); geom[i] = True
        if p and c.pocket_z and pocket_mode != 'none':
            pc = torch.tensor(c.pocket_pos)
            order = torch.argsort(((pc-pos[i,c.anchor_index])**2).sum(-1))[:p]
            k = len(order)
            pz[i,:k], pp[i,:k] = torch.tensor(c.pocket_z)[order],pc[order]
    if pocket_mode == 'shuffle':
        if b < 2:
            raise ValueError('Pocket shuffle requires at least two distinct conditions')
        centers = (pos*anchor.unsqueeze(-1)).sum(1)
        pp = (pp-centers[:,None,:]).roll(1,0) + centers[:,None,:]
        pz = pz.roll(1,0)
    elif pocket_mode not in ('correct','none'):
        raise ValueError('Unknown pocket intervention')
    result = {'z':z,'bonds':bonds,'charges':charges,'aromatic':aromatic,'anchor':anchor,
              'pos':pos,'geometry_available':geom,'pocket_z':pz,'pocket_pos':pp,'mask':z.ne(0)}
    if any(c.legacy_atom_z is not None for c in conditions):
        if not all(c.legacy_atom_z is not None and len(c.legacy_atom_z)==len(c.z) for c in conditions):
            raise ValueError('Mixed or invalid historical atom order')
        historical=torch.zeros_like(z)
        for i,c in enumerate(conditions):historical[i,:len(c.z)]=torch.tensor(c.legacy_atom_z)
        result['legacy_atom_z']=historical
    return {k:v.to(device) for k,v in result.items()}


def collate_supervised(rows, tokenizer, device='cpu', **kwargs):
    from condmolgen.v4_data import encode_v4_sequence
    result = collate_conditions([Condition.from_record(r) for r in rows],device,**kwargs)
    encoded = [encode_v4_sequence(tokenizer,r['scaffold_smiles'],r['rooted_decoration_smiles']) for r in rows]
    if any(tokenizer.unk_token_id in e['input_ids'] for e in encoded):
        raise ValueError('Out-of-training-vocabulary record; record unsupported instead of silent substitution')
    t = max(len(e['input_ids']) for e in encoded)
    ids = torch.full((len(rows),t),tokenizer.pad_token_id,dtype=torch.long,device=device)
    labels = torch.full_like(ids,-100)
    for i,e in enumerate(encoded):
        ids[i,:len(e['input_ids'])] = torch.tensor(e['input_ids'],device=device)
        labels[i,:len(e['labels'])] = torch.tensor(e['labels'],device=device)
    result.update(input_ids=ids,labels=labels,anchor_z=torch.tensor([r['anchor_z'] for r in rows],device=device))
    return add_scaffold_tokens(result, [r['scaffold_smiles'] for r in rows], tokenizer)


def add_scaffold_tokens(batch, scaffolds, tokenizer):
    """Never derive encoder input by slicing a teacher-forcing sequence."""
    encoded = [tokenizer.encode_smiles(s) for s in scaffolds]
    if any(not e or tokenizer.unk_token_id in e for e in encoded):
        raise ValueError('Empty or out-of-vocabulary scaffold')
    tokens = torch.full((len(encoded), max(map(len, encoded))), tokenizer.pad_token_id,
                        dtype=torch.long, device=batch['z'].device)
    for i, values in enumerate(encoded):
        tokens[i, :len(values)] = torch.tensor(values, device=tokens.device)
    return dict(batch, scaffold_input_ids=tokens)
