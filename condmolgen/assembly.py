from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for scaffold-decoration assembly.") from exc
    return Chem, AllChem


ATTACH_PREFIX = "[ATTACH_"
ATTACH_SUFFIX = "]"


def attachment_token(site_id: int) -> str:
    return f"{ATTACH_PREFIX}{site_id}{ATTACH_SUFFIX}"


def split_full_decoration_text(text: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    if not text:
        return result
    cursor = 0
    while True:
        start = text.find(ATTACH_PREFIX, cursor)
        if start == -1:
            break
        end = text.find(ATTACH_SUFFIX, start)
        if end == -1:
            break
        site_id = int(text[start + len(ATTACH_PREFIX):end])
        next_start = text.find(ATTACH_PREFIX, end + 1)
        fragment = text[end + 1:] if next_start == -1 else text[end + 1:next_start]
        result[site_id] = fragment.strip()
        if next_start == -1:
            break
        cursor = next_start
    return result


def format_decoration_targets(site_to_smiles: Dict[int, str]) -> str:
    parts = []
    for site_id in sorted(site_to_smiles):
        parts.append(f"{attachment_token(site_id)}{site_to_smiles[site_id]}")
    return "".join(parts)


def extract_single_attachment_record(smiles: str) -> Optional[Dict]:
    record = extract_scaffold_and_decorations(smiles)
    if record is None:
        return None
    if record.get("attachment_count") != 1:
        return None
    decoration_map = split_full_decoration_text(record["decoration_smiles"])
    if 1 not in decoration_map:
        return None
    plain_decoration = strip_single_attachment_dummy(decoration_map[1])
    if plain_decoration is None:
        return None
    return {
        **record,
        "single_decoration_smiles": decoration_map[1],
        "plain_decoration_smiles": plain_decoration,
    }


def assemble_single_attachment(scaffold_smiles: str, fragment_smiles: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    scaffold = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    fragment = Chem.MolFromSmiles(fragment_smiles, sanitize=True)
    if scaffold is None or fragment is None:
        return None

    scaffold_dummy_idx = None
    scaffold_attach_idx = None
    for atom in scaffold.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            scaffold_dummy_idx = atom.GetIdx()
            scaffold_attach_idx = neighbors[0].GetIdx()
            break
    if scaffold_dummy_idx is None or scaffold_attach_idx is None:
        return None

    frag_dummy_idx = None
    frag_attach_idx = None
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            frag_dummy_idx = atom.GetIdx()
            frag_attach_idx = neighbors[0].GetIdx()
            break
    if frag_dummy_idx is None or frag_attach_idx is None:
        return None

    combo = Chem.CombineMols(scaffold, fragment)
    rw = Chem.RWMol(combo)
    offset = scaffold.GetNumAtoms()
    rw.AddBond(scaffold_attach_idx, offset + frag_attach_idx, Chem.BondType.SINGLE)
    for idx in sorted([scaffold_dummy_idx, offset + frag_dummy_idx], reverse=True):
        rw.RemoveAtom(idx)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def strip_single_attachment_dummy(fragment_smiles: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    fragment = Chem.MolFromSmiles(fragment_smiles, sanitize=True)
    if fragment is None:
        return None

    rw = Chem.RWMol(fragment)
    dummy_idx = None
    attach_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            dummy_idx = atom.GetIdx()
            attach_idx = neighbors[0].GetIdx()
            break
    if dummy_idx is None or attach_idx is None:
        return None

    rw.RemoveAtom(dummy_idx)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def strip_single_attachment_dummy_with_root(fragment_smiles: str) -> Optional[Tuple[str, int]]:
    Chem, _ = _import_rdkit()
    fragment = Chem.MolFromSmiles(fragment_smiles, sanitize=True)
    if fragment is None:
        return None

    rw = Chem.RWMol(fragment)
    dummy_idx = None
    attach_idx = None
    anchor_z = None
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            dummy_idx = atom.GetIdx()
            attach_idx = neighbors[0].GetIdx()
            anchor_z = neighbors[0].GetAtomicNum()
            break
    if dummy_idx is None or attach_idx is None or anchor_z is None:
        return None

    rw.RemoveAtom(dummy_idx)
    if attach_idx > dummy_idx:
        attach_idx -= 1
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        rooted = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=attach_idx)
        rooted_mol = Chem.MolFromSmiles(rooted, sanitize=True)
        if rooted_mol is None:
            return None
        return rooted, int(anchor_z)
    except Exception:
        return None


def assemble_single_attachment_plain(scaffold_smiles: str, plain_fragment_smiles: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    scaffold = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    fragment = Chem.MolFromSmiles(plain_fragment_smiles, sanitize=True)
    if scaffold is None or fragment is None:
        return None
    if fragment.GetNumAtoms() == 0:
        return None

    scaffold_dummy_idx = None
    scaffold_attach_idx = None
    for atom in scaffold.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            scaffold_dummy_idx = atom.GetIdx()
            scaffold_attach_idx = neighbors[0].GetIdx()
            break
    if scaffold_dummy_idx is None or scaffold_attach_idx is None:
        return None

    combo = Chem.CombineMols(scaffold, fragment)
    rw = Chem.RWMol(combo)
    offset = scaffold.GetNumAtoms()
    fragment_attach_idx = offset + 0
    rw.AddBond(scaffold_attach_idx, fragment_attach_idx, Chem.BondType.SINGLE)
    rw.RemoveAtom(scaffold_dummy_idx)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def assemble_single_attachment_rooted(scaffold_smiles: str, rooted_fragment_smiles: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    scaffold = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    fragment = Chem.MolFromSmiles(rooted_fragment_smiles, sanitize=True)
    if scaffold is None or fragment is None:
        return None
    if fragment.GetNumAtoms() == 0:
        return None

    scaffold_dummy_idx = None
    scaffold_attach_idx = None
    for atom in scaffold.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                return None
            scaffold_dummy_idx = atom.GetIdx()
            scaffold_attach_idx = neighbors[0].GetIdx()
            break
    if scaffold_dummy_idx is None or scaffold_attach_idx is None:
        return None

    combo = Chem.CombineMols(scaffold, fragment)
    rw = Chem.RWMol(combo)
    offset = scaffold.GetNumAtoms()
    fragment_attach_idx = offset + 0
    rw.AddBond(scaffold_attach_idx, fragment_attach_idx, Chem.BondType.SINGLE)
    rw.RemoveAtom(scaffold_dummy_idx)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def strip_scaffold_dummy(scaffold_smiles: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    mol = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummy_indices:
        try:
            Chem.SanitizeMol(rw)
            return Chem.MolToSmiles(rw, canonical=True)
        except Exception:
            return None
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    try:
        core = rw.GetMol()
        Chem.SanitizeMol(core)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return None


def _find_attachment_sites(scaffold_mol) -> List[Tuple[int, int]]:
    Chem, _ = _import_rdkit()
    sites = []
    for atom in scaffold_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                continue
            sites.append((atom.GetIdx(), neighbors[0].GetIdx()))
    return sites


def _renumber_scaffold_dummy_atoms(scaffold_mol) -> Tuple[object, Dict[int, int]]:
    Chem, _ = _import_rdkit()
    scaffold = Chem.RWMol(scaffold_mol)
    mapping: Dict[int, int] = {}
    site_id = 1
    for atom in scaffold.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(site_id)
            atom.SetIsotope(site_id)
            mapping[site_id] = atom.GetIdx()
            site_id += 1
    return scaffold.GetMol(), mapping


def _single_site_fragment_from_bond(mol, keep_atom_idx: int, cut_atom_idx: int, site_id: int):
    Chem, _ = _import_rdkit()
    rw = Chem.RWMol(mol)
    dummy_idx = rw.AddAtom(Chem.Atom(0))
    dummy_atom = rw.GetAtomWithIdx(dummy_idx)
    dummy_atom.SetAtomMapNum(site_id)
    dummy_atom.SetIsotope(site_id)

    bond = rw.GetBondBetweenAtoms(keep_atom_idx, cut_atom_idx)
    if bond is None:
        return None
    bond_type = bond.GetBondType()
    rw.RemoveBond(keep_atom_idx, cut_atom_idx)
    rw.AddBond(cut_atom_idx, dummy_idx, bond_type)
    fragment_mol = rw.GetMol()

    frags = Chem.GetMolFrags(fragment_mol, asMols=True, sanitizeFrags=False)
    candidate = None
    for frag in frags:
        has_dummy = any(atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == site_id for atom in frag.GetAtoms())
        if has_dummy:
            candidate = frag
            break
    if candidate is None:
        return None
    try:
        Chem.SanitizeMol(candidate)
    except Exception:
        try:
            candidate.UpdatePropertyCache(strict=False)
        except Exception:
            return None
    return candidate


def extract_scaffold_and_decorations(smiles: str) -> Optional[Dict]:
    Chem, AllChem = _import_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None

    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None

    scaffold_with_dummies = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    _ = scaffold_with_dummies

    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    scaffold_core = GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold_core)
    if not match:
        return None

    scaffold_atom_set = set(match)
    bonds_to_cut = []
    site_neighbors = []
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in scaffold_atom_set:
                bonds_to_cut.append((atom_idx, nbr_idx))
                site_neighbors.append((atom_idx, nbr_idx))

    if not bonds_to_cut:
        return None

    if len({nbr for _, nbr in site_neighbors}) != len(site_neighbors):
        return None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    mol_3d = Chem.RemoveHs(mol_h)

    bond_indices = []
    for a, b in bonds_to_cut:
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return None
        bond_indices.append(bond.GetIdx())

    try:
        fragmented = Chem.FragmentOnBonds(mol, bondIndices=bond_indices, addDummies=True)
    except Exception:
        return None

    fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    scaffold_fragment = None
    for frag in fragments:
        heavy_atoms = [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() > 0]
        if len(heavy_atoms) == scaffold_core.GetNumAtoms():
            scaffold_fragment = frag
            break
    if scaffold_fragment is None:
        return None

    scaffold_fragment, _ = _renumber_scaffold_dummy_atoms(scaffold_fragment)
    scaffold_smiles = Chem.MolToSmiles(scaffold_fragment, canonical=True)

    decoration_map: Dict[int, str] = {}
    site_id = 1
    for scaffold_atom_idx, neighbor_idx in site_neighbors:
        frag = _single_site_fragment_from_bond(mol, scaffold_atom_idx, neighbor_idx, site_id)
        if frag is None:
            return None
        frag_smiles = Chem.MolToSmiles(frag, canonical=True)
        decoration_map[site_id] = frag_smiles
        site_id += 1

    conf = mol_3d.GetConformer()
    cond_z = []
    cond_pos = []
    for atom_idx in match:
        atom = mol_3d.GetAtomWithIdx(atom_idx)
        pos = conf.GetAtomPosition(atom_idx)
        cond_z.append(int(atom.GetAtomicNum()))
        cond_pos.append([float(pos.x), float(pos.y), float(pos.z)])

    return {
        "source_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "cond_smiles": scaffold_smiles,
        "scaffold_smiles": scaffold_smiles,
        "cond_z": cond_z,
        "cond_pos": cond_pos,
        "decoration_smiles": format_decoration_targets(decoration_map),
        "attachment_count": len(decoration_map),
    }


def assemble_scaffold_and_decorations(scaffold_smiles: str, decoration_text: str) -> Optional[str]:
    Chem, _ = _import_rdkit()
    decoration_map = split_full_decoration_text(decoration_text)
    scaffold = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
    if scaffold is None:
        return None

    rw = Chem.RWMol(scaffold)
    site_to_dummy = {}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            site_id = atom.GetAtomMapNum() or atom.GetIsotope()
            if site_id:
                site_to_dummy[site_id] = atom.GetIdx()

    for site_id, dummy_idx in sorted(site_to_dummy.items()):
        fragment_smiles = decoration_map.get(site_id)
        if not fragment_smiles:
            continue
        fragment = Chem.MolFromSmiles(fragment_smiles, sanitize=True)
        if fragment is None:
            return None

        frag_dummy_idx = None
        frag_attach_idx = None
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0 and (atom.GetAtomMapNum() == site_id or atom.GetIsotope() == site_id):
                frag_dummy_idx = atom.GetIdx()
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) != 1:
                    return None
                frag_attach_idx = neighbors[0].GetIdx()
                break
        if frag_dummy_idx is None or frag_attach_idx is None:
            return None

        combo = Chem.CombineMols(rw.GetMol(), fragment)
        combo_rw = Chem.RWMol(combo)
        offset = rw.GetNumAtoms()

        scaffold_dummy = combo_rw.GetAtomWithIdx(dummy_idx)
        scaffold_neighbors = list(scaffold_dummy.GetNeighbors())
        if len(scaffold_neighbors) != 1:
            return None
        scaffold_attach_idx = scaffold_neighbors[0].GetIdx()

        combo_rw.AddBond(scaffold_attach_idx, offset + frag_attach_idx, Chem.BondType.SINGLE)
        remove_indices = sorted([dummy_idx, offset + frag_dummy_idx], reverse=True)
        for idx in remove_indices:
            combo_rw.RemoveAtom(idx)
        rw = combo_rw

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
