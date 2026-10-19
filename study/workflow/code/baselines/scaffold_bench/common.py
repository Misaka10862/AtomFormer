import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scaffold_metrics import scaffold_checks, protocol_metadata

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def canonical(s):
    from rdkit import Chem
    if not s:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None or not m.GetNumAtoms() or any((a.GetAtomicNum() == 0 for a in m.GetAtoms())):
        return None
    return Chem.MolToSmiles(m, canonical=True)

def normalize_dummy(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError('unparseable scaffold')
    d = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(d) != 1 or d[0].GetDegree() != 1:
        raise ValueError('expected one terminal attachment dummy')
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
    return Chem.MolToSmiles(m, canonical=True)

def core_smiles(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    rw = Chem.RWMol(m)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            rw.ReplaceAtom(a.GetIdx(), Chem.Atom(1))
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m), canonical=True)

def decoration_with_dummy(rooted, consume_h=False):
    from rdkit import Chem
    m = Chem.MolFromSmiles(rooted)
    if m is None or not m.GetNumAtoms():
        raise ValueError('invalid rooted decoration')
    if any((a.GetAtomicNum() == 0 for a in m.GetAtoms())):
        raise ValueError('rooted decoration must not contain an attachment dummy')
    rw = Chem.RWMol(m)
    anchor = rw.GetAtomWithIdx(0)
    if consume_h:
        if anchor.GetNumExplicitHs() < 1:
            raise ValueError('no explicit attachment hydrogen')
        anchor.SetNumExplicitHs(anchor.GetNumExplicitHs() - 1)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(dummy, 0, Chem.BondType.SINGLE)
    result = rw.GetMol()
    Chem.SanitizeMol(result)
    return normalize_dummy(Chem.MolToSmiles(result))

def join(scaffold, decoration):
    from rdkit import Chem
    try:
        a, b = (Chem.MolFromSmiles(scaffold), Chem.MolFromSmiles(decoration))
        if a is None or b is None or (not b.GetNumAtoms()):
            return None
        scaffold_dummies = [x for x in a.GetAtoms() if x.GetAtomicNum() == 0]
        decoration_dummies = [x for x in b.GetAtoms() if x.GetAtomicNum() == 0]
        if len(scaffold_dummies) != 1 or scaffold_dummies[0].GetDegree() != 1:
            return None
        if len(decoration_dummies) != 1 or decoration_dummies[0].GetDegree() != 1:
            return None
        offset = a.GetNumAtoms()
        combo = Chem.CombineMols(a, b)
        sd = scaffold_dummies[0].GetIdx()
        dd = offset + decoration_dummies[0].GetIdx()
        sa = scaffold_dummies[0].GetNeighbors()[0].GetIdx()
        da = offset + decoration_dummies[0].GetNeighbors()[0].GetIdx()
        replacement = {sd: da, dd: sa}
        removed = sorted((sd, dd))

        def mapped(i):
            i = replacement.get(i, i)
            return i - sum((x < i for x in removed))
        tetra = [(x.GetIdx(), x.GetChiralTag(), [n.GetIdx() for n in x.GetNeighbors()]) for x in combo.GetAtoms() if x.GetIdx() not in removed]
        double = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetStereo(), list(bond.GetStereoAtoms())) for bond in combo.GetBonds() if bond.GetStereoAtoms() and bond.GetBeginAtomIdx() not in removed and (bond.GetEndAtomIdx() not in removed)]
        source_bond = combo.GetBondBetweenAtoms(sd, sa)
        rw = Chem.RWMol(combo)
        bridge_begin = replacement.get(source_bond.GetBeginAtomIdx(), source_bond.GetBeginAtomIdx())
        bridge_end = replacement.get(source_bond.GetEndAtomIdx(), source_bond.GetEndAtomIdx())
        rw.AddBond(bridge_begin, bridge_end, Chem.BondType.SINGLE)
        rw.GetBondBetweenAtoms(bridge_begin, bridge_end).SetBondDir(source_bond.GetBondDir())
        for i in reversed(removed):
            rw.RemoveAtom(i)
        m = rw.GetMol()
        for old_i, tag, neighbours in tetra:
            atom = m.GetAtomWithIdx(mapped(old_i))
            atom.SetChiralTag(tag)
            if tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                logical = [mapped(i) for i in neighbours]
                actual = [n.GetIdx() for n in atom.GetNeighbors()]
                if set(logical) != set(actual):
                    raise ValueError('Attachment stereo neighbours differ')
                order = [logical.index(i) for i in actual]
                if sum((order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))) % 2:
                    atom.InvertChirality()
        for begin, end, stereo, neighbours in double:
            bond = m.GetBondBetweenAtoms(mapped(begin), mapped(end))
            bond.SetStereoAtoms(*[mapped(i) for i in neighbours])
            bond.SetStereo(stereo)
        Chem.SanitizeMol(m)
        Chem.SetDoubleBondNeighborDirections(m)
        return canonical(Chem.MolToSmiles(m))
    except (ValueError, RuntimeError):
        return None

def same_molecule(left, right):
    from rdkit import Chem
    if not left or not right:
        return False
    if left == right:
        return True
    a, b = (Chem.MolFromSmiles(left), Chem.MolFromSmiles(right))
    if a is None or b is None:
        return False

    def labels(m):
        return sorted(((x.GetAtomicNum(), x.GetIsotope(), x.GetFormalCharge(), x.GetTotalNumHs(), x.GetNumRadicalElectrons()) for x in m.GetAtoms()))
    return a.GetNumAtoms() == b.GetNumAtoms() and a.GetNumBonds() == b.GetNumBonds() and (labels(a) == labels(b)) and a.HasSubstructMatch(b, useChirality=True) and b.HasSubstructMatch(a, useChirality=True)

def derive_decoration(source, scaffold):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(source)
    query = Chem.MolFromSmarts(scaffold)
    template = Chem.MolFromSmiles(scaffold)
    dummy = next((x.GetIdx() for x in template.GetAtoms() if x.GetAtomicNum() == 0))
    anchor = template.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    candidates = set()
    for match in mol.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=4096):
        core = {idx for i, idx in enumerate(match) if i != dummy}
        boundary = [b for b in mol.GetBonds() if (b.GetBeginAtomIdx() in core) != (b.GetEndAtomIdx() in core)]
        if len(boundary) != 1 or boundary[0].GetBondType() != Chem.BondType.SINGLE:
            continue
        if {boundary[0].GetBeginAtomIdx(), boundary[0].GetEndAtomIdx()} != {match[anchor], match[dummy]}:
            continue
        rw = Chem.RWMol(mol)
        attach = match[dummy] - sum((i < match[dummy] for i in core))
        for i in sorted(core, reverse=True):
            rw.RemoveAtom(i)
        label = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(attach, label, Chem.BondType.SINGLE)
        try:
            fragment = rw.GetMol()
            Chem.SanitizeMol(fragment)
            candidate = normalize_dummy(Chem.MolToSmiles(fragment))
            if same_molecule(join(scaffold, candidate), source):
                candidates.add(candidate)
        except (ValueError, RuntimeError):
            continue
    if not candidates:
        raise ValueError(f'cannot recover a single-attachment complement: source={source!r}, scaffold={scaffold!r}')
    return (sorted(candidates)[0], len(candidates))

def convert(item):
    index, record = item
    source = canonical(record.get('canonical_smiles') or record.get('source_smiles'))
    if source is None:
        raise ValueError(f'invalid source record {index}')
    scaff = normalize_dummy(record['scaffold_smiles'])
    original = record.get('single_decoration_smiles')
    if not original:
        tagged = record.get('decoration_smiles', '')
        if tagged.startswith('[ATTACH_1]') and tagged.count('[ATTACH_') == 1:
            original = tagged[len('[ATTACH_1]'):]
    if original:
        dec = normalize_dummy(original)
    else:
        dec = None
        for consume_h in (False, True):
            try:
                candidate = decoration_with_dummy(record['rooted_decoration_smiles'], consume_h)
                if same_molecule(join(scaff, candidate), source):
                    dec = candidate
                    break
            except (ValueError, RuntimeError):
                pass
        if dec is None:
            raise ValueError(f'cannot restore labelled decoration record {index}')
    assembled = join(scaff, dec)
    origin, candidates = ('stored_or_rooted', 1)
    if not same_molecule(assembled, source):
        dec, candidates = derive_decoration(source, scaff)
        origin = 'source_graph_complement'
    return {'record_id': index, 'source': source, 'scaffold': scaff, 'core': core_smiles(scaff), 'decoration': dec, 'decoration_origin': origin, 'complement_candidates': candidates}
_SA = None

def score(item):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, RDConfig
    import sys
    global _SA
    RDLogger.DisableLog('rdApp.*')
    if _SA is None:
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / 'SA_Score'))
        import sascorer
        _SA = sascorer
    row, output, decoration, sample_id = item
    if output == '__assemble__':
        output = join(row['scaffold'], decoration)
    smi = canonical(output)
    result = {'record_id': row['record_id'], 'sample_id': sample_id, 'scaffold': row['scaffold'], 'generated': smi, 'raw_output': output, 'decoration': decoration, 'valid': smi is not None, 'core_match': False, 'core_retention': False, 'qed': None, 'sas': None}
    if smi:
        m = Chem.MolFromSmiles(smi)
        checks = scaffold_checks(smi, row['core'])
        if not checks['eligible']:
            raise ValueError(f"invalid target core for valid product: record {row['record_id']}")
        result.update(core_match=checks['core_match'], core_retention=checks['core_retention'], core_match_kekule_rescued=checks['kekule_rescued'], qed=float(QED.qed(m)), sas=float(_SA.calculateScore(m)))
    result.update(protocol_metadata())
    return result

def reward(result, reference):
    if not result['valid'] or not result['core_match'] or result['qed'] is None or (result['sas'] is None):
        return -2.0
    return result['qed'] + 0.2 * (result['generated'] not in reference) - 0.2 * result['sas']

def augmented_likelihood_score(result, reference):
    return max(0.0, min(1.0, (reward(result, reference) + 2.0) / 3.0))

def validate_attempts(rows, records, samples):
    expected = records * samples
    seen = bytearray(expected)
    count = 0
    for r in rows:
        i, j = (r['record_id'], r['sample_id'])
        if not (0 <= i < records and 0 <= j < samples):
            raise ValueError('attempt id out of range')
        k = i * samples + j
        if seen[k]:
            raise ValueError('duplicate attempt')
        seen[k] = 1
        count += 1
    if count != expected:
        raise ValueError(f'missing attempts: {count}/{expected}')
    return count
