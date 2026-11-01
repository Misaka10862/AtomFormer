"""Task gates and property calculations shared by every aligned optimizer."""
from dataclasses import asdict, dataclass
from statistics import median

from scaffold_metrics import scaffold_checks, scaffold_core_from_scaffold
from .io import digest


@dataclass(frozen=True)
class RewardSpec:
    task: str = 'murcko'
    protocol: str = 'aligned'
    method: str = 'atomformer'
    qed_weight: float = 1.0
    novelty_weight: float = 0.2
    sas_weight: float = 0.2
    failure: float = -2.0
    repetition_weight: float = 0.0

    def __post_init__(self):
        if self.task not in ('murcko', 'pocket') or self.protocol not in ('aligned', 'native'):
            raise ValueError('Unknown task/reward protocol')
        if self.method not in ('atomformer', 'libinvent', 'promptsmiles', 'drugex', 'diffdec'):
            raise ValueError('Unknown method')


def canonical(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def properties(smiles):
    from rdkit import Chem
    from rdkit.Chem import QED
    from rdkit.Contrib.SA_Score import sascorer
    molecule = Chem.MolFromSmiles(smiles)
    return float(QED.qed(molecule)), float(sascorer.calculateScore(molecule))


def retained_subgraph(smiles, core_smiles):
    """Induced scaffold graph, charge and specified stereochemistry preserved."""
    from rdkit import Chem
    mol, core = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(core_smiles)
    if mol is None or core is None:
        return False
    for match in mol.GetSubstructMatches(core, useChirality=True, uniquify=False):
        if any(mol.GetAtomWithIdx(j).GetFormalCharge() != core.GetAtomWithIdx(i).GetFormalCharge() for i, j in enumerate(match)):
            continue
        inside = set(match)
        if sum(b.GetBeginAtomIdx() in inside and b.GetEndAtomIdx() in inside for b in mol.GetBonds()) == core.GetNumBonds():
            return True
    return False


def score(smiles, condition, reference, spec=RewardSpec(), repeats=0):
    row = {'canonical_smiles': None, 'valid': False, 'core_match': False,
           'core_retention': False, 'qed': None, 'sas': None, 'novelty': 0.0,
           'reward': spec.failure, 'gate_passed': False,
           'failure_reason': None, 'reward_protocol': asdict(spec)}
    try:
        canon = canonical(smiles)
        row.update(canonical_smiles=canon, valid=bool(canon))
        if canon is None:
            row['failure_reason'] = 'invalid_or_assembly_failure'
            return row
        core = condition.get('core_smiles') or scaffold_core_from_scaffold(condition['scaffold_smiles'])
        if spec.task == 'pocket':
            retained = retained_subgraph(canon, core)
            row.update(core_match=retained, core_retention=retained)
        else:
            checks = scaffold_checks(canon, core)
            row.update(core_match=checks['core_match'], core_retention=checks['core_retention'])
        qed, sas = properties(canon)
        import math
        if not all(math.isfinite(v) for v in (qed, sas)):
            raise ValueError('Non-finite property')
        row.update(qed=qed, sas=sas, novelty=float(canon not in reference))
        strict = spec.protocol == 'aligned' or spec.method == 'atomformer'
        gate = row['core_match'] and (row['core_retention'] if strict else True)
        row['gate_passed'] = bool(gate)
        if not gate:
            row['failure_reason'] = 'core_match_failure' if not row['core_match'] else 'exact_core_failure'
        elif spec.protocol == 'native' and spec.method == 'drugex':
            row['reward'] = None
            row['pareto_objectives'] = [qed, (10.0 - sas) / 9.0]
        else:
            row['reward'] = (spec.qed_weight * qed + spec.novelty_weight * row['novelty']
                             - spec.sas_weight * sas - spec.repetition_weight * repeats)
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        row.update(reward=spec.failure, gate_passed=False, failure_reason='scoring_exception',
                   scoring_error_type=type(error).__name__, qed=None, sas=None)
    return row


def training_reference(records):
    records = list(records)
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('Novelty reference requires the complete training split')
    values = {canonical(r['canonical_smiles']) for r in records}
    if None in values:
        raise ValueError('Invalid molecule in training reference')
    values = sorted(values)
    return set(values), digest(values)


def training_thresholds(records):
    records = list(records)
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('Thresholds must be fitted on training only')
    values = [properties(r['canonical_smiles']) for r in records]
    return {'qed_min': median(q for q, s in values), 'sas_max': median(s for q, s in values),
            'source': 'training_medians', 'record_ids_sha256': digest(sorted(r['record_id'] for r in records))}


def augmented_score(reward):
    return max(0.0, min(1.0, (reward + 2.0) / 3.0))
