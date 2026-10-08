from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _import_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import QED, AllChem, RDConfig
        from rdkit.Chem.Scaffolds import MurckoScaffold
        import os
        import sys

        contrib_dir = getattr(RDConfig, "RDContribDir", None)
        if contrib_dir:
            sa_dir = os.path.join(contrib_dir, "SA_Score")
            if sa_dir not in sys.path:
                sys.path.append(sa_dir)
        import sascorer
    except ImportError as exc:
        raise RuntimeError("RDKit with SA_Score support is required for metrics.") from exc
    return Chem, DataStructs, QED, AllChem, MurckoScaffold, sascorer


def canonicalize_smiles(smiles: str) -> Optional[str]:
    Chem, _, _, _, _, _ = _import_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles: str):
    Chem, _, _, _, _, _ = _import_rdkit()
    return Chem.MolFromSmiles(smiles)


def validity(smiles_list: Sequence[str]) -> Tuple[List[Optional[str]], float]:
    canonical = [canonicalize_smiles(smiles) for smiles in smiles_list]
    valid_count = sum(item is not None for item in canonical)
    rate = valid_count / max(len(smiles_list), 1)
    return canonical, rate


def qed_scores(valid_smiles: Sequence[str]) -> List[float]:
    Chem, _, QED, _, _, _ = _import_rdkit()
    scores = []
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scores.append(float(QED.qed(mol)))
    return scores


def sas_scores(valid_smiles: Sequence[str]) -> List[float]:
    Chem, _, _, _, _, sascorer = _import_rdkit()
    scores = []
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                score = float(sascorer.calculateScore(mol))
            except Exception:
                continue
            if score == score:
                scores.append(score)
    return scores


def _fingerprints(valid_smiles: Sequence[str]):
    Chem, _, _, AllChem, _, _ = _import_rdkit()
    fps = []
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def diversity_score(valid_smiles: Sequence[str]) -> float:
    _, DataStructs, _, _, _, _ = _import_rdkit()
    # Avoid quadratic blow-up on multi-sample test runs; deterministic cap.
    if len(valid_smiles) > 5000:
        step = max(1, len(valid_smiles) // 5000)
        valid_smiles = list(valid_smiles)[::step][:5000]
    fps = _fingerprints(valid_smiles)
    n = len(fps)
    if n <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            count += 1
    return total / max(count, 1)


def novelty_score(valid_smiles: Sequence[str], train_smiles: Iterable[str]) -> float:
    train_canonical: Set[str] = set()
    for smiles in train_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            train_canonical.add(canonical)
    if not valid_smiles:
        return 0.0
    novel = sum(smiles not in train_canonical for smiles in valid_smiles)
    return novel / len(valid_smiles)


def scaffold_core_rates(valid_smiles, core_smiles):
    """Core match and retention share the valid-molecule denominator."""
    from scaffold_metrics import scaffold_checks
    if len(valid_smiles) != len(core_smiles):
        raise ValueError('generated/core length mismatch')
    matches = retained = 0
    for generated, core in zip(valid_smiles, core_smiles):
        checks = scaffold_checks(generated, core)
        if not checks['eligible']:
            raise ValueError('Valid-product scaffold scoring requires a parseable, nonempty target core')
        matches += checks['core_match']
        retained += checks['core_retention']
    denominator = max(len(valid_smiles), 1)
    return matches / denominator, retained / denominator


def summarize_metric(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_median": 0.0}
    import statistics

    return {
        f"{prefix}_mean": float(statistics.mean(values)),
        f"{prefix}_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_median": float(statistics.median(values)),
    }


def evaluate_generations(
    generated_smiles: Sequence[str],
    scaffold_smiles: Sequence[str],
    train_smiles: Iterable[str],
) -> Dict[str, float]:
    canonical, valid_rate = validity(generated_smiles)
    valid_smiles = [smiles for smiles in canonical if smiles is not None]
    qed = qed_scores(valid_smiles)
    sas = sas_scores(valid_smiles)
    diversity = diversity_score(valid_smiles)
    novelty = novelty_score(valid_smiles, train_smiles)
    valid_cores = [s for s, c in zip(scaffold_smiles, canonical) if c is not None]
    if len(generated_smiles) != len(scaffold_smiles):
        raise ValueError('generated/core length mismatch')
    match_rate, retention_rate = scaffold_core_rates(valid_smiles, valid_cores)

    metrics = {
        "validity": valid_rate,
        "diversity": diversity,
        "novelty": novelty,
        "substructure_match_rate": match_rate,
        "scaffold_retention_rate": retention_rate,
        "num_generated": len(generated_smiles),
        "num_valid": len(valid_smiles),
    }
    metrics.update(summarize_metric(qed, "qed"))
    metrics.update(summarize_metric(sas, "sas"))
    metrics['scaffold_core_metric_n'] = len(valid_smiles)
    return metrics
