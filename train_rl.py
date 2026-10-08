import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from rdkit import RDLogger

from condmolgen.checkpoint import load_checkpoint, save_checkpoint
from condmolgen.data import load_records
from condmolgen.metrics import canonicalize_smiles
from condmolgen.train_utils import seed_everything, split_records, write_json
from condmolgen.decoration_data import DecorationBatchCollator, DecorationDataset
from condmolgen.decoration_model import DecorationGenerator
from condmolgen.representation import assemble_with_representation, get_representation_spec
from condmolgen.run_manifest import write_run_manifest
from scaffold_metrics import scaffold_checks, scaffold_core_from_scaffold, protocol_metadata


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight REINFORCE fine-tuning for AtomFormer conditional autoregressive decoder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=0.2)
    parser.add_argument("--qed-weight", type=float, default=1.0)
    parser.add_argument("--novelty-weight", type=float, default=0.2)
    parser.add_argument("--sas-weight", type=float, default=0.05)
    parser.add_argument("--invalid-reward", type=float, default=-2.0)
    parser.add_argument("--core-failure-reward", type=float, default=-2.0)
    parser.add_argument("--reward-baseline", default="batch_mean", choices=["none", "batch_mean"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--representation", choices=["rooted", "plain"], default="rooted")
    parser.add_argument("--expected-rdkit", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--reward-workers", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--require-full-coverage",
        type=int,
        default=0,
        help="Fail if max-length filtering drops any RL input record.",
    )
    return parser.parse_args()


def _import_rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import QED, RDConfig
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import os
    import sys

    RDLogger.DisableLog("rdApp.*")
    contrib_dir = getattr(RDConfig, "RDContribDir", None)
    if contrib_dir:
        sa_dir = os.path.join(contrib_dir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer
    return Chem, QED, MurckoScaffold, sascorer


def loader_kwargs(args: argparse.Namespace) -> Dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_train_canonical(records: Iterable[Dict]) -> set:
    canonical = set()
    for record in records:
        smiles = record.get("canonical_smiles")
        if smiles:
            canon = canonicalize_smiles(smiles)
            if canon:
                canonical.add(canon)
    return canonical


def scaffold_core_ok(assembled_smiles: str, scaffold_smiles: str) -> bool:
    core = scaffold_core_from_scaffold(scaffold_smiles)
    if not core:
        raise ValueError('Cannot construct target scaffold core')
    return scaffold_checks(assembled_smiles, core)['core_retention']


def compute_rewards(
    decorations: List[str],
    scaffolds: List[str],
    train_canonical: set,
    args: argparse.Namespace,
    representation: str = "rooted",
) -> Tuple[torch.Tensor, List[Dict]]:
    Chem, QED, _, sascorer = _import_rdkit()
    rewards: List[float] = []
    rows: List[Dict] = []

    for decoration, scaffold in zip(decorations, scaffolds):
        assembled = assemble_with_representation(representation, scaffold, decoration)
        row = {
            "scaffold_smiles": scaffold,
            "generated_decoration": decoration,
            "assembled_smiles": assembled,
            "assembly_success": assembled is not None,
            "validity": False,
            "scaffold_core_ok": False,
            "qed": None,
            "sas": None,
            "novelty": 0.0,
            "reward": args.invalid_reward,
        }
        if assembled is None:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue

        mol = Chem.MolFromSmiles(assembled)
        if mol is None:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue
        row["validity"] = True

        if not scaffold_core_ok(assembled, scaffold):
            row["reward"] = args.core_failure_reward
            rewards.append(args.core_failure_reward)
            rows.append(row)
            continue
        row["scaffold_core_ok"] = True

        try:
            qed = float(QED.qed(mol))
            sas = float(sascorer.calculateScore(mol))
        except Exception:
            rewards.append(args.invalid_reward)
            rows.append(row)
            continue

        canon = canonicalize_smiles(assembled)
        novelty = 1.0 if canon and canon not in train_canonical else 0.0
        reward = args.qed_weight * qed + args.novelty_weight * novelty - args.sas_weight * sas
        row.update(
            {
                "qed": qed,
                "sas": sas,
                "novelty": novelty,
                "reward": reward,
            }
        )
        rewards.append(reward)
        rows.append(row)

    return torch.tensor(rewards, dtype=torch.float32), rows


def summarize_reward_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    valid = [row for row in rows if row["validity"]]
    assembled = [row for row in rows if row["assembly_success"]]
    core_ok = [row for row in rows if row["scaffold_core_ok"]]
    qeds = [row["qed"] for row in rows if row["qed"] is not None]
    sas = [row["sas"] for row in rows if row["sas"] is not None]
    rewards = [row["reward"] for row in rows]
    return {
        "reward_mean": float(sum(rewards) / max(len(rewards), 1)),
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "assembly_success_rate": len(assembled) / max(len(rows), 1),
        "validity": len(valid) / max(len(rows), 1),
        "scaffold_core_ok_rate": len(core_ok) / max(len(rows), 1),
        "qed_mean": float(sum(qeds) / max(len(qeds), 1)) if qeds else 0.0,
        "sas_mean": float(sum(sas) / max(len(sas), 1)) if sas else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print('CLI_VALIDATED');return
    if args.expected_rdkit and args.expected_rdkit != protocol_metadata()['rdkit_version']:
        raise RuntimeError('RDKit version differs from the requested training environment')
    from condmolgen.rl_training import run
    run(args)


if __name__ == "__main__":
    main()
