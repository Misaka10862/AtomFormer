import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from rdkit import RDLogger

from condmolgen.checkpoint import load_checkpoint
from condmolgen.data import load_records
from condmolgen.metrics import canonicalize_smiles, evaluate_generations
from condmolgen.decoration_data import DecorationBatchCollator, DecorationDataset
from condmolgen.decoration_model import DecorationGenerator
from condmolgen.run_manifest import write_run_manifest
from condmolgen.representation import assemble_with_representation, get_representation_spec
from scaffold_metrics import scaffold_core_from_scaffold, protocol_metadata


INVALID_GENERATION = "__INVALID__"
RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AtomFormer conditional autoregressive decoration generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-path", required=True)
    parser.add_argument("--novelty-reference-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-samples-jsonl", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", type=int, default=1)
    parser.add_argument("--samples-per-scaffold", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--expected-rdkit", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--representation", choices=["rooted", "plain"], default="rooted")
    parser.add_argument(
        "--require-full-coverage",
        type=int,
        default=0,
        help="Fail if max-length filtering drops any evaluation record.",
    )
    return parser.parse_args()


def _import_rdkit():
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import QED, RDConfig
    import os
    import sys

    RDLogger.DisableLog("rdApp.*")
    contrib_dir = getattr(RDConfig, "RDContribDir", None)
    if contrib_dir:
        sa_dir = os.path.join(contrib_dir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer
    return Chem, QED, sascorer


def decoration_validity(decoration_smiles: List[str]) -> float:
    valid = 0
    for smiles in decoration_smiles:
        if canonicalize_smiles(smiles):
            valid += 1
    return valid / max(len(decoration_smiles), 1)


def score_assembled(assembled: str) -> Dict:
    if not assembled:
        return {"qed": None, "sas": None}
    Chem, QED, sascorer = _import_rdkit()
    mol = Chem.MolFromSmiles(assembled)
    if mol is None:
        return {"qed": None, "sas": None}
    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = None
    try:
        sas = float(sascorer.calculateScore(mol))
    except Exception:
        sas = None
    return {"qed": qed, "sas": sas}


def score_row(item):
    decoration, scaffold, reference, target_decoration, sample_idx, representation = item
    assembled = assemble_with_representation(representation, scaffold, decoration)
    return {
        "scaffold_smiles": scaffold, "reference_smiles": reference,
        "target_decoration": target_decoration, "sample_idx": sample_idx,
        "generated_decoration": decoration, "assembled_smiles": assembled,
        "assembly_success": assembled is not None,
        **score_assembled(assembled or ""),
    }


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print('CLI_VALIDATED');return
    if args.expected_rdkit and args.expected_rdkit != protocol_metadata()['rdkit_version']:
        raise RuntimeError('RDKit version differs from the requested evaluation environment')
    if Path(args.output_json).exists() or (args.output_samples_jsonl and Path(args.output_samples_jsonl).exists()):
        raise RuntimeError('Output files already exist; choose new output paths')
    representation = get_representation_spec(args.representation)
    model, tokenizer, extra = load_checkpoint(args.checkpoint, map_location=args.device)
    config = model.config
    decoration_model = DecorationGenerator(config)
    decoration_model.load_state_dict(model.state_dict(), strict=True)
    decoration_model = decoration_model.to(torch.device(args.device))
    decoration_model.eval()

    records = load_records(args.processed_path)
    if args.max_samples:
        records = records[: args.max_samples]
    novelty_reference_records = load_records(args.novelty_reference_path) if args.novelty_reference_path else records

    dataset = DecorationDataset(
        records,
        tokenizer,
        max_length=args.max_length,
        target_key=representation.target_key,
    )
    if args.require_full_coverage and len(dataset) != len(records):
        raise ValueError(
            f"{representation.name} representation filtered evaluation records: "
            f"expected={len(records)} actual={len(dataset)}"
        )
    collator = DecorationBatchCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    generated_smiles: List[str] = []
    scaffold_core_smiles: List[str] = []
    generated_decorations: List[str] = []
    sample_rows: List[Dict] = []
    assembly_success = 0
    pool = mp.Pool(args.workers) if args.workers > 1 else None

    for batch in loader:
        z = batch["z"].to(args.device)
        pos = batch["pos"].to(args.device)
        total_samples = max(1, args.samples_per_scaffold)
        for sample_start in range(0, total_samples, 4):
            sample_indices = list(range(sample_start, min(sample_start + 4, total_samples)))
            repeat = len(sample_indices)
            torch.manual_seed(args.sampling_seed + sample_start)
            if repeat > 1:
                z_gen = z.repeat_interleave(repeat, dim=0)
                pos_gen = pos.repeat_interleave(repeat, dim=0)
                scaffolds_gen = [s for s in batch["scaffold_smiles"] for _ in range(repeat)]
            else:
                z_gen, pos_gen, scaffolds_gen = z, pos, batch["scaffold_smiles"]
            decorations, _ = decoration_model.generate_decorations(
                z=z_gen, pos=pos_gen, tokenizer=tokenizer, scaffold_smiles=scaffolds_gen,
                max_length=args.max_length, temperature=args.temperature,
                greedy=bool(args.greedy),
            )
            refs = [r for r in batch["canonical_smiles"] for _ in range(repeat)]
            targets = [r for r in batch["target_decoration_smiles"] for _ in range(repeat)]
            items = list(zip(
                decorations,
                scaffolds_gen, refs, targets,
            ))
            items = [
                (*item, sample_indices[i % repeat], representation.name)
                for i, item in enumerate(items)
            ]
            rows = pool.map(score_row, items) if pool else [score_row(item) for item in items]
            for row in rows:
                assembled = row["assembled_smiles"]
                assembly_success += int(row["assembly_success"])
                generated_smiles.append(assembled if assembled is not None else INVALID_GENERATION)
                generated_decorations.append(row["generated_decoration"])
                scaffold_core_smiles.append(scaffold_core_from_scaffold(row["scaffold_smiles"]) or "")
                sample_rows.append(row)

    if pool:
        pool.close(); pool.join()

    metrics = evaluate_generations(
        generated_smiles=generated_smiles,
        scaffold_smiles=scaffold_core_smiles,
        train_smiles=[
            record["canonical_smiles"]
            for record in novelty_reference_records
            if record.get("canonical_smiles")
        ],
    )
    metrics["assembly_success_rate"] = assembly_success / max(len(sample_rows), 1)
    metrics["decoration_validity"] = decoration_validity(generated_decorations)
    metrics["scaffold_core_match_rate"] = metrics["substructure_match_rate"]
    metrics["scaffold_core_retention_rate"] = metrics["scaffold_retention_rate"]

    payload = {
        **protocol_metadata(),
        "metrics": metrics,
        "checkpoint_extra": extra,
        "num_samples": len(sample_rows),
        "args": vars(args),
        "task": "generation_evaluation",
        "representation": representation.name,
        "target_key": representation.target_key,
        "assembly_mode": representation.assembly_mode,
        "input_records": len(records),
        "dataset_records": len(dataset),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_manifest(
        output_json.parent / "run_manifest.json",
        data_paths=[args.processed_path]
        + ([args.novelty_reference_path] if args.novelty_reference_path else []),
        model_config=config.to_dict(),
        eval_params=vars(args),
        checkpoint_lineage=args.checkpoint,
        outputs=[args.output_json, args.output_samples_jsonl]
        if args.output_samples_jsonl
        else [args.output_json],
        experiment_metadata={
            **protocol_metadata(),
            "representation": representation.name,
            "target_key": representation.target_key,
            "assembly_mode": representation.assembly_mode,
            "input_records": len(records),
            "dataset_records": len(dataset),
            "attempts": len(sample_rows),
        },
    )

    if args.output_samples_jsonl:
        jsonl_path = Path(args.output_samples_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
