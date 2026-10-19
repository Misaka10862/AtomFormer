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
from condmolgen.v4_data import V4BatchCollator, V4DecorationAutoregDataset
from condmolgen.v4_model import V4ConditionalAutoregGenerator
from condmolgen.run_manifest import write_run_manifest
from condmolgen.representation import assemble_with_representation, get_representation_spec
from scaffold_metrics import scaffold_core_from_scaffold, protocol_metadata

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate v4 conditional autoregressive decoration generation.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--processed-path', required=True)
    parser.add_argument('--novelty-reference-path', default=None)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-samples-jsonl', default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--greedy', type=int, default=1)
    parser.add_argument('--samples-per-scaffold', type=int, default=1)
    parser.add_argument('--sampling-seed', type=int, default=0)
    parser.add_argument('--expected-rdkit', default=None)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--representation', choices=['rooted', 'plain'], default='rooted')
    parser.add_argument('--require-full-coverage', type=int, default=0, help='Fail if max-length filtering drops any evaluation record.')
    return parser.parse_args()
