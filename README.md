# AtomFormer

AtomFormer generates molecular decorations conditioned on a three-dimensional scaffold. A geometric scaffold encoder supplies atom-type and coordinate features to an autoregressive decoder, and a graph assembly step attaches the generated decoration to the requested scaffold. Supervised learning can be followed by reinforcement learning using molecular quality and scaffold-preservation rewards.

The implementation supports single-attachment scaffold–decoration pairs. It includes data preparation, scaffold-disjoint splitting, training, checkpoint recovery, generation, and molecular evaluation.

## Installation

Use Python 3.11 and a virtual environment:

```bash
git clone https://github.com/Misaka10862/AtomFormer.git
cd AtomFormer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build compatible with your system using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). The commands below use `--device cuda`; use `--device cpu` for small checks. RDKit must include its `SA_Score` module.

## Prepare data

Provide your own UTF-8 SMILES file at `data/molecules.smi`, with one SMILES string per line and no header. Preprocessing extracts eligible single-attachment scaffold–decoration pairs and computes scaffold coordinates. Molecules that cannot be represented or embedded are skipped, and their count is printed.

```bash
python preprocess.py \
  --input-smi data/molecules.smi \
  --output-path data/processed.pt \
  --num-workers 8

python split_scaffold_disjoint.py \
  --input-path data/processed.pt \
  --output-dir data/splits
```

The split groups canonical scaffold cores into approximately 80% training, 10% validation, and 10% test groups. Record counts depend on group sizes. It deduplicates canonical molecules and writes an overlap audit in `split_manifest.json`.

Processed `.pt` files contain a list of dictionaries. The training fields are `scaffold_smiles`, `rooted_decoration_smiles`, `cond_z` (scaffold atomic numbers), `cond_pos` (an N×3 coordinate array), `anchor_z`, and `canonical_smiles`. Load only data and checkpoints from sources you trust.

## Train

Supervised training:

```bash
python train.py --stage supervised \
  --train-path data/splits/train.pt \
  --val-path data/splits/validation.pt \
  --output-dir runs/supervised \
  --device cuda
```

Reinforcement learning:

```bash
python train.py --stage rl \
  --checkpoint runs/supervised/last.pt \
  --train-path data/splits/train.pt \
  --output-dir runs/rl \
  --device cuda
```

RL checkpoints preserve model and optimizer state, random-number state, and the position within the shuffled batches. To recover an interrupted run, repeat its command with `--resume`. Keep its inputs, code, and training parameters unchanged. `--reward-workers` controls parallel CPU reward scoring, and `--checkpoint-every` controls the checkpoint interval in batches. RL requires every selected training record to fit the tokenizer and sequence-length limit.

Training writes `last.pt`, `history.json`, and `run_manifest.json`; supervised training also writes `tokenizer.json`. RL additionally writes `resume.pt`, a configuration contract, and reward traces. `train_supervised.py` and `train_rl.py` expose the individual training stages. Run an entry point with `--help` for its options.

## Generate and evaluate

```bash
python evaluate.py \
  --checkpoint runs/rl/last.pt \
  --processed-path data/splits/test.pt \
  --novelty-reference-path data/splits/train.pt \
  --output-json generated/summary.json \
  --output-samples-jsonl generated/samples.jsonl \
  --greedy 0 --samples-per-scaffold 20 \
  --require-full-coverage 1 \
  --device cuda
```

The evaluator generates and assembles decorations, then reports validity, assembly success, QED, synthetic accessibility, novelty, diversity, scaffold core match, and scaffold core retention. Novelty uses the supplied reference dataset. Core match and retention use valid products as their denominator; validity and assembly success use all attempts. Retention additionally requires core match. Diversity uses a deterministic cap of 5,000 valid products.

`summary.json` contains aggregate metrics and settings, `samples.jsonl` contains individual outputs, and `run_manifest.json` records inputs and provenance. Use fresh output paths to avoid overwriting an evaluation. Datasets, checkpoints, and generated outputs are not included in this repository.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use small synthetic CPU fixtures for chemistry, model/checkpoint behavior, command-line workflows, and RL recovery.
