# AtomFormer

AtomFormer generates molecular decorations conditioned on a fixed scaffold. The default model uses scaffold atom identities and a scaffold-SMILES prefix, without coordinate features. An autoregressive decoder generates an attachment-rooted decoration, and explicit graph assembly attaches it to the requested scaffold. Supervised learning is followed by balanced reinforcement learning with molecular-property rewards and a supervised anchor.

The implementation supports single-attachment scaffold–decoration pairs. It includes data preparation, scaffold-disjoint splitting, training, checkpoint recovery, generation, and molecular evaluation.

## Research materials

Fixed study inputs, GDB13 reconstruction selectors, nine primary-model weights,
and saved evaluation outputs have been prepared for the associated
[Zenodo deposit](https://doi.org/10.5281/zenodo.23468121) (reserved DOI, version 1.0.0).
The deposit is currently a draft and its files are not yet publicly accessible.
The [study guide](study/README.md) describes their use and the fixed evaluation
protocol. See [data sources](study/data_sources.json) for original dataset versions
and citations. Use the `v1.1.0` code tag with this archive. Original software is distributed
under GPL-3.0-or-later, with the upstream notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Data and model artifacts have their own licenses and are distributed separately.
The [recorded workflow supplement](study/workflow/README.txt) supplies per-run
settings, observed environments, pinned baseline adapters and paired-statistics code.

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

Provide your own UTF-8 SMILES file at `data/molecules.smi`, with one SMILES string per line and no header. Preprocessing extracts eligible single-attachment scaffold–decoration pairs. To retain the evaluated data-selection protocol and compatibility with the optional geometric encoder, it still computes scaffold coordinates. Molecules that cannot be represented or embedded are skipped, and their count is printed. The default atom-type encoder ignores these coordinates; this model change does not make the preprocessing pipeline conformer-independent.

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

Processed `.pt` files contain a list of dictionaries. The training fields are `scaffold_smiles`, `rooted_decoration_smiles`, `cond_z` (scaffold atomic numbers), `cond_pos` (an N×3 coordinate array retained for data-format compatibility), `anchor_z`, and `canonical_smiles`. `cond_pos` is required by the shared data loader but is not used by the default model's encoder. Load only data and checkpoints from sources you trust.

## Train

Supervised training:

```bash
python train.py --stage supervised \
  --train-path data/splits/train.pt \
  --val-path data/splits/validation.pt \
  --output-dir runs/supervised \
  --encoder-type atom --representation rooted \
  --epochs 12 --batch-size 256 \
  --learning-rate 3e-4 --use-amp 1 \
  --device cuda
```

Reinforcement learning:

```bash
python train.py --stage rl \
  --checkpoint runs/supervised/last.pt \
  --train-path data/splits/train.pt \
  --output-dir runs/rl \
  --representation rooted --epochs 4 \
  --learning-rate 2e-6 --batch-size 64 \
  --max-train-samples 32768 --supervised-weight 0.2 \
  --qed-weight 1.0 --novelty-weight 0.2 --sas-weight 0.2 \
  --device cuda
```

These are the default no-3D rooted/balanced settings in both the wrapper and individual stage entry points. A smaller physical batch can be used with `--grad-accum-steps` for general training, but the microbatch weighting is not a guarantee of the archived training trajectory. See the study guide before an exact replay. `--block-size` sets the supervised sequence limit; use a compatible `--max-length` for RL and evaluation. The examples use the default limit of 128; use 256 consistently when the dataset requires longer sequences.

Balanced RL uses `QED + 0.2 × novelty − 0.2 × SAS` for valid, strictly scaffold-retaining products and a failure reward of −2. The total loss combines the policy loss with 0.2 times the supervised loss, including the configured anchor auxiliary term. This balances property optimization and candidate variety; it does not imply that the supervised term improves all molecular properties.

RL and evaluation load the encoder architecture from the checkpoint. `--encoder-type atom` applies to supervised training and does not convert an existing geometric checkpoint. Use `--encoder-type geo` at the supervised stage for the optional 3D comparison. For a plain-decoration comparison, use `--representation plain` consistently across supervised training, RL and evaluation. For Pure RL, start from the same rooted supervised checkpoint and set `--supervised-weight 0`; the supervised forward pass remains in place to preserve the random-number consumption path.

RL checkpoints preserve model and optimizer state, random-number state, and the position within the shuffled batches. To recover an interrupted run, repeat its command with `--resume`. Keep its inputs, code, and training parameters unchanged. Existing runs created with different defaults must retain their original explicit arguments and code. `--reward-workers` controls parallel CPU reward scoring, and `--checkpoint-every` controls the checkpoint interval in batches. RL requires every selected training record to fit the tokenizer and sequence-length limit.

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
