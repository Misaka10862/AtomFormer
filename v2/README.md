# AtomFormer 2.0

AtomFormer performs coordinate-free, single-attachment scaffold decoration with an atom encoder, a scaffold SMILES prefix, rooted decoration generation, deterministic molecular assembly, and supervised-anchored reinforcement learning.

This version supplies the implementation and reference results for the updated ChEMBL37, MOSES and GDB13 study. The main anchor weight is **0.10 on all three datasets**. Controls are supervised AtomFormer and unanchored RL (0). LibINVENT, PromptSMILES and DrugEx are **frozen public reference systems**, not retrained benchmark replicas.

## Research artifacts

The version 2.0.0 data archive is published at **[10.5281/zenodo.23862166](https://doi.org/10.5281/zenodo.23862166)**. The [1.0.0 archive](https://doi.org/10.5281/zenodo.23468121) preserves the earlier study and the GDB13 reconstruction selectors.

The published archive contains 27 AtomFormer checkpoints, fixed ChEMBL37/MOSES inputs, 108 final scored generation ledgers (54 validation-check and 54 test evaluations), their summaries, and source/asset checksums. Internal monitoring, approval records, debugging histories and manuscript revisions are excluded.

## Installation

Run the commands below from this `v2/` directory, in Python 3.8 with a suitable Linux environment. The recorded scoring environment is RDKit 2022.09.5 and NumPy 1.24.4; the recorded CPU implementation used PyTorch 1.13.1. Install the appropriate CUDA build for GPU generation/training. External systems have additional pinned requirements and source revisions listed in `baseline_registry.json` and `provenance/upstream.lock.json`.

```sh
python -m pip install -r requirements.txt
python atomformer.py --help
```

The implementation modules retain their serialized schema identifiers for compatibility with archived checkpoints and generation-event IDs. The public method and software name is AtomFormer.

## Generate and score

After downloading the published archive and extracting its models/inputs:

```sh
python atomformer.py generate \
  --checkpoint /path/to/models/chembl37/atomformer_rl-s42.pt \
  --inputs /path/to/inputs/chembl37/test.jsonl.gz \
  --sampling-seed 142 --device cuda --output /path/to/new-generation

python atomformer.py score \
  --events /path/to/new-generation/samples.jsonl.gz \
  --inputs /path/to/inputs/chembl37/test.jsonl.gz \
  --reference /path/to/inputs/chembl37/reference.json \
  --output /path/to/new-scoring
```

The same `score` command accepts the archived `events.jsonl.gz` ledgers. Use a fresh output directory. Generation uses 20 scheduled attempts per input, eval/no_grad, original support masks and per-record seeds; invalid, repeated and unsupported attempts are retained. It does not read reference decorations as generation conditions. Archived model files omit optimizer/controller histories; model tensors and tokenizers are preserved, with tensor equality checked during export. Their original checkpoint identities are retained for stable generation-event IDs.

Exact training or generation replay across differing devices, libraries and batch implementations is not asserted. The archived event ledgers and checksums are the reference for the reported experiment. Scoring applies the narrow aromatic-carbanion boundary-hydrogen handling used by the final evaluator, identically for every method.

## Training

Each checkpoint has a companion JSON model specification. Supervision uses the full training split and selects its best checkpoint only on the selection-validation subset. RL uses a fixed, stratified 29,491-record training list for four epochs, effective batch 64, and the **last** checkpoint; both reward novelty and reported novelty use the full benchmark training reference. The source checkpoint for RL is the matching supervised best checkpoint.

```sh
python atomformer.py train --spec /path/to/model-spec.json \
  --data /path/to/inputs/chembl37 --output /path/to/new-training --device cuda
# For an RL spec also provide --initial-checkpoint /path/to/supervised-best.pt
```

The default source path in an exported spec is a portable placeholder; supply the initial checkpoint explicitly for RL. No official external model training is implemented by this public entry point. The frozen external inference adapters, native decoding settings and weight hashes are in `re_atomformer/scheme_a/baselines.py` and `baseline_registry.json`. The directory name is an implementation namespace, not a separate model name.

## GDB13 reconstruction

Upstream GDB13 parent structures are not redistributed. Obtain the licensed upstream `gdb13.1M.freq.ll.smi.gz` and the selectors from the [historical artifact record](https://doi.org/10.5281/zenodo.23468121). Reconstruct the historical ordered `.pt` inputs with the repository's `study/restore_gdb13_inputs.py`, then run:

```sh
python restore_gdb13.py --legacy-inputs /path/to/reconstructed-gdb13 \
  --output /path/to/new-gdb13-inputs
```

This applies the source-identity-based scaffold mapping and reproduces the selection/check split. Compare the produced scientific records against `gdb13_record_hashes.json`; gzip container byte hashes can differ due to timestamps. The restoration entry point was checked against all 229281 fixed records with exact scientific-field equality. Expected sizes are train 181472, selection-validation 17469, check-validation 4367 and test 24441. The excluded 1532 validation records overlap training scoring cores. The nonoverlap test subset has 24003 records. Historical rooted sulfur-H assembly limitations remain part of the fixed inputs; reconstruction does not silently remove them.

## Metrics and statistical interpretation

`reference_results/` contains full-precision final summaries for the common-support, native-support, full-request and GDB13 nonoverlap populations. QED/SAS main values average valid complete products; retained properties are conditional supplementary metrics. Y uses the requested attempt budget; U and duplicate-preserving D average all reported inputs, including zero-output inputs. Aggregate diversity samples up to 5000 valid events uniformly using seed 20261025, without molecular deduplication.

Three AtomFormer repeats vary training and sampling; external repeats vary sampling of a fixed checkpoint. Core bootstrap intervals are conditional on evaluated checkpoints and do not represent new-training variability. Common-support ChEMBL37 test size is 16521; validation-check size is 3466. Do not mix these populations or use legacy diversity values in place of the event-uniform metric.

## Licenses

Code uses the repository's GPL-3.0-or-later license with retained third-party notices. Dataset/weight licenses are separate. Follow the original GDB13, ChEMBL and MOSES terms and the official reference checkpoint licenses. Official reference weights are identified by upstream download links and hashes rather than duplicated here.
