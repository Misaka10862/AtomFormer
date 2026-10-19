# Study materials

The primary model used an atom-only encoder, attachment-first decoration targets,
a fixed scaffold-SMILES prefix, deterministic assembly, and balanced reinforcement
learning. The three datasets were trained and evaluated separately.

## Download the fixed artifacts

Data deposit: https://doi.org/10.5281/zenodo.23468121 (reserved DOI, version 1.0.0).
The deposit is currently a draft. The download and restoration instructions below
apply after its public release.
Use the included README, LICENSES and SHA256SUMS files before loading artifacts.
The ChEMBL37 and MOSES input packages include the fixed ordered tensors and
readable molecule identity lists. GDB13 is supplied as source-line selectors and
study-produced numerical annotations because its upstream structures must be
obtained from their original provider under its terms.

To restore GDB13 after obtaining the official upstream gzip and extracting the
selector package:

```bash
python study/restore_gdb13_inputs.py \
  --upstream-gz /path/to/gdb13.1M.freq.ll.smi.gz \
  --selectors /path/to/extracted/selectors \
  --output /path/to/new/gdb13_inputs
```

The nine `AtomFormer_Model_<dataset>_run<index>.zip` files contain
`gdb13/run1.pt` through `moses/run3.pt`. Use the matching
dataset, tokenizer, and sequence limit with `evaluate.py`. The result ZIPs contain
three gzip JSONL files per dataset/configuration archive, one for each run.
These preserve failed attempts and original
record/sample identifiers. They are separate from the code repository.

## Reference files

- `protocol.json`: architecture, training settings and metric denominators.
- `run_metrics.json`: 81 dataset/configuration/run evaluations used for the main
  comparison, component analysis and geometric-encoder comparison.
- `comparison_summary.json`: stored across-run means and sample standard deviations
  for the main model, external baselines and geometric extension.
- `component_comparisons.json`: stored paired differences and conditional intervals.
- `data_sources.json`: source versions, citations and input checksums.

These numbers were copied from the evaluated results, not recomputed for this
release. Reused main-model evaluations appear once. `atomformer` denotes the
coordinate-free balanced model. `geometry_extension` denotes the geometric
encoder comparison. Run 1–3 labels identify independent trained repetitions
within each dataset. The archived training seed is provided in machine-readable
metadata to identify a repetition, not to assert universal deterministic replay.

## Interpretation

Validity used generation attempts. Exact core retention used valid products.
Requested-budget retained yield included unsupported inputs. Native-support and
requested-budget metrics are not interchangeable. Baseline reward gates and
input support differed. The component bootstrap intervals were conditional on
three fitted run pairs and did not include uncertainty over possible training
runs. The fixed test sets were reused for the component analysis.

The study GDB13 working input was verified to equal the first SMILES column
of the official `gdb13.1M.freq.ll.smi.gz` file, preserving source order. Its
exact SHA-256 is recorded in `data_sources.json`. Fixed source-line selectors
and numerical annotations are archived so that no new random subset is needed.
Use `restore_gdb13_inputs.py` to reconstruct the ordered splits from the
upstream file. The script verifies all archived field hashes and reuses the
original scaffold coordinates rather than generating new conformers. A
serialized `.pt` container may have a different byte hash after reconstruction,
but the decoded scientific fields must match. Use RDKit 2022.09.5 if a
representation hash differs on a newer version. The primary network did not use coordinates, but shared data
selection and the tensor format retained the conformer-based preparation.

## Reuse and reproducibility

The public code implements the architecture and provides training and evaluation
utilities. The primary-model weights and saved evaluation outputs are the
reference for model reuse and result inspection. The command-line training and
sampling utilities are not a guarantee of bitwise reproduction of historical
training or generation trajectories. In particular, microbatch loss weighting,
sampling batch boundaries, random-number consumption and library versions must
be matched before claiming an exact replay. Paper property scoring used RDKit
2022.09.5. No new numerical result is substituted for the archived evaluations.
