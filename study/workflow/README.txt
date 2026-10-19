AtomFormer recorded configurations and scientific workflow
Version 1.1.0

SCOPE
This supplement provides recorded settings for the 81 dataset/configuration/run evaluations in the data archive, observed software versions, pinned external baseline sources, baseline adapters, the scientific training/generation implementations, and common scoring/paired-statistics implementations. Use it with the fixed input and result archives described in ../README.md. The Zenodo deposit is still a draft pending the authors' review.

FILES
run_parameters.json: one entry for each dataset/configuration/run. It distinguishes observed training information, recorded argument blocks, the five coordinate-free configurations, and sampling metadata. Null/NR denotes missing historical information and is not a nominal value substituted for an actual total.
generation_parameters.json: whitelisted parameters recovered from actual generation records for 72 evaluations. Nine historical GDB13 baseline evaluations have no separate generation-contract file in this export. Their available sampling settings remain in run_parameters.json's baseline configuration blocks.
environment_versions.json: observed core package versions for model/common-scoring and external-baseline environments. This is not a complete transitive dependency or operating-system lock.
upstream.lock.json and runtime_files.lock.json: upstream repository revisions and 35 recorded LibINVENT/SMILES-RNN/PromptSMILES runtime file hashes.
source_hashes.json: identities of the included scientific implementation files. Platform-specific data/resource lookup is replaced by explicit local inputs. Scheduling permissions are replaced by a content-based execution identity. Comments/docstrings from the original source snapshots are omitted. Model and metric calculations are preserved.
public_models.json: identities of the nine exported primary weights. The compatibility generator restores only the protocol marker in memory after a public-weight hash match. It does not change weight tensors or the checkpoint on disk.
code/: the scientific implementation modules. Scheduler/controller, monitoring, release bookkeeping and previous source iterations are excluded.

SOFTWARE ENVIRONMENTS
The primary-model and final common-scoring environment used Python 3.8.10, PyTorch 1.11.0+cu113, NumPy 1.22.4 and RDKit 2022.09.5.
The restored baseline environment used Python 3.10.21, PyTorch 1.13.1+cu117, NumPy 1.26.4 and RDKit 2026.03.5.
Run final cross-method scoring and chemical reconstruction with RDKit 2022.09.5, even when generation used the separate baseline environment. Preserve the recorded training settings and sampling boundaries. Cross-version or bitwise training/generation replay has not been certified.

UPSTREAM BASELINES
Retrieve the pinned sources into a NEW directory:
  python study/workflow/fetch_upstream.py --destination /path/to/new/upstream
The script downloads only the needed upstream packages and license files, applies the supplied functional LibINVENT compatibility patch, and checks the recorded 35-file runtime hashes. Existing directories are not replaced. DrugEx is fixed to its recorded upstream commit. Install the dependencies required by these upstream revisions into the corresponding environment. No upstream pretrained baseline weights are included.

SELECT A RUN WITHOUT COMPUTATION
  python study/workflow/run_stage.py --dataset chembl37 --configuration atomformer --run 1 --stage supervised --show
This prints the selected record without loading data, creating a model or requiring a GPU. Configuration labels match the result archive names. Run labels identify the three recorded training repetitions. Seeds are retained in machine-readable parameters.

PRIMARY-MODEL STAGES
Run formal stages on a GPU compute host with the matching environment. Set DATA to the directory containing train.pt, validation.pt, test.pt and tokenizer.json from the fixed-input archive (or the restored GDB13 inputs). Use a separate WORK directory for generated artifacts.

  python study/workflow/run_stage.py --dataset chembl37 --configuration atomformer --run 1 --stage supervised --data-dir DATA --work-root WORK
  python study/workflow/run_stage.py --dataset chembl37 --configuration atomformer --run 1 --stage reinforcement --data-dir DATA --work-root WORK --checkpoint WORK/models/chembl37__atomformer__run1/supervised/last.pt
  python study/workflow/run_stage.py --dataset chembl37 --configuration atomformer --run 1 --stage generate --data-dir DATA --work-root WORK --checkpoint /path/to/exported/chembl37/run1.pt

The five coordinate-free configurations select the recorded representation and RL implementation. Legacy supervised training preserves its per-epoch scaler behavior. Weighted supervision preserves token/record-weighted microbatch accumulation. Memory-recomputed RL preserves its recorded rollout/backpropagation implementation. These implementations are supplied separately from the general-purpose public CLI.

BASELINE STAGES
For ChEMBL37 or MOSES, choose drugex, libinvent or promptsmiles as --configuration and add --upstream /path/to/upstream. Run prepare before supervised, then reinforcement with the selected prior and generate with the trained RL model. The adapter preserves explicit input coverage, vocabulary construction, action space and method-specific reward differences.
GDB13 baselines use their recorded legacy argument/configuration blocks and the included native baseline functions. The DrugEx legacy generation CLI is code/baselines/drugex/evaluate_drugex.py. Its available settings must be matched to the recorded arguments; do not interpret an unrecorded evaluation batch as a verified default. For the geometric extension, consult its per-run argument blocks and the included legacy or weighted training functions before claiming an exact rerun. The convenience dispatcher is not certification of a complete historical GPU replay.

COMMON SCORING AND STATISTICS
Saved result ZIPs contain gzip JSONL files. Keep their original ordering and failed attempts. For a native-support summary with the full requested budget retained separately:
  python study/workflow/analyze.py summary --samples /path/to/atomformer/run1.jsonl.gz --train-input DATA/train.pt --intended-records 20000 --output /path/to/new/summary.json
Use 24441 intended records for GDB13 and 20000 for ChEMBL37/MOSES. The summary uses the recorded common scoring formulas and stored QED/SAS values. It excludes explicit input_supported=false slots from native-support metrics, while retained yield keeps the full intended budget. The working native JSONL is retained in a temporary working directory for inspection.
The underlying code/rerun_campaign/scoring.py also contains the original chemical score_one and normalization functions for raw model outputs. Analyze.py reads the already scored public projection. It does not substitute new RDKit property values for the stored scores.

For the paired component intervals, supply three matching runs on each side:
  python study/workflow/analyze.py component-ci --left LEFT/run1.jsonl.gz LEFT/run2.jsonl.gz LEFT/run3.jsonl.gz --right RIGHT/run1.jsonl.gz RIGHT/run2.jsonl.gz RIGHT/run3.jsonl.gz --train-input DATA/train.pt --test-input DATA/test.pt --output /path/to/new/component.json
The default is 2000 resamples of hydrogen-capped scoring-target cores with the three fitted run pairs fixed. The program reconstructs per-record sufficient statistics using the archived inputs. It checks the complete 20-attempt record budget and pairs the same record/core identities across both sides and all runs. These intervals do not quantify training-seed uncertainty. The source paired-bootstrap function is unchanged from the evaluated implementation.

VALIDATION AND BOUNDARIES
The supplement passed small synthetic checks of parameter coverage, distinct environments, failed-attempt retention, native/requested denominators and paired core intervals. Three representative public primary-model checkpoints strictly loaded into the supplied architecture. The pinned upstream runtime passed all 35 recorded file hashes. No full-data statistics, training or generation were rerun to create this supplement. Stored paper results remain the numerical reference. Historical missing update totals remain explicitly unavailable.

LICENSES
Original software is GPL-3.0-or-later, with the repository COPYRIGHT and THIRD_PARTY_NOTICES. The LibINVENT compatibility patch is supplied with its upstream Apache-2.0 license, and upstream notices are retained in licenses/. Recorded scientific parameter data follow the study data license, CC BY-SA 3.0. The supplement grants no additional rights over upstream datasets or third-party software. The data archive's LICENSES.txt remains applicable.
