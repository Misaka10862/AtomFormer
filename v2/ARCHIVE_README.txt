AtomFormer research artifacts, version 2.0.0
Creators: Pengkai Zhang and Yongna Yuan, Lanzhou University
Code: https://github.com/Misaka10862/AtomFormer/tree/v2.0.0/v2
Reserved data DOI: 10.5281/zenodo.23862166

This deposit is a draft pending author approval. Do not describe it as publicly available until publication.

CONTENTS
AtomFormer_Models_<dataset>_v2.0.0.zip: three supervised best, three balanced-RL last (lambda 0.10), and three unanchored-RL last (lambda 0) checkpoints per dataset. Tensors and tokenizers are preserved; optimizer states, monitoring histories and execution-control records are omitted. Model-tensor equality is verified during export. The model JSON files provide training/architecture settings; replace portable source-checkpoint placeholders when training a fresh continuation.
AtomFormer_Inputs_chembl37_v2.0.0.zip and AtomFormer_Inputs_moses_v2.0.0.zip: fixed full training data, selection-validation, check-validation, test inputs, tokenizers, full-training novelty reference and three stratified policy lists.
AtomFormer_Results_<dataset>_<split>_v2.0.0.zip: scored event ledgers and scalar summaries for six systems and three repeats per split. Each events.jsonl.gz retains generated outputs, attempt IDs, original model/event identity, support flags, failed attempts and scores. Separate duplicate raw ledgers and per-record summary copies are omitted. Original events can be rescored from assembled_output with the fixed input cores and training novelty reference.
REFERENCE_RESULTS.zip: full-precision final summaries, core-bootstrap intervals, paired internal differences, official baseline registry and pinned source identities.
MANIFEST.json: archive SHA256/size inventory, checkpoint provenance and input checksums.
LICENSES.txt: artifact and upstream license/attribution information.

GDB13
Obtain gdb13.1M.freq.ll.smi.gz from its original provider under the provider's terms. Use the published selectors/numerical annotations in https://doi.org/10.5281/zenodo.23468121 and study/restore_gdb13_inputs.py to reconstruct the historical ordered inputs, then v2/restore_gdb13.py to obtain the input representation for this version. Parent GDB13 structures are not redistributed in an input package here. GDB13 generated outputs and derived evaluation annotations are included.

SCIENTIFIC SETTINGS
Main lambda 0.10 across all datasets; lambda 0 and supervised controls. Supervision: 12 epochs, selection-validation best. RL: 29491 stratified training inputs, 4 epochs, effective batch 64, last checkpoint. Reward and evaluation novelty both use the complete benchmark training set. Frozen official LibINVENT/PromptSMILES/DrugEx checkpoints are not retrained. Their upstream assets are identified by hashes; external model binaries are not included.
There are 108 final evaluations: 54 validation-check and 54 test, each with 20 scheduled attempts per requested input. AtomFormer training seeds 42/43/44 correspond to sampling seeds 142/143/144; official systems reuse a single upstream checkpoint with three sampling seeds.
Y uses scheduled attempts; U and D average all reporting inputs including zero-output inputs. D retains molecular duplicates. Aggregate diversity samples up to 5000 valid events uniformly, without deduplication, using seed 20261025. Core-bootstrap intervals are conditional on evaluated checkpoints, not new-training variance.

USE
Read the v2 code README before generation, scoring or training. Verify MANIFEST checksums. Keep common-support, native-support and full-request populations distinct. Use RDKit 2022.09.5 and the supplied final scoring implementation. These files support reuse and result verification; no bitwise cross-environment training replay is asserted.
