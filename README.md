# AtomFormer

Scaffold-preserving molecular decoration with coordinate-free atom encoding, attachment-first autoregressive generation, and balanced reinforcement learning.

## Current version

The main branch carries the updated implementation and study in **[v2/](v2/README.md)**. The main supervised-anchor weight is 0.10 on ChEMBL37, MOSES and GDB13, with supervised and zero-anchor controls and three frozen official reference systems.

The version 2.0.0 data archive is published at **[10.5281/zenodo.23862166](https://doi.org/10.5281/zenodo.23862166)**. Full-precision compact reference results are included in [`v2/reference_results/`](v2/reference_results/).

## Earlier version

The [v1.1.0 code tag](https://github.com/Misaka10862/AtomFormer/tree/v1.1.0) and the [published data archive 1.0.0](https://doi.org/10.5281/zenodo.23468121) preserve the earlier study. That archive is publicly released, not an unpublished draft. Root-level legacy training commands and `study/` documents belong to that earlier version; use the v2 entry points for the updated study.

Code is licensed under GPL-3.0-or-later. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the dataset-specific license notices. Data, checkpoints and generated molecules are distributed separately from Git.
