# Third-party acknowledgements

The optional geometric encoder in `condmolgen/geometry.py` follows the
Geoformer implementation and interatomic positional-encoding method from
[Microsoft AI2BMD, Geoformer branch](https://github.com/microsoft/AI2BMD/tree/Geoformer).
Its upstream MIT notice is preserved verbatim in
[`LICENSES/AI2BMD-Geoformer-MIT.txt`](LICENSES/AI2BMD-Geoformer-MIT.txt).
The AtomFormer implementation adapts the encoder to scaffold-conditioned
decoration generation. It does not include the upstream pretrained models.

Method reference: Wang, Y. et al. *Geometric Transformer with Interatomic
Positional Encoding*. Advances in Neural Information Processing Systems 36
(2023), 55981–55994.

PyTorch, RDKit (including SA_Score), NumPy, and any separately installed
baseline implementations retain their own copyright and license terms.
They are dependencies, not relicensed copies distributed by this source pack.
GDB13, ChEMBL, and MOSES data are not included in this repository. Their
licensing and attribution must accompany any later data archive.
