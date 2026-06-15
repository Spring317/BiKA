# Related work: binary / multiply-free neural networks

Papers with mechanics related to BiKA's per-connection learnable
threshold (`bias`), sign-based forward, and straight-through-estimator
(STE) backward with weight clamping.

- **BinaryConnect** — Courbariaux, Bengio, David (2015)
  https://arxiv.org/abs/1511.00363
  Origin of the `clamp(w, -1, 1)` trick: binarize weights in the forward
  pass, keep full-precision shadow weights, clamp after each update.

- **Binarized Neural Networks (BNN)** — Courbariaux, Hubara, Soudry,
  El-Yaniv, Bengio (2016)
  https://arxiv.org/abs/1602.02830
  Extends BinaryConnect to binarize activations too; formalizes the
  hard-tanh STE gradient gate (`|z| <= 1` passes gradient, else zero).

- **XNOR-Net** — Rastegari, Ordonez, Redmon, Farhadi (2016)
  https://arxiv.org/abs/1603.05279
  Binary weights/activations plus a per-filter scaling factor; motivated
  by multiply-free, comparator/accumulator-style hardware.

- **DoReFa-Net** — Zhou, Wu, Ni, Zhou, Wen, Zou (2016)
  https://arxiv.org/abs/1606.06160
  Generalizes to low-bitwidth weights, activations, and gradients.

- **Bi-Real Net** — Liu, Wu, Luo, Yang, Liu, Cheng (2018)
  https://arxiv.org/abs/1808.00278
  Addresses gradient mismatch / saturated connections from binarization
  using a piecewise-polynomial STE instead of hard-tanh.

- **ReActNet** — Liu, Shen, Savvides, Cheng (2020)
  https://arxiv.org/abs/2003.03488
  Introduces RSign: a learnable per-channel threshold added before the
  sign function (`sign(x - tau)`), plus RPReLU. Closest precedent for
  BiKA's learnable per-connection threshold (bias).
