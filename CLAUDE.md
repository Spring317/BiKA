# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BiKA is a Kolmogorov-Arnold-Network-inspired, multiply-free neural network layer
(`BiKA_Linear`, `BiKA_Conv2d`) implemented as a CUDA/C++ PyTorch extension, plus
experiments that use these layers (MNIST/FashionMNIST/CIFAR-10 classifiers and a
BDD100K road-segmentation U-Net).

## Repo layout

- `src/bika/` — the installable `bika` Python package.
  - `functional.py` — `_BiKALinearFn` / `_BiKAConv2dFn` autograd.Function wrappers around
    the compiled `bika._C` extension (forward/backward call into CUDA kernels).
  - `BiKA_Linear.py`, `BiKA_Conv2d.py` — `nn.Module` layers. Both use a **per-connection
    bias**: bias has the same shape as the weight tensor (not just per-output-channel).
  - `csrc/` — CUDA/C++ kernel sources, built via `setup.py` (`CUDAExtension`, module name `bika._C`).
  - `segmentation.py` — `BiKASegNet`, a small U-Net built from `BiKAConvBlock`s
    (each block = two `BiKA_Conv2d` + BatchNorm + ReLU), exported from `bika`.
- `experiment/` — training scripts/configs for CNN-like (CIFAR10/MNIST) and MLP-like
  (MNIST/FashionMNIST) BiKA experiments, plus saved run outputs
  (`bdd100k_BiKASegNet/`, `segBiKA_cont/`: checkpoints, `config.yml`, `log.csv`).
- `road_segmentation/` — standalone BDD100K segmentation pipeline (separate from
  `experiment/`, has its own `models/`/`data/`/`training/` packages mirroring `src/bika`'s
  `BiKASegNet`/dataset/loss code for that experiment).
- `test/BiKA_Lib_Test.ipynb` — install/sanity-check notebook for the `bika` package.

## Build / install

The CUDA extension must be built before `bika` can be imported (PyTorch must already
be importable, since the extension links against it):

```bash
python -m pip install -e . --no-build-isolation
```

Sanity check after (re)building:

```bash
python -c "import torch; import bika; print(bika.__version__)"
```

Re-run the install command after any change to `src/bika/csrc/*.cu` / `*.cpp` — there
is no separate build step, the extension is compiled by setuptools/ninja on install.

## Running road segmentation training

Run from inside `road_segmentation/` (it does `import models as archs`, `from data import ...`,
`import training.losses as losses` as top-level package imports relative to that directory):

```bash
cd road_segmentation
python train.py \
  --bdd100k_base /path/to/bdd100k/seg \
  --output_dir outputs \
  --batch_size 8 \
  --epochs 200
```

Key things to know:
- Expects BDD100K seg layout: `<bdd100k_base>/{images,labels}/{train,val}/...`, masks named
  `<id>_train_id.png`.
- `--arch`/`--model_name` choices come from `road_segmentation/models/__init__.__all__`
  (currently only `BiKASegNet`); `--loss` choices come from `training/losses.__all__`
  (`CrossEntropyDiceLoss`, `FocalLoss`) plus `CrossEntropyLoss`.
- Supports DDP (set via `RANK`/`WORLD_SIZE`/`LOCAL_RANK` env vars) and AMP
  (`--use_amp`, `--amp_dtype`); requires CUDA — `setup_distributed()` raises if
  `torch.cuda.is_available()` is False.
- `--resume true` auto-discovers `checkpoint_last.pth` in `outputs/<name>/` unless
  `--checkpoint_path` is given. Each run writes `config.yml`, `log.csv`,
  `checkpoint_last.pth`, `checkpoint_best.pth`, `model_best.pth` to `outputs/<name>/`,
  and copies `train.py` itself into that dir for reproducibility.
- `road_segmentation/plot_log.py` plots loss/IoU/LR curves from a run's `log.csv`.

Note: `road_segmentation/config.py` (yacs-based `_C` config) and `configs.yaml` (a conda
env spec, despite the name) are currently unused by `train.py`, which takes its config
purely from argparse and dumps it to `config.yml`.

## Conventions

- `BiKA_Conv2d`/`BiKA_Linear` currently support only `dilation=1`, `groups=1`; weight and
  bias share shape `(out, in, kh, kw)` / `(out, in)` (per-connection bias).
- Segmentation losses/metrics use `ignore_index=255` for unlabeled pixels.
