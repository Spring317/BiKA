# BiKA Road Segmentation

Lightweight road semantic segmentation model using BiKA layers and the same
data preprocessing pipeline as KAN-Road-Segmentation (BDD100K).

## Dataset layout (BDD100K seg)

```
<bdd100k_base>/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── labels/
    ├── train/*_train_id.png
    └── val/*_train_id.png
```

## Training

```
python road_segmentation/train.py \
  --bdd100k_base /path/to/bdd100k/seg \
  --output_dir outputs \
  --batch_size 8 \
  --epochs 200
```

## Model

`BiKASegNet` is a lightweight U-Net-style model built from `BiKA_Conv2d` blocks.
The default `base_channels=16` keeps the parameter count small while preserving
skip connections and multi-scale context.

## Notes

- Uses the same preprocessing as `KAN-Road-Segmentation/Seg_UKAN/train.py`:
  `RandomRotate90`, `HorizontalFlip`, `Resize`, `Normalize`.
- For AMP, use `--use_amp true` (default).
