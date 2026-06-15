"""Visualize BiKASegNet predictions vs ground truth on BDD100K val images.

Saves a grid PNG with rows = samples, columns = [input | ground truth | prediction],
colorized with the BDD100K palette. Makes failure modes obvious at a glance
(e.g. rare classes never predicted, vehicles all collapsed into 'car').

Run from road_segmentation/:
    python plot_predictions.py \
      --checkpoint outputs/bika_segmentation_v5/model_best.pth \
      --base_channels 16 --num_samples 6
"""
import argparse
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import (
    BDD100KDataset, BDD100K_NUM_CLASSES, BDD100K_COLOR_DICT,
    LABEL_GROUPINGS, colorize_mask, inverse_normalize,
)
from models import BiKASegNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="",
                   help="model_best.pth / checkpoint_*.pth. Empty = random "
                        "weights (only useful to preview ground-truth masks).")
    p.add_argument("--bdd100k_base",
                   default="/storage/student11/bdd100k_seg/bdd100k/seg")
    p.add_argument("--base_channels", default=16, type=int,
                   help="MUST match the trained model (v5=16, v6/v7=32).")
    p.add_argument("--num_samples", default=6, type=int)
    p.add_argument("--input_h", default=192, type=int)
    p.add_argument("--input_w", default=256, type=int)
    p.add_argument("--out", default="predictions.png")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--label_grouping", default="none",
                   choices=["none"] + list(LABEL_GROUPINGS))
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "BiKA kernels need CUDA."

    val_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(args.bdd100k_base, "labels", "val", "*.png")))
    ]
    assert val_ids, f"No val masks under {args.bdd100k_base}/labels/val"
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(val_ids), size=min(args.num_samples, len(val_ids)), replace=False)

    if args.label_grouping != "none":
        g = LABEL_GROUPINGS[args.label_grouping]
        num_classes, color_dict, label_map = g["num_classes"], g["color_dict"], g["label_map"]
    else:
        num_classes, color_dict, label_map = BDD100K_NUM_CLASSES, BDD100K_COLOR_DICT, None

    ds = BDD100KDataset(
        img_ids=[val_ids[i] for i in pick],
        img_dir=os.path.join(args.bdd100k_base, "images", "val"),
        mask_dir=os.path.join(args.bdd100k_base, "labels", "val"),
        num_classes=num_classes,
        input_h=args.input_h, input_w=args.input_w,
        is_training=False, mask_suffix="_train_id", label_map=label_map,
    )

    model = BiKASegNet(num_classes=num_classes,
                       base_channels=args.base_channels).cuda()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        print(f"loaded {args.checkpoint}")
    else:
        print("WARNING: no checkpoint, predictions are from random weights")
    model.eval()

    n = len(ds)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 4, n * 3))
    if n == 1:
        axes = axes[None, :]
    axes[0, 0].set_title("input")
    axes[0, 1].set_title("ground truth")
    axes[0, 2].set_title("prediction")

    with torch.no_grad():
        for r in range(n):
            img, mask, meta = ds[r]
            logits = model(img.unsqueeze(0).cuda())
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

            rgb = inverse_normalize(img.numpy())  # (H,W,3) uint8
            gt_col = colorize_mask(mask.numpy().astype(np.uint8), color_dict)
            pr_col = colorize_mask(pred, color_dict)

            for c, im in enumerate((rgb, gt_col, pr_col)):
                axes[r, c].imshow(im)
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(meta["img_id"][:12], rotation=0, labelpad=40,
                                  fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
