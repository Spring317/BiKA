"""Per-class IoU breakdown for a trained BiKASegNet checkpoint.

A single mIoU hides which classes the model fails on. This evaluates a
checkpoint on the BDD100K val split and prints per-class IoU, sorted, so
you can see whether the ceiling is resolution (small/thin classes ~0) or
raw capacity (most classes mediocre).

Run from road_segmentation/, e.g.:
    python eval_per_class.py \
      --checkpoint outputs/bika_segmentation_v5/model_best.pth \
      --base_channels 16
"""
import argparse
import os
from glob import glob

import torch

from data import (
    BDD100KDataset, BDD100K_NUM_CLASSES, BDD100K_CLASSES, LABEL_GROUPINGS,
)
from models import BiKASegNet
from training.metrics import SegmentationMetric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="model_best.pth (raw state_dict) or checkpoint_*.pth")
    p.add_argument("--bdd100k_base",
                   default="/storage/student11/bdd100k_seg/bdd100k/seg")
    p.add_argument("--base_channels", default=16, type=int,
                   help="MUST match the trained model (v5=16, v6/v7=32)")
    p.add_argument("--input_h", default=192, type=int)
    p.add_argument("--input_w", default=256, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--label_grouping", default="none",
                   choices=["none"] + list(LABEL_GROUPINGS))
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "Eval needs CUDA (BiKA kernels)."

    val_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(args.bdd100k_base, "labels", "val", "*.png")))
    ]
    if args.label_grouping != "none":
        g = LABEL_GROUPINGS[args.label_grouping]
        num_classes, class_names, label_map = g["num_classes"], g["classes"], g["label_map"]
    else:
        num_classes, class_names, label_map = BDD100K_NUM_CLASSES, BDD100K_CLASSES, None

    val_ds = BDD100KDataset(
        img_ids=val_ids,
        img_dir=os.path.join(args.bdd100k_base, "images", "val"),
        mask_dir=os.path.join(args.bdd100k_base, "labels", "val"),
        num_classes=num_classes,
        input_h=args.input_h, input_w=args.input_w,
        is_training=False, mask_suffix="_train_id", label_map=label_map,
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = BiKASegNet(num_classes=num_classes,
                       base_channels=args.base_channels).cuda()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    metric = SegmentationMetric(num_classes)
    with torch.no_grad():
        for inp, target, _ in loader:
            out = model(inp.cuda(non_blocking=True))
            metric.update(out, target.cuda(non_blocking=True))

    miou, mdice, per_class = metric.compute()
    rows = sorted(
        ((class_names.get(i, str(i)), v) for i, v in enumerate(per_class)),
        key=lambda r: (r[1] != r[1], r[1]),  # NaN last, then ascending
    )
    print(f"\n{'class':<16} {'IoU':>8}")
    print("-" * 26)
    for name, v in rows:
        print(f"{name:<16} {v:>8.4f}")
    print("-" * 26)
    print(f"{'mean IoU':<16} {miou:>8.4f}")
    print(f"{'mean Dice':<16} {mdice:>8.4f}")
    near_zero = [n for n, v in rows if v == v and v < 0.05]
    if near_zero:
        print(f"\nclasses < 0.05 IoU ({len(near_zero)}): {', '.join(near_zero)}")


if __name__ == "__main__":
    main()
