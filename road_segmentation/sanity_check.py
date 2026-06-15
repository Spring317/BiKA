"""Overfit a handful of BDD100K images to sanity-check the BiKA stack.

If BiKASegNet + the BiKA CUDA backward are healthy, the loss on a fixed
batch of ~10 images should drop sharply and train mIoU should climb well
above the ~0.23 plateau seen in full runs. If the loss refuses to go down
even here, the problem is in the model/kernels, not the training recipe.

Usage (from road_segmentation/):
    python sanity_check.py --bdd100k_base /path/to/bdd100k/seg
"""

import argparse
import os
from glob import glob

import torch
import torch.nn as nn

from data import BDD100KDataset, BDD100K_NUM_CLASSES
from models import BiKASegNet
from training.losses import CrossEntropyDiceLoss
from training.metrics import SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bdd100k_base", default="/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg"
    )
    parser.add_argument("--num_images", default=10, type=int)
    parser.add_argument("--iters", default=600, type=int)
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--input_w", default=256, type=int)
    parser.add_argument("--input_h", default=192, type=int)
    parser.add_argument("--base_channels", default=16, type=int)
    parser.add_argument("--print_every", default=20, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "Sanity check requires CUDA (BiKA kernels)."

    img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(args.bdd100k_base, "labels", "train", "*.png")))
    ][: args.num_images]
    assert img_ids, f"No training masks found under {args.bdd100k_base}"

    dataset = BDD100KDataset(
        img_ids=img_ids,
        img_dir=os.path.join(args.bdd100k_base, "images", "train"),
        mask_dir=os.path.join(args.bdd100k_base, "labels", "train"),
        num_classes=BDD100K_NUM_CLASSES,
        input_h=args.input_h,
        input_w=args.input_w,
        is_training=False,  # deterministic: same batch every iteration
        mask_suffix="_train_id",
    )

    images = torch.stack([dataset[i][0] for i in range(len(dataset))]).cuda()
    masks = torch.stack([dataset[i][1] for i in range(len(dataset))]).cuda()
    print(f"Overfitting {len(dataset)} images at {args.input_w}x{args.input_h}")

    model = BiKASegNet(
        num_classes=BDD100K_NUM_CLASSES, base_channels=args.base_channels
    ).cuda()
    criterion = CrossEntropyDiceLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    first_loss = None
    for it in range(1, args.iters + 1):
        output = model(images)
        loss = criterion(output, masks)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        if it % args.print_every == 0 or it == 1:
            metric = SegmentationMetric(BDD100K_NUM_CLASSES)
            metric.update(output.detach(), masks)
            miou, _, _ = metric.compute()
            print(f"iter {it:4d}  loss {loss.item():.4f}  train mIoU {miou:.4f}")

    final_loss = loss.item()
    metric = SegmentationMetric(BDD100K_NUM_CLASSES)
    metric.update(model(images).detach(), masks)
    final_miou, _, _ = metric.compute()

    print("-" * 50)
    print(f"loss: {first_loss:.4f} -> {final_loss:.4f}, final train mIoU {final_miou:.4f}")
    # Loss alone is misleading: it can halve just by reaching the chance
    # plateau (~ln(20)+dice). Memorizing a tiny fixed batch must show in mIoU.
    if final_miou > 0.4:
        print("PASS: model memorizes the batch — BiKA stack looks healthy.")
    else:
        print(
            "FAIL: cannot overfit a tiny fixed batch. "
            "Suspect the model definition, init, or BiKA kernels."
        )


if __name__ == "__main__":
    main()
