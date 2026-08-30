#!/usr/bin/env python3
"""
Fine-tuning script to transfer weights from a trained BiKASegNet checkpoint
to the ultra-fast BiKASegNet_Light model.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bika.segmentation import BiKASegNet_Light
from road_segmentation.data import BDD100KDataset, LABEL_GROUPINGS
from road_segmentation.training.metrics import SegmentationMetric
import road_segmentation.training.losses as losses


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BiKASegNet_Light from pretrained BiKASegNet checkpoint")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to pretrained model_best.pth (e.g. BiKASegNet_lane_fg_32ch_Round3_Fixes/model_best.pth)")
    p.add_argument("--base_channels", type=int, default=32, choices=[16, 32],
                   help="Base channels (16 or 32, should match checkpoint)")
    p.add_argument("--label_grouping", type=str, default="lane_fg", choices=["lane_fg", "drive5", "bench19"],
                   help="Segmentation task grouping")
    p.add_argument("--bdd100k_base", type=str, default="/mnt/storage/M2_internship/bdd100k_seg/bdd100k/seg",
                   help="BDD100K segmentation dataset root")
    p.add_argument("--input_h", type=int, default=192)
    p.add_argument("--input_w", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for pretrained backbone")
    p.add_argument("--lr_decoder", type=float, default=5e-4, help="Learning rate for newly initialized decoder")
    p.add_argument("--output_dir", type=str, default="road_segmentation/outputs/BiKASegNet_Light_finetuned")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--legacy_mode", action="store_true", default=True,
                   help="Use ReLU (fusable at inference for 2x CPU speedup). Use --no_legacy_mode for RPReLU.")
    p.add_argument("--no_legacy_mode", dest="legacy_mode", action="store_false",
                   help="Use RPReLU (non-fusable, slower inference but potentially better training)")
    return p.parse_args()


def load_pretrained_backbone(model: BiKASegNet_Light, checkpoint_path: str):
    """Transfer pretrained weights from BiKASegNet to BiKASegNet_Light."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    
    clean_state = {}
    for k, v in state.items():
        clean_k = k.replace("module.", "")
        # Map old sequential names if needed
        clean_k = clean_k.replace(".block.0.", ".conv1.")
        clean_k = clean_k.replace(".block.1.", ".bn1.")
        clean_k = clean_k.replace(".block.3.", ".conv2.")
        clean_k = clean_k.replace(".block.4.", ".bn2.")
        clean_state[clean_k] = v

    model_state = model.state_dict()
    transferred = []
    skipped = []

    # Mapping logic:
    # enc2 -> stage1
    # enc3 -> stage2
    # bottleneck -> bottleneck
    key_mapping = {
        "enc2.": "stage1.",
        "enc3.": "stage2.",
        "bottleneck.": "bottleneck.",
    }

    for k, v in clean_state.items():
        mapped_k = None
        for src_prefix, dst_prefix in key_mapping.items():
            if k.startswith(src_prefix):
                mapped_k = k.replace(src_prefix, dst_prefix, 1)
                break
        
        if mapped_k and mapped_k in model_state and model_state[mapped_k].shape == v.shape:
            model_state[mapped_k] = v
            transferred.append(mapped_k)
        elif k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            transferred.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_state, strict=False)
    print(f"\n[Weight Transfer] Successfully loaded {len(transferred)} layers from '{checkpoint_path}'.")
    print(f"[Weight Transfer] Newly initialized decoder layers: {[k for k in model_state.keys() if k not in transferred]}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = LABEL_GROUPINGS[args.label_grouping]["num_classes"]
    print("=" * 80)
    print(f"  Fine-Tuning BiKASegNet_Light ({args.base_channels} base channels)")
    print(f"  Pretrained Checkpoint: {args.checkpoint}")
    print(f"  Task / Classes       : {args.label_grouping} ({num_classes} classes)")
    print(f"  Resolution           : {args.input_h} × {args.input_w}")
    print(f"  Output Dir           : {args.output_dir}")
    print(f"  legacy_mode          : {args.legacy_mode} ({'ReLU, fusable' if args.legacy_mode else 'RPReLU, non-fusable'})")
    print("=" * 80)

    # 1. Instantiate BiKASegNet_Light
    model = BiKASegNet_Light(
        num_classes=num_classes,
        base_channels=args.base_channels,
        legacy_mode=args.legacy_mode,
    ).to(device)

    # 2. Transfer pretrained backbone weights
    load_pretrained_backbone(model, args.checkpoint)

    # 3. Differential Parameter Groups
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj" in name or "fuse" in name:
            decoder_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
        {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_cuda = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    criterion = losses.CrossEntropyDiceLoss(ignore_index=255)

    # 4. Data Loaders
    from glob import glob
    bdd = args.bdd100k_base
    label_map = LABEL_GROUPINGS[args.label_grouping]["label_map"] if args.label_grouping != "none" else None

    train_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "train", "*.png")))
    ]
    val_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "val", "*.png")))
    ]

    print(f"[Dataset] Loaded {len(train_img_ids)} training images, {len(val_img_ids)} validation images.")

    train_dataset = BDD100KDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(bdd, "images", "train"),
        mask_dir=os.path.join(bdd, "labels", "train"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=num_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        is_training=True,
        mask_suffix="_train_id",
        label_map=label_map,
    )
    val_dataset = BDD100KDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(bdd, "images", "val"),
        mask_dir=os.path.join(bdd, "labels", "val"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=num_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        is_training=False,
        mask_suffix="_train_id",
        label_map=label_map,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    best_miou = 0.0

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            images, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_cuda):
                preds = model(images)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        metric = SegmentationMetric(num_classes=num_classes)
        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch[0].to(device), batch[1].to(device)
                preds = model(images)
                metric.update(preds, targets)

        miou, mdice, iou_per_class = metric.compute()
        print(f"Epoch {epoch:02d} | Val mIoU: {miou*100:.2f}% | Loss: {total_loss/len(train_loader):.4f}")

        # Save Best Model
        if miou > best_miou:
            best_miou = miou
            save_path = os.path.join(args.output_dir, "model_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "label_grouping": args.label_grouping,
                "base_channels": args.base_channels,
                "legacy_mode": args.legacy_mode,
            }, save_path)
            print(f"  >>> Saved new best model to '{save_path}' (mIoU: {best_miou*100:.2f}%)")

        # Save last checkpoint every epoch (crash recovery)
        last_path = os.path.join(args.output_dir, "checkpoint_last.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "label_grouping": args.label_grouping,
            "base_channels": args.base_channels,
            "legacy_mode": args.legacy_mode,
        }, last_path)

    print(f"\nTraining complete. Best mIoU: {best_miou*100:.2f}%")


if __name__ == "__main__":
    main()
