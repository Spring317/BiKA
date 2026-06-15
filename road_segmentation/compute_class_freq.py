"""Compute per-class pixel frequencies over the BDD100K train masks and
derive median-frequency-balanced (MFB) loss weights.

MFB weight for class c:  median(freq) / freq(c), where freq(c) is the
fraction of (labeled) pixels belonging to c. Rare classes get weights > 1,
dominant classes < 1. Absent classes get weight 0 (won't be predicted).

Run from road_segmentation/:
    python compute_class_freq.py --bdd100k_base /path/to/bdd100k/seg

Outputs: prints a table + a comma-separated weight string, and saves
class_weights.npy for use via:  train.py --class_weights class_weights.npy
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np

from data import BDD100K_NUM_CLASSES, BDD100K_CLASSES

IGNORE_INDEX = 255


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bdd100k_base",
                   default="/storage/student11/bdd100k_seg/bdd100k/seg")
    p.add_argument("--out", default="class_weights.npy")
    p.add_argument("--max_images", default=0, type=int,
                   help="0 = use all train masks; >0 = sample this many (faster).")
    p.add_argument("--clip", default=0.0, type=float,
                   help="If >0, clip weights to [1/clip, clip] to avoid "
                        "extreme values for ultra-rare classes.")
    return p.parse_args()


def main():
    args = parse_args()
    mask_paths = sorted(glob(os.path.join(args.bdd100k_base, "labels", "train", "*.png")))
    if args.max_images > 0:
        mask_paths = mask_paths[: args.max_images]
    assert mask_paths, f"No masks under {args.bdd100k_base}/labels/train"

    counts = np.zeros(BDD100K_NUM_CLASSES, dtype=np.int64)
    for i, mp in enumerate(mask_paths):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        valid = m[m != IGNORE_INDEX]
        binc = np.bincount(valid, minlength=BDD100K_NUM_CLASSES)
        counts += binc[:BDD100K_NUM_CLASSES]
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(mask_paths)} masks")

    total = counts.sum()
    freq = counts / max(total, 1)
    present = freq > 0
    med = np.median(freq[present]) if present.any() else 0.0
    weights = np.zeros(BDD100K_NUM_CLASSES, dtype=np.float64)
    weights[present] = med / freq[present]
    if args.clip > 0:
        weights[present] = np.clip(weights[present], 1.0 / args.clip, args.clip)

    print(f"\n{'class':<16} {'pixels%':>9} {'MFB weight':>11}")
    print("-" * 40)
    for c in range(BDD100K_NUM_CLASSES):
        name = BDD100K_CLASSES.get(c, str(c))
        print(f"{name:<16} {100 * freq[c]:>8.4f}% {weights[c]:>11.3f}")
    print("-" * 40)
    print(f"total labeled pixels: {total:,}  over {len(mask_paths)} masks")

    np.save(args.out, weights.astype(np.float32))
    print(f"\nsaved -> {args.out}")
    print("comma string:")
    print(",".join(f"{w:.4f}" for w in weights))


if __name__ == "__main__":
    main()
