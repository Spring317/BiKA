import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data

from .transforms import (
    IGNORE_INDEX,
    color_jitter,
    horizontal_flip,
    normalize,
    random_scale_crop,
    resize,
    to_chw,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

BDD100K_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "unknown",
}

BDD100K_COLOR_DICT = {
    0: (0.7, 0.7, 0.7),
    1: (0.9, 0.9, 0.2),
    2: (1.0, 0.4980392156862745, 0.054901960784313725),
    3: (1.0, 0.7333333333333333, 0.47058823529411764),
    4: (0.8, 0.5, 0.1),
    5: (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    6: (0.325, 0.196, 0.361),
    7: (1.0, 0.596078431372549, 0.5882352941176471),
    8: (0.2, 0.6, 0.2),
    9: (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    10: (0.5, 0.7, 1.0),
    11: (1.0, 0.0, 0.0),
    12: (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    13: (0.0, 0.0, 1.0),
    14: (0.0, 0.0, 1.0),
    15: (0.0, 0.0, 1.0),
    16: (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    17: (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    18: (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    19: (0, 0, 0),
}

BDD100K_NUM_CLASSES = 20


class BDD100KDataset(torch.utils.data.Dataset):
    """BDD100K semantic segmentation dataset with cv2-based preprocessing.

    Returns
    -------
    image : torch.FloatTensor   (3, H, W)  — ImageNet-normalised
    mask  : torch.LongTensor    (H, W)     — class indices [0..19], 255 = ignore
    meta  : dict                           — {"img_id": str}
    """

    def __init__(
        self,
        img_ids,
        img_dir,
        mask_dir,
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=BDD100K_NUM_CLASSES,
        input_h=192,
        input_w=256,
        is_training=False,
        ignore_index=IGNORE_INDEX,
        mask_suffix="",
    ):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.input_h = input_h
        self.input_w = input_w
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # --- Read image (BGR -> RGB) ---
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Read mask (grayscale class indices) ---
        mask_path = os.path.join(
            self.mask_dir, img_id + self.mask_suffix + self.mask_ext
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Keep 255 as the ignore index — do NOT remap it to a valid class.
        # Pixels with value 255 will be excluded from the loss via
        # ignore_index=255 in CrossEntropyLoss.

        # --- Augmentations & Resize ---
        if self.is_training:
            # Random scale + crop (handles resize internally)
            img, mask = random_scale_crop(
                img, mask, self.input_h, self.input_w,
                scale_range=(0.5, 2.0), ignore_index=self.ignore_index,
            )
            # Random horizontal flip
            if random.random() > 0.5:
                img, mask = horizontal_flip(img, mask)
            # Color jitter (before normalization, on uint8)
            img = color_jitter(img)
        else:
            # Validation: deterministic resize only
            img, mask = resize(img, mask, self.input_h, self.input_w)

        # --- Normalize image (uint8 -> float32, ImageNet stats) ---
        img = normalize(img)
        img = to_chw(img)

        # --- Convert to tensors ---
        img_tensor = torch.from_numpy(img)                      # (3, H, W) float32
        mask_tensor = torch.from_numpy(mask.copy()).long()       # (H, W)   int64

        return img_tensor, mask_tensor, {"img_id": img_id}


__all__ = [
    "BDD100K_CLASSES",
    "BDD100K_COLOR_DICT",
    "BDD100K_NUM_CLASSES",
    "BDD100KDataset",
]
