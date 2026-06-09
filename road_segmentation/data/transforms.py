import cv2
import numpy as np
import random

# ImageNet statistics (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Preprocessing transforms (cv2-based)
# ---------------------------------------------------------------------------

def resize(image: np.ndarray, mask: np.ndarray, height: int, width: int):
    """Resize image with bilinear interpolation, mask with nearest-neighbor."""
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask


def horizontal_flip(image: np.ndarray, mask: np.ndarray):
    """Flip image and mask horizontally (left-right)."""
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
    return image, mask


def random_scale_crop(
    image: np.ndarray,
    mask: np.ndarray,
    target_h: int,
    target_w: int,
    scale_range: tuple = (0.5, 2.0),
    ignore_index: int = IGNORE_INDEX,
):
    """Randomly scale the image/mask then crop to (target_h, target_w).

    This is one of the most effective augmentations for semantic segmentation
    because it forces the model to handle objects at multiple scales.
    """
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad if scaled image is smaller than target crop
    pad_h = max(target_h - new_h, 0)
    pad_w = max(target_w - new_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        mask = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=ignore_index
        )

    # Random crop
    crop_h, crop_w = image.shape[:2]
    y = random.randint(0, crop_h - target_h)
    x = random.randint(0, crop_w - target_w)
    image = image[y : y + target_h, x : x + target_w]
    mask = mask[y : y + target_h, x : x + target_w]

    return image, mask


def color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> np.ndarray:
    """Apply random brightness, contrast, and saturation jitter.

    Operates on uint8 RGB images.
    """
    img = image.astype(np.float32)

    # Brightness
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = img * factor

    # Contrast
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean

    # Saturation (convert to HSV, scale S, convert back)
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-saturation, saturation)
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def normalize(image: np.ndarray,
              mean: np.ndarray = IMAGENET_MEAN,
              std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """Scale uint8 [0,255] -> float32 [0,1] then apply ImageNet normalization."""
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def to_chw(image: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) numpy array to (C, H, W) float32."""
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


def inverse_normalize(image: np.ndarray,
                      mean: np.ndarray = IMAGENET_MEAN,
                      std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """Undo ImageNet normalization for visualization. Input: (C,H,W) or (H,W,C)."""
    chw = image.ndim == 3 and image.shape[0] == 3
    if chw:
        image = image.transpose(1, 2, 0)
    image = (image * std + mean) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def colorize_mask(mask: np.ndarray, color_dict: dict) -> np.ndarray:
    """Colorize a class-index mask for visualization."""
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        colored[mask == class_id] = [int(c * 255) for c in color]
    return colored


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility / visualization scripts)
# ---------------------------------------------------------------------------

def mask_to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class-index mask to one-hot encoding."""
    h, w = mask.shape[:2]
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        one_hot[:, :, c] = (mask == c).astype(np.float32)
    return one_hot


def onehot_to_mask(one_hot: np.ndarray) -> np.ndarray:
    """Convert one-hot encoded mask back to class indices."""
    if one_hot.shape[0] < one_hot.shape[-1]:
        one_hot = one_hot.transpose(1, 2, 0)
    return np.argmax(one_hot, axis=-1)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IGNORE_INDEX",
    "resize",
    "horizontal_flip",
    "random_scale_crop",
    "color_jitter",
    "normalize",
    "to_chw",
    "inverse_normalize",
    "colorize_mask",
    "mask_to_onehot",
    "onehot_to_mask",
]
