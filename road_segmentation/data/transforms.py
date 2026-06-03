import numpy as np


def mask_to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = mask.shape[:2]
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        one_hot[:, :, c] = (mask == c).astype(np.float32)
    return one_hot


def onehot_to_mask(one_hot: np.ndarray) -> np.ndarray:
    if one_hot.shape[0] < one_hot.shape[-1]:
        one_hot = one_hot.transpose(1, 2, 0)
    return np.argmax(one_hot, axis=-1)


def colorize_mask(mask: np.ndarray, color_dict: dict) -> np.ndarray:
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        colored[mask == class_id] = [int(c * 255) for c in color]
    return colored


__all__ = ["mask_to_onehot", "onehot_to_mask", "colorize_mask"]
