import torch


def iou_score(
    output: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 20,
    ignore_index: int = 255,
):
    """Compute mean IoU and mean Dice for multi-class semantic segmentation.

    Parameters
    ----------
    output : (B, C, H, W) raw logits from the model.
    target : (B, H, W) class-index ground truth, with *ignore_index* for
             pixels to exclude.
    num_classes : number of valid classes (excluding the ignore class).
    ignore_index : label value to ignore (default 255).

    Returns
    -------
    mean_iou  : float
    mean_dice : float
    per_class_iou : list[float]  (length = num_classes)
    """
    smooth = 1e-6

    # (B, H, W) predicted class per pixel
    preds = torch.argmax(output, dim=1)

    # Mask out ignored pixels
    valid = target != ignore_index

    per_class_iou = []
    per_class_dice = []

    for c in range(num_classes):
        pred_c = (preds == c) & valid
        target_c = (target == c) & valid

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        iou = (intersection + smooth) / (union + smooth)
        dice = (2.0 * intersection + smooth) / (pred_c.sum().float() + target_c.sum().float() + smooth)

        per_class_iou.append(iou.item())
        per_class_dice.append(dice.item())

    mean_iou = sum(per_class_iou) / len(per_class_iou)
    mean_dice = sum(per_class_dice) / len(per_class_dice)

    return mean_iou, mean_dice, per_class_iou


def pixel_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """Compute overall pixel accuracy, ignoring *ignore_index* pixels."""
    preds = torch.argmax(output, dim=1)
    valid = target != ignore_index
    correct = ((preds == target) & valid).sum().float()
    total = valid.sum().float()
    return (correct / (total + 1e-8)).item()


__all__ = ["iou_score", "pixel_accuracy"]
