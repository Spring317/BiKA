import math

import torch
import torch.distributed as dist


class SegmentationMetric:
    """Accumulates a confusion matrix over an epoch and computes mIoU / mDice.

    Unlike per-batch mIoU averaging, this gives one consistent epoch-level
    number. Classes absent from both prediction and target are excluded from
    the means instead of counting as a perfect score.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None

    @torch.no_grad()
    def update(self, output: torch.Tensor, target: torch.Tensor):
        """output: (B, C, H, W) logits; target: (B, H, W) class indices."""
        preds = torch.argmax(output, dim=1).flatten()
        target = target.flatten()
        valid = (target != self.ignore_index) & (target < self.num_classes)
        idx = target[valid] * self.num_classes + preds[valid]
        mat = torch.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )
        if self.mat is None:
            self.mat = mat
        else:
            self.mat += mat

    def all_reduce(self):
        if self.mat is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.mat)

    def compute(self):
        """Returns (mean_iou, mean_dice, per_class_iou).

        per_class_iou has NaN for classes absent from both pred and target.
        """
        if self.mat is None:
            return 0.0, 0.0, [math.nan] * self.num_classes

        h = self.mat.float()
        tp = torch.diag(h)
        fp = h.sum(dim=0) - tp
        fn = h.sum(dim=1) - tp
        union = tp + fp + fn
        present = union > 0

        iou = torch.full_like(tp, math.nan)
        dice = torch.full_like(tp, math.nan)
        iou[present] = tp[present] / union[present]
        dice[present] = 2.0 * tp[present] / (2.0 * tp[present] + fp[present] + fn[present])

        if not present.any():
            return 0.0, 0.0, iou.tolist()
        return iou[present].mean().item(), dice[present].mean().item(), iou.tolist()


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
    per_class_iou : list[float]  (length = num_classes, NaN for classes
                    absent from both prediction and target)
    """
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

        # A class absent from both pred and target must not count as a
        # perfect score — exclude it from the means entirely.
        if union == 0:
            per_class_iou.append(math.nan)
            per_class_dice.append(math.nan)
            continue

        iou = intersection / union
        dice = 2.0 * intersection / (pred_c.sum().float() + target_c.sum().float())

        per_class_iou.append(iou.item())
        per_class_dice.append(dice.item())

    present_iou = [v for v in per_class_iou if not math.isnan(v)]
    present_dice = [v for v in per_class_dice if not math.isnan(v)]
    mean_iou = sum(present_iou) / len(present_iou) if present_iou else 0.0
    mean_dice = sum(present_dice) / len(present_dice) if present_dice else 0.0

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


__all__ = ["SegmentationMetric", "iou_score", "pixel_accuracy"]
