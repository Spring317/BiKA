import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyDiceLoss(nn.Module):
    """Combined CrossEntropy + Dice loss for multi-class semantic segmentation.

    Works with raw logits (B, C, H, W) and class-index targets (B, H, W).
    Properly ignores pixels with *ignore_index* in both loss terms.
    """

    def __init__(self, ignore_index=255, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input  : (B, C, H, W) raw logits
        target : (B, H, W) class indices, 255 = ignore
        """
        # --- Cross-Entropy term ---
        ce_loss = self.ce(input, target)

        # --- Dice term (per-class, ignoring ignore_index) ---
        num_classes = input.shape[1]
        smooth = 1e-5

        # Build valid mask and clamp target for one-hot scatter
        valid = (target != self.ignore_index)          # (B, H, W)
        target_clamped = target.clone()
        target_clamped[~valid] = 0                     # safe index for scatter

        # Probabilities: (B, C, H, W)
        probs = F.softmax(input, dim=1)

        # One-hot encode target: (B, C, H, W)
        target_oh = torch.zeros_like(probs)
        target_oh.scatter_(1, target_clamped.unsqueeze(1), 1.0)

        # Zero-out ignored pixels in both predictions and targets
        valid_mask = valid.unsqueeze(1).float()        # (B, 1, H, W)
        probs = probs * valid_mask
        target_oh = target_oh * valid_mask

        # Per-class Dice
        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * target_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)

        dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for multi-class semantic segmentation.

    Reduces the contribution of easy-to-classify pixels so the model
    focuses on hard examples.
    """

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            input, target, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce

        valid = target != self.ignore_index
        return focal[valid].mean()


__all__ = ["CrossEntropyDiceLoss", "FocalLoss"]
