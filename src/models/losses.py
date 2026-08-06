"""
Training losses for crack segmentation (50/50 BCE + Dice by default).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + soft-Dice loss for extreme class imbalance.

    With ``aux_logits`` (deep supervision), each auxiliary head contributes
    its own BCE + Dice term, weighted by ``aux_weight`` and averaged.
    """
    def __init__(self, bce_weight: float = 0.5, aux_weight: float = 0.25):
        super().__init__()
        self.bce_weight = bce_weight
        self.aux_weight = aux_weight
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _dice_term(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()

    def _head_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return self.bce_weight * self.bce(logits, targets) + (1.0 - self.bce_weight) * self._dice_term(probs, targets)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: list | None = None,
    ) -> torch.Tensor:
        loss = self._head_loss(logits, targets)
        if aux_logits:
            aux_loss = torch.stack([self._head_loss(a, targets) for a in aux_logits]).mean()
            loss = loss + self.aux_weight * aux_loss
        return loss
