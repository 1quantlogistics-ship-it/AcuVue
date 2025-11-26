"""
Loss Functions for Glaucoma Classification
==========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.full_like(inputs, smoothing_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        log_probs = F.log_softmax(inputs, dim=1)
        return -(one_hot * log_probs).sum(dim=1).mean()


def get_loss_function(
    loss_type: str = "cross_entropy",
    num_classes: int = 2,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: Type of loss ("cross_entropy", "focal", "label_smoothing")
        num_classes: Number of classes
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Smoothing parameter for label smoothing
        class_weights: Optional class weights tensor
        
    Returns:
        Loss function module
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(smoothing=label_smoothing, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
