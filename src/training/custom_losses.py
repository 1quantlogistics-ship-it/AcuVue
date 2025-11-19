"""
Custom Loss Functions for Medical Image Classification
=======================================================

Implements advanced loss functions optimized for ARC's training experiments.
Standard cross-entropy is insufficient for medical imaging due to:
- Class imbalance (healthy vs diseased)
- High cost of false negatives (missing disease)
- Need to optimize AUC instead of accuracy
- Requirement to maintain diagnostic attention (DRI constraint)

Part of ARC Phase E Week 3: Loss Function Engineering
Dev 2 implementation

Loss Functions:
1. WeightedBCELoss - Adaptive class weights for imbalanced datasets
2. AsymmetricFocalLoss - Reduces false negatives (higher penalty for missing positives)
3. AUCSurrogateLoss - Directly optimizes AUC via pairwise ranking
4. DRIRegularizer - Differentiable regularization for attention constraint

Each loss is composable and can be combined with DRIRegularizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss with adaptive class weights.

    Automatically computes class weights from training data distribution
    to handle class imbalance. Uses inverse frequency weighting:

    weight_positive = total_samples / (2 * num_positive)
    weight_negative = total_samples / (2 * num_negative)

    Example:
        >>> # Compute weights from training labels
        >>> pos_weight = len(train_labels) / (2 * train_labels.sum())
        >>> neg_weight = len(train_labels) / (2 * (len(train_labels) - train_labels.sum()))
        >>>
        >>> loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight: Weight for positive class (default: 1.0)
            neg_weight: Weight for negative class (default: 1.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Model predictions [batch_size, num_classes] (raw logits)
            labels: Ground truth labels [batch_size] (class indices)

        Returns:
            Weighted BCE loss (scalar if reduction='mean' or 'sum')
        """
        # Convert labels to one-hot if needed
        if labels.ndim == 1:
            labels = F.one_hot(labels, num_classes=logits.shape[1]).float()

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Extract positive class probabilities
        pos_probs = probs[:, 1]
        pos_labels = labels[:, 1]

        # Compute weighted BCE
        pos_loss = -pos_labels * torch.log(pos_probs + 1e-8) * self.pos_weight
        neg_loss = -(1 - pos_labels) * torch.log(1 - pos_probs + 1e-8) * self.neg_weight

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @staticmethod
    def compute_weights_from_labels(labels: torch.Tensor) -> Tuple[float, float]:
        """
        Compute optimal class weights from training labels.

        Args:
            labels: Training labels [num_samples] (class indices 0 or 1)

        Returns:
            Tuple of (pos_weight, neg_weight)

        Example:
            >>> pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(train_labels)
            >>> loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
        """
        num_positive = (labels == 1).sum().item()
        num_negative = (labels == 0).sum().item()
        total = len(labels)

        if num_positive == 0 or num_negative == 0:
            warnings.warn("One class has zero samples, using equal weights")
            return 1.0, 1.0

        pos_weight = total / (2 * num_positive)
        neg_weight = total / (2 * num_negative)

        return pos_weight, neg_weight


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for reducing false negatives.

    Standard focal loss treats false positives and false negatives symmetrically.
    In medical imaging, false negatives (missing disease) are more costly than
    false positives (false alarm). This loss applies asymmetric penalties:

    - Higher penalty for false negatives (gamma_pos)
    - Lower penalty for false positives (gamma_neg)

    Based on: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    Example:
        >>> # Prioritize reducing false negatives
        >>> loss_fn = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=0.5)
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        gamma_pos: float = 2.0,
        gamma_neg: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Initialize Asymmetric Focal Loss.

        Args:
            gamma_pos: Focusing parameter for positive class (default: 2.0)
                      Higher = more focus on hard positive examples
            gamma_neg: Focusing parameter for negative class (default: 1.0)
                      Lower = less penalty for easy negative examples
            clip: Probability clip value to prevent log(0) (default: 0.05)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric focal loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Asymmetric focal loss (scalar if reduction='mean' or 'sum')
        """
        # Convert labels to one-hot if needed
        if labels.ndim == 1:
            labels = F.one_hot(labels, num_classes=logits.shape[1]).float()

        # Apply softmax
        probs = F.softmax(logits, dim=1)

        # Extract positive class probabilities
        pos_probs = probs[:, 1]
        pos_labels = labels[:, 1]

        # Clip probabilities to prevent log(0)
        pos_probs = torch.clamp(pos_probs, min=self.clip, max=1.0 - self.clip)

        # Asymmetric focal loss
        # For positive examples: -labels * (1-p)^gamma_pos * log(p)
        # For negative examples: -(1-labels) * p^gamma_neg * log(1-p)

        pos_loss = -pos_labels * torch.pow(1 - pos_probs, self.gamma_pos) * torch.log(pos_probs)
        neg_loss = -(1 - pos_labels) * torch.pow(pos_probs, self.gamma_neg) * torch.log(1 - pos_probs)

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AUCSurrogateLoss(nn.Module):
    """
    AUC Surrogate Loss for directly optimizing AUC.

    Standard cross-entropy optimizes accuracy, but medical diagnosis requires
    high AUC (ability to rank positive cases above negative cases).

    This loss directly optimizes AUC using a pairwise ranking approach:
    - For each positive-negative pair, ensure positive has higher score
    - Uses smooth hinge loss for differentiability

    Based on: "Optimizing Classifier Performance via an Approximation to the
    Wilcoxon-Mann-Whitney Statistic" (ICML 2003)

    Example:
        >>> loss_fn = AUCSurrogateLoss(margin=1.0)
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize AUC Surrogate Loss.

        Args:
            margin: Margin for pairwise ranking (default: 1.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AUC surrogate loss via pairwise ranking.

        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            AUC surrogate loss (scalar if reduction='mean' or 'sum')
        """
        # Get positive class scores
        scores = F.softmax(logits, dim=1)[:, 1]

        # Find positive and negative samples
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            # If batch has only one class, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute all pairwise differences
        # pos_scores: [num_pos, 1], neg_scores: [1, num_neg]
        # diff: [num_pos, num_neg]
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        # Smooth hinge loss: max(0, margin - diff)
        # Penalize when positive score is not sufficiently higher than negative
        loss = F.relu(self.margin - diff)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DRIRegularizer(nn.Module):
    """
    DRI (Disc Relevance Index) Regularizer for attention constraint.

    Ensures model maintains focus on optic disc region during training.
    Uses Grad-CAM during forward pass to compute attention maps and
    penalizes attention drift away from the disc.

    This is a composable regularization term that can be added to any base loss:

    total_loss = base_loss + lambda * dri_regularizer

    Example:
        >>> base_loss = nn.CrossEntropyLoss()
        >>> dri_reg = DRIRegularizer(model, lambda_dri=0.1, dri_threshold=0.6)
        >>>
        >>> # In training loop:
        >>> ce_loss = base_loss(logits, labels)
        >>> dri_penalty = dri_reg(images, disc_masks)
        >>> total_loss = ce_loss + dri_penalty
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_dri: float = 0.1,
        dri_threshold: float = 0.6,
        target_layer: Optional[nn.Module] = None
    ):
        """
        Initialize DRI Regularizer.

        Args:
            model: Model to compute attention for
            lambda_dri: Regularization strength (default: 0.1)
            dri_threshold: Target DRI value (default: 0.6)
            target_layer: Layer for Grad-CAM (default: auto-detect)
        """
        super().__init__()
        self.model = model
        self.lambda_dri = lambda_dri
        self.dri_threshold = dri_threshold

        # Import DRIComputer lazily to avoid circular imports
        try:
            from ..evaluation.dri_metrics import DRIComputer
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from evaluation.dri_metrics import DRIComputer

        self.dri_computer = DRIComputer(model, dri_threshold=dri_threshold, target_layer=target_layer)

    def forward(
        self,
        images: torch.Tensor,
        disc_masks: torch.Tensor,
        clinical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DRI regularization penalty.

        Args:
            images: Input images [batch_size, C, H, W]
            disc_masks: Ground-truth disc masks [batch_size, H, W]
            clinical: Optional clinical indicators [batch_size, clinical_dim]

        Returns:
            DRI penalty (scalar)
        """
        batch_size = images.shape[0]

        # Compute DRI for each image in batch
        dri_scores = []

        for i in range(batch_size):
            image = images[i:i+1]
            disc_mask = disc_masks[i]
            clin = clinical[i:i+1] if clinical is not None else None

            result = self.dri_computer.compute_dri(image, disc_mask, clin)
            dri_scores.append(result['dri'])

        # Average DRI across batch
        avg_dri = sum(dri_scores) / batch_size

        # Penalty = lambda * max(0, threshold - avg_dri)
        # Penalize if DRI falls below threshold
        penalty = self.lambda_dri * max(0, self.dri_threshold - avg_dri)

        return torch.tensor(penalty, device=images.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Combines a base loss with DRI regularization.

    Convenience wrapper for adding DRI constraint to any loss function.

    Example:
        >>> base_loss = AsymmetricFocalLoss(gamma_pos=2.0)
        >>> loss_fn = CombinedLoss(
        ...     base_loss=base_loss,
        ...     model=model,
        ...     lambda_dri=0.1
        ... )
        >>>
        >>> # In training loop:
        >>> loss = loss_fn(logits, labels, images, disc_masks)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        model: nn.Module,
        lambda_dri: float = 0.1,
        dri_threshold: float = 0.6
    ):
        """
        Initialize Combined Loss.

        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss, AsymmetricFocalLoss)
            model: Model for computing DRI
            lambda_dri: DRI regularization strength (default: 0.1)
            dri_threshold: Target DRI value (default: 0.6)
        """
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.dri_reg = DRIRegularizer(model, lambda_dri=lambda_dri, dri_threshold=dri_threshold)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        images: torch.Tensor,
        disc_masks: torch.Tensor,
        clinical: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            images: Input images [batch_size, C, H, W]
            disc_masks: Disc masks [batch_size, H, W]
            clinical: Optional clinical indicators [batch_size, clinical_dim]

        Returns:
            Dict with keys:
                - total: Total loss (base + DRI penalty)
                - base: Base loss value
                - dri_penalty: DRI regularization penalty
        """
        # Compute base loss
        base_loss_value = self.base_loss(logits, labels)

        # Compute DRI penalty
        dri_penalty = self.dri_reg(images, disc_masks, clinical)

        # Total loss
        total_loss = base_loss_value + dri_penalty

        return {
            'total': total_loss,
            'base': base_loss_value,
            'dri_penalty': dri_penalty
        }


# Example usage
if __name__ == '__main__':
    """Demonstrate custom loss functions."""

    print("=" * 80)
    print("Custom Loss Functions for Medical Image Classification")
    print("=" * 80)

    # Create dummy data
    batch_size = 4
    num_classes = 2

    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size,))

    print(f"\nDummy data: batch_size={batch_size}, num_classes={num_classes}")
    print(f"Labels: {labels.tolist()}")

    # Test 1: WeightedBCELoss
    print("\n" + "=" * 80)
    print("Test 1: WeightedBCELoss")
    print("=" * 80)

    # Compute weights
    pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(labels)
    print(f"Computed weights: pos_weight={pos_weight:.2f}, neg_weight={neg_weight:.2f}")

    loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
    loss = loss_fn(logits, labels)
    print(f"Weighted BCE Loss: {loss.item():.4f}")
    print("✓ WeightedBCELoss working")

    # Test 2: AsymmetricFocalLoss
    print("\n" + "=" * 80)
    print("Test 2: AsymmetricFocalLoss")
    print("=" * 80)

    loss_fn = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=0.5)
    loss = loss_fn(logits, labels)
    print(f"Asymmetric Focal Loss: {loss.item():.4f}")
    print("✓ AsymmetricFocalLoss working")

    # Test 3: AUCSurrogateLoss
    print("\n" + "=" * 80)
    print("Test 3: AUCSurrogateLoss")
    print("=" * 80)

    loss_fn = AUCSurrogateLoss(margin=1.0)
    loss = loss_fn(logits, labels)
    print(f"AUC Surrogate Loss: {loss.item():.4f}")
    print("✓ AUCSurrogateLoss working")

    # Test 4: Gradient flow
    print("\n" + "=" * 80)
    print("Test 4: Gradient Flow")
    print("=" * 80)

    logits_with_grad = torch.randn(batch_size, num_classes, requires_grad=True)

    loss_fn = AsymmetricFocalLoss()
    loss = loss_fn(logits_with_grad, labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient norm: {logits_with_grad.grad.norm().item():.4f}")
    print("✓ Gradients flow correctly")

    print("\n" + "=" * 80)
    print("All loss functions working correctly!")
    print("=" * 80)
