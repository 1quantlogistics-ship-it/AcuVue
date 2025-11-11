"""
Evaluation metrics for segmentation tasks.

Includes Dice coefficient, IoU, pixel accuracy, and per-class metrics.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).

    Args:
        pred: Predicted mask (B, 1, H, W) or (H, W) with values in [0, 1]
        target: Target mask (B, 1, H, W) or (H, W) with values in [0, 1]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Dice coefficient in [0, 1] (higher is better)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Compute Intersection over Union (IoU) / Jaccard index.

    Args:
        pred: Predicted mask (B, 1, H, W) or (H, W) with values in [0, 1]
        target: Target mask (B, 1, H, W) or (H, W) with values in [0, 1]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        IoU score in [0, 1] (higher is better)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute pixel-wise accuracy.

    Args:
        pred: Predicted mask (B, 1, H, W) or (H, W) with values in [0, 1]
        target: Target mask (B, 1, H, W) or (H, W) with values in [0, 1]
        threshold: Threshold to binarize predictions

    Returns:
        Pixel accuracy in [0, 1] (higher is better)
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    correct = (pred_binary == target_binary).float().sum()
    total = pred_binary.numel()

    accuracy = correct / total
    return accuracy.item()


def sensitivity_specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity.

    Args:
        pred: Predicted mask with values in [0, 1]
        target: Target mask with values in [0, 1]
        threshold: Threshold to binarize predictions

    Returns:
        Tuple of (sensitivity, specificity)
    """
    pred_binary = (pred > threshold).float().view(-1)
    target_binary = (target > threshold).float().view(-1)

    # True positives, false positives, true negatives, false negatives
    tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    tn = ((pred_binary == 0) & (target_binary == 0)).float().sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()

    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn + 1e-7)

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp + 1e-7)

    return sensitivity.item(), specificity.item()


def precision_recall_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        pred: Predicted mask with values in [0, 1]
        target: Target mask with values in [0, 1]
        threshold: Threshold to binarize predictions

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_binary = (pred > threshold).float().view(-1)
    target_binary = (target > threshold).float().view(-1)

    tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return precision.item(), recall.item(), f1.item()


class SegmentationMetrics:
    """
    Comprehensive metrics tracker for segmentation tasks.

    Tracks and averages metrics across batches/epochs.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.accuracies = []
        self.sensitivities = []
        self.specificities = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a new batch.

        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Target masks (B, 1, H, W)
        """
        # Compute metrics
        dice = dice_coefficient(pred, target)
        iou = iou_score(pred, target)
        acc = pixel_accuracy(pred, target)
        sens, spec = sensitivity_specificity(pred, target)
        prec, rec, f1 = precision_recall_f1(pred, target)

        # Store
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.accuracies.append(acc)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.precisions.append(prec)
        self.recalls.append(rec)
        self.f1_scores.append(f1)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get average metrics.

        Returns:
            Dictionary of metric names to values
        """
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'accuracy': np.mean(self.accuracies) if self.accuracies else 0.0,
            'sensitivity': np.mean(self.sensitivities) if self.sensitivities else 0.0,
            'specificity': np.mean(self.specificities) if self.specificities else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0.0,
        }

    def print_metrics(self, prefix: str = ""):
        """
        Print current metrics.

        Args:
            prefix: Optional prefix for output (e.g., "Train", "Val")
        """
        metrics = self.get_metrics()

        print(f"\n{prefix} Metrics:" if prefix else "Metrics:")
        print(f"  Dice:        {metrics['dice']:.4f}")
        print(f"  IoU:         {metrics['iou']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        pred: Predicted masks
        target: Target masks

    Returns:
        Dictionary of all metrics
    """
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    acc = pixel_accuracy(pred, target)
    sens, spec = sensitivity_specificity(pred, target)
    prec, rec, f1 = precision_recall_f1(pred, target)

    return {
        'dice': dice,
        'iou': iou,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
