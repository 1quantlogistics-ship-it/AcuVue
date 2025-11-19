"""
Loss Function Factory
=====================

Central factory for building loss functions from specifications.
Enables ARC's Explorer agent to propose different loss functions.

Part of ARC Phase E Week 3: Loss Function Engineering
Dev 2 implementation

Usage:
    >>> spec = {
    ...     "loss_type": "asymmetric_focal",
    ...     "gamma_pos": 2.0,
    ...     "gamma_neg": 0.5
    ... }
    >>> loss_fn = build_loss_from_spec(spec)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .custom_losses import (
    WeightedBCELoss,
    AsymmetricFocalLoss,
    AUCSurrogateLoss,
    DRIRegularizer,
    CombinedLoss
)


def validate_loss_spec(spec: Dict[str, Any]) -> bool:
    """
    Validate loss specification.

    Args:
        spec: Loss specification dict

    Returns:
        True if valid

    Raises:
        ValueError: If spec is invalid
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Loss spec must be dict, got {type(spec)}")

    if "loss_type" not in spec:
        raise ValueError("Loss spec missing 'loss_type' field")

    valid_types = ["weighted_bce", "asymmetric_focal", "auc_surrogate", "cross_entropy"]
    if spec["loss_type"] not in valid_types:
        raise ValueError(
            f"Invalid loss_type: {spec['loss_type']}. "
            f"Valid types: {valid_types}"
        )

    # Validate parameters for each loss type
    if spec["loss_type"] == "weighted_bce":
        if "pos_weight" in spec and spec["pos_weight"] <= 0:
            raise ValueError("pos_weight must be positive")
        if "neg_weight" in spec and spec["neg_weight"] <= 0:
            raise ValueError("neg_weight must be positive")

    elif spec["loss_type"] == "asymmetric_focal":
        if "gamma_pos" in spec and spec["gamma_pos"] < 0:
            raise ValueError("gamma_pos must be non-negative")
        if "gamma_neg" in spec and spec["gamma_neg"] < 0:
            raise ValueError("gamma_neg must be non-negative")

    elif spec["loss_type"] == "auc_surrogate":
        if "margin" in spec and spec["margin"] <= 0:
            raise ValueError("margin must be positive")

    return True


def build_loss_from_spec(
    spec: Dict[str, Any],
    model: Optional[nn.Module] = None,
    train_labels: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Build loss function from specification.

    Args:
        spec: Loss specification with keys:
            - loss_type: "weighted_bce", "asymmetric_focal", "auc_surrogate", "cross_entropy"
            - dri_regularization: bool (optional, default False)
            - lambda_dri: float (optional, default 0.1)
            - <loss-specific params>
        model: Model for DRI regularization (required if dri_regularization=True)
        train_labels: Training labels for auto-computing weights (optional)

    Returns:
        Loss function (nn.Module)

    Example:
        >>> spec = {
        ...     "loss_type": "asymmetric_focal",
        ...     "gamma_pos": 2.0,
        ...     "gamma_neg": 0.5,
        ...     "dri_regularization": True,
        ...     "lambda_dri": 0.1
        ... }
        >>> loss_fn = build_loss_from_spec(spec, model=model)
    """
    # Validate spec
    validate_loss_spec(spec)

    loss_type = spec["loss_type"]
    use_dri = spec.get("dri_regularization", False)

    # Build base loss
    if loss_type == "weighted_bce":
        pos_weight = spec.get("pos_weight", 1.0)
        neg_weight = spec.get("neg_weight", 1.0)

        # Auto-compute weights if training labels provided
        if train_labels is not None and (pos_weight == 1.0 and neg_weight == 1.0):
            pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(train_labels)

        base_loss = WeightedBCELoss(
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            reduction=spec.get("reduction", "mean")
        )

    elif loss_type == "asymmetric_focal":
        base_loss = AsymmetricFocalLoss(
            gamma_pos=spec.get("gamma_pos", 2.0),
            gamma_neg=spec.get("gamma_neg", 1.0),
            clip=spec.get("clip", 0.05),
            reduction=spec.get("reduction", "mean")
        )

    elif loss_type == "auc_surrogate":
        base_loss = AUCSurrogateLoss(
            margin=spec.get("margin", 1.0),
            reduction=spec.get("reduction", "mean")
        )

    elif loss_type == "cross_entropy":
        base_loss = nn.CrossEntropyLoss(reduction=spec.get("reduction", "mean"))

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Add DRI regularization if requested
    if use_dri:
        if model is None:
            raise ValueError("Model required for DRI regularization")

        lambda_dri = spec.get("lambda_dri", 0.1)
        dri_threshold = spec.get("dri_threshold", 0.6)

        return CombinedLoss(
            base_loss=base_loss,
            model=model,
            lambda_dri=lambda_dri,
            dri_threshold=dri_threshold
        )

    return base_loss


def get_loss_summary(loss_fn: nn.Module) -> Dict[str, Any]:
    """
    Get summary of loss function.

    Args:
        loss_fn: Loss function

    Returns:
        Dict with loss function info
    """
    summary = {
        "loss_class": loss_fn.__class__.__name__
    }

    if isinstance(loss_fn, WeightedBCELoss):
        summary["pos_weight"] = loss_fn.pos_weight
        summary["neg_weight"] = loss_fn.neg_weight

    elif isinstance(loss_fn, AsymmetricFocalLoss):
        summary["gamma_pos"] = loss_fn.gamma_pos
        summary["gamma_neg"] = loss_fn.gamma_neg

    elif isinstance(loss_fn, AUCSurrogateLoss):
        summary["margin"] = loss_fn.margin

    elif isinstance(loss_fn, CombinedLoss):
        summary["base_loss"] = loss_fn.base_loss.__class__.__name__
        summary["lambda_dri"] = loss_fn.dri_reg.lambda_dri
        summary["dri_threshold"] = loss_fn.dri_reg.dri_threshold

    return summary


# Example usage
if __name__ == '__main__':
    """Demonstrate loss factory."""
    import torch

    print("=" * 80)
    print("Loss Function Factory")
    print("=" * 80)

    # Test 1: Build weighted BCE loss
    print("\nTest 1: Weighted BCE Loss")
    spec = {
        "loss_type": "weighted_bce",
        "pos_weight": 2.0,
        "neg_weight": 1.0
    }

    loss_fn = build_loss_from_spec(spec)
    summary = get_loss_summary(loss_fn)

    print(f"Spec: {spec}")
    print(f"Summary: {summary}")
    print("✓ Weighted BCE loss created")

    # Test 2: Build asymmetric focal loss
    print("\nTest 2: Asymmetric Focal Loss")
    spec = {
        "loss_type": "asymmetric_focal",
        "gamma_pos": 2.0,
        "gamma_neg": 0.5
    }

    loss_fn = build_loss_from_spec(spec)
    summary = get_loss_summary(loss_fn)

    print(f"Spec: {spec}")
    print(f"Summary: {summary}")
    print("✓ Asymmetric focal loss created")

    # Test 3: Build AUC surrogate loss
    print("\nTest 3: AUC Surrogate Loss")
    spec = {
        "loss_type": "auc_surrogate",
        "margin": 1.0
    }

    loss_fn = build_loss_from_spec(spec)
    summary = get_loss_summary(loss_fn)

    print(f"Spec: {spec}")
    print(f"Summary: {summary}")
    print("✓ AUC surrogate loss created")

    # Test 4: Validation
    print("\nTest 4: Spec Validation")

    try:
        spec = {"loss_type": "invalid_loss"}
        build_loss_from_spec(spec)
    except ValueError as e:
        print(f"✓ Correctly rejected invalid spec: {e}")

    try:
        spec = {"loss_type": "weighted_bce", "pos_weight": -1.0}
        build_loss_from_spec(spec)
    except ValueError as e:
        print(f"✓ Correctly rejected invalid parameter: {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
