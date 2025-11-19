"""
Model Factory for Architecture Grammar System
==============================================

Central factory for building complete models from architecture specifications.
Assembles: Backbone → Fusion Module → Classification Head

Part of ARC Phase E: Architecture Grammar System
Dev 2 implementation - Week 1

This factory enables ARC's Architect agent to propose architecture_spec dicts
and have them automatically built into PyTorch models for training.

Example architecture_spec:
{
    "backbone": "efficientnet-b3",
    "fusion_type": "film",
    "clinical_dim": 4,
    "head_config": {
        "hidden_dim": 256,
        "num_classes": 2,
        "dropout": 0.3
    }
}
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import warnings

# Import our custom modules
from .backbones import create_backbone, get_backbone_feature_dim, EfficientNetBackbone, ConvNeXtBackbone, DeiTBackbone
from .fusion_modules import create_fusion_module, FiLMLayer, CrossAttentionFusion, GatedFusion, LateFusion


class MultiModalClassifier(nn.Module):
    """
    Complete multi-modal classification model.

    Architecture: Image Input → Backbone → Fusion (with clinical indicators) → Classification Head

    This is the complete model that will be trained by ARC's Executor agent.

    Args:
        backbone: Backbone module (EfficientNet, ConvNeXt, or DeiT)
        fusion: Fusion module (FiLM, CrossAttention, Gated, or Late)
        num_classes: Number of output classes (2 for binary glaucoma classification)
        dropout: Dropout probability for classification head

    Forward:
        image: [B, 3, H, W] - Input fundus images
        clinical: [B, K] - Clinical indicators (CDR, ISNT, vessel density, etc.)
        -> logits: [B, num_classes]
    """

    def __init__(
        self,
        backbone: nn.Module,
        fusion: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.num_classes = num_classes

        # Classification head
        fusion_output_dim = fusion.output_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_output_dim, num_classes)
        )

    def forward(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            image: [B, 3, H, W] - Input images
            clinical: [B, K] - Clinical indicators

        Returns:
            logits: [B, num_classes] - Class logits (not probabilities)
        """
        # Extract CNN features
        cnn_features = self.backbone(image)  # [B, C, H, W] or [B, C]

        # Handle DeiT case (returns pooled features [B, C])
        if len(cnn_features.shape) == 2:
            # DeiT returns [B, C], need to reshape to [B, C, 1, 1] for fusion modules
            B, C = cnn_features.shape
            cnn_features = cnn_features.view(B, C, 1, 1)

        # Fuse with clinical indicators
        fused = self.fusion(cnn_features, clinical)  # [B, fusion_output_dim]

        # Classification
        logits = self.classifier(fused)  # [B, num_classes]

        return logits

    def get_feature_embeddings(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fused feature embeddings (before classification head).

        Useful for visualization, clustering, or downstream tasks.

        Args:
            image: [B, 3, H, W] - Input images
            clinical: [B, K] - Clinical indicators

        Returns:
            embeddings: [B, fusion_output_dim] - Fused feature vectors
        """
        cnn_features = self.backbone(image)

        if len(cnn_features.shape) == 2:
            B, C = cnn_features.shape
            cnn_features = cnn_features.view(B, C, 1, 1)

        fused = self.fusion(cnn_features, clinical)

        return fused


def build_model_from_spec(
    architecture_spec: Dict[str, Any],
    num_classes: int = 2
) -> MultiModalClassifier:
    """
    Build a complete model from an architecture specification dict.

    This is the main entry point for ARC's Architect agent. The agent generates
    an architecture_spec dict, and this function builds the corresponding PyTorch model.

    Args:
        architecture_spec: Dictionary with keys:
            - backbone: str (e.g., 'efficientnet-b3', 'convnext-tiny', 'deit-small')
            - fusion_type: str (e.g., 'film', 'cross_attention', 'gated', 'late')
            - clinical_dim: int (dimension of clinical indicator vector)
            - head_config: dict with optional keys:
                - hidden_dim: int (default 256)
                - dropout: float (default 0.3)
            - backbone_config: dict with optional keys:
                - pretrained: bool (default True)
                - freeze_backbone: bool (default False)
            - fusion_config: dict with optional kwargs for fusion module
        num_classes: Number of output classes (default 2)

    Returns:
        model: MultiModalClassifier instance ready for training

    Example:
        >>> spec = {
        ...     "backbone": "efficientnet-b3",
        ...     "fusion_type": "film",
        ...     "clinical_dim": 4,
        ...     "head_config": {"dropout": 0.3}
        ... }
        >>> model = build_model_from_spec(spec)
        >>> logits = model(images, clinical_indicators)
    """
    # Extract configuration
    backbone_name = architecture_spec['backbone']
    fusion_type = architecture_spec['fusion_type']
    clinical_dim = architecture_spec['clinical_dim']

    # Get optional configs
    head_config = architecture_spec.get('head_config', {})
    backbone_config = architecture_spec.get('backbone_config', {})
    fusion_config = architecture_spec.get('fusion_config', {})

    # Default values
    dropout = head_config.get('dropout', 0.3)
    fusion_output_dim = head_config.get('hidden_dim', 256)
    pretrained = backbone_config.get('pretrained', True)
    freeze_backbone = backbone_config.get('freeze_backbone', False)

    # Build backbone
    # DeiT doesn't support feature_extraction parameter
    if backbone_name.startswith('deit-'):
        backbone = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        backbone = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_extraction=True  # Always return feature maps, not pooled
        )

    # Get backbone feature dimension
    if hasattr(backbone, 'feature_dim'):
        cnn_channels = backbone.feature_dim
    else:
        cnn_channels = get_backbone_feature_dim(backbone_name)

    # Build fusion module
    fusion = create_fusion_module(
        fusion_type=fusion_type,
        cnn_channels=cnn_channels,
        clinical_dim=clinical_dim,
        output_dim=fusion_output_dim,
        **fusion_config
    )

    # Build complete model
    model = MultiModalClassifier(
        backbone=backbone,
        fusion=fusion,
        num_classes=num_classes,
        dropout=dropout
    )

    return model


def validate_architecture_spec(architecture_spec: Dict[str, Any]) -> bool:
    """
    Validate that an architecture specification is well-formed.

    This can be used by ARC's Critic agent to check proposals before training.

    Args:
        architecture_spec: Architecture specification dict

    Returns:
        is_valid: True if spec is valid, raises ValueError otherwise

    Raises:
        ValueError: If spec is malformed or contains invalid values
    """
    # Check required fields
    required_fields = ['backbone', 'fusion_type', 'clinical_dim']
    for field in required_fields:
        if field not in architecture_spec:
            raise ValueError(f"Missing required field: {field}")

    # Validate backbone
    valid_backbones = [
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'convnext-tiny', 'convnext-small', 'convnext-base',
        'deit-tiny', 'deit-small', 'deit-base'
    ]
    backbone = architecture_spec['backbone'].lower()
    if backbone not in valid_backbones:
        raise ValueError(
            f"Invalid backbone: {backbone}. Must be one of {valid_backbones}"
        )

    # Validate fusion type
    valid_fusion_types = ['film', 'cross_attention', 'gated', 'late']
    fusion_type = architecture_spec['fusion_type'].lower()
    if fusion_type not in valid_fusion_types:
        raise ValueError(
            f"Invalid fusion_type: {fusion_type}. Must be one of {valid_fusion_types}"
        )

    # Validate clinical_dim
    clinical_dim = architecture_spec['clinical_dim']
    if not isinstance(clinical_dim, int) or clinical_dim <= 0:
        raise ValueError(f"clinical_dim must be a positive integer, got {clinical_dim}")

    # Validate head_config if present
    if 'head_config' in architecture_spec:
        head_config = architecture_spec['head_config']
        if 'dropout' in head_config:
            dropout = head_config['dropout']
            if not (0.0 <= dropout < 1.0):
                raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        if 'hidden_dim' in head_config:
            hidden_dim = head_config['hidden_dim']
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")

    return True


def get_model_summary(model: MultiModalClassifier) -> Dict[str, Any]:
    """
    Get a summary of model architecture and parameter counts.

    Useful for logging and comparison in ARC's Historian.

    Args:
        model: MultiModalClassifier instance

    Returns:
        summary: Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'backbone_parameters': backbone_params,
        'fusion_parameters': fusion_params,
        'classifier_parameters': classifier_params,
        'backbone_type': model.backbone.__class__.__name__,
        'fusion_type': model.fusion.__class__.__name__,
        'num_classes': model.num_classes
    }

    return summary


# Example usage and testing
if __name__ == '__main__':
    """
    Example usage demonstrating all supported architecture combinations.
    """
    import time

    # Test all 12 combinations (4 fusion × 3 backbones)
    backbones = ['efficientnet-b3', 'convnext-tiny', 'deit-small']
    fusion_types = ['film', 'cross_attention', 'gated', 'late']

    print("=" * 80)
    print("Model Factory - Testing All Architecture Combinations")
    print("=" * 80)

    for backbone in backbones:
        for fusion_type in fusion_types:
            print(f"\nBuilding: {backbone} + {fusion_type}")

            spec = {
                'backbone': backbone,
                'fusion_type': fusion_type,
                'clinical_dim': 4,  # CDR, ISNT, vessel density, entropy
                'head_config': {
                    'hidden_dim': 256,
                    'dropout': 0.3
                },
                'backbone_config': {
                    'pretrained': False,  # Don't download weights for testing
                    'freeze_backbone': False
                }
            }

            # Validate spec
            try:
                validate_architecture_spec(spec)
                print("  ✓ Spec validation passed")
            except ValueError as e:
                print(f"  ✗ Spec validation failed: {e}")
                continue

            # Build model
            try:
                start = time.time()
                model = build_model_from_spec(spec, num_classes=2)
                build_time = time.time() - start
                print(f"  ✓ Model built in {build_time:.3f}s")
            except Exception as e:
                print(f"  ✗ Model build failed: {e}")
                continue

            # Get summary
            summary = get_model_summary(model)
            print(f"  ✓ Total parameters: {summary['total_parameters']:,}")
            print(f"  ✓ Trainable parameters: {summary['trainable_parameters']:,}")

            # Test forward pass (CPU-compatible)
            try:
                batch_size = 2
                image_size = 224
                clinical_dim = 4

                dummy_images = torch.randn(batch_size, 3, image_size, image_size)
                dummy_clinical = torch.randn(batch_size, clinical_dim)

                with torch.no_grad():
                    logits = model(dummy_images, dummy_clinical)

                assert logits.shape == (batch_size, 2), f"Expected shape (2, 2), got {logits.shape}"
                print(f"  ✓ Forward pass successful: {logits.shape}")
            except Exception as e:
                print(f"  ✗ Forward pass failed: {e}")

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)
