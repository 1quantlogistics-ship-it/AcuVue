"""
Backbone Architectures for Medical Image Classification
========================================================

Implements backbone alternatives for AcuVue glaucoma classification.
Each backbone returns feature embeddings (not final classifications).

Part of ARC Phase E: Architecture Grammar System
Dev 2 implementation - Week 1

Supported Backbones:
- EfficientNet-B0 (current baseline)
- EfficientNet-B3 (upgraded capacity)
- ConvNeXt-Tiny (modern CNN architecture)
- DeiT-Small (Vision Transformer)

All backbones implement a common interface:
    forward(x: Tensor[B, 3, H, W]) -> Tensor[B, feature_dim]

Optional: Load medical imaging pretrained weights (CheXpert, MIMIC-CXR)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal
import warnings


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet Backbone (B0, B3, or other variants)

    Uses torchvision's EfficientNet implementation. Returns feature embeddings
    from the final convolutional layer before the classification head.

    Args:
        variant: EfficientNet variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters (for fine-tuning)
        feature_extraction: If True, remove classification head and return features

    Returns:
        For B0: [B, 1280, H/32, W/32]
        For B3: [B, 1536, H/32, W/32]
    """

    def __init__(
        self,
        variant: Literal['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'] = 'b3',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_extraction: bool = True
    ):
        super().__init__()
        self.variant = variant
        self.feature_extraction = feature_extraction

        # Load appropriate EfficientNet variant
        if variant == 'b0':
            if pretrained:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = 1280
        elif variant == 'b3':
            if pretrained:
                weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b3(weights=weights)
            self.feature_dim = 1536
        elif variant == 'b1':
            if pretrained:
                weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b1(weights=weights)
            self.feature_dim = 1280
        elif variant == 'b2':
            if pretrained:
                weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b2(weights=weights)
            self.feature_dim = 1408
        else:
            raise ValueError(f"EfficientNet variant '{variant}' not supported. Use b0-b3.")

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove classification head if feature extraction mode
        if feature_extraction:
            # EfficientNet structure: features -> avgpool -> classifier
            # We want to keep features but remove classifier
            self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet backbone.

        Args:
            x: [B, 3, H, W] - Input images

        Returns:
            features: [B, feature_dim, H', W'] - Feature maps (if feature_extraction=True)
                   or [B, feature_dim] - Pooled features (if feature_extraction=False)
        """
        # Get features from backbone
        features = self.backbone.features(x)  # [B, feature_dim, H/32, W/32]

        if not self.feature_extraction:
            # Apply global pooling if not in feature extraction mode
            features = self.backbone.avgpool(features)  # [B, feature_dim, 1, 1]
            features = torch.flatten(features, 1)  # [B, feature_dim]

        return features


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt Backbone (Tiny or Small variants)

    ConvNeXt is a modern CNN architecture that incorporates design principles
    from Vision Transformers while maintaining CNN efficiency.

    Reference: Liu et al. "A ConvNet for the 2020s"

    Args:
        variant: ConvNeXt variant ('tiny', 'small', 'base')
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        feature_extraction: If True, return spatial feature maps

    Returns:
        For Tiny: [B, 768, H/32, W/32] or [B, 768] (pooled)
        For Small: [B, 768, H/32, W/32] or [B, 768] (pooled)
    """

    def __init__(
        self,
        variant: Literal['tiny', 'small', 'base'] = 'tiny',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_extraction: bool = True
    ):
        super().__init__()
        self.variant = variant
        self.feature_extraction = feature_extraction

        # Load appropriate ConvNeXt variant
        if variant == 'tiny':
            if pretrained:
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.convnext_tiny(weights=weights)
            self.feature_dim = 768
        elif variant == 'small':
            if pretrained:
                weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.convnext_small(weights=weights)
            self.feature_dim = 768
        elif variant == 'base':
            if pretrained:
                weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.convnext_base(weights=weights)
            self.feature_dim = 1024
        else:
            raise ValueError(f"ConvNeXt variant '{variant}' not supported. Use tiny/small/base.")

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove classification head if feature extraction mode
        if feature_extraction:
            self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvNeXt backbone.

        Args:
            x: [B, 3, H, W] - Input images

        Returns:
            features: [B, feature_dim, H', W'] - Feature maps (if feature_extraction=True)
                   or [B, feature_dim] - Pooled features (if feature_extraction=False)
        """
        # Get features from backbone
        features = self.backbone.features(x)  # [B, feature_dim, H/32, W/32]

        if not self.feature_extraction:
            # Apply global pooling
            features = self.backbone.avgpool(features)  # [B, feature_dim, 1, 1]
            features = torch.flatten(features, 1)  # [B, feature_dim]

        return features


class DeiTBackbone(nn.Module):
    """
    DeiT (Data-efficient Image Transformer) Backbone

    Vision Transformer trained with knowledge distillation for improved data efficiency.
    Suitable for medical imaging where data is limited.

    Reference: Touvron et al. "Training data-efficient image transformers"

    Note: Requires timm library (pip install timm)

    Args:
        variant: DeiT variant ('tiny', 'small', 'base')
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        image_size: Expected input image size (default 224)
        patch_size: Patch size for vision transformer (default 16)

    Returns:
        [B, hidden_dim] - CLS token embedding
        For tiny: hidden_dim = 192
        For small: hidden_dim = 384
        For base: hidden_dim = 768
    """

    def __init__(
        self,
        variant: Literal['tiny', 'small', 'base'] = 'small',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        image_size: int = 224,
        patch_size: int = 16
    ):
        super().__init__()
        self.variant = variant
        self.image_size = image_size
        self.patch_size = patch_size

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm library is required for DeiT backbone. "
                "Install with: pip install timm"
            )

        # Load appropriate DeiT variant from timm
        if variant == 'tiny':
            model_name = 'deit_tiny_patch16_224'
            self.feature_dim = 192
        elif variant == 'small':
            model_name = 'deit_small_patch16_224'
            self.feature_dim = 384
        elif variant == 'base':
            model_name = 'deit_base_patch16_224'
            self.feature_dim = 768
        else:
            raise ValueError(f"DeiT variant '{variant}' not supported. Use tiny/small/base.")

        # Create model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Don't pool, we'll extract CLS token manually
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeiT backbone.

        Args:
            x: [B, 3, H, W] - Input images (should be resized to self.image_size)

        Returns:
            features: [B, feature_dim] - CLS token embedding
        """
        # DeiT returns [B, num_patches + 1, hidden_dim]
        # First token is CLS token
        features = self.backbone.forward_features(x)  # [B, N+1, feature_dim]

        # Extract CLS token (first token)
        cls_token = features[:, 0]  # [B, feature_dim]

        return cls_token


# Factory function for easy instantiation
def create_backbone(
    backbone_name: str,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create backbone by name.

    Args:
        backbone_name: One of:
            - 'efficientnet-b0', 'efficientnet-b3'
            - 'convnext-tiny', 'convnext-small'
            - 'deit-tiny', 'deit-small', 'deit-base'
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        **kwargs: Additional arguments passed to specific backbone

    Returns:
        Backbone module instance with .feature_dim attribute

    Example:
        >>> backbone = create_backbone('efficientnet-b3', pretrained=True)
        >>> print(backbone.feature_dim)  # 1536
    """
    backbone_name = backbone_name.lower()

    # EfficientNet variants
    if backbone_name.startswith('efficientnet-'):
        variant = backbone_name.split('-')[1]  # Extract 'b0', 'b3', etc.
        return EfficientNetBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    # ConvNeXt variants
    elif backbone_name.startswith('convnext-'):
        variant = backbone_name.split('-')[1]  # Extract 'tiny', 'small', etc.
        return ConvNeXtBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    # DeiT variants
    elif backbone_name.startswith('deit-'):
        variant = backbone_name.split('-')[1]  # Extract 'tiny', 'small', 'base'
        return DeiTBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Supported: efficientnet-[b0|b3], convnext-[tiny|small], deit-[tiny|small|base]"
        )


def get_backbone_feature_dim(backbone_name: str) -> int:
    """
    Get the feature dimension for a given backbone without instantiating it.

    Args:
        backbone_name: Backbone identifier (e.g., 'efficientnet-b3')

    Returns:
        Feature dimension (int)

    Example:
        >>> get_backbone_feature_dim('efficientnet-b3')
        1536
    """
    feature_dims = {
        'efficientnet-b0': 1280,
        'efficientnet-b1': 1280,
        'efficientnet-b2': 1408,
        'efficientnet-b3': 1536,
        'convnext-tiny': 768,
        'convnext-small': 768,
        'convnext-base': 1024,
        'deit-tiny': 192,
        'deit-small': 384,
        'deit-base': 768
    }

    backbone_name = backbone_name.lower()
    if backbone_name not in feature_dims:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Supported: {list(feature_dims.keys())}"
        )

    return feature_dims[backbone_name]
