"""
EfficientNet Classifier for Glaucoma Detection
===============================================

Simple EfficientNet-based classifier for binary classification.
Used by train_classification.py.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier.
    
    Args:
        model_name: EfficientNet variant (efficientnet_b0, efficientnet_b3, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=""  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                feature_dim = features.shape[1]
            else:
                feature_dim = features.shape[-1]
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        
        # Global average pooling if needed
        if len(features.shape) == 4:
            features = self.pool(features)
            features = features.flatten(1)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    **kwargs
) -> EfficientNetClassifier:
    """
    Factory function to create an EfficientNet classifier.
    
    Args:
        model_name: EfficientNet variant
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        
    Returns:
        EfficientNetClassifier instance
    """
    return EfficientNetClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
