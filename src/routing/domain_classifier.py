"""
Domain Classifier
=================

Lightweight CNN for classifying fundus image domains.
Classifies images into: rimone, refuge2, g1020, unknown

This classifier identifies which dataset family an image belongs to,
enabling the multi-head pipeline to route to the appropriate expert model.
"""

import torch
import torch.nn as nn
import timm
from typing import List


class DomainClassifier(nn.Module):
    """
    Lightweight domain classifier using MobileNetV3-Small.

    This model classifies fundus images by their source domain (dataset family).
    It uses a small, fast backbone suitable for real-time routing decisions.

    Architecture:
        - Backbone: MobileNetV3-Small (fast inference, ~2.5M params)
        - Global Pool: Average pooling
        - Classifier: Dropout(0.2) -> Linear(576 -> num_domains)
        - Input: 224x224 RGB images (ImageNet normalized)
        - Output: num_domains class logits

    Attributes:
        DOMAINS: List of domain names in order of class indices
        backbone: MobileNetV3-Small feature extractor
        classifier: Classification head
        num_features: Number of features from backbone (576)

    Example:
        >>> model = DomainClassifier(num_domains=4)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)  # -> (1, 4)
    """

    # Domain names in order of class indices
    DOMAINS: List[str] = ['rimone', 'refuge2', 'g1020', 'unknown']

    def __init__(self, num_domains: int = 4, dropout: float = 0.2):
        """
        Initialize domain classifier.

        Args:
            num_domains: Number of domain classes (default: 4)
            dropout: Dropout probability before classifier (default: 0.2)
        """
        super().__init__()

        self.num_domains = num_domains

        # Load MobileNetV3-Small backbone
        # This is a lightweight model suitable for fast routing decisions
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=False,  # We'll load our own weights
            num_classes=0,     # Remove classifier head
            global_pool='avg'  # Keep average pooling (returns 2D features)
        )

        # Feature dimension for MobileNetV3-Small
        self.num_features = self.backbone.num_features  # 576

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W) where H=W=224

        Returns:
            Logits tensor (B, num_domains)
        """
        features = self.backbone(x)  # (B, 576) due to global_pool='avg'
        logits = self.classifier(features)
        return logits

    def get_domain_name(self, class_idx: int) -> str:
        """Get domain name for a class index."""
        if 0 <= class_idx < len(self.DOMAINS):
            return self.DOMAINS[class_idx]
        return 'unknown'

    @classmethod
    def get_num_params(cls) -> int:
        """Get approximate number of parameters."""
        model = cls()
        return sum(p.numel() for p in model.parameters())


def create_domain_classifier(
    num_domains: int = 4,
    dropout: float = 0.2,
    pretrained_backbone: bool = True
) -> DomainClassifier:
    """
    Factory function to create a domain classifier.

    Args:
        num_domains: Number of domain classes
        dropout: Dropout probability
        pretrained_backbone: If True, use ImageNet pretrained backbone

    Returns:
        DomainClassifier instance
    """
    model = DomainClassifier(num_domains=num_domains, dropout=dropout)

    if pretrained_backbone:
        # Load pretrained backbone weights
        pretrained = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        model.backbone.load_state_dict(pretrained.state_dict())

    return model
