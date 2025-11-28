"""
Inference Configuration
=======================

Configuration dataclasses for the production inference pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class InferenceConfig:
    """
    Configuration for glaucoma inference.

    Attributes:
        model_path: Path to model checkpoint (.pt file)
        device: Device to run inference on ('cuda', 'cpu', or 'auto')
        input_size: Expected input image size (height, width)
        class_names: List of class names in order
        confidence_threshold: Minimum confidence for positive prediction
    """
    model_path: str = "models/production/glaucoma_efficientnet_b0_v1.pt"
    device: str = "auto"
    input_size: Tuple[int, int] = (224, 224)
    class_names: List[str] = field(default_factory=lambda: ['normal', 'glaucoma'])
    confidence_threshold: float = 0.5

    def __post_init__(self):
        """Validate configuration."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if len(self.class_names) != 2:
            raise ValueError("class_names must have exactly 2 classes for binary classification")


@dataclass
class ModelMetadata:
    """
    Metadata about a trained model.

    Attributes:
        version: Model version string
        architecture: Model architecture name
        library: Library used (timm, torchvision)
        input_size: Expected input size
        num_classes: Number of output classes
        classes: Class names
        metrics: Dictionary of performance metrics
        training_config: Training configuration used
    """
    version: str
    architecture: str
    library: str = "timm"
    input_size: int = 224
    num_classes: int = 2
    classes: List[str] = field(default_factory=lambda: ['normal', 'glaucoma'])
    metrics: dict = field(default_factory=dict)
    training_config: Optional[dict] = None


# Default production model metadata
PRODUCTION_V1_METADATA = ModelMetadata(
    version="v1",
    architecture="efficientnet_b0",
    library="timm",
    input_size=224,
    num_classes=2,
    classes=['normal', 'glaucoma'],
    metrics={
        'test_auc': 0.937,
        'test_accuracy': 0.765,
        'test_sensitivity': 0.744,
        'test_specificity': 0.917,
    },
    training_config={
        'optimizer': 'adamw',
        'lr': 0.0001,
        'epochs': 30,
        'batch_size': 32,
        'splitting_strategy': 'hospital_based',
        'test_hospitals': ['r1'],
        'train_val_hospitals': ['r2', 'r3'],
    }
)
