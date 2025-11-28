"""
Glaucoma Predictor
==================

Production inference classes for glaucoma detection from fundus images.
"""

import torch
import torch.nn as nn
import timm
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from PIL import Image

from .preprocessing import preprocess_image, load_and_preprocess
from .config import InferenceConfig, PRODUCTION_V1_METADATA


@dataclass
class PredictionResult:
    """
    Result from a single prediction.

    Attributes:
        prediction: Class name ('normal' or 'glaucoma')
        confidence: Confidence score for the predicted class (0-1)
        probabilities: Dictionary of class probabilities
        image_path: Path to image (if provided)
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    image_path: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"PredictionResult(prediction='{self.prediction}', "
            f"confidence={self.confidence:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'image_path': self.image_path,
        }


class GlaucomaClassifier(nn.Module):
    """
    EfficientNet-B0 based glaucoma classifier.

    This architecture matches the production model exactly:
    - Backbone: EfficientNet-B0 (timm, with global_pool='avg')
    - Classifier: Dropout(0.3) -> Linear(1280, 2)

    Args:
        num_classes: Number of output classes (default: 2)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Load EfficientNet-B0 backbone with integrated pooling
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,  # We load our own weights
            num_classes=0,     # Remove classifier head
            global_pool='avg'  # Keep average pooling (returns 2D features)
        )

        # Feature dimension for EfficientNet-B0
        self.num_features = self.backbone.num_features  # 1280

        # Classifier head (matches training)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Logits tensor (B, num_classes)
        """
        features = self.backbone(x)  # (B, 1280) due to global_pool='avg'
        logits = self.classifier(features)
        return logits


class GlaucomaPredictor:
    """
    High-level inference wrapper for glaucoma detection.

    Provides a simple API for loading models and running predictions
    on single images or batches.

    Example:
        >>> predictor = GlaucomaPredictor.from_checkpoint(
        ...     "models/production/glaucoma_efficientnet_b0_v1.pt"
        ... )
        >>> result = predictor.predict("path/to/fundus.png")
        >>> print(f"{result.prediction}: {result.confidence:.1%}")
    """

    def __init__(
        self,
        model: GlaucomaClassifier,
        device: str = 'cpu',
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize predictor with a loaded model.

        Use GlaucomaPredictor.from_checkpoint() for standard usage.

        Args:
            model: Loaded GlaucomaClassifier model
            device: Device model is on
            config: Inference configuration
        """
        self.model = model
        self.device = device
        self.config = config or InferenceConfig(device=device)
        self.class_names = self.config.class_names

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        config: Optional[InferenceConfig] = None
    ) -> 'GlaucomaPredictor':
        """
        Load predictor from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            config: Optional inference configuration

        Returns:
            Initialized GlaucomaPredictor

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint is invalid
        """
        # Resolve device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Verify checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model
        model = GlaucomaClassifier(num_classes=2, dropout=0.3)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Create config if not provided
        if config is None:
            config = InferenceConfig(
                model_path=str(checkpoint_path),
                device=device
            )

        return cls(model=model, device=device, config=config)

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        return_features: bool = False
    ) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            image: Image path or PIL Image
            return_features: If True, include feature vector (not implemented)

        Returns:
            PredictionResult with prediction, confidence, and probabilities
        """
        # Handle path vs PIL Image
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            input_tensor = load_and_preprocess(
                image_path,
                input_size=self.config.input_size[0],
                device=self.device
            )
        else:
            input_tensor = preprocess_image(
                image,
                input_size=self.config.input_size[0],
                device=self.device
            )

        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        # Build result
        prediction = self.class_names[pred_idx]
        confidence = probs[0, pred_idx].item()
        probabilities = {
            name: probs[0, i].item()
            for i, name in enumerate(self.class_names)
        }

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            image_path=image_path
        )

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """
        Run inference on multiple images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images to process at once

        Returns:
            List of PredictionResult objects
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Process batch
            batch_tensors = []
            batch_paths = []

            for img in batch_images:
                if isinstance(img, (str, Path)):
                    batch_paths.append(str(img))
                    tensor = load_and_preprocess(
                        str(img),
                        input_size=self.config.input_size[0],
                        device=self.device
                    )
                else:
                    batch_paths.append(None)
                    tensor = preprocess_image(
                        img,
                        input_size=self.config.input_size[0],
                        device=self.device
                    )
                batch_tensors.append(tensor)

            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0)

            # Run inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)

            # Build results
            for j, (pred_idx, path) in enumerate(zip(pred_indices, batch_paths)):
                pred_idx = pred_idx.item()
                prediction = self.class_names[pred_idx]
                confidence = probs[j, pred_idx].item()
                probabilities = {
                    name: probs[j, k].item()
                    for k, name in enumerate(self.class_names)
                }

                results.append(PredictionResult(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    image_path=path
                ))

        return results

    def __repr__(self) -> str:
        return (
            f"GlaucomaPredictor(device='{self.device}', "
            f"classes={self.class_names})"
        )
