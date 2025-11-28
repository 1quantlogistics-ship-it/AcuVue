"""
Domain Router
=============

Routes fundus images to appropriate expert heads based on domain classification.

The router identifies which dataset family an image belongs to (rimone, refuge2,
g1020, or unknown), enabling the multi-head pipeline to select the appropriate
expert model for diagnosis.

IMPORTANT: The router does NOT diagnose glaucoma - it only identifies the image
domain. The actual diagnosis is performed by the expert head selected based on
the routing result.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image

from .domain_classifier import DomainClassifier
from src.inference.preprocessing import preprocess_image


@dataclass
class RoutingResult:
    """
    Result from domain routing.

    Attributes:
        domain: Predicted domain ('rimone', 'refuge2', 'g1020', 'unknown')
        confidence: Confidence score for the predicted domain (0-1)
        all_scores: Dictionary mapping each domain to its probability score
    """
    domain: str
    confidence: float
    all_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'domain': self.domain,
            'confidence': self.confidence,
            'all_scores': self.all_scores,
        }


class DomainRouter:
    """
    Routes images to appropriate expert based on domain classification.

    This router uses a lightweight MobileNetV3-Small classifier to identify
    which dataset family a fundus image belongs to. The routing result is
    used by the multi-head pipeline to select the appropriate expert model.

    Example:
        >>> router = DomainRouter("models/routing/domain_classifier_v1.pt")
        >>> result = router.route("path/to/fundus_image.png")
        >>> print(f"Domain: {result.domain} ({result.confidence:.1%})")
        Domain: rimone (94.2%)

    Attributes:
        device: Torch device (cuda or cpu)
        model: DomainClassifier instance
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize domain router.

        Args:
            checkpoint_path: Path to trained classifier weights.
                           If None, uses untrained model (for testing).
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = DomainClassifier()

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        """
        Load model weights from checkpoint.

        Handles both raw state_dict and wrapped checkpoint formats.

        Args:
            path: Path to checkpoint file
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def classify_domain(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Classify the domain of an image.

        Args:
            image: Image path or PIL Image

        Returns:
            Domain name: 'rimone', 'refuge2', 'g1020', or 'unknown'
        """
        result = self._predict(image)
        return result.domain

    def get_routing_confidence(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, float]:
        """
        Get confidence scores for all domains.

        Args:
            image: Image path or PIL Image

        Returns:
            Dictionary mapping domain names to probability scores
        """
        result = self._predict(image)
        return result.all_scores

    def route(self, image: Union[str, Path, Image.Image]) -> RoutingResult:
        """
        Route an image with full result details.

        This is the main method for routing decisions, providing the predicted
        domain, confidence score, and scores for all domains.

        Args:
            image: Image path or PIL Image

        Returns:
            RoutingResult with domain, confidence, and all_scores
        """
        return self._predict(image)

    def _predict(self, image: Union[str, Path, Image.Image]) -> RoutingResult:
        """
        Internal prediction method.

        Args:
            image: Image path or PIL Image

        Returns:
            RoutingResult with prediction details
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Preprocess image
        tensor = preprocess_image(image, input_size=224, device=self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Build result
        domains = DomainClassifier.DOMAINS
        scores = {d: probs[i].item() for i, d in enumerate(domains)}
        best_idx = torch.argmax(probs).item()

        return RoutingResult(
            domain=domains[best_idx],
            confidence=probs[best_idx].item(),
            all_scores=scores
        )

    def __repr__(self) -> str:
        return f"DomainRouter(device='{self.device}')"
