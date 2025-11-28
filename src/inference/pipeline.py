"""
Multi-Head Inference Pipeline
=============================

Orchestrates domain routing and expert head inference for glaucoma detection.

The pipeline automatically routes fundus images to the appropriate expert model
based on domain classification, enabling optimized predictions across different
dataset families.

Architecture:
    1. Domain Router: Classifies image domain (rimone, refuge2, g1020, unknown)
    2. Expert Head Selection: Routes to appropriate trained model
    3. Prediction: Expert head provides glaucoma diagnosis

Example:
    >>> from src.inference.pipeline import MultiHeadPipeline
    >>> pipeline = MultiHeadPipeline.from_config("configs/pipeline_v1.yaml")
    >>> result = pipeline.predict("path/to/fundus_image.png")
    >>> print(f"{result.prediction}: {result.confidence:.1%}")
    >>> print(f"Routed via: {result.routed_domain} -> {result.head_used}")
"""

from typing import Dict, Optional, List, Union, Any
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from omegaconf import OmegaConf

from .predictor import GlaucomaPredictor, PredictionResult
from .head_registry import get_head_for_domain, EXPERT_HEADS, HeadConfig, DOMAIN_HEAD_MAPPING
from src.routing import DomainRouter, RoutingResult


@dataclass
class PipelineResult:
    """
    Result from multi-head pipeline prediction.

    Attributes:
        prediction: Predicted class ('normal' or 'glaucoma')
        confidence: Confidence score for the prediction (0-1)
        probabilities: Dictionary mapping class names to probabilities
        routed_domain: Domain identified by router
        routing_confidence: Router's confidence in domain classification
        head_used: Name of the expert head used for prediction
        image_path: Path to input image (if provided as path)
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    routed_domain: str
    routing_confidence: float
    head_used: str
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'routed_domain': self.routed_domain,
            'routing_confidence': self.routing_confidence,
            'head_used': self.head_used,
            'image_path': self.image_path,
        }


class MultiHeadPipeline:
    """
    Routes images to appropriate expert heads based on domain classification.

    The pipeline combines domain routing with expert head inference to provide
    optimized glaucoma detection across different fundus image sources.

    Example:
        >>> pipeline = MultiHeadPipeline.from_config("configs/pipeline_v1.yaml")
        >>> result = pipeline.predict(image)
        >>> print(f"Domain: {result.routed_domain} -> Head: {result.head_used}")
        >>> print(f"Prediction: {result.prediction} ({result.confidence:.1%})")

    Attributes:
        router: DomainRouter for classifying image domains
        heads: Dictionary of expert head predictors
        domain_head_mapping: Maps domains to head names
    """

    def __init__(
        self,
        router: DomainRouter,
        heads: Dict[str, GlaucomaPredictor],
        domain_head_mapping: Dict[str, str]
    ):
        """
        Initialize multi-head pipeline.

        Args:
            router: Initialized DomainRouter
            heads: Dictionary mapping head names to GlaucomaPredictor instances
            domain_head_mapping: Dictionary mapping domains to head names
        """
        self.router = router
        self.heads = heads
        self.domain_head_mapping = domain_head_mapping

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: Optional[str] = None
    ) -> 'MultiHeadPipeline':
        """
        Load pipeline from YAML configuration.

        Args:
            config_path: Path to pipeline configuration file
            device: Device to use ('cuda', 'cpu', or None for auto-detect)

        Returns:
            Initialized MultiHeadPipeline

        Example config (configs/pipeline_v1.yaml):
            router:
              checkpoint: models/routing/domain_classifier_v1.pt
            heads:
              glaucoma_rimone_v1:
                checkpoint: models/production/glaucoma_efficientnet_b0_v1.pt
            domain_mapping:
              rimone: glaucoma_rimone_v1
              refuge2: glaucoma_rimone_v1
        """
        config = OmegaConf.load(config_path)

        # Load router
        router_config = config.get('router', {})
        router_checkpoint = router_config.get('checkpoint', None)

        router = DomainRouter(
            checkpoint_path=router_checkpoint,
            device=device
        )

        # Load expert heads
        heads = {}
        heads_config = config.get('heads', {})
        for head_name, head_config in heads_config.items():
            checkpoint_path = head_config.get('checkpoint')
            if checkpoint_path:
                heads[head_name] = GlaucomaPredictor.from_checkpoint(
                    checkpoint_path,
                    device=device
                )

        # Get domain mapping
        domain_mapping = dict(config.get('domain_mapping', DOMAIN_HEAD_MAPPING))

        return cls(
            router=router,
            heads=heads,
            domain_head_mapping=domain_mapping
        )

    @classmethod
    def from_registry(
        cls,
        router_checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ) -> 'MultiHeadPipeline':
        """
        Create pipeline from the head registry.

        Uses the default domain-to-head mappings from the registry.

        Args:
            router_checkpoint: Path to domain classifier weights
            device: Device to use

        Returns:
            Initialized MultiHeadPipeline
        """
        # Initialize router
        router = DomainRouter(
            checkpoint_path=router_checkpoint,
            device=device
        )

        # Load all registered heads
        heads = {}
        for head_name, head_config in EXPERT_HEADS.items():
            if Path(head_config.checkpoint_path).exists():
                heads[head_name] = GlaucomaPredictor.from_checkpoint(
                    head_config.checkpoint_path,
                    device=device
                )

        return cls(
            router=router,
            heads=heads,
            domain_head_mapping=dict(DOMAIN_HEAD_MAPPING)
        )

    def predict(
        self,
        image: Union[str, Path, Image.Image]
    ) -> PipelineResult:
        """
        Route image and return prediction from appropriate expert head.

        This is the main inference method. It:
        1. Classifies the image domain using the router
        2. Selects the appropriate expert head
        3. Returns the head's glaucoma prediction

        Args:
            image: Image path or PIL Image

        Returns:
            PipelineResult with prediction and routing info
        """
        # Get image path for result
        image_path = str(image) if isinstance(image, (str, Path)) else None

        # Route to domain
        routing = self.router.route(image)

        # Get appropriate head
        head_name = self.domain_head_mapping.get(
            routing.domain,
            list(self.heads.keys())[0]  # Fallback to first head
        )

        if head_name not in self.heads:
            # Fallback if configured head not loaded
            head_name = list(self.heads.keys())[0]

        head = self.heads[head_name]

        # Run inference
        result = head.predict(image)

        return PipelineResult(
            prediction=result.prediction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            routed_domain=routing.domain,
            routing_confidence=routing.confidence,
            head_used=head_name,
            image_path=image_path
        )

    def predict_with_ensemble(
        self,
        image: Union[str, Path, Image.Image],
        head_names: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Use multiple heads and combine predictions via averaging.

        Useful when domain is uncertain or when you want to leverage
        multiple expert opinions.

        Args:
            image: Image path or PIL Image
            head_names: List of head names to use (defaults to all heads)

        Returns:
            PipelineResult with averaged predictions
        """
        head_names = head_names or list(self.heads.keys())

        # Get image path for result
        image_path = str(image) if isinstance(image, (str, Path)) else None

        # Collect predictions from all specified heads
        all_probs = []
        for name in head_names:
            if name in self.heads:
                result = self.heads[name].predict(image)
                all_probs.append(result.probabilities)

        if not all_probs:
            raise ValueError(f"No valid heads found in: {head_names}")

        # Average probabilities
        avg_probs = {}
        for key in all_probs[0].keys():
            avg_probs[key] = sum(p[key] for p in all_probs) / len(all_probs)

        # Determine prediction
        prediction = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[prediction]

        # Get routing info
        routing = self.router.route(image)

        return PipelineResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=avg_probs,
            routed_domain=routing.domain,
            routing_confidence=routing.confidence,
            head_used=f"ensemble({','.join(head_names)})",
            image_path=image_path
        )

    def get_routing_info(
        self,
        image: Union[str, Path, Image.Image]
    ) -> RoutingResult:
        """
        Get domain routing information without running prediction.

        Args:
            image: Image path or PIL Image

        Returns:
            RoutingResult with domain and confidence scores
        """
        return self.router.route(image)

    def get_loaded_heads(self) -> List[str]:
        """Get list of currently loaded expert heads."""
        return list(self.heads.keys())

    def __repr__(self) -> str:
        return (
            f"MultiHeadPipeline("
            f"heads={list(self.heads.keys())}, "
            f"domains={list(self.domain_head_mapping.keys())})"
        )
