"""
Production Inference Pipeline
=============================

This module provides production-ready inference for glaucoma detection
from fundus images using the validated EfficientNet-B0 model.

Quick Start:
    >>> from src.inference import GlaucomaPredictor
    >>> predictor = GlaucomaPredictor.from_checkpoint(
    ...     "models/production/glaucoma_efficientnet_b0_v1.pt"
    ... )
    >>> result = predictor.predict("path/to/fundus.png")
    >>> print(f"{result.prediction}: {result.confidence:.1%}")

Classes:
    GlaucomaPredictor: High-level inference wrapper
    GlaucomaClassifier: EfficientNet-B0 based model
    PredictionResult: Dataclass for prediction results
    InferenceConfig: Configuration settings

Model Performance (Hospital-Based Evaluation):
    - Test AUC: 93.7%
    - Test Accuracy: 76.5%
    - Test Sensitivity: 74.4%
    - Test Specificity: 91.7%
"""

from .predictor import GlaucomaPredictor, GlaucomaClassifier, PredictionResult
from .config import InferenceConfig, ModelMetadata, PRODUCTION_V1_METADATA
from .preprocessing import (
    preprocess_image,
    load_and_preprocess,
    get_inference_transforms,
    unnormalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .model_registry import (
    get_model_path,
    get_model_config,
    get_model_metadata,
    get_available_versions,
    get_model_info,
    validate_model_exists,
    PRODUCTION_MODELS,
    DEFAULT_MODEL_VERSION,
)
from .batch_processor import BatchProcessor, BatchResult

__all__ = [
    # Main classes
    'GlaucomaPredictor',
    'GlaucomaClassifier',
    'PredictionResult',
    'BatchProcessor',
    'BatchResult',

    # Configuration
    'InferenceConfig',
    'ModelMetadata',
    'PRODUCTION_V1_METADATA',

    # Preprocessing
    'preprocess_image',
    'load_and_preprocess',
    'get_inference_transforms',
    'unnormalize',
    'IMAGENET_MEAN',
    'IMAGENET_STD',

    # Model registry
    'get_model_path',
    'get_model_config',
    'get_model_metadata',
    'get_available_versions',
    'get_model_info',
    'validate_model_exists',
    'PRODUCTION_MODELS',
    'DEFAULT_MODEL_VERSION',
]
