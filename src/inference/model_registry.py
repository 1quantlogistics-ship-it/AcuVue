"""
Model Registry
==============

Production model registry for AcuVue inference.
Manages model versions, paths, and metadata.
"""

from pathlib import Path
from typing import Dict, Optional, Any
import json

from .config import ModelMetadata, PRODUCTION_V1_METADATA


# Registry of production models
PRODUCTION_MODELS: Dict[str, Dict[str, Any]] = {
    "v1": {
        "path": "models/production/glaucoma_efficientnet_b0_v1.pt",
        "architecture": "efficientnet_b0",
        "library": "timm",
        "input_size": 224,
        "classes": ["normal", "glaucoma"],
        "dropout": 0.3,
        "metrics": {
            "test_auc": 0.937,
            "test_accuracy": 0.765,
            "test_sensitivity": 0.744,
            "test_specificity": 0.917,
        },
        "training": {
            "splitting_strategy": "hospital_based",
            "test_hospitals": ["r1"],
            "train_val_hospitals": ["r2", "r3"],
            "epochs": 30,
            "optimizer": "adamw",
            "lr": 0.0001,
        },
        "metadata": PRODUCTION_V1_METADATA,
    }
}

# Default model version
DEFAULT_MODEL_VERSION = "v1"


def get_available_versions() -> list:
    """Get list of available model versions."""
    return list(PRODUCTION_MODELS.keys())


def get_model_path(version: Optional[str] = None, project_root: Optional[Path] = None) -> Path:
    """
    Get the path to a model checkpoint.

    Args:
        version: Model version (default: latest)
        project_root: Project root directory (default: auto-detect)

    Returns:
        Absolute path to model checkpoint

    Raises:
        KeyError: If version doesn't exist
    """
    version = version or DEFAULT_MODEL_VERSION

    if version not in PRODUCTION_MODELS:
        available = ", ".join(get_available_versions())
        raise KeyError(f"Unknown model version: {version}. Available: {available}")

    relative_path = PRODUCTION_MODELS[version]["path"]

    if project_root is None:
        # Try to find project root by looking for src/ directory
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src").exists() and (parent / "models").exists():
                project_root = parent
                break
        else:
            project_root = Path.cwd()

    return project_root / relative_path


def get_model_config(version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for a model version.

    Args:
        version: Model version (default: latest)

    Returns:
        Model configuration dictionary

    Raises:
        KeyError: If version doesn't exist
    """
    version = version or DEFAULT_MODEL_VERSION

    if version not in PRODUCTION_MODELS:
        available = ", ".join(get_available_versions())
        raise KeyError(f"Unknown model version: {version}. Available: {available}")

    return PRODUCTION_MODELS[version].copy()


def get_model_metadata(version: Optional[str] = None) -> ModelMetadata:
    """
    Get metadata for a model version.

    Args:
        version: Model version (default: latest)

    Returns:
        ModelMetadata object

    Raises:
        KeyError: If version doesn't exist
    """
    config = get_model_config(version)
    return config.get("metadata", PRODUCTION_V1_METADATA)


def register_model(
    version: str,
    path: str,
    architecture: str = "efficientnet_b0",
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
) -> None:
    """
    Register a new model version.

    Args:
        version: Version string (e.g., "v2")
        path: Relative path to checkpoint from project root
        architecture: Model architecture name
        metrics: Dictionary of performance metrics
        **kwargs: Additional configuration
    """
    if version in PRODUCTION_MODELS:
        raise ValueError(f"Model version {version} already exists")

    PRODUCTION_MODELS[version] = {
        "path": path,
        "architecture": architecture,
        "library": "timm",
        "input_size": kwargs.get("input_size", 224),
        "classes": kwargs.get("classes", ["normal", "glaucoma"]),
        "dropout": kwargs.get("dropout", 0.3),
        "metrics": metrics or {},
        **kwargs
    }


def validate_model_exists(version: Optional[str] = None, project_root: Optional[Path] = None) -> bool:
    """
    Check if model checkpoint file exists.

    Args:
        version: Model version to check
        project_root: Project root directory

    Returns:
        True if model file exists
    """
    try:
        path = get_model_path(version, project_root)
        return path.exists()
    except KeyError:
        return False


def get_model_info(version: Optional[str] = None) -> str:
    """
    Get human-readable model information.

    Args:
        version: Model version

    Returns:
        Formatted string with model information
    """
    config = get_model_config(version)
    metrics = config.get("metrics", {})

    lines = [
        f"Model: {config['architecture']} ({version or DEFAULT_MODEL_VERSION})",
        f"Library: {config['library']}",
        f"Input Size: {config['input_size']}x{config['input_size']}",
        f"Classes: {', '.join(config['classes'])}",
        "",
        "Performance Metrics:",
    ]

    for metric, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {metric}: {value:.1%}")
        else:
            lines.append(f"  {metric}: {value}")

    return "\n".join(lines)
