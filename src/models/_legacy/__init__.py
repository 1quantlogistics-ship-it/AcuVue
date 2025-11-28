"""
Legacy Models Module
====================

DEPRECATED: This module contains legacy model implementations.

For production inference, use:
    from src.inference import GlaucomaPredictor

Legacy classes are preserved for backwards compatibility only.
Do not use for new development.

Migration Date: November 2024
"""

import warnings

from .efficientnet_classifier_v1 import EfficientNetClassifier, create_classifier


def __getattr__(name):
    """Emit deprecation warning when accessing legacy classes."""
    if name in ('EfficientNetClassifier', 'create_classifier'):
        warnings.warn(
            f"src.models._legacy.{name} is deprecated. "
            "Use src.inference.GlaucomaPredictor for production inference.",
            DeprecationWarning,
            stacklevel=2
        )
        if name == 'EfficientNetClassifier':
            return EfficientNetClassifier
        return create_classifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['EfficientNetClassifier', 'create_classifier']
