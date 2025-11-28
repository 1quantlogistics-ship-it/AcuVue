"""
Domain Routing Module
=====================

Routes fundus images to appropriate expert heads based on image domain.

The domain router is a lightweight classifier that identifies which dataset
family an image belongs to (RIM-ONE, REFUGE2, G1020, or unknown). This
information is used by the multi-head pipeline to route images to the
appropriate expert model.

Key Principle: The router does NOT diagnose glaucoma - it only identifies
image domain. The expert heads perform the actual diagnosis.

Quick Start:
    >>> from src.routing import DomainRouter
    >>> router = DomainRouter("models/routing/domain_classifier_v1.pt")
    >>> domain = router.classify_domain(image)
    >>> print(f"Domain: {domain}")  # -> 'rimone', 'refuge2', 'g1020', or 'unknown'

Classes:
    DomainRouter: High-level routing interface
    DomainClassifier: PyTorch model for domain classification
    RoutingResult: Dataclass for routing results
"""

from .router import DomainRouter, RoutingResult
from .domain_classifier import DomainClassifier

__all__ = [
    'DomainRouter',
    'DomainClassifier',
    'RoutingResult',
]
