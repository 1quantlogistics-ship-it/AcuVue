"""
Head Registry
=============

Registry of available expert heads for multi-head inference.

Each expert head is trained on a specific dataset domain (RIM-ONE, REFUGE2, G1020)
and optimized for that domain's characteristics. The registry maintains metadata
about available heads and their domain mappings.

Example:
    >>> from src.inference.head_registry import get_head_for_domain
    >>> config = get_head_for_domain('rimone')
    >>> print(f"Using: {config.name} at {config.checkpoint_path}")
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HeadConfig:
    """
    Configuration for an expert head.

    Attributes:
        name: Unique identifier for the head
        checkpoint_path: Path to model weights
        domains: List of domains this head handles
        architecture: Model architecture name
        metrics: Performance metrics from training
    """
    name: str
    checkpoint_path: str
    domains: List[str]
    architecture: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'checkpoint_path': self.checkpoint_path,
            'domains': self.domains,
            'architecture': self.architecture,
            'metrics': self.metrics,
        }


# Registry of expert heads
# Each head is trained on specific dataset domains
EXPERT_HEADS: Dict[str, HeadConfig] = {
    "glaucoma_rimone_v1": HeadConfig(
        name="glaucoma_rimone_v1",
        checkpoint_path="models/production/glaucoma_efficientnet_b0_v1.pt",
        domains=["rimone"],
        architecture="efficientnet_b0",
        metrics={"test_auc": 0.937, "test_accuracy": 0.765}
    ),
    # Future heads will be added here as training completes:
    # "glaucoma_refuge2_v1": HeadConfig(
    #     name="glaucoma_refuge2_v1",
    #     checkpoint_path="models/production/glaucoma_refuge2_v1.pt",
    #     domains=["refuge2"],
    #     architecture="efficientnet_b0",
    #     metrics={"test_auc": 0.0, "test_accuracy": 0.0}
    # ),
    # "glaucoma_g1020_v1": HeadConfig(
    #     name="glaucoma_g1020_v1",
    #     checkpoint_path="models/production/glaucoma_g1020_v1.pt",
    #     domains=["g1020"],
    #     architecture="efficientnet_b0",
    #     metrics={"test_auc": 0.0, "test_accuracy": 0.0}
    # ),
}

# Domain to head mapping
# Maps image domains to the expert head that should handle them
DOMAIN_HEAD_MAPPING: Dict[str, str] = {
    "rimone": "glaucoma_rimone_v1",
    "refuge2": "glaucoma_rimone_v1",  # Fallback until REFUGE2 head trained
    "g1020": "glaucoma_rimone_v1",    # Fallback until G1020 head trained
    "unknown": "glaucoma_rimone_v1",  # Default fallback
}


def get_head_for_domain(domain: str) -> HeadConfig:
    """
    Get the expert head config for a given domain.

    Args:
        domain: Domain name ('rimone', 'refuge2', 'g1020', 'unknown')

    Returns:
        HeadConfig for the appropriate expert head
    """
    head_name = DOMAIN_HEAD_MAPPING.get(domain, "glaucoma_rimone_v1")
    return EXPERT_HEADS[head_name]


def get_available_heads() -> List[str]:
    """
    List all registered expert heads.

    Returns:
        List of head names
    """
    return list(EXPERT_HEADS.keys())


def get_available_domains() -> List[str]:
    """
    List all domains with configured mappings.

    Returns:
        List of domain names
    """
    return list(DOMAIN_HEAD_MAPPING.keys())


def register_head(config: HeadConfig) -> None:
    """
    Register a new expert head.

    Args:
        config: HeadConfig to register
    """
    EXPERT_HEADS[config.name] = config


def update_domain_mapping(domain: str, head_name: str) -> None:
    """
    Update the domain to head mapping.

    Args:
        domain: Domain name
        head_name: Name of the head to handle this domain

    Raises:
        ValueError: If head_name not in registry
    """
    if head_name not in EXPERT_HEADS:
        raise ValueError(f"Unknown head: {head_name}. Available: {get_available_heads()}")
    DOMAIN_HEAD_MAPPING[domain] = head_name
