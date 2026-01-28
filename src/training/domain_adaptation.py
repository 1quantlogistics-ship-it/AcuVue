"""
Domain adaptation components for cross-dataset training.

Implements gradient reversal and domain-adversarial training to learn
domain-invariant features across different fundus imaging datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Dict, Any


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for domain-adversarial training.

    Forward pass: Identity function (output = input)
    Backward pass: Reverses gradient and scales by lambda

    Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by
    Backpropagation", ICML 2015
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient and scale by lambda
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).

    Reverses gradients during backpropagation to enable domain-adversarial training.
    The domain classifier tries to discriminate domains, while the feature extractor
    tries to confuse the domain classifier.
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Initialize GRL.

        Args:
            lambda_: Gradient reversal strength (0.0 = no reversal, 1.0 = full reversal)
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """Forward pass with gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda during training (commonly scheduled)."""
        self.lambda_ = lambda_


class DomainClassifier(nn.Module):
    """
    Domain classifier for discriminating between datasets.

    Takes features and predicts which dataset they came from.
    Used in domain-adversarial training to learn domain-invariant features.
    """

    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        """
        Initialize domain classifier.

        Args:
            feature_dim: Dimension of input features
            num_domains: Number of domains/datasets
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_domains = num_domains

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, features):
        """
        Classify domain.

        Args:
            features: Feature tensor [batch_size, feature_dim]

        Returns:
            Domain logits [batch_size, num_domains]
        """
        return self.classifier(features)


class DomainAdversarialLoss(nn.Module):
    """
    Combined loss for domain-adversarial training.

    Combines task loss (e.g., classification) with domain adversarial loss.
    The feature extractor minimizes task loss while maximizing domain confusion.
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_domain: float = 0.1,
        schedule_lambda: bool = True
    ):
        """
        Initialize domain-adversarial loss.

        Args:
            task_loss_fn: Task loss function (e.g., CrossEntropyLoss)
            lambda_domain: Weight for domain loss
            schedule_lambda: Whether to schedule lambda during training
        """
        super().__init__()

        self.task_loss_fn = task_loss_fn
        self.lambda_domain = lambda_domain
        self.schedule_lambda = schedule_lambda
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            task_logits: Task predictions [batch_size, num_classes]
            task_labels: Task ground truth [batch_size]
            domain_logits: Domain predictions [batch_size, num_domains]
            domain_labels: Domain ground truth [batch_size]
            epoch: Current epoch (for lambda scheduling)
            max_epochs: Total epochs (for lambda scheduling)

        Returns:
            Dict with 'total', 'task', 'domain', 'lambda' losses
        """
        # Compute task loss
        task_loss = self.task_loss_fn(task_logits, task_labels)

        # Compute domain loss
        domain_loss = self.domain_loss_fn(domain_logits, domain_labels)

        # Schedule lambda if requested
        lambda_current = self.lambda_domain
        if self.schedule_lambda and epoch is not None and max_epochs is not None:
            # Progressive schedule: increases from 0 to lambda_domain
            # Formula: lambda * (2 / (1 + exp(-10 * p)) - 1)
            # where p = epoch / max_epochs
            import math
            p = epoch / max_epochs
            lambda_current = self.lambda_domain * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

        # Total loss
        total_loss = task_loss + lambda_current * domain_loss

        return {
            'total': total_loss,
            'task': task_loss,
            'domain': domain_loss,
            'lambda': lambda_current  # Return scalar, not tensor
        }


def compute_domain_labels(
    batch_datasets: list,
    dataset_to_id: Dict[str, int]
) -> torch.Tensor:
    """
    Convert dataset names to domain labels.

    Args:
        batch_datasets: List of dataset names for each sample
        dataset_to_id: Mapping from dataset name to domain ID

    Returns:
        Domain labels tensor [batch_size]

    Example:
        >>> batch_datasets = ["REFUGE", "ORIGA", "REFUGE"]
        >>> dataset_to_id = {"REFUGE": 0, "ORIGA": 1, "Drishti": 2}
        >>> labels = compute_domain_labels(batch_datasets, dataset_to_id)
        >>> # tensor([0, 1, 0])
    """
    domain_ids = [dataset_to_id[name] for name in batch_datasets]
    return torch.tensor(domain_ids, dtype=torch.long)


class DomainAdaptationWrapper(nn.Module):
    """
    Wrapper that adds domain adaptation to any model.

    Wraps an existing model and adds:
    - Gradient reversal layer
    - Domain classifier
    - Domain-adversarial training
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_extractor_name: str,
        feature_dim: int,
        num_domains: int,
        lambda_domain: float = 0.1,
        grl_lambda: float = 1.0
    ):
        """
        Initialize domain adaptation wrapper.

        Args:
            base_model: Base model to wrap
            feature_extractor_name: Name of feature extractor attribute
                (e.g., "encoder" or "backbone")
            feature_dim: Dimension of features from extractor
            num_domains: Number of domains/datasets
            lambda_domain: Weight for domain adversarial loss
            grl_lambda: Gradient reversal strength
        """
        super().__init__()

        self.base_model = base_model
        self.feature_extractor_name = feature_extractor_name

        # Domain adaptation components
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.domain_classifier = DomainClassifier(
            feature_dim=feature_dim,
            num_domains=num_domains
        )

        self.lambda_domain = lambda_domain

    def forward(
        self,
        images: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        return_features: bool = False
    ):
        """
        Forward pass with optional domain prediction.

        Args:
            images: Input images [batch_size, C, H, W]
            clinical: Optional clinical features [batch_size, clinical_dim]
            return_features: Whether to return intermediate features

        Returns:
            If return_features=False: task_logits
            If return_features=True: (task_logits, features, domain_logits)
        """
        # Get features from base model
        feature_extractor = getattr(self.base_model, self.feature_extractor_name)
        features = feature_extractor(images)

        # Task prediction (use base model)
        if clinical is not None:
            task_logits = self.base_model(images, clinical)
        else:
            task_logits = self.base_model(images)

        if return_features:
            # Domain prediction (with gradient reversal)
            reversed_features = self.grl(features)
            domain_logits = self.domain_classifier(reversed_features)
            return task_logits, features, domain_logits

        return task_logits


if __name__ == '__main__':
    # Quick test
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(4, 512, requires_grad=True)
    y = grl(x)
    y.sum().backward()
    print(f"GRL: forward identity={torch.allclose(y, x)}, grad reversed={(x.grad < 0).all()}")

    domain_clf = DomainClassifier(feature_dim=512, num_domains=3)
    logits = domain_clf(torch.randn(8, 512))
    print(f"Domain classifier: output shape={logits.shape}")
