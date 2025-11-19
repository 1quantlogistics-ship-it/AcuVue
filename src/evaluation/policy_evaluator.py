"""
Fast Augmentation Policy Evaluator
===================================

Evaluates augmentation policies using fast 5-epoch proxy training instead of
full 50-epoch training. Enables ARC to rapidly iterate through policy search.

Part of ARC Phase E Week 2: Augmentation Policy Search
Dev 2 implementation

Evaluation Protocol:
1. Train model for 5 epochs with proposed augmentation policy
2. Measure validation AUC
3. Compute DRI (Disc Relevance Index) on validation set
4. Return fitness score: AUC (if DRI ≥ 0.6), else 0.0

This allows population-based search to evaluate ~50 policies per day
instead of ~5 policies per day with full training.

Usage:
    >>> evaluator = PolicyEvaluator(model, train_loader, val_loader)
    >>> policy = [{"operation": "rotate", "probability": 0.5, "magnitude": 10.0}]
    >>> result = evaluator.evaluate_policy(policy)
    >>> print(f"Fitness: {result['fitness']:.3f}, DRI: {result['dri']:.3f}")
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
import copy
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path

# Handle imports for both module execution and script execution
try:
    from ..data.policy_augmentor import PolicyAugmentor
    from .dri_metrics import DRIComputer
except ImportError:
    # Add parent directory to path for script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.policy_augmentor import PolicyAugmentor
    from evaluation.dri_metrics import DRIComputer


class PolicyEvaluator:
    """
    Fast evaluator for augmentation policies.

    Trains model for 5 epochs and measures validation AUC + DRI.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs: int = 5,
        learning_rate: float = 1e-3,
        dri_threshold: float = 0.6
    ):
        """
        Initialize policy evaluator.

        Args:
            model: Model architecture to train (will be reset for each policy)
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training ('cuda' or 'cpu')
            num_epochs: Number of training epochs (default: 5)
            learning_rate: Learning rate for optimizer (default: 1e-3)
            dri_threshold: Minimum DRI for valid policy (default: 0.6)
        """
        self.model_template = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dri_threshold = dri_threshold

        # Store initial model weights for reset
        self.initial_state_dict = copy.deepcopy(model.state_dict())

    def reset_model(self):
        """Reset model to initial weights."""
        self.model_template.load_state_dict(self.initial_state_dict)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        augmentor: Optional[PolicyAugmentor] = None
    ) -> float:
        """
        Train for one epoch with optional augmentation policy.

        Args:
            model: Model to train
            train_loader: Training data
            optimizer: Optimizer
            criterion: Loss function
            augmentor: Optional augmentation policy

        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Unpack batch (handle different formats)
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
                clinical = batch.get('clinical', None)
            elif len(batch) >= 3:
                images, labels = batch[0], batch[1]
                clinical = batch[2] if len(batch) > 2 else None
            else:
                images, labels = batch
                clinical = None

            # Apply augmentation policy if provided
            if augmentor is not None:
                augmented_images = []
                for img in images:
                    aug_img = augmentor(img)
                    # Ensure tensor format
                    if not isinstance(aug_img, torch.Tensor):
                        import torchvision.transforms.functional as TF
                        aug_img = TF.to_tensor(aug_img)
                    augmented_images.append(aug_img)
                images = torch.stack(augmented_images)

            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            if clinical is not None:
                clinical = clinical.to(self.device)

            # Forward pass
            optimizer.zero_grad()

            if clinical is not None:
                logits = model(images, clinical)
            else:
                logits = model(images)

            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            model: Model to evaluate
            val_loader: Validation data
            criterion: Loss function

        Returns:
            Dict with loss, accuracy, and AUC
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch
                if isinstance(batch, dict):
                    images = batch['image']
                    labels = batch['label']
                    clinical = batch.get('clinical', None)
                elif len(batch) >= 3:
                    images, labels = batch[0], batch[1]
                    clinical = batch[2] if len(batch) > 2 else None
                else:
                    images, labels = batch
                    clinical = None

                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                if clinical is not None:
                    clinical = clinical.to(self.device)

                # Forward pass
                if clinical is not None:
                    logits = model(images, clinical)
                else:
                    logits = model(images)

                loss = criterion(logits, labels)

                # Collect predictions
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # AUC
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5  # Default for single-class validation set

        # Accuracy
        all_preds = (all_probs >= 0.5).astype(int)
        accuracy = (all_preds == all_labels).mean()

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }

    def compute_validation_dri(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        num_samples: int = 10
    ) -> float:
        """
        Compute average DRI on validation set.

        Args:
            model: Trained model
            val_loader: Validation data loader
            num_samples: Number of samples to evaluate (default: 10)

        Returns:
            Average DRI score
        """
        dri_computer = DRIComputer(model, dri_threshold=self.dri_threshold)

        dri_scores = []
        samples_evaluated = 0

        model.eval()

        # NOTE: No torch.no_grad() here because Grad-CAM needs gradients
        for batch in val_loader:
            if samples_evaluated >= num_samples:
                break

            # Unpack batch
            if isinstance(batch, dict):
                images = batch['image']
                disc_masks = batch.get('disc_mask', batch.get('mask'))
                clinical = batch.get('clinical', None)
            elif len(batch) >= 3:
                # Format: (images, labels, disc_masks) or (images, disc_masks, clinical)
                images = batch[0]
                # Check if second item is labels (1D) or disc_masks (3D)
                if batch[1].ndim == 1:  # labels
                    disc_masks = batch[2]
                    clinical = None
                else:  # disc_masks
                    disc_masks = batch[1]
                    clinical = batch[2] if len(batch) > 2 else None
            else:
                images, disc_masks = batch[:2]
                clinical = None

            # Evaluate DRI for each image in batch
            batch_size = images.shape[0]

            for i in range(min(batch_size, num_samples - samples_evaluated)):
                image = images[i:i+1].to(self.device)
                disc_mask = disc_masks[i].to(self.device)
                clin = clinical[i:i+1].to(self.device) if clinical is not None else None

                result = dri_computer.compute_dri(image, disc_mask, clin)
                dri_scores.append(result['dri'])
                samples_evaluated += 1

                if samples_evaluated >= num_samples:
                    break

        return np.mean(dri_scores) if dri_scores else 0.0

    def evaluate_policy(
        self,
        policy: List[Dict[str, Any]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate an augmentation policy using fast proxy training.

        Args:
            policy: Augmentation policy (list of sub-policies)
            verbose: Print progress (default: False)

        Returns:
            Dict with keys:
                - fitness: Fitness score (AUC if DRI valid, else 0.0)
                - auc: Validation AUC
                - dri: Validation DRI
                - valid: Whether DRI meets threshold
                - train_time: Training time in seconds
        """
        start_time = time.time()

        # Reset model to initial weights
        self.reset_model()
        model = self.model_template.to(self.device)

        # Create augmentor from policy
        augmentor = PolicyAugmentor(policy)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        if verbose:
            print(f"Training with policy ({len(policy)} operations)...")

        # Train for num_epochs
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(
                model,
                self.train_loader,
                optimizer,
                criterion,
                augmentor
            )

            if verbose:
                print(f"  Epoch {epoch+1}/{self.num_epochs} - Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_metrics = self.evaluate_epoch(model, self.val_loader, criterion)
        auc = val_metrics['auc']

        if verbose:
            print(f"  Validation AUC: {auc:.3f}")

        # Compute DRI on validation set
        dri = self.compute_validation_dri(model, self.val_loader, num_samples=10)
        valid = dri >= self.dri_threshold

        if verbose:
            print(f"  Validation DRI: {dri:.3f} ({'✓ VALID' if valid else '✗ INVALID'})")

        # Compute fitness (AUC if DRI valid, else 0.0)
        fitness = auc if valid else 0.0

        train_time = time.time() - start_time

        return {
            'fitness': fitness,
            'auc': auc,
            'dri': dri,
            'valid': valid,
            'train_time': train_time,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        }

    def compare_policies(
        self,
        policies: List[List[Dict[str, Any]]],
        policy_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple augmentation policies.

        Args:
            policies: List of augmentation policies
            policy_names: Optional names for policies (default: Policy 1, 2, ...)

        Returns:
            List of evaluation results, sorted by fitness (descending)
        """
        if policy_names is None:
            policy_names = [f"Policy {i+1}" for i in range(len(policies))]

        results = []

        print(f"Comparing {len(policies)} augmentation policies...\n")

        for i, (policy, name) in enumerate(zip(policies, policy_names)):
            print(f"Evaluating {name}...")
            result = self.evaluate_policy(policy, verbose=False)
            result['policy_name'] = name
            result['policy'] = policy
            results.append(result)

            print(f"  Fitness: {result['fitness']:.3f}, "
                  f"AUC: {result['auc']:.3f}, "
                  f"DRI: {result['dri']:.3f} "
                  f"({'✓' if result['valid'] else '✗'})\n")

        # Sort by fitness (descending)
        results.sort(key=lambda x: x['fitness'], reverse=True)

        return results


def rank_policies_by_fitness(
    policies: List[List[Dict[str, Any]]],
    evaluator: PolicyEvaluator,
    top_k: int = 5
) -> List[Tuple[List[Dict[str, Any]], float]]:
    """
    Rank augmentation policies by fitness score.

    This is the main function ARC's Explorer agent will use for policy search.

    Args:
        policies: List of augmentation policies to evaluate
        evaluator: PolicyEvaluator instance
        top_k: Return top K policies (default: 5)

    Returns:
        List of (policy, fitness) tuples, sorted by fitness

    Example:
        >>> policies = [policy1, policy2, policy3, ...]
        >>> top_policies = rank_policies_by_fitness(policies, evaluator, top_k=5)
        >>> best_policy, best_fitness = top_policies[0]
        >>> print(f"Best policy has fitness {best_fitness:.3f}")
    """
    results = evaluator.compare_policies(policies)

    # Extract top K
    top_results = results[:top_k]

    # Return (policy, fitness) tuples
    return [(r['policy'], r['fitness']) for r in top_results]


# Example usage
if __name__ == '__main__':
    """Demonstrate fast policy evaluation."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    print("=" * 80)
    print("Fast Augmentation Policy Evaluator")
    print("=" * 80)

    # Create dummy dataset
    print("\nCreating dummy dataset...")
    num_train = 50
    num_val = 20

    train_images = torch.randn(num_train, 3, 224, 224)
    train_labels = torch.randint(0, 2, (num_train,))
    train_disc_masks = torch.zeros(num_train, 224, 224)

    val_images = torch.randn(num_val, 3, 224, 224)
    val_labels = torch.randint(0, 2, (num_val,))
    val_disc_masks = torch.zeros(num_val, 224, 224)

    # Create circular disc masks
    for masks in [train_disc_masks, val_disc_masks]:
        for i in range(masks.shape[0]):
            center_y, center_x = 112, 112
            radius = 40
            for y in range(224):
                for x in range(224):
                    if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                        masks[i, y, x] = 1.0

    train_dataset = TensorDataset(train_images, train_labels, train_disc_masks)
    val_dataset = TensorDataset(val_images, val_labels, val_disc_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"✓ Train set: {num_train} samples")
    print(f"✓ Val set: {num_val} samples")

    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 2)

        def forward(self, x, clinical=None):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleModel()
    print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create evaluator
    evaluator = PolicyEvaluator(
        model,
        train_loader,
        val_loader,
        device='cpu',
        num_epochs=2,  # Fast test
        learning_rate=1e-3
    )
    print("\n✓ Created PolicyEvaluator (2 epochs for testing)")

    # Define test policies
    try:
        from ..data.policy_augmentor import create_random_policy
    except ImportError:
        from data.policy_augmentor import create_random_policy

    print("\n" + "=" * 80)
    print("Evaluating Augmentation Policies")
    print("=" * 80)

    policy1 = [
        {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
        {"operation": "brightness", "probability": 0.3, "magnitude": 0.1}
    ]

    policy2 = [
        {"operation": "hflip", "probability": 0.5, "magnitude": 1.0},
        {"operation": "contrast", "probability": 0.4, "magnitude": 0.1},
        {"operation": "gaussian_blur", "probability": 0.2, "magnitude": 2.0}
    ]

    policy3 = create_random_policy(num_operations=3, seed=42)

    # Compare policies
    policies = [policy1, policy2, policy3]
    policy_names = ["Rotation+Brightness", "Flip+Contrast+Blur", "Random"]

    results = evaluator.compare_policies(policies, policy_names)

    # Show best policy
    print("=" * 80)
    print("Best Policy")
    print("=" * 80)

    best = results[0]
    print(f"\nWinner: {best['policy_name']}")
    print(f"  • Fitness: {best['fitness']:.3f}")
    print(f"  • AUC: {best['auc']:.3f}")
    print(f"  • DRI: {best['dri']:.3f}")
    print(f"  • Valid: {best['valid']}")
    print(f"  • Train time: {best['train_time']:.1f}s")

    print(f"\nPolicy operations:")
    for i, op in enumerate(best['policy']):
        print(f"  {i+1}. {op['operation']:15s} - prob={op['probability']:.2f}, mag={op['magnitude']:.2f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
