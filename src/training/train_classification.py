"""
Phase 03 classification training script with validation loop, metrics tracking, and checkpointing.

Features:
- Binary glaucoma classification
- Train/val/test splits
- Multiple metrics (Accuracy, AUC-ROC, sensitivity, specificity)
- Best model checkpointing based on validation AUC
- Training history logging
- Configurable via Hydra
- GPU fail-safe system
"""
import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_classifier import create_classifier
from src.data.fundus_dataset import FundusDataset
from src.evaluation.metrics import ClassificationMetrics
from src.utils.checkpoint import CheckpointManager
from src.data.samplers import get_sampler
from src.training.losses import get_loss_function

# Setup logging
logger = logging.getLogger(__name__)


def setup_device(config: DictConfig) -> torch.device:
    """Setup compute device (CUDA or CPU)."""
    # GPU FAIL-SAFE: Prevent accidental local CPU training
    require_gpu = config.system.get("require_gpu", True)

    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU FAIL-SAFE TRIGGERED: CUDA not available but require_gpu=True.\n"
            "This prevents accidental CPU training on local machines.\n"
            "\n"
            "Solutions:\n"
            "  1. Run on RunPod GPU: bash scripts/run_on_gpu.sh\n"
            "  2. Override (NOT recommended): Add 'system.require_gpu=False' to config\n"
            "\n"
            "For production training, always use RunPod GPU infrastructure."
        )

    if config.system.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.system.device)

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("Running on CPU - training will be SLOW!")

    return device


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


def create_weighted_loss(class_weights: list, device: torch.device) -> nn.Module:
    """
    Create weighted cross-entropy loss for handling class imbalance.

    Args:
        class_weights: List of weights for each class
        device: Device to move loss to

    Returns:
        Weighted CrossEntropyLoss
    """
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"Using weighted loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard cross-entropy loss")

    return criterion


def compute_class_distribution(dataset: FundusDataset, num_classes: int = 2) -> dict:
    """
    Compute class distribution in the dataset.

    Args:
        dataset: FundusDataset instance
        num_classes: Number of classes

    Returns:
        Dictionary with class counts and percentages
    """
    class_counts = np.bincount([label for _, label in dataset], minlength=num_classes)
    total = len(dataset)

    distribution = {
        'counts': class_counts.tolist(),
        'percentages': (class_counts / total * 100).tolist(),
        'total': total
    }

    return distribution


def create_weighted_sampler(dataset: FundusDataset, num_classes: int = 2) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler for balanced batch sampling.

    Args:
        dataset: FundusDataset instance
        num_classes: Number of classes

    Returns:
        WeightedRandomSampler instance
    """
    # Get all labels
    labels = np.array([label for _, label in dataset])

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its class
    sample_weights = class_weights[labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    logger.info(f"Created WeightedRandomSampler:")
    logger.info(f"  Class counts: {class_counts.tolist()}")
    logger.info(f"  Class weights: {class_weights.tolist()}")

    return sampler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> tuple:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    metrics_tracker = ClassificationMetrics(
        num_classes=2,
        class_names=['Normal', 'Glaucoma']
    )

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        metrics_tracker.update(logits.detach(), labels)

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    # Compute average loss
    avg_loss = epoch_loss / len(dataloader)

    # Get metrics
    metrics = metrics_tracker.get_metrics()

    return avg_loss, metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    epoch_loss = 0.0
    metrics_tracker = ClassificationMetrics(
        num_classes=2,
        class_names=['Normal', 'Glaucoma']
    )

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # Compute loss
            loss = criterion(logits, labels)

            # Track metrics
            epoch_loss += loss.item()
            metrics_tracker.update(logits, labels)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

    # Compute average loss
    avg_loss = epoch_loss / len(dataloader)

    # Get metrics
    metrics = metrics_tracker.get_metrics()

    return avg_loss, metrics, metrics_tracker


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Test model on test set."""
    model.eval()
    test_loss = 0.0
    metrics_tracker = ClassificationMetrics(
        num_classes=2,
        class_names=['Normal', 'Glaucoma']
    )

    logger.info("\nEvaluating on test set...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Track metrics
            test_loss += loss.item()
            metrics_tracker.update(logits, labels)

    # Compute average loss
    avg_loss = test_loss / len(dataloader)

    # Get metrics
    metrics = metrics_tracker.get_metrics()
    metrics['loss'] = avg_loss

    # Print detailed report
    metrics_tracker.print_detailed_report(prefix="Test")

    return metrics


@hydra.main(config_path="../../configs", config_name="phase03_classification", version_base=None)
def main(config: DictConfig):
    """Main training function."""
    # Print config
    logger.info("\n" + "="*60)
    logger.info("Phase 03: Classification Training")
    logger.info("="*60)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(config))

    # Setup device
    device = setup_device(config)

    # Set seed
    set_seed(config.system.seed)

    # Create datasets
    logger.info("\nLoading datasets...")

    # Extract ImageNet normalization flag from config (Phase 03e)
    use_imagenet_norm = config.data.get('use_imagenet_norm', False)

    train_dataset = FundusDataset(
        data_root=config.data.data_root,
        split='train',
        task='classification',
        image_size=config.data.image_size,
        augment=True,
        augmentation_params=dict(config.data.augmentation) if 'augmentation' in config.data else None,
        use_imagenet_norm=use_imagenet_norm
    )

    val_dataset = FundusDataset(
        data_root=config.data.data_root,
        split='val',
        task='classification',
        image_size=config.data.image_size,
        augment=False,
        use_imagenet_norm=use_imagenet_norm
    )

    test_dataset = FundusDataset(
        data_root=config.data.data_root,
        split='test',
        task='classification',
        image_size=config.data.image_size,
        augment=False,
        use_imagenet_norm=use_imagenet_norm
    )

    # Log class distributions
    logger.info("\nDataset class distributions:")
    for split_name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        dist = compute_class_distribution(dataset, num_classes=config.model.num_classes)
        logger.info(f"  {split_name}: {dist['counts']} ({dist['percentages'][0]:.1f}% / {dist['percentages'][1]:.1f}%)")

    # Create dataloaders with optional sampling strategies
    # Phase 03d: Support for balanced dataset sampler
    use_balanced_sampler = config.data.get('use_balanced_sampler', False)
    use_weighted_sampler = config.training.get('use_weighted_sampler', False)

    if use_balanced_sampler:
        # Phase 03d: Dataset-aware balanced sampling
        sampler_mode = config.data.get('sampler_mode', 'balanced')
        logger.info(f"\nUsing dataset-aware sampler: {sampler_mode}")
        train_sampler = get_sampler(
            train_dataset,
            batch_size=config.training.batch_size,
            mode=sampler_mode,
            drop_last=config.data.get('drop_last', True),
            seed=config.system.seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.system.num_workers,
            pin_memory=(device.type == 'cuda')
        )
    elif use_weighted_sampler:
        logger.info("\nUsing WeightedRandomSampler for training")
        train_sampler = create_weighted_sampler(train_dataset, num_classes=config.model.num_classes)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.system.num_workers,
            pin_memory=(device.type == 'cuda')
        )
    else:
        logger.info("\nUsing standard shuffled training")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.system.num_workers,
            pin_memory=(device.type == 'cuda')
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")

    # Create model
    logger.info("\nCreating model...")
    model = create_classifier(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        freeze_backbone_epochs=config.model.freeze_backbone_epochs,
        device=device
    )

    # Create loss function
    # Phase 03d: Support for focal loss and other loss functions
    loss_type = config.training.get('loss_type', 'ce')  # 'ce', 'focal', 'weighted_focal'

    if loss_type in ['focal', 'weighted_focal']:
        # Phase 03d: Focal loss for sensitivity/specificity balance
        logger.info(f"\nUsing {loss_type} loss function")

        # Compute class counts for weighted focal loss
        train_labels = np.array([label for _, label in train_dataset])
        class_counts = np.bincount(train_labels, minlength=config.model.num_classes)

        # Get focal loss parameters
        focal_gamma = config.training.get('focal_gamma', 2.0)
        focal_alpha = config.training.get('focal_alpha', None)

        logger.info(f"  Focal gamma: {focal_gamma}")
        logger.info(f"  Focal alpha: {focal_alpha}")
        logger.info(f"  Class counts: {class_counts.tolist()}")

        criterion = get_loss_function(
            loss_type=loss_type,
            num_classes=config.model.num_classes,
            class_counts=class_counts,
            gamma=focal_gamma,
            alpha=focal_alpha,
            device=device
        )
    else:
        # Standard cross-entropy (with optional class weighting)
        class_weights = config.training.get('class_weights', None)
        criterion = create_weighted_loss(class_weights, device)

    # Create optimizer
    if config.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.get('weight_decay', 0.0)
        )
    elif config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.get('weight_decay', 1e-4)
        )
    elif config.training.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            momentum=config.training.get('momentum', 0.9),
            weight_decay=config.training.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    logger.info(f"Optimizer: {config.training.optimizer}")
    logger.info(f"Learning rate: {config.training.lr}")

    # Create checkpoint manager
    checkpoint_dir = Path(config.paths.models_dir)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name='auc',  # Track AUC for classification
        mode='max'
    )

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_auc': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
    }

    # Training loop
    logger.info(f"\nStarting training for {config.training.epochs} epochs...")
    logger.info("="*60)

    best_val_auc = 0.0
    freeze_backbone_epochs = config.model.freeze_backbone_epochs

    for epoch in range(1, config.training.epochs + 1):
        # Unfreeze backbone after specified epochs
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
            # Optionally reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            logger.info(f"\nBackbone unfrozen at epoch {epoch}, LR reduced by 10x\n")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, config.training.epochs
        )

        # Validate
        val_loss, val_metrics, val_tracker = validate_epoch(
            model, val_loader, criterion, device,
            epoch, config.training.epochs
        )

        # Log metrics
        logger.info(f"\nEpoch {epoch}/{config.training.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        logger.info(f"  Train AUC:  {train_metrics['auc']:.4f} | Val AUC:  {val_metrics['auc']:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

            # Save checkpoint if best
        # Note: checkpoint_manager uses 'auc' metric (set in __init__)
        metrics_with_auc = {'auc': val_metrics['auc'], **val_metrics}
        is_best = checkpoint_manager.update_best(model, optimizer, epoch, metrics_with_auc)

        if is_best:
            best_val_auc = val_metrics['auc']
            logger.info(f"  ✓ New best model! Val AUC: {best_val_auc:.4f}")

            # Print confusion matrix for best model
            logger.info("\nValidation Confusion Matrix (Best Model):")
            val_tracker.print_confusion_matrix()

    # Training complete
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Best Val AUC: {best_val_auc:.4f}")
    logger.info("="*60)

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nTraining history saved to: {history_path}")

    # Load best model for testing
    logger.info("\nLoading best model for testing...")
    best_path = checkpoint_manager.get_best_checkpoint_path()

    if best_path and best_path.exists():
        checkpoint_manager.load_checkpoint(best_path, model, device=str(device))
    else:
        logger.warning("No best checkpoint found, using final model")

    # Test on test set
    test_metrics = test_model(model, test_loader, criterion, device)

    # Save test results
    test_results_path = checkpoint_dir / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"\nTest results saved to: {test_results_path}")

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Final Results")
    logger.info("="*60)
    logger.info(f"Best Val AUC:  {best_val_auc:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC:      {test_metrics['auc']:.4f}")
    logger.info(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    logger.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
