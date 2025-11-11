"""
Phase 02 training script with validation loop, metrics tracking, and checkpointing.

Features:
- Train/val/test splits
- Multiple metrics (Dice, IoU, accuracy, sensitivity, specificity)
- Best model checkpointing
- Training history logging
- Configurable via Hydra
"""
import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.unet_disc_cup import UNet, dice_loss
from src.data.fundus_dataset import create_dataloaders
from src.evaluation.metrics import SegmentationMetrics
from src.utils.checkpoint import CheckpointManager

# Setup logging
logger = logging.getLogger(__name__)


def setup_device(config: DictConfig) -> torch.device:
    """Setup compute device (CUDA or CPU)."""
    if config.system.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.system.device)

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.Module,
    device: torch.device,
    epoch: int,
    config: DictConfig
) -> tuple:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    metrics_tracker = SegmentationMetrics()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.training.epochs} [Train]")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)

        # Compute loss (BCE + Dice)
        bce = criterion_bce(predictions, masks)
        dice = dice_loss(predictions, masks)
        loss = bce + dice

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_loss = loss.item()
        epoch_loss += batch_loss

        with torch.no_grad():
            metrics_tracker.update(predictions, masks)

        # Update progress bar
        if (batch_idx + 1) % config.logging.log_every_n_steps == 0:
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })

    avg_loss = epoch_loss / len(dataloader)
    metrics = metrics_tracker.get_metrics()

    return avg_loss, metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_bce: nn.Module,
    device: torch.device,
    epoch: int,
    config: DictConfig
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    epoch_loss = 0.0
    metrics_tracker = SegmentationMetrics()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.training.epochs} [Val]  ")

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)

        # Compute loss
        bce = criterion_bce(predictions, masks)
        dice = dice_loss(predictions, masks)
        loss = bce + dice

        epoch_loss += loss.item()
        metrics_tracker.update(predictions, masks)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = epoch_loss / len(dataloader)
    metrics = metrics_tracker.get_metrics()

    return avg_loss, metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="phase02_baseline")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Print configuration
    logger.info("=" * 70)
    logger.info("Phase 02: Baseline Training with Validation")
    logger.info("=" * 70)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_seed(cfg.system.seed)

    # Setup device
    device = setup_device(cfg)

    # Create dataloaders
    logger.info(f"\nLoading dataset from: {cfg.data.data_root}")

    dataloaders = create_dataloaders(
        data_root=cfg.data.data_root,
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.system.num_workers,
        augment_train=cfg.data.use_augmentation,
        augmentation_params=cfg.get('augmentation', {})
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    logger.info("\nInitializing U-Net model...")
    model = UNet().to(device)
    logger.info(model.summary())

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.optimizer.get('weight_decay', 0.0)
    )

    # Loss function
    criterion_bce = nn.BCELoss()

    logger.info(f"Optimizer: Adam (lr={cfg.training.learning_rate})")
    logger.info("Loss: BCE + Dice")

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.checkpoint.save_dir,
        metric_name=cfg.checkpoint.metric_name,
        mode=cfg.checkpoint.metric_mode
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    # Training loop
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Starting training for {cfg.training.epochs} epochs")
    logger.info(f"{'=' * 70}\n")

    best_val_dice = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_bce, device, epoch, cfg
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion_bce, device, epoch, cfg
        )

        # Log metrics
        logger.info(f"\nEpoch {epoch}/{cfg.training.epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        logger.info(f"  Train IoU:  {train_metrics['iou']:.4f} | Val IoU:  {val_metrics['iou']:.4f}")
        logger.info(f"  Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Save checkpoint
        metrics_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }

        is_best = checkpoint_manager.update_best(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics_dict,
            config=dict(cfg)
        )

        if is_best:
            best_val_dice = val_metrics['dice']
            logger.info(f"  ✓ New best model! Val Dice: {best_val_dice:.4f}")

    # Save training history
    checkpoint_manager.save_training_history(history)

    # Final evaluation on test set
    logger.info(f"\n{'=' * 70}")
    logger.info("Final Evaluation on Test Set")
    logger.info(f"{'=' * 70}\n")

    # Load best model
    best_checkpoint_path = checkpoint_manager.get_best_checkpoint_path()
    if best_checkpoint_path:
        checkpoint_manager.load_checkpoint(best_checkpoint_path, model, device=device)

    test_loss, test_metrics = validate_epoch(
        model, test_loader, criterion_bce, device, cfg.training.epochs, cfg
    )

    logger.info("\nTest Set Results:")
    logger.info(f"  Loss:        {test_loss:.4f}")
    logger.info(f"  Dice:        {test_metrics['dice']:.4f}")
    logger.info(f"  IoU:         {test_metrics['iou']:.4f}")
    logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    logger.info(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")

    # Save test metrics
    test_results = {
        'test_loss': test_loss,
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }

    results_path = Path(cfg.checkpoint.save_dir) / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"\n✓ Test results saved: {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Phase 02 Training: COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nBest model saved to: {best_checkpoint_path}")
    logger.info(f"Best val_dice: {best_val_dice:.4f}")
    logger.info(f"Test dice: {test_metrics['dice']:.4f}")


if __name__ == "__main__":
    train()
