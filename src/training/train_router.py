"""
Domain Router Training Script
=============================

Train a lightweight domain classifier that determines which expert head
should process a given fundus image based on its acquisition characteristics.

The router learns WHERE an image came from (device/hospital), NOT diagnosis.

Usage:
    python src/training/train_router.py --config configs/router_training_v1.yaml

    Or with command line args:
    python src/training/train_router.py \
        --data-roots data/rimone data/refuge data/g1020 \
        --epochs 20 \
        --output-dir outputs/router_v1
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.domain_dataset import (
    DomainDataset,
    MultiSourceDomainDataset,
    create_domain_dataloaders,
)
from src.data.domain_labels import DOMAIN_CLASSES, NUM_DOMAINS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_router_model(
    backbone: str = 'mobilenetv3_small',
    num_classes: int = NUM_DOMAINS,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create a lightweight domain classification model.

    Uses timm for consistent model loading (same as expert heads).

    Args:
        backbone: Model backbone ('mobilenetv3_small', 'efficientnet_b0', etc.)
        num_classes: Number of domain classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout probability

    Returns:
        PyTorch model
    """
    import timm

    # Create model with timm
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=0,  # Remove classifier
        drop_rate=dropout,
    )

    # Get number of features
    num_features = model.num_features

    # Add custom classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes),
    )

    # Replace forward to use our classifier
    original_forward = model.forward_features

    def forward(x):
        features = original_forward(x)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # Global average pooling
        return model.classifier(features)

    model.forward = forward

    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100*correct/total:.2f}%"
        })

    return total_loss / len(loader), correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = {i: 0 for i in range(NUM_DOMAINS)}
    class_total = {i: 0 for i in range(NUM_DOMAINS)}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{split}]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Per-class tracking
            for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': correct / total,
    }

    # Per-class accuracy
    for i in range(NUM_DOMAINS):
        domain_name = DOMAIN_CLASSES[i]
        if class_total[i] > 0:
            metrics[f'accuracy_{domain_name}'] = class_correct[i] / class_total[i]
        else:
            metrics[f'accuracy_{domain_name}'] = 0.0

    # Confusion matrix
    confusion = np.zeros((NUM_DOMAINS, NUM_DOMAINS), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1
    metrics['confusion_matrix'] = confusion.tolist()

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    output_dir: Path,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # Save latest
    torch.save(checkpoint, output_dir / 'router_latest.pt')

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'router_best.pt')
        logger.info(f"Saved best model at epoch {epoch}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def train_router(
    data_roots: list,
    output_dir: str,
    backbone: str = 'mobilenetv3_small_100',
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    image_size: int = 224,
    num_workers: int = 4,
    device: Optional[str] = None,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train the domain router.

    Args:
        data_roots: List of dataset root directories or dict mapping domain->path
        output_dir: Directory for outputs
        backbone: Model backbone architecture
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        image_size: Input image size
        num_workers: Data loading workers
        device: Device to train on (None for auto)
        resume: Path to checkpoint to resume from

    Returns:
        Dictionary with training results and final metrics
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training on device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    if isinstance(data_roots, dict):
        dataloaders = create_domain_dataloaders(
            data_roots=data_roots,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
    else:
        dataloaders = create_domain_dataloaders(
            data_roots=data_roots,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    logger.info(f"Creating model: {backbone}")
    model = create_router_model(
        backbone=backbone,
        num_classes=NUM_DOMAINS,
        pretrained=True,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    best_accuracy = 0.0

    if resume and Path(resume).exists():
        logger.info(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('metrics', {}).get('accuracy', 0.0)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, "val")

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
            f"LR={current_lr:.6f}"
        )

        # Log per-class accuracy
        for domain_idx in range(NUM_DOMAINS):
            domain_name = DOMAIN_CLASSES[domain_idx]
            domain_acc = val_metrics.get(f'accuracy_{domain_name}', 0)
            logger.info(f"  {domain_name}: {domain_acc:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(current_lr)

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_accuracy
        if is_best:
            best_accuracy = val_metrics['accuracy']

        save_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir, is_best
        )

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save final model in simple format (for deployment)
    torch.save(model.state_dict(), output_dir / 'router_final.pt')

    # Final results
    results = {
        'best_accuracy': best_accuracy,
        'final_val_metrics': val_metrics,
        'epochs_trained': epochs,
        'backbone': backbone,
        'output_dir': str(output_dir),
    }

    logger.info(f"Training complete! Best accuracy: {best_accuracy:.4f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train domain router')

    # Data arguments
    parser.add_argument(
        '--data-roots',
        nargs='+',
        help='List of dataset root directories'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    # Model arguments
    parser.add_argument(
        '--backbone',
        type=str,
        default='mobilenetv3_small_100',
        help='Model backbone (default: mobilenetv3_small_100)'
    )

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/router',
        help='Output directory'
    )
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    # Device
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)

        # Override with command line args
        data_roots = args.data_roots or config.get('data', {}).get('roots', [])
        backbone = args.backbone or config.get('model', {}).get('backbone', 'mobilenetv3_small_100')
        epochs = args.epochs or config.get('training', {}).get('epochs', 20)
        batch_size = args.batch_size or config.get('training', {}).get('batch_size', 32)
        lr = args.lr or config.get('training', {}).get('lr', 0.001)
        weight_decay = args.weight_decay or config.get('training', {}).get('weight_decay', 0.01)
        image_size = args.image_size or config.get('data', {}).get('image_size', 224)
        output_dir = args.output_dir or config.get('output_dir', 'outputs/router')
    else:
        data_roots = args.data_roots
        backbone = args.backbone
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        weight_decay = args.weight_decay
        image_size = args.image_size
        output_dir = args.output_dir

    if not data_roots:
        logger.error("No data roots provided! Use --data-roots or --config")
        sys.exit(1)

    # Train
    results = train_router(
        data_roots=data_roots,
        output_dir=output_dir,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        image_size=image_size,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"Output Directory: {results['output_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
