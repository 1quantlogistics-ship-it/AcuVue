"""
Training script for optic disc/cup segmentation using U-Net.

Phase 01: Smoke test with dummy data and single epoch.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.unet_disc_cup import UNet, dice_loss
from src.data.segmentation_dataset import SegmentationDataset, create_dummy_dataset

# Setup logging
logger = logging.getLogger(__name__)


def setup_device(config: DictConfig) -> torch.device:
    """
    Setup compute device (CUDA or CPU).

    Args:
        config: Hydra configuration

    Returns:
        torch.device object
    """
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


@hydra.main(version_base=None, config_path="../../configs", config_name="phase01_smoke_test")
def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra configuration loaded from YAML
    """
    # Print configuration
    logger.info("=" * 60)
    logger.info("Phase 01: Smoke Test - Segmentation Training")
    logger.info("=" * 60)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_seed(cfg.system.seed)

    # Setup device
    device = setup_device(cfg)

    # Create dummy dataset for smoke test
    logger.info(f"\nGenerating {cfg.training.num_dummy_samples} dummy image-mask pairs...")
    images, masks = create_dummy_dataset(
        num_samples=cfg.training.num_dummy_samples,
        image_size=cfg.data.image_size
    )

    # Create dataset and dataloader
    dataset = SegmentationDataset(
        images=images,
        masks=masks,
        augment=cfg.data.use_augmentation
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for smoke test
        pin_memory=device.type == "cuda"
    )
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Number of batches: {len(loader)}")

    # Initialize model
    logger.info("\nInitializing U-Net model...")
    model = UNet().to(device)
    logger.info(model.summary())

    # Setup optimizer and loss functions
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate
    )
    bce_loss = nn.BCELoss()

    logger.info(f"Optimizer: Adam (lr={cfg.training.learning_rate})")
    logger.info("Loss: BCE + Dice")

    # Training loop
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting training for {cfg.training.epochs} epoch(s)")
    logger.info(f"{'=' * 60}\n")

    model.train()

    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0
        batch_losses = []

        # Progress bar for batches
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")

        for batch_idx, (images_batch, masks_batch) in enumerate(pbar):
            # Move data to device
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            # Forward pass
            predictions = model(images_batch)

            # Compute loss
            bce = bce_loss(predictions, masks_batch)
            dice = dice_loss(predictions, masks_batch)
            loss = bce + dice

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(loader)
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.epochs} - Average Loss: {avg_epoch_loss:.4f}")

    # Save checkpoint
    logger.info(f"\nSaving checkpoint to: {cfg.checkpoint.save_path}")
    os.makedirs(os.path.dirname(cfg.checkpoint.save_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint.save_path)

    # Verify checkpoint was saved
    checkpoint_path = Path(cfg.checkpoint.save_path)
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Checkpoint saved successfully ({size_mb:.2f} MB)")

        # Test loading the checkpoint
        logger.info("Testing checkpoint load...")
        test_model = UNet()
        test_model.load_state_dict(torch.load(cfg.checkpoint.save_path, map_location='cpu'))
        logger.info("✓ Checkpoint loads successfully")
    else:
        logger.error("✗ Checkpoint save failed!")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 01 Smoke Test: COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
