"""
Checkpoint management utilities for model saving and loading.

Handles best model tracking, checkpoint metadata, and resuming training.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints with best model tracking.

    Features:
    - Save best model based on validation metric
    - Save last epoch checkpoint
    - Load checkpoints with metadata
    - Resume training from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str = "models",
        metric_name: str = "val_dice",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Metric to track for best model (e.g., "val_dice", "val_loss")
            mode: "max" (higher is better) or "min" (lower is better)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode

        if mode == "max":
            self.best_metric = float('-inf')
            self.is_better = lambda new, best: new > best
        elif mode == "min":
            self.best_metric = float('inf')
            self.is_better = lambda new, best: new < best
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

        self.best_epoch = 0

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any = None,
        is_best: bool = False,
        filename: str = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch number
            metrics: Dictionary of metrics
            config: Optional configuration object
            is_best: Whether this is the best model
            filename: Optional custom filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
            'timestamp': datetime.now().isoformat(),
        }

        if config is not None:
            checkpoint['config'] = config

        # Save checkpoint
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path} (epoch {epoch}, {self.metric_name}={metrics.get(self.metric_name, 0):.4f})")

        # Always save last checkpoint
        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(checkpoint, last_path)

    def update_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any = None
    ) -> bool:
        """
        Check if current model is best and save if so.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Optional configuration

        Returns:
            True if this is the best model
        """
        current_metric = metrics.get(self.metric_name, None)

        if current_metric is None:
            print(f"Warning: {self.metric_name} not found in metrics")
            return False

        is_best = self.is_better(current_metric, self.best_metric)

        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch

        self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            is_best=is_best
        )

        return is_best

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            device: Device to load tensors to

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best metric ({self.metric_name}): {checkpoint.get('best_metric', 'unknown')}")

        return checkpoint

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None

    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get path to last model checkpoint."""
        last_path = self.checkpoint_dir / "last_model.pt"
        return last_path if last_path.exists() else None

    def save_training_history(
        self,
        history: Dict[str, list],
        filename: str = "training_history.json"
    ) -> None:
        """
        Save training history to JSON.

        Args:
            history: Dictionary of lists (e.g., {'train_loss': [...], 'val_dice': [...]})
            filename: Filename to save
        """
        history_path = self.checkpoint_dir / filename

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"✓ Training history saved: {history_path}")

    def load_training_history(
        self,
        filename: str = "training_history.json"
    ) -> Optional[Dict[str, list]]:
        """
        Load training history from JSON.

        Args:
            filename: Filename to load

        Returns:
            Dictionary of training history or None if not found
        """
        history_path = self.checkpoint_dir / filename

        if not history_path.exists():
            return None

        with open(history_path, 'r') as f:
            history = json.load(f)

        return history

    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading model weights.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        return {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
            'best_metric': checkpoint.get('best_metric'),
            'metric_name': checkpoint.get('metric_name'),
            'timestamp': checkpoint.get('timestamp'),
        }


def save_simple_checkpoint(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Simple checkpoint save (state dict only).

    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        metadata: Optional metadata dictionary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if metadata is not None:
        checkpoint.update(metadata)

    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_simple_checkpoint(
    model: nn.Module,
    load_path: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Simple checkpoint load (state dict only).

    Args:
        model: PyTorch model
        load_path: Path to checkpoint
        device: Device to load to

    Returns:
        Checkpoint metadata
    """
    load_path = Path(load_path)

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Checkpoint loaded: {load_path}")
    return checkpoint
