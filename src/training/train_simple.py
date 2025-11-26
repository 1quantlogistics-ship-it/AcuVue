"""
Simple Classification Training Script
=====================================

Minimal training script for classification that works with folder-based datasets.
No Hydra dependency - uses simple argparse.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_classifier import create_classifier
from src.data.classification_dataset import ClassificationDataset, get_dataloaders
from src.training.losses import get_loss_function

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
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
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, split="val"):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{split}]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
    
    # Calculate metrics
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_roc": auc
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="cross_entropy")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--experiment_id", type=str, default="exp")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info(f"Creating model: {args.model_name}")
    model = create_classifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Create dataloaders
    logger.info(f"Loading dataset from: {args.dataset_path}")
    loaders = get_dataloaders(
        data_root=args.dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True
    )
    
    if "train" not in loaders:
        raise ValueError("No training data found!")
    
    # Loss and optimizer
    criterion = get_loss_function(args.loss_type, num_classes=args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "best_auc": 0.0,
        "best_epoch": 0
    }
    
    best_auc = 0.0
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # Validate
        if "val" in loaders:
            val_metrics = evaluate(model, loaders["val"], criterion, device, "val")
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_auc"].append(val_metrics["auc_roc"])
            
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_metrics["loss"]:.4f}, val_acc={val_metrics["accuracy"]:.4f}, "
                f"val_auc={val_metrics["auc_roc"]:.4f}"
            )
            
            # Save best model
            if val_metrics["auc_roc"] > best_auc:
                best_auc = val_metrics["auc_roc"]
                history["best_auc"] = best_auc
                history["best_epoch"] = epoch
                
                checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "args": vars(args)
                }, checkpoint_path)
                logger.info(f"Saved best model (AUC: {best_auc:.4f})")
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        
        scheduler.step()
    
    # Final test evaluation
    if "test" in loaders:
        logger.info("Evaluating on test set...")
        # Load best model
        best_checkpoint = Path(args.checkpoint_dir) / "best_model.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        test_metrics = evaluate(model, loaders["test"], criterion, device, "test")
        logger.info(f"Test Results: {test_metrics}")
        history["test_metrics"] = test_metrics
    
    # Save training history
    history_path = Path(args.log_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete! Best AUC: {best_auc:.4f} at epoch {history["best_epoch"]}")
    
    # Return results for ARC
    return {
        "status": "success",
        "best_auc": best_auc,
        "best_epoch": history["best_epoch"],
        "final_train_acc": history["train_acc"][-1],
        "checkpoint_path": str(Path(args.checkpoint_dir) / "best_model.pt"),
        "test_metrics": history.get("test_metrics", {})
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
