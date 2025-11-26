"""
Classification Dataset for Folder-Based Structure
=================================================

Handles datasets organized as:
    data_root/
        train/
            glaucoma/
            normal/
        val/
            glaucoma/
            normal/
        test/
            glaucoma/
            normal/
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torchvision.transforms as T


class ClassificationDataset(Dataset):
    """
    Dataset for classification with folder-based structure.
    
    Expects: data_root/split/class_name/*.png
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
        use_imagenet_norm: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.use_imagenet_norm = use_imagenet_norm
        
        # Load samples
        self.samples = self._load_samples()
        self.labels = [s["label"] for s in self.samples]
        
        # Build transforms
        self.transform = self._build_transforms()
        
        print(f"ClassificationDataset: Loaded {len(self.samples)} samples for {split}")
        print(f"  Classes: {self.class_names}")
        print(f"  Class counts: {self._get_class_counts()}")
        
    def _load_samples(self) -> List[Dict]:
        """Load samples from folder structure."""
        samples = []
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Get class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        self.num_classes = len(self.class_names)
        
        # Class to label mapping
        class_to_label = {name: idx for idx, name in enumerate(self.class_names)}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = class_to_label[class_name]
            
            # Get all image files
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in sorted(class_dir.glob(ext)):
                    samples.append({
                        "image_path": img_path,
                        "label": label,
                        "class_name": class_name
                    })
        
        return samples
    
    def _get_class_counts(self) -> Dict[str, int]:
        """Get count per class."""
        counts = {}
        for s in self.samples:
            cls = s["class_name"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    def _build_transforms(self):
        """Build image transforms."""
        if self.augment:
            transforms = [
                T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
            ]
        else:
            transforms = [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ]
        
        if self.use_imagenet_norm:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return T.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        return img_tensor, sample["label"]


def get_dataloaders(
    data_root: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        data_root: Dataset root directory
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted sampling for class balance
        
    Returns:
        Dict with "train", "val", "test" dataloaders
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    loaders = {}
    
    for split in ["train", "val", "test"]:
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            continue
            
        dataset = ClassificationDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            augment=(split == "train")
        )
        
        # Create sampler for training
        sampler = None
        shuffle = (split == "train")
        
        if split == "train" and use_weighted_sampler:
            # Calculate class weights
            labels = np.array(dataset.labels)
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders
