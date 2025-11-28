"""
Domain Classification Dataset for Training the Router

This dataset is used to train a lightweight domain classifier that determines
WHERE a fundus image came from (RIM-ONE vs REFUGE vs G1020). The classifier
is NOT for diagnosing glaucoma - it learns device/acquisition characteristics.

The router uses this classifier to route images to the appropriate expert head.

Key Design Decisions:
- Augmentation keeps domain-identifying features (device artifacts, color profile)
- No heavy geometric transforms that could destroy domain signatures
- Light color jitter to improve robustness, but not so much to confuse domains

Usage:
    >>> from src.data.domain_dataset import DomainDataset
    >>> dataset = DomainDataset(
    ...     data_roots=['data/rimone', 'data/refuge', 'data/g1020'],
    ...     split='train'
    ... )
    >>> image, domain_label = dataset[0]
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import random

from .domain_labels import (
    extract_domain,
    DOMAIN_CLASSES,
    DOMAIN_TO_IDX,
    NUM_DOMAINS,
    generate_domain_labels,
    create_domain_split,
    get_domain_distribution,
)


class DomainDataset(Dataset):
    """
    Dataset for training a domain classifier.

    Each sample returns (image, domain_label) where domain_label is
    the integer class index for the image's source domain.

    Supports:
    - Single dataset root with metadata.json
    - Multiple dataset roots (will be combined)
    - Pre-computed domain labels file
    """

    def __init__(
        self,
        data_root: Union[str, List[str]],
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True,
        domain_labels_path: Optional[str] = None,
        metadata_key: str = 'samples',
        use_imagenet_norm: bool = True,
    ):
        """
        Initialize domain dataset.

        Args:
            data_root: Root directory or list of root directories
            split: 'train' or 'val'
            image_size: Target image size (default 224 for MobileNet/EfficientNet)
            augment: Whether to apply augmentation (only for training)
            domain_labels_path: Optional pre-computed labels file
            metadata_key: Key in metadata.json for samples list
            use_imagenet_norm: Whether to apply ImageNet normalization
        """
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.use_imagenet_norm = use_imagenet_norm

        # Handle single or multiple data roots
        if isinstance(data_root, str):
            data_root = [data_root]
        self.data_roots = [Path(r) for r in data_root]

        # Load samples from all data roots
        self.samples = self._load_all_samples(metadata_key)

        # Generate or load domain labels
        if domain_labels_path and Path(domain_labels_path).exists():
            self._load_domain_labels(domain_labels_path)
        else:
            self._generate_domain_labels()

        # Create or load splits
        self._setup_split()

        # Setup transforms
        self.transform = self._get_transforms()

        print(f"DomainDataset: {len(self)} samples for {split} split")
        print(f"  Domain distribution: {self._get_split_distribution()}")

    def _load_all_samples(self, metadata_key: str) -> List[Dict[str, Any]]:
        """Load samples from all data roots."""
        all_samples = []

        for data_root in self.data_roots:
            metadata_path = data_root / 'metadata.json'

            if not metadata_path.exists():
                # Try to infer samples from images directory
                images_dir = data_root / 'images'
                if images_dir.exists():
                    for img_path in sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg')):
                        all_samples.append({
                            'image_path': str(img_path),
                            'image_filename': img_path.name,
                            'data_root': str(data_root),
                        })
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            samples = metadata.get(metadata_key, [])
            for sample in samples:
                # Add data_root reference
                sample['data_root'] = str(data_root)

                # Resolve image path
                if 'image_path' not in sample:
                    filename = sample.get('image_filename', sample.get('filename', ''))
                    sample['image_path'] = str(data_root / 'images' / filename)

                all_samples.append(sample)

        return all_samples

    def _generate_domain_labels(self):
        """Generate domain labels from sample metadata."""
        self.indices, self.labels = generate_domain_labels(
            self.samples,
            exclude_unknown=True
        )

    def _load_domain_labels(self, path: str):
        """Load pre-computed domain labels."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.indices = data['indices']
        self.labels = data['labels']

    def _setup_split(self):
        """Setup train/val split."""
        # Create splits based on domain labels
        splits = create_domain_split(
            [self.samples[i] for i in self.indices],
            train_ratio=0.8,
            val_ratio=0.2,
            stratified=True,
            seed=42
        )

        # Remap to original indices
        if self.split == 'train':
            split_indices = splits['train']
        else:
            split_indices = splits['val']

        # Filter to current split
        self.split_indices = [self.indices[i] for i in split_indices]
        self.split_labels = [self.labels[i] for i in split_indices]

    def _get_transforms(self) -> transforms.Compose:
        """
        Get transforms for domain classification.

        Key: Keep domain-identifying features while adding robustness.
        - Light color jitter (domains have different color profiles)
        - Minimal geometric transforms (preserve device artifacts)
        """
        if self.augment:
            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Light rotation - domains don't depend on orientation
                transforms.RandomRotation(degrees=10),
                # Very light color jitter - preserve domain color signatures
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05,  # Light - color profile is domain signal
                    hue=0.02          # Very light - hue is device-specific
                ),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]

        if self.use_imagenet_norm:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return transforms.Compose(transform_list)

    def _get_split_distribution(self) -> Dict[str, int]:
        """Get domain distribution for current split."""
        distribution = {domain: 0 for domain in DOMAIN_CLASSES[:NUM_DOMAINS]}

        for label in self.split_labels:
            domain = DOMAIN_CLASSES[label]
            distribution[domain] += 1

        return distribution

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, domain_label) where:
            - image_tensor: (3, H, W) normalized tensor
            - domain_label: integer domain class index
        """
        sample_idx = self.split_indices[idx]
        sample = self.samples[sample_idx]
        domain_label = self.split_labels[idx]

        # Load image
        image_path = sample.get('image_path')
        if not image_path:
            data_root = sample.get('data_root', str(self.data_roots[0]))
            filename = sample.get('image_filename', sample.get('filename', ''))
            image_path = str(Path(data_root) / 'images' / filename)

        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image)

        return image_tensor, domain_label

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        sample_idx = self.split_indices[idx]
        sample = self.samples[sample_idx]
        domain_label = self.split_labels[idx]

        return {
            'sample_idx': sample_idx,
            'domain_label': domain_label,
            'domain_name': DOMAIN_CLASSES[domain_label],
            'metadata': sample,
        }


class MultiSourceDomainDataset(Dataset):
    """
    Domain dataset that explicitly combines multiple data sources.

    Use this when you have separate directories for each domain and want
    explicit control over domain assignment.
    """

    def __init__(
        self,
        domain_roots: Dict[str, str],
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True,
        samples_per_domain: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize multi-source domain dataset.

        Args:
            domain_roots: Dictionary mapping domain names to data directories
                Example: {'rimone': 'data/rimone', 'refuge': 'data/refuge'}
            split: 'train' or 'val'
            image_size: Target image size
            augment: Whether to apply augmentation
            samples_per_domain: Limit samples per domain (for balancing)
            seed: Random seed
        """
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.samples_per_domain = samples_per_domain

        np.random.seed(seed)
        random.seed(seed)

        # Collect samples from each domain
        self.samples = []
        self.labels = []

        for domain_name, data_root in domain_roots.items():
            if domain_name not in DOMAIN_TO_IDX:
                print(f"Warning: Unknown domain '{domain_name}', skipping")
                continue

            domain_idx = DOMAIN_TO_IDX[domain_name]
            domain_samples = self._load_domain_samples(data_root, domain_idx)

            if samples_per_domain and len(domain_samples) > samples_per_domain:
                indices = np.random.choice(
                    len(domain_samples),
                    samples_per_domain,
                    replace=False
                )
                domain_samples = [domain_samples[i] for i in indices]

            for sample in domain_samples:
                self.samples.append(sample)
                self.labels.append(domain_idx)

        # Create train/val split
        self._setup_split(seed)

        # Setup transforms
        self.transform = self._get_transforms()

        print(f"MultiSourceDomainDataset: {len(self)} samples for {split}")

    def _load_domain_samples(
        self,
        data_root: str,
        domain_idx: int
    ) -> List[Dict[str, Any]]:
        """Load samples from a single domain directory."""
        data_root = Path(data_root)
        samples = []

        # Check for metadata.json
        metadata_path = data_root / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            for sample in metadata.get('samples', []):
                filename = sample.get('image_filename', sample.get('filename', ''))
                sample['image_path'] = str(data_root / 'images' / filename)
                sample['domain_idx'] = domain_idx
                samples.append(sample)

            return samples

        # Fall back to scanning images directory
        images_dir = data_root / 'images'
        if images_dir.exists():
            for img_path in sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg')):
                samples.append({
                    'image_path': str(img_path),
                    'image_filename': img_path.name,
                    'domain_idx': domain_idx,
                })

        return samples

    def _setup_split(self, seed: int):
        """Create stratified train/val split."""
        np.random.seed(seed)

        indices = np.arange(len(self.samples))
        labels = np.array(self.labels)

        train_indices = []
        val_indices = []

        # Stratified split
        for domain_idx in range(NUM_DOMAINS):
            domain_mask = labels == domain_idx
            domain_indices = indices[domain_mask]
            np.random.shuffle(domain_indices)

            n_train = int(len(domain_indices) * 0.8)
            train_indices.extend(domain_indices[:n_train].tolist())
            val_indices.extend(domain_indices[n_train:].tolist())

        if self.split == 'train':
            self.split_indices = train_indices
        else:
            self.split_indices = val_indices

        np.random.shuffle(self.split_indices)

    def _get_transforms(self) -> transforms.Compose:
        """Get transforms for domain classification."""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05,
                    hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        sample_idx = self.split_indices[idx]
        sample = self.samples[sample_idx]
        domain_label = self.labels[sample_idx]

        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = self.transform(image)

        return image_tensor, domain_label


def create_domain_dataloaders(
    data_roots: Union[str, List[str], Dict[str, str]],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    augment_train: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for domain classification.

    Args:
        data_roots: Single path, list of paths, or dict mapping domain->path
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data

    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    dataloaders = {}

    for split in ['train', 'val']:
        if isinstance(data_roots, dict):
            dataset = MultiSourceDomainDataset(
                domain_roots=data_roots,
                split=split,
                image_size=image_size,
                augment=augment_train and split == 'train',
            )
        else:
            dataset = DomainDataset(
                data_root=data_roots,
                split=split,
                image_size=image_size,
                augment=augment_train and split == 'train',
            )

        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders
