"""
Unified dataset loader for fundus images (synthetic and real).

Supports loading from disk with train/val/test splits.
Handles both segmentation and classification tasks.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Union
import random


class FundusDataset(Dataset):
    """
    Dataset for fundus image segmentation and/or classification.

    Supports:
    - Segmentation datasets (with masks): synthetic, REFUGE 2
    - Classification datasets (no masks): RIM-ONE r3
    - Mixed datasets with both masks and labels
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        task: str = 'segmentation',
        image_size: int = 512,
        augment: bool = True,
        augmentation_params: Optional[Dict] = None,
        return_labels: bool = True,
        use_imagenet_norm: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing images/ and optionally masks/
            split: 'train', 'val', or 'test'
            task: 'segmentation', 'classification', or 'both'
            image_size: Target image size
            augment: Whether to apply data augmentation
            augmentation_params: Dict with augmentation parameters
            return_labels: Whether to return classification labels
            use_imagenet_norm: Whether to apply ImageNet normalization (mean/std)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.image_size = image_size
        self.augment = augment and (split == 'train')  # Only augment training data
        self.return_labels = return_labels
        self.use_imagenet_norm = use_imagenet_norm

        # Default augmentation parameters
        self.aug_params = augmentation_params or {
            'horizontal_flip': 0.5,
            'rotation_degrees': 5,
            'brightness': 0.1,
            'contrast': 0.1
        }

        # Load metadata (contains labels and other info)
        self.metadata = self.load_metadata()

        # Check if dataset has masks
        self.has_masks = self.metadata.get('has_masks', True)

        # Load splits
        self.load_splits()

        # Get file paths
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split ({self.task} task)")

    def load_metadata(self) -> Dict:
        """Load metadata.json if available."""
        metadata_file = self.data_root / 'metadata.json'

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            # Return minimal metadata if file doesn't exist
            return {'has_masks': True, 'has_labels': False}

    def load_splits(self):
        """Load train/val/test split indices."""
        splits_file = self.data_root / 'splits.json'

        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            self.indices = splits.get(self.split, [])
        else:
            # If no splits file, use all samples
            print(f"Warning: No splits.json found, using all samples")
            self.indices = None

    def _load_samples(self) -> List[Dict]:
        """Load sample file paths and metadata."""
        samples = []

        images_dir = self.data_root / 'images'

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Check for masks directory if needed
        # Try 'masks' first (standard), fallback to 'combined_masks' (REFUGE2)
        masks_dir = self.data_root / 'masks'
        if not masks_dir.exists():
            masks_dir = self.data_root / 'combined_masks'

        if self.has_masks and not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.data_root / 'masks'} or {self.data_root / 'combined_masks'}")

        # Get metadata samples if available
        metadata_samples = self.metadata.get('samples', [])
        metadata_dict = {s['sample_id']: s for s in metadata_samples} if metadata_samples else {}

        # Get all image files
        image_files = sorted(images_dir.glob('*.png'))

        # Filter by split indices if available
        if self.indices is not None:
            # Map indices to actual samples
            for idx in self.indices:
                if idx < len(image_files):
                    img_path = image_files[idx]
                elif idx in metadata_dict:
                    # Try to find by filename from metadata
                    img_name = metadata_dict[idx].get('image_filename')
                    img_path = images_dir / img_name if img_name else None
                else:
                    continue

                if img_path is None or not img_path.exists():
                    continue

                sample = {'image': img_path, 'sample_id': idx}

                # Add mask if dataset has masks
                if self.has_masks:
                    mask_path = masks_dir / img_path.name
                    if mask_path.exists():
                        sample['mask'] = mask_path
                    else:
                        # For classification tasks, missing masks are OK
                        if self.task == 'segmentation':
                            print(f"Warning: Missing mask for {img_path.name} (skipping)")
                            continue
                        # else: classification task, proceed without mask

                # Add label from metadata if available
                if idx in metadata_dict:
                    sample['label'] = metadata_dict[idx].get('label', -1)
                    sample['label_name'] = metadata_dict[idx].get('label_name', 'unknown')
                    # Add institution/hospital for proper evaluation splitting
                    sample['institution'] = metadata_dict[idx].get('source_hospital',
                                            metadata_dict[idx].get('institution', None))
                else:
                    sample['label'] = -1
                    sample['label_name'] = 'unknown'
                    sample['institution'] = None

                samples.append(sample)
        else:
            # Use all images if no split indices
            for img_path in image_files:
                sample = {'image': img_path}

                if self.has_masks:
                    mask_path = masks_dir / img_path.name
                    if mask_path.exists():
                        sample['mask'] = mask_path

                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.

        Args:
            image: Input image (H, W, 3)
            mask: Input mask (H, W)

        Returns:
            Augmented (image, mask)
        """
        # Horizontal flip
        if random.random() < self.aug_params.get('horizontal_flip', 0.5):
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Random rotation
        rotation_degrees = self.aug_params.get('rotation_degrees', 5)
        if rotation_degrees > 0:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))

        # Brightness adjustment
        brightness = self.aug_params.get('brightness', 0.1)
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast adjustment
        contrast = self.aug_params.get('contrast', 0.1)
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Depending on task and return_labels:
            - Segmentation: (image_tensor, mask_tensor) or (image, mask, label)
            - Classification: (image_tensor, label_tensor)
            - Both: (image_tensor, mask_tensor, label_tensor)

            Where:
            - image_tensor: (3, H, W) float32 in [0, 1]
            - mask_tensor: (1, H, W) float32 in [0, 1]
            - label_tensor: int64 scalar (class label)
        """
        sample = self.samples[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(str(sample['image']))
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask if available
        mask = None
        if self.has_masks and 'mask' in sample:
            mask = cv2.imread(str(sample['mask']), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {sample['mask']}")

        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))
            if mask is not None:
                mask = cv2.resize(mask, (self.image_size, self.image_size),
                                interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.augment:
            if mask is not None:
                image, mask = self.apply_augmentation(image, mask)
            else:
                image = self.apply_augmentation_image_only(image)

        # Convert image to tensor
        # Image: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Apply ImageNet normalization if requested
        if self.use_imagenet_norm:
            # ImageNet mean and std (RGB)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        # Convert mask to tensor if available
        mask_tensor = None
        if mask is not None:
            # Mask: (H, W) -> (1, H, W), keep raw values (0, 1, 2, ...)
            # Don't binarize - preserve multi-class segmentation
            mask_tensor = torch.from_numpy(mask[None, :, :]).long()

        # Get label if requested
        label = sample.get('label', -1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Return based on task
        if self.task == 'classification' or not self.has_masks:
            return image_tensor, label_tensor
        elif self.task == 'segmentation' and not self.return_labels:
            return image_tensor, mask_tensor
        else:  # Both or segmentation with labels
            return image_tensor, mask_tensor, label_tensor

    def apply_augmentation_image_only(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to image only (for classification datasets without masks).

        Args:
            image: Input image (H, W, 3)

        Returns:
            Augmented image
        """
        # Horizontal flip
        horizontal_flip_prob = self.aug_params.get('horizontal_flip_prob',
                                                    0.5 if self.aug_params.get('horizontal_flip', False) else 0.0)
        if random.random() < horizontal_flip_prob:
            image = np.fliplr(image).copy()

        # Vertical flip (new)
        vertical_flip_prob = self.aug_params.get('vertical_flip_prob',
                                                  0.5 if self.aug_params.get('vertical_flip', False) else 0.0)
        if random.random() < vertical_flip_prob:
            image = np.flipud(image).copy()

        # Random rotation
        rotation_degrees = self.aug_params.get('rotation_degrees', 5)
        if rotation_degrees > 0:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))

        # Brightness adjustment
        brightness = self.aug_params.get('brightness', 0.1)
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast adjustment
        contrast = self.aug_params.get('contrast', 0.1)
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Saturation adjustment (new)
        saturation = self.aug_params.get('saturation', 0.0)
        if saturation > 0:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Adjust saturation channel
            factor = 1.0 + random.uniform(-saturation, saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            # Convert back to RGB
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Hue adjustment (new)
        hue = self.aug_params.get('hue', 0.0)
        if hue > 0:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Adjust hue channel (0-179 in OpenCV)
            shift = random.uniform(-hue, hue) * 179
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            # Convert back to RGB
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gaussian blur (new)
        gaussian_blur_prob = self.aug_params.get('gaussian_blur_prob', 0.0)
        if random.random() < gaussian_blur_prob:
            sigma_range = self.aug_params.get('gaussian_blur_sigma', [0.1, 2.0])
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            kernel_size = int(sigma * 3) * 2 + 1  # Ensure odd kernel size
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        return image


def create_dataloaders(
    data_root: str,
    batch_size: int,
    image_size: int = 512,
    num_workers: int = 0,
    augment_train: bool = True,
    augmentation_params: Optional[Dict] = None,
    use_imagenet_norm: bool = False
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for dataloaders
        image_size: Image size
        num_workers: Number of worker processes
        augment_train: Whether to augment training data
        augmentation_params: Augmentation parameters
        use_imagenet_norm: Whether to apply ImageNet normalization

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        dataset = FundusDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            augment=augment_train if split == 'train' else False,
            augmentation_params=augmentation_params,
            use_imagenet_norm=use_imagenet_norm
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        dataloaders[split] = dataloader

    return dataloaders
