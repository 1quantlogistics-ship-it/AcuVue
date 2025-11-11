"""
PyTorch Dataset for optic disc/cup segmentation.

Handles image-mask pairs with optional data augmentation.
"""
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
from typing import List, Tuple


class SegmentationDataset(Dataset):
    """
    Dataset for segmentation tasks with image-mask pairs.

    Args:
        images: List of numpy arrays (H, W, 3) representing RGB images.
        masks: List of numpy arrays (H, W) representing binary masks.
        augment: Whether to apply random augmentation (flip, rotation).
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        augment: bool = True
    ):
        assert len(images) == len(masks), "Number of images and masks must match"
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image-mask pair.

        Returns:
            img_tensor: (3, H, W) float32 tensor normalized to [0, 1]
            mask_tensor: (1, H, W) float32 tensor normalized to [0, 1]
        """
        img = self.images[idx]
        mask = self.masks[idx]

        # Apply augmentation if enabled
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # Small random rotation (-5 to +5 degrees)
            angle = random.uniform(-5, 5)
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            img = img.rotate(angle)
            mask = mask.rotate(angle)
            img = np.array(img)
            mask = np.array(mask)

        # Convert to tensors and normalize
        # Image: (H, W, C) -> (C, H, W) and scale to [0, 1]
        img_tensor = torch.tensor(
            img.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        )

        # Mask: (H, W) -> (1, H, W) and normalize
        mask_tensor = torch.tensor(
            mask[None, :, :] / 255.0 if mask.max() > 1 else mask[None, :, :],
            dtype=torch.float32
        )

        return img_tensor, mask_tensor


def create_dummy_dataset(
    num_samples: int = 10,
    image_size: int = 512
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create dummy image-mask pairs for smoke testing.

    Args:
        num_samples: Number of image-mask pairs to generate.
        image_size: Size of square images (height and width).

    Returns:
        Tuple of (images, masks) as lists of numpy arrays.
    """
    images = [
        np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        for _ in range(num_samples)
    ]
    masks = [
        np.random.randint(0, 2, (image_size, image_size), dtype=np.uint8) * 255
        for _ in range(num_samples)
    ]
    return images, masks
