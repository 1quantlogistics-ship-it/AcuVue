"""
Inference Preprocessing
=======================

Image preprocessing transforms for the production inference pipeline.
Matches the transforms used during training for consistent results.
"""

from torchvision import transforms
from typing import Tuple
from PIL import Image
import torch


# ImageNet normalization constants (used since backbone was pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms matching training validation transforms.

    Args:
        input_size: Target image size (height and width)

    Returns:
        Composed transforms for inference
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(
    image: Image.Image,
    input_size: int = 224,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Preprocess a single PIL image for inference.

    Args:
        image: PIL Image (will be converted to RGB if needed)
        input_size: Target image size
        device: Device to move tensor to

    Returns:
        Preprocessed tensor with batch dimension (1, 3, H, W)
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transforms
    transform = get_inference_transforms(input_size)
    tensor = transform(image)

    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device)

    return tensor


def load_and_preprocess(
    image_path: str,
    input_size: int = 224,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load an image from path and preprocess for inference.

    Args:
        image_path: Path to image file
        input_size: Target image size
        device: Device to move tensor to

    Returns:
        Preprocessed tensor with batch dimension (1, 3, H, W)
    """
    image = Image.open(image_path).convert('RGB')
    return preprocess_image(image, input_size, device)


def unnormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD)
) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Unnormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean
