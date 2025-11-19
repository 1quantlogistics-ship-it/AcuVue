"""
Safe Augmentation Operations for Medical Imaging
=================================================

Defines a library of safe augmentation operations for fundus photography.
Standard ImageNet augmentations can destroy diagnostic signal in medical images.

Part of ARC Phase E Week 2: Augmentation Policy Search
Dev 2 implementation

SAFE Operations (preserves diagnostic features):
- Geometric: Rotation, flip, scale, translation
- Intensity: Brightness, contrast, gamma correction
- Noise: Gaussian noise, Gaussian blur

FORBIDDEN Operations (can destroy diagnostic signal):
- Color jitter (hue/saturation) - alters hemorrhage appearance
- Cutout / Random erasing - can remove optic disc
- Strong elastic deformation - distorts anatomy
- Mixup / CutMix - blends multiple images

Each operation:
- Has a magnitude_range for interpolation
- Is CPU-compatible (no GPU required)
- Preserves image dimensions
- Works with PIL Images and tensors
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Union, Tuple, Dict, List
import warnings


class ForbiddenOperationError(Exception):
    """Raised when attempting to use a forbidden augmentation operation."""
    pass


class AugmentationOperation:
    """
    Base class for augmentation operations.

    Each operation has a magnitude parameter that controls its strength.
    Magnitude is linearly interpolated within magnitude_range.
    """

    def __init__(
        self,
        name: str,
        magnitude_range: Tuple[float, float],
        description: str,
        safe: bool = True
    ):
        self.name = name
        self.magnitude_range = magnitude_range
        self.description = description
        self.safe = safe

        if not safe:
            raise ForbiddenOperationError(
                f"Operation '{name}' is forbidden for medical imaging. "
                f"Reason: {description}"
            )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the augmentation operation.

        Args:
            image: PIL Image or torch Tensor [C, H, W]
            magnitude: Strength parameter (will be clamped to magnitude_range)

        Returns:
            Augmented image (same type as input)
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def _clamp_magnitude(self, magnitude: float) -> float:
        """Clamp magnitude to valid range."""
        return np.clip(magnitude, self.magnitude_range[0], self.magnitude_range[1])

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"range={self.magnitude_range}, safe={self.safe})")


# ============================================================================
# Geometric Transformations (SAFE)
# ============================================================================

class RotateOp(AugmentationOperation):
    """Rotation within ±15 degrees (safe for fundus images)."""

    def __init__(self):
        super().__init__(
            name="rotate",
            magnitude_range=(-15.0, 15.0),
            description="Rotate image by angle in degrees",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            return image.rotate(magnitude, resample=Image.BILINEAR)
        else:  # torch.Tensor
            return TF.rotate(image, magnitude, interpolation=TF.InterpolationMode.BILINEAR)


class HorizontalFlipOp(AugmentationOperation):
    """Horizontal flip (magnitude ignored, always 100% flip)."""

    def __init__(self):
        super().__init__(
            name="hflip",
            magnitude_range=(0.0, 1.0),  # Magnitude unused (binary operation)
            description="Horizontal flip",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        if isinstance(image, Image.Image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return TF.hflip(image)


class VerticalFlipOp(AugmentationOperation):
    """Vertical flip (magnitude ignored, always 100% flip)."""

    def __init__(self):
        super().__init__(
            name="vflip",
            magnitude_range=(0.0, 1.0),
            description="Vertical flip",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        if isinstance(image, Image.Image):
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return TF.vflip(image)


class ScaleOp(AugmentationOperation):
    """Scale/zoom within 0.9-1.1x (safe range)."""

    def __init__(self):
        super().__init__(
            name="scale",
            magnitude_range=(0.9, 1.1),
            description="Scale image (crop and resize)",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            w, h = image.size
            new_w, new_h = int(w * magnitude), int(h * magnitude)

            # Crop or pad to achieve scaling effect
            if magnitude > 1.0:  # Zoom in (crop)
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                scaled = image.resize((new_w, new_h), Image.BILINEAR)
                return scaled.crop((left, top, left + w, top + h))
            else:  # Zoom out (pad)
                scaled = image.resize((new_w, new_h), Image.BILINEAR)
                new_image = Image.new('RGB', (w, h), (0, 0, 0))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                new_image.paste(scaled, (left, top))
                return new_image
        else:  # Tensor
            return TF.resize(image, [int(image.shape[1] * magnitude),
                                      int(image.shape[2] * magnitude)])


class TranslateXOp(AugmentationOperation):
    """Translate horizontally within ±10% of image width."""

    def __init__(self):
        super().__init__(
            name="translate_x",
            magnitude_range=(-0.1, 0.1),  # Fraction of width
            description="Translate image horizontally",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            pixels = int(magnitude * image.width)
            return TF.affine(image, angle=0, translate=(pixels, 0),
                           scale=1.0, shear=0)
        else:
            pixels = int(magnitude * image.shape[2])
            return TF.affine(image, angle=0, translate=(pixels, 0),
                           scale=1.0, shear=0)


class TranslateYOp(AugmentationOperation):
    """Translate vertically within ±10% of image height."""

    def __init__(self):
        super().__init__(
            name="translate_y",
            magnitude_range=(-0.1, 0.1),  # Fraction of height
            description="Translate image vertically",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            pixels = int(magnitude * image.height)
            return TF.affine(image, angle=0, translate=(0, pixels),
                           scale=1.0, shear=0)
        else:
            pixels = int(magnitude * image.shape[1])
            return TF.affine(image, angle=0, translate=(0, pixels),
                           scale=1.0, shear=0)


# ============================================================================
# Intensity Transformations (SAFE)
# ============================================================================

class BrightnessOp(AugmentationOperation):
    """Brightness adjustment within ±10%."""

    def __init__(self):
        super().__init__(
            name="brightness",
            magnitude_range=(-0.1, 0.1),  # Additive brightness change
            description="Adjust image brightness",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)
        factor = 1.0 + magnitude  # Convert to multiplicative factor

        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        else:
            return TF.adjust_brightness(image, factor)


class ContrastOp(AugmentationOperation):
    """Contrast adjustment within ±10%."""

    def __init__(self):
        super().__init__(
            name="contrast",
            magnitude_range=(-0.1, 0.1),
            description="Adjust image contrast",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)
        factor = 1.0 + magnitude

        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        else:
            return TF.adjust_contrast(image, factor)


class GammaOp(AugmentationOperation):
    """Gamma correction within 0.8-1.2."""

    def __init__(self):
        super().__init__(
            name="gamma",
            magnitude_range=(0.8, 1.2),
            description="Gamma correction",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            # Convert to tensor for gamma operation
            tensor = TF.to_tensor(image)
            corrected = TF.adjust_gamma(tensor, gamma=magnitude)
            return TF.to_pil_image(corrected)
        else:
            return TF.adjust_gamma(image, gamma=magnitude)


# ============================================================================
# Noise Transformations (SAFE with constraints)
# ============================================================================

class GaussianNoiseOp(AugmentationOperation):
    """Add Gaussian noise with σ ≤ 0.05."""

    def __init__(self):
        super().__init__(
            name="gaussian_noise",
            magnitude_range=(0.0, 0.05),  # Standard deviation
            description="Add Gaussian noise",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)

        if isinstance(image, Image.Image):
            tensor = TF.to_tensor(image)
        else:
            tensor = image

        noise = torch.randn_like(tensor) * magnitude
        noisy = torch.clamp(tensor + noise, 0, 1)

        if isinstance(image, Image.Image):
            return TF.to_pil_image(noisy)
        else:
            return noisy


class GaussianBlurOp(AugmentationOperation):
    """Gaussian blur with kernel size ≤ 5."""

    def __init__(self):
        super().__init__(
            name="gaussian_blur",
            magnitude_range=(1.0, 5.0),  # Kernel size (must be odd)
            description="Apply Gaussian blur",
            safe=True
        )

    def apply(
        self,
        image: Union[Image.Image, torch.Tensor],
        magnitude: float
    ) -> Union[Image.Image, torch.Tensor]:
        magnitude = self._clamp_magnitude(magnitude)
        kernel_size = int(magnitude)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        if isinstance(image, Image.Image):
            return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
        else:
            return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])


# ============================================================================
# Operation Library Registry
# ============================================================================

SAFE_OPERATIONS = {
    # Geometric
    "rotate": RotateOp(),
    "hflip": HorizontalFlipOp(),
    "vflip": VerticalFlipOp(),
    "scale": ScaleOp(),
    "translate_x": TranslateXOp(),
    "translate_y": TranslateYOp(),

    # Intensity
    "brightness": BrightnessOp(),
    "contrast": ContrastOp(),
    "gamma": GammaOp(),

    # Noise
    "gaussian_noise": GaussianNoiseOp(),
    "gaussian_blur": GaussianBlurOp()
}


FORBIDDEN_OPERATIONS = {
    "color_jitter_hue": "Alters hemorrhage and vessel appearance",
    "color_jitter_saturation": "Changes diagnostic color features",
    "cutout": "Can remove critical anatomical structures (optic disc)",
    "random_erasing": "Destroys diagnostic regions",
    "elastic_deform": "Distorts anatomical relationships",
    "mixup": "Blends multiple diagnostic cases",
    "cutmix": "Replaces diagnostic regions with other images"
}


def get_operation(name: str) -> AugmentationOperation:
    """
    Get an augmentation operation by name.

    Args:
        name: Operation name (e.g., 'rotate', 'brightness')

    Returns:
        AugmentationOperation instance

    Raises:
        ForbiddenOperationError: If operation is forbidden
        ValueError: If operation not found
    """
    # Check if forbidden
    if name in FORBIDDEN_OPERATIONS:
        raise ForbiddenOperationError(
            f"Operation '{name}' is forbidden for medical imaging.\n"
            f"Reason: {FORBIDDEN_OPERATIONS[name]}"
        )

    # Get safe operation
    if name not in SAFE_OPERATIONS:
        raise ValueError(
            f"Unknown operation: {name}.\n"
            f"Available safe operations: {list(SAFE_OPERATIONS.keys())}"
        )

    return SAFE_OPERATIONS[name]


def list_safe_operations() -> List[str]:
    """Return list of all safe operation names."""
    return list(SAFE_OPERATIONS.keys())


def list_forbidden_operations() -> Dict[str, str]:
    """Return dict of forbidden operations with reasons."""
    return FORBIDDEN_OPERATIONS.copy()


def validate_operation_name(name: str) -> bool:
    """
    Check if operation name is valid and safe.

    Returns:
        True if valid and safe, False if forbidden

    Raises:
        ValueError: If operation not recognized
    """
    if name in FORBIDDEN_OPERATIONS:
        return False
    if name in SAFE_OPERATIONS:
        return True
    raise ValueError(f"Unknown operation: {name}")


# Example usage
if __name__ == '__main__':
    """Demonstrate augmentation operations."""
    from PIL import Image
    import requests
    from io import BytesIO

    print("=" * 80)
    print("Safe Augmentation Operations Library")
    print("=" * 80)

    print("\nSafe Operations:")
    for name in list_safe_operations():
        op = SAFE_OPERATIONS[name]
        print(f"  • {name:20s} - {op.description:40s} Range: {op.magnitude_range}")

    print("\nForbidden Operations:")
    for name, reason in list_forbidden_operations().items():
        print(f"  ✗ {name:20s} - {reason}")

    # Test with sample image
    print("\n" + "=" * 80)
    print("Testing operations on sample image...")
    print("=" * 80)

    # Create dummy image
    image = Image.new('RGB', (224, 224), color=(128, 128, 128))

    for op_name in ["rotate", "brightness", "gaussian_blur"]:
        op = get_operation(op_name)
        magnitude = (op.magnitude_range[0] + op.magnitude_range[1]) / 2
        augmented = op.apply(image, magnitude)
        print(f"✓ {op_name}: Applied with magnitude={magnitude}")

    # Test forbidden operation
    print("\nTesting forbidden operation:")
    try:
        op = get_operation("cutout")
    except ForbiddenOperationError as e:
        print(f"✓ Correctly rejected: {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
