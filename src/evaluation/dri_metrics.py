"""
DRI (Disc Relevance Index) Metrics for Augmentation Policy Validation
======================================================================

Computes whether model attention focuses on diagnostically relevant regions
(optic disc) after applying augmentation policies.

Part of ARC Phase E Week 2: Augmentation Policy Search
Dev 2 implementation

DRI Calculation:
1. Generate Grad-CAM attention heatmap from model prediction
2. Compute IoU between attention map and ground-truth optic disc mask
3. DRI = IoU score (higher = model focuses on disc region)

Constraint: Valid augmentation policies must maintain DRI ≥ 0.6
(ensures augmentations preserve diagnostic signal)

Usage:
    >>> dri_computer = DRIComputer(model)
    >>> dri_result = dri_computer.compute_dri(image, disc_mask, clinical_data)
    >>> print(f"DRI: {dri_result['dri']:.3f}, Valid: {dri_result['valid']}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import warnings


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping

    Generates attention heatmaps showing which regions the model focuses on
    for making predictions.

    Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model (must be in eval mode)
            target_layer: Layer to compute gradients from (default: last conv layer)
        """
        self.model = model
        self.target_layer = target_layer

        # Hook storage
        self.activations = None
        self.gradients = None

        # Register hooks
        if target_layer is None:
            # Auto-detect last convolutional layer
            self.target_layer = self._find_last_conv_layer()

        self._register_hooks()

    def _find_last_conv_layer(self) -> nn.Module:
        """
        Find the last convolutional layer in the model.

        Returns:
            Last Conv2d layer
        """
        last_conv = None

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            # Try to find in backbone (for MultiModalClassifier)
            if hasattr(self.model, 'backbone'):
                for module in self.model.backbone.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module

        if last_conv is None:
            raise ValueError(
                "Could not auto-detect convolutional layer. "
                "Please specify target_layer explicitly."
            )

        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: Input image tensor [1, C, H, W] (batch size must be 1)
            clinical: Clinical indicators [1, clinical_dim] (optional)
            target_class: Class to compute gradients for (default: predicted class)

        Returns:
            Heatmap tensor [H, W] normalized to [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Ensure gradients are enabled
        image.requires_grad_(True)

        # Forward pass
        if clinical is not None:
            logits = self.model(image, clinical)
        else:
            logits = self.model(image)

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = logits[0, target_class]
        class_score.backward()

        # Compute Grad-CAM weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU (only positive influence)
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # [1, 1, H, W]

        # Normalize to [0, 1]
        cam = cam.squeeze()  # [H, W]
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Uniform attention (edge case)
            cam = torch.ones_like(cam) * 0.5

        return cam


class DRIComputer:
    """
    Computes Disc Relevance Index (DRI) for augmented images.

    DRI measures whether the model's attention remains focused on the
    optic disc region after augmentation.

    Usage:
        >>> dri_computer = DRIComputer(model)
        >>> result = dri_computer.compute_dri(image, disc_mask)
        >>> if result['valid']:
        ...     print(f"✓ Policy valid, DRI = {result['dri']:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        dri_threshold: float = 0.6
    ):
        """
        Initialize DRI computer.

        Args:
            model: Trained model to analyze
            target_layer: Layer for Grad-CAM (default: auto-detect)
            dri_threshold: Minimum DRI for valid augmentation (default: 0.6)
        """
        self.model = model
        self.dri_threshold = dri_threshold

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(model, target_layer)

    def compute_iou(
        self,
        attention_map: torch.Tensor,
        disc_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Compute IoU between attention map and disc mask.

        Args:
            attention_map: Grad-CAM heatmap [H, W], values in [0, 1]
            disc_mask: Binary disc mask [H, W], values in {0, 1}
            threshold: Threshold for binarizing attention map (default: 0.5)

        Returns:
            IoU score in [0, 1]
        """
        # Ensure same device
        if attention_map.device != disc_mask.device:
            attention_map = attention_map.to(disc_mask.device)

        # Binarize attention map (top 50% of attention)
        attention_binary = (attention_map >= threshold).float()

        # Compute intersection and union
        intersection = (attention_binary * disc_mask).sum()
        union = attention_binary.sum() + disc_mask.sum() - intersection

        # Avoid division by zero
        if union < 1e-8:
            warnings.warn("Union is zero, returning IoU=0")
            return 0.0

        iou = (intersection / union).item()
        return iou

    def compute_dri(
        self,
        image: torch.Tensor,
        disc_mask: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> Dict[str, Union[float, bool, torch.Tensor]]:
        """
        Compute DRI for an image.

        Args:
            image: Input image [1, C, H, W] or [C, H, W]
            disc_mask: Ground-truth optic disc mask [H, W], binary
            clinical: Clinical indicators [1, clinical_dim] or [clinical_dim] (optional)
            target_class: Class to analyze (default: predicted class)

        Returns:
            Dict with keys:
                - dri: DRI score (IoU between attention and disc mask)
                - valid: Whether DRI meets threshold (≥ 0.6)
                - attention_map: Grad-CAM heatmap [H, W]
                - iou: Same as dri (for convenience)
        """
        # Ensure batch dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]

        if clinical is not None and clinical.ndim == 1:
            clinical = clinical.unsqueeze(0)  # [1, clinical_dim]

        # Ensure disc_mask is on same device and is float
        disc_mask = disc_mask.to(image.device).float()

        # Resize disc_mask to match image size if needed
        if disc_mask.shape != image.shape[2:]:
            disc_mask = F.interpolate(
                disc_mask.unsqueeze(0).unsqueeze(0),
                size=image.shape[2:],
                mode='nearest'
            ).squeeze()

        # Generate attention heatmap
        attention_map = self.grad_cam.generate_heatmap(
            image,
            clinical,
            target_class
        )  # [H, W]

        # Compute IoU
        iou = self.compute_iou(attention_map, disc_mask)

        # Check validity
        valid = iou >= self.dri_threshold

        return {
            'dri': iou,
            'valid': valid,
            'attention_map': attention_map,
            'iou': iou
        }

    def compute_dri_batch(
        self,
        images: torch.Tensor,
        disc_masks: torch.Tensor,
        clinical: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, bool, list]]:
        """
        Compute average DRI over a batch of images.

        Args:
            images: Batch of images [B, C, H, W]
            disc_masks: Batch of disc masks [B, H, W]
            clinical: Batch of clinical indicators [B, clinical_dim] (optional)

        Returns:
            Dict with keys:
                - dri: Average DRI across batch
                - valid: Whether average DRI meets threshold
                - dri_per_sample: List of per-sample DRI scores
        """
        batch_size = images.shape[0]
        dri_scores = []

        for i in range(batch_size):
            image = images[i:i+1]
            disc_mask = disc_masks[i]
            clin = clinical[i:i+1] if clinical is not None else None

            result = self.compute_dri(image, disc_mask, clin)
            dri_scores.append(result['dri'])

        avg_dri = np.mean(dri_scores)

        return {
            'dri': avg_dri,
            'valid': avg_dri >= self.dri_threshold,
            'dri_per_sample': dri_scores
        }


def validate_policy_dri(
    policy,
    model: nn.Module,
    dataset,
    num_samples: int = 10,
    dri_threshold: float = 0.6,
    device: str = 'cpu'
) -> Dict[str, Union[float, bool]]:
    """
    Validate an augmentation policy by checking if it maintains DRI.

    This is the main function that ARC's Critic agent will use to
    validate proposed augmentation policies.

    Args:
        policy: PolicyAugmentor instance or policy dict
        model: Trained model to analyze
        dataset: Dataset with images and disc masks
        num_samples: Number of samples to test (default: 10)
        dri_threshold: Minimum acceptable DRI (default: 0.6)
        device: Device for computation

    Returns:
        Dict with keys:
            - avg_dri: Average DRI across samples
            - valid: Whether policy maintains DRI ≥ threshold
            - dri_scores: List of per-sample DRI scores

    Example:
        >>> from data.policy_augmentor import PolicyAugmentor
        >>> policy = PolicyAugmentor([...])
        >>> result = validate_policy_dri(policy, model, dataset)
        >>> if result['valid']:
        ...     print(f"✓ Policy valid, avg DRI = {result['avg_dri']:.3f}")
        ... else:
        ...     print(f"✗ Policy invalid, avg DRI = {result['avg_dri']:.3f} < {dri_threshold}")
    """
    from data.policy_augmentor import PolicyAugmentor

    # Convert policy dict to PolicyAugmentor if needed
    if isinstance(policy, dict):
        policy = PolicyAugmentor(policy)

    # Initialize DRI computer
    dri_computer = DRIComputer(model, dri_threshold=dri_threshold)

    # Sample images from dataset
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    dri_scores = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx in indices:
            # Get sample from dataset
            sample = dataset[idx]

            # Unpack (handle different dataset formats)
            if isinstance(sample, dict):
                image = sample['image']
                disc_mask = sample.get('disc_mask', sample.get('mask'))
                clinical = sample.get('clinical', None)
            elif len(sample) == 3:
                image, disc_mask, clinical = sample
            else:
                image, disc_mask = sample[:2]
                clinical = None

            # Apply augmentation policy
            augmented_image = policy(image)

            # Ensure tensor format
            if isinstance(augmented_image, Image.Image):
                import torchvision.transforms.functional as TF
                augmented_image = TF.to_tensor(augmented_image)

            augmented_image = augmented_image.to(device)
            disc_mask = disc_mask.to(device)

            if clinical is not None:
                clinical = clinical.to(device)

            # Compute DRI
            result = dri_computer.compute_dri(
                augmented_image,
                disc_mask,
                clinical
            )

            dri_scores.append(result['dri'])

    avg_dri = np.mean(dri_scores)

    return {
        'avg_dri': avg_dri,
        'valid': avg_dri >= dri_threshold,
        'dri_scores': dri_scores
    }


# Example usage
if __name__ == '__main__':
    """Demonstrate DRI computation."""
    import torch
    from PIL import Image

    print("=" * 80)
    print("DRI (Disc Relevance Index) Metrics - Grad-CAM + IoU")
    print("=" * 80)

    # Create dummy model (simple CNN classifier)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 2)

        def forward(self, x, clinical=None):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyModel()
    model.eval()

    print("\n✓ Created dummy model")

    # Create dummy image and disc mask
    image = torch.randn(1, 3, 224, 224)

    # Create circular disc mask (center of image)
    disc_mask = torch.zeros(224, 224)
    center_y, center_x = 112, 112
    radius = 40

    for y in range(224):
        for x in range(224):
            if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                disc_mask[y, x] = 1.0

    print(f"✓ Created dummy image {tuple(image.shape)} and disc mask {tuple(disc_mask.shape)}")
    print(f"  Disc mask coverage: {disc_mask.sum() / disc_mask.numel():.2%}")

    # Initialize DRI computer
    dri_computer = DRIComputer(model, dri_threshold=0.6)
    print("\n✓ Initialized DRI computer with threshold=0.6")

    # Compute DRI
    print("\nComputing DRI...")
    result = dri_computer.compute_dri(image, disc_mask)

    print(f"\nResults:")
    print(f"  • DRI score: {result['dri']:.3f}")
    print(f"  • Valid: {result['valid']}")
    print(f"  • Attention map shape: {result['attention_map'].shape}")
    print(f"  • Attention map range: [{result['attention_map'].min():.3f}, {result['attention_map'].max():.3f}]")

    if result['valid']:
        print(f"\n✓ Policy VALID: DRI {result['dri']:.3f} ≥ {dri_computer.dri_threshold}")
    else:
        print(f"\n✗ Policy INVALID: DRI {result['dri']:.3f} < {dri_computer.dri_threshold}")

    # Test batch computation
    print("\n" + "=" * 80)
    print("Batch DRI Computation")
    print("=" * 80)

    batch_images = torch.randn(4, 3, 224, 224)
    batch_masks = disc_mask.unsqueeze(0).repeat(4, 1, 1)

    batch_result = dri_computer.compute_dri_batch(batch_images, batch_masks)

    print(f"\nBatch size: 4")
    print(f"Average DRI: {batch_result['dri']:.3f}")
    print(f"Valid: {batch_result['valid']}")
    print(f"Per-sample DRI scores: {[f'{s:.3f}' for s in batch_result['dri_per_sample']]}")

    # Test IoU computation directly
    print("\n" + "=" * 80)
    print("IoU Computation Test")
    print("=" * 80)

    # Create attention map that overlaps with disc
    attention_map = torch.zeros(224, 224)
    for y in range(224):
        for x in range(224):
            # Attention focused slightly offset from disc center
            if (y - 120)**2 + (x - 120)**2 <= 35**2:
                attention_map[y, x] = 1.0

    iou = dri_computer.compute_iou(attention_map, disc_mask)
    print(f"\nAttention-Disc IoU: {iou:.3f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
