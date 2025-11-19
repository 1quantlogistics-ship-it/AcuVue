"""
Augmentation Visualization Tool
================================

Visualizes the effects of augmentation policies on fundus images.
Useful for debugging policies and understanding their impact.

Part of ARC Phase E Week 2: Augmentation Policy Search
Dev 2 implementation

Features:
- Side-by-side comparison of original vs augmented images
- Grid view showing multiple augmented versions
- Attention heatmap overlay (Grad-CAM + DRI visualization)
- Save visualizations to disk

Usage:
    >>> from data.visualize_augmentations import visualize_policy
    >>> policy = [{"operation": "rotate", "probability": 0.5, "magnitude": 10.0}]
    >>> visualize_policy(image, policy, save_path="aug_viz.png")
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings

try:
    from .policy_augmentor import PolicyAugmentor
    from .augmentation_ops import list_safe_operations
except ImportError:
    from policy_augmentor import PolicyAugmentor
    from augmentation_ops import list_safe_operations


def visualize_policy(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    policy: List[Dict[str, Any]],
    num_samples: int = 4,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize augmentation policy effects on an image.

    Creates a grid showing:
    - Original image (top-left)
    - Multiple augmented versions

    Args:
        image: Input image (PIL Image, torch Tensor, or numpy array)
        policy: Augmentation policy (list of sub-policies)
        num_samples: Number of augmented versions to show (default: 4)
        save_path: Path to save visualization (optional)
        show: Whether to display with plt.show() (default: True)
    """
    # Convert to PIL Image if needed
    if isinstance(image, torch.Tensor):
        import torchvision.transforms.functional as TF
        image = TF.to_pil_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Create augmentor
    augmentor = PolicyAugmentor(policy)

    # Generate augmented versions
    augmented_images = [image]  # Include original as first image
    for i in range(num_samples):
        aug = augmentor(image)
        if isinstance(aug, torch.Tensor):
            import torchvision.transforms.functional as TF
            aug = TF.to_pil_image(aug)
        augmented_images.append(aug)

    # Create grid visualization
    grid_size = int(np.ceil(np.sqrt(num_samples + 1)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    # Plot images
    for idx, (ax, img) in enumerate(zip(axes, augmented_images)):
        ax.imshow(img)
        ax.axis('off')
        if idx == 0:
            ax.set_title("Original", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"Augmented {idx}", fontsize=10)

    # Hide unused subplots
    for idx in range(len(augmented_images), len(axes)):
        axes[idx].axis('off')

    # Add policy description
    policy_text = "Policy:\n"
    for i, sub_policy in enumerate(policy):
        policy_text += f"{i+1}. {sub_policy['operation']} (p={sub_policy['probability']:.2f}, m={sub_policy['magnitude']:.2f})\n"

    fig.suptitle(policy_text, fontsize=10, ha='left', x=0.02, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_policy_with_heatmap(
    image: Union[Image.Image, torch.Tensor],
    policy: List[Dict[str, Any]],
    model: torch.nn.Module,
    disc_mask: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize augmentation policy with attention heatmap overlay.

    Shows:
    - Original image
    - Augmented image
    - Attention heatmap (Grad-CAM) for original
    - Attention heatmap for augmented
    - DRI scores

    Args:
        image: Input image
        policy: Augmentation policy
        model: Trained model for Grad-CAM
        disc_mask: Ground-truth optic disc mask (optional, for DRI computation)
        save_path: Path to save visualization (optional)
        show: Whether to display with plt.show() (default: True)
    """
    import sys
    from pathlib import Path

    try:
        from ..evaluation.dri_metrics import DRIComputer
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from evaluation.dri_metrics import DRIComputer

    import torchvision.transforms.functional as TF

    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        image_tensor = TF.to_tensor(image).unsqueeze(0)
    else:
        if image.ndim == 3:
            image_tensor = image.unsqueeze(0)
        else:
            image_tensor = image

    # Apply augmentation
    augmentor = PolicyAugmentor(policy)
    aug_image = augmentor(image)

    if isinstance(aug_image, Image.Image):
        aug_tensor = TF.to_tensor(aug_image).unsqueeze(0)
    else:
        if aug_image.ndim == 3:
            aug_tensor = aug_image.unsqueeze(0)
        else:
            aug_tensor = aug_image

    # Compute attention heatmaps
    dri_computer = DRIComputer(model)

    model.eval()

    # Original image attention
    if disc_mask is not None:
        result_orig = dri_computer.compute_dri(image_tensor, disc_mask)
        heatmap_orig = result_orig['attention_map']
        dri_orig = result_orig['dri']
    else:
        heatmap_orig = dri_computer.grad_cam.generate_heatmap(image_tensor)
        dri_orig = None

    # Augmented image attention
    if disc_mask is not None:
        result_aug = dri_computer.compute_dri(aug_tensor, disc_mask)
        heatmap_aug = result_aug['attention_map']
        dri_aug = result_aug['dri']
    else:
        heatmap_aug = dri_computer.grad_cam.generate_heatmap(aug_tensor)
        dri_aug = None

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Convert tensors to PIL for display
    if isinstance(image, torch.Tensor):
        image_display = TF.to_pil_image(image_tensor.squeeze())
    else:
        image_display = image

    if isinstance(aug_image, torch.Tensor):
        aug_display = TF.to_pil_image(aug_tensor.squeeze())
    else:
        aug_display = aug_image

    # Original image
    axes[0, 0].imshow(image_display)
    axes[0, 0].axis('off')
    title_orig = "Original"
    if dri_orig is not None:
        title_orig += f" (DRI: {dri_orig:.3f})"
    axes[0, 0].set_title(title_orig, fontsize=12, fontweight='bold')

    # Augmented image
    axes[0, 1].imshow(aug_display)
    axes[0, 1].axis('off')
    title_aug = "Augmented"
    if dri_aug is not None:
        title_aug += f" (DRI: {dri_aug:.3f})"
    axes[0, 1].set_title(title_aug, fontsize=12, fontweight='bold')

    # Original attention heatmap
    axes[1, 0].imshow(image_display)
    axes[1, 0].imshow(heatmap_orig.cpu().numpy(), alpha=0.5, cmap='jet')
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Original Attention (Grad-CAM)", fontsize=12)

    # Augmented attention heatmap
    axes[1, 1].imshow(aug_display)
    axes[1, 1].imshow(heatmap_aug.cpu().numpy(), alpha=0.5, cmap='jet')
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Augmented Attention (Grad-CAM)", fontsize=12)

    # Add disc mask overlay if provided
    if disc_mask is not None:
        disc_contour = disc_mask.cpu().numpy()
        axes[1, 0].contour(disc_contour, colors='white', linewidths=2, levels=[0.5])
        axes[1, 1].contour(disc_contour, colors='white', linewidths=2, levels=[0.5])

    # Add policy description
    policy_text = "Policy:\n"
    for i, sub_policy in enumerate(policy):
        policy_text += f"{i+1}. {sub_policy['operation']} (p={sub_policy['probability']:.2f}, m={sub_policy['magnitude']:.2f})\n"

    fig.suptitle(policy_text, fontsize=10, ha='left', x=0.02, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved heatmap visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compare_policies(
    image: Union[Image.Image, torch.Tensor],
    policies: List[List[Dict[str, Any]]],
    policy_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Compare multiple augmentation policies side-by-side.

    Args:
        image: Input image
        policies: List of augmentation policies
        policy_names: Names for each policy (optional)
        save_path: Path to save visualization (optional)
        show: Whether to display with plt.show() (default: True)
    """
    import torchvision.transforms.functional as TF

    if policy_names is None:
        policy_names = [f"Policy {i+1}" for i in range(len(policies))]

    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)

    # Generate augmented versions for each policy
    augmented_sets = []
    for policy in policies:
        augmentor = PolicyAugmentor(policy)
        aug = augmentor(image)
        if isinstance(aug, torch.Tensor):
            aug = TF.to_pil_image(aug)
        augmented_sets.append(aug)

    # Create grid: original + all policies
    num_policies = len(policies)
    fig, axes = plt.subplots(1, num_policies + 1, figsize=(4 * (num_policies + 1), 4))

    if num_policies == 0:
        axes = [axes]

    # Original
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title("Original", fontsize=12, fontweight='bold')

    # Augmented versions
    for idx, (ax, aug, name) in enumerate(zip(axes[1:], augmented_sets, policy_names)):
        ax.imshow(aug)
        ax.axis('off')
        ax.set_title(name, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved policy comparison to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_operation_effects(
    image: Union[Image.Image, torch.Tensor],
    operation_name: str,
    magnitudes: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize the effect of a single operation at different magnitudes.

    Args:
        image: Input image
        operation_name: Name of operation to visualize
        magnitudes: List of magnitude values to test (optional, uses operation range)
        save_path: Path to save visualization (optional)
        show: Whether to display with plt.show() (default: True)
    """
    from augmentation_ops import get_operation
    import torchvision.transforms.functional as TF

    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)

    # Get operation
    op = get_operation(operation_name)

    # Generate magnitude values if not provided
    if magnitudes is None:
        mag_min, mag_max = op.magnitude_range
        magnitudes = np.linspace(mag_min, mag_max, 5)

    # Apply operation at different magnitudes
    augmented_images = [image]  # Include original
    magnitude_labels = ["Original"]

    for mag in magnitudes:
        aug = op.apply(image, mag)
        if isinstance(aug, torch.Tensor):
            aug = TF.to_pil_image(aug)
        augmented_images.append(aug)
        magnitude_labels.append(f"mag={mag:.2f}")

    # Create grid
    num_images = len(augmented_images)
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

    if num_images == 1:
        axes = [axes]

    for ax, img, label in zip(axes, augmented_images, magnitude_labels):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label, fontsize=10)

    fig.suptitle(f"Operation: {operation_name}", fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved operation visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_operations(
    image: Union[Image.Image, torch.Tensor],
    save_dir: Optional[str] = None
) -> None:
    """
    Visualize the effect of all safe operations on an image.

    Args:
        image: Input image
        save_dir: Directory to save visualizations (optional)
    """
    from pathlib import Path

    operations = list_safe_operations()

    for op_name in operations:
        print(f"Visualizing operation: {op_name}")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(save_dir) / f"{op_name}_effect.png")
        else:
            save_path = None

        visualize_operation_effects(
            image,
            op_name,
            save_path=save_path,
            show=False
        )

    print(f"\n✓ Visualized all {len(operations)} operations")


# Example usage
if __name__ == '__main__':
    """Demonstrate augmentation visualization."""
    import torch

    print("=" * 80)
    print("Augmentation Visualization Tool")
    print("=" * 80)

    # Create dummy fundus image (grayscale circular region)
    img_size = 224
    image_np = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Create circular fundus appearance
    center_y, center_x = img_size // 2, img_size // 2
    radius = img_size // 3

    for y in range(img_size):
        for x in range(img_size):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist <= radius:
                # Gradient from center (lighter) to edge (darker)
                intensity = int(200 - (dist / radius) * 100)
                image_np[y, x] = [intensity, int(intensity * 0.8), int(intensity * 0.6)]

    image = Image.fromarray(image_np)
    print(f"✓ Created dummy fundus image: {image.size}")

    # Test 1: Visualize single policy
    print("\n" + "=" * 80)
    print("Test 1: Visualize Single Policy")
    print("=" * 80)

    policy = [
        {"operation": "rotate", "probability": 0.7, "magnitude": 15.0},
        {"operation": "brightness", "probability": 0.5, "magnitude": 0.1}
    ]

    print("\nPolicy:")
    for i, sp in enumerate(policy):
        print(f"  {i+1}. {sp['operation']} (p={sp['probability']:.2f}, m={sp['magnitude']:.2f})")

    visualize_policy(image, policy, num_samples=3, show=False)
    print("✓ Policy visualization complete")

    # Test 2: Compare multiple policies
    print("\n" + "=" * 80)
    print("Test 2: Compare Multiple Policies")
    print("=" * 80)

    policy1 = [{"operation": "rotate", "probability": 0.5, "magnitude": 10.0}]
    policy2 = [{"operation": "hflip", "probability": 1.0, "magnitude": 1.0}]
    policy3 = [{"operation": "brightness", "probability": 0.5, "magnitude": 0.2}]

    policies = [policy1, policy2, policy3]
    policy_names = ["Rotation", "Horizontal Flip", "Brightness"]

    compare_policies(image, policies, policy_names, show=False)
    print("✓ Policy comparison complete")

    # Test 3: Visualize operation effects
    print("\n" + "=" * 80)
    print("Test 3: Visualize Operation Effects")
    print("=" * 80)

    visualize_operation_effects(image, "rotate", magnitudes=[-15, -7.5, 0, 7.5, 15], show=False)
    print("✓ Operation effect visualization complete")

    print("\n" + "=" * 80)
    print("All visualization tests passed!")
    print("=" * 80)
    print("\nNote: Set show=True to display visualizations interactively")
