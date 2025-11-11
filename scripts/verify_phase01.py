#!/usr/bin/env python3
"""
Phase 01 Verification Script

Checks that the environment is properly configured for the smoke test:
- All imports resolve
- CUDA is available (on GPU environments)
- Model can be instantiated
- Forward pass works
- Checkpoint exists and loads (if already trained)
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text.center(60)}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print an error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text):
    """Print an info message."""
    print(f"  {text}")


def check_imports():
    """Verify all required packages can be imported."""
    print_header("Checking Imports")

    imports_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("tqdm", "tqdm"),
    ]

    all_imports_ok = True

    for module_name, display_name in imports_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_success(f"{display_name} ({version})")
        except ImportError as e:
            print_error(f"{display_name} - IMPORT FAILED")
            print_info(f"Error: {e}")
            all_imports_ok = False

    return all_imports_ok


def check_cuda():
    """Verify CUDA availability and GPU info."""
    print_header("Checking CUDA/GPU")

    import torch

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print_success("CUDA is available")
        gpu_count = torch.cuda.device_count()
        print_info(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            total_memory = gpu_props.total_memory / (1024 ** 3)  # Convert to GB
            print_info(f"GPU {i}: {gpu_name}")
            print_info(f"  Memory: {total_memory:.2f} GB")
            print_info(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        print_warning("CUDA is not available - will use CPU")
        print_info("This is OK for testing, but GPU is recommended for training")

    return True  # Not a failure if CUDA is unavailable


def check_model_instantiation():
    """Verify model can be instantiated and used."""
    print_header("Checking Model")

    try:
        import torch
        from src.models.unet_disc_cup import UNet

        # Instantiate model
        model = UNet()
        print_success("Model instantiated successfully")

        # Count parameters
        param_count = model.count_parameters()
        print_info(f"Trainable parameters: {param_count:,}")

        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print_info(f"Model moved to device: {device}")

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
        print_info(f"Created dummy input: {dummy_input.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print_info(f"Forward pass output: {output.shape}")

        # Verify output shape
        expected_shape = (batch_size, 1, 512, 512)
        if output.shape == expected_shape:
            print_success(f"Output shape correct: {output.shape}")
        else:
            print_error(f"Output shape incorrect: {output.shape} (expected {expected_shape})")
            return False

        # Verify output range (sigmoid should be 0-1)
        if output.min() >= 0 and output.max() <= 1:
            print_success(f"Output range correct: [{output.min():.3f}, {output.max():.3f}]")
        else:
            print_error(f"Output range incorrect: [{output.min():.3f}, {output.max():.3f}]")
            return False

        return True

    except Exception as e:
        print_error("Model instantiation or forward pass failed")
        print_info(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset():
    """Verify dataset can be created."""
    print_header("Checking Dataset")

    try:
        from src.data.segmentation_dataset import create_dummy_dataset, SegmentationDataset

        # Create dummy data
        images, masks = create_dummy_dataset(num_samples=5, image_size=512)
        print_success(f"Created {len(images)} dummy image-mask pairs")

        # Create dataset
        dataset = SegmentationDataset(images, masks, augment=False)
        print_success(f"Dataset instantiated with {len(dataset)} samples")

        # Test __getitem__
        img, mask = dataset[0]
        print_info(f"Image tensor shape: {img.shape}")
        print_info(f"Mask tensor shape: {mask.shape}")

        # Verify shapes
        if img.shape == (3, 512, 512) and mask.shape == (1, 512, 512):
            print_success("Dataset returns correct tensor shapes")
        else:
            print_error(f"Dataset shapes incorrect: img={img.shape}, mask={mask.shape}")
            return False

        return True

    except Exception as e:
        print_error("Dataset creation failed")
        print_info(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_checkpoint():
    """Check if checkpoint exists and can be loaded."""
    print_header("Checking Checkpoint (Optional)")

    checkpoint_path = project_root / "models" / "unet_disc_cup.pt"

    if not checkpoint_path.exists():
        print_warning("Checkpoint not found (expected for first run)")
        print_info(f"Path: {checkpoint_path}")
        print_info("Run training script to generate checkpoint")
        return True  # Not a failure

    try:
        import torch
        from src.models.unet_disc_cup import UNet

        print_info(f"Found checkpoint: {checkpoint_path}")

        # Check file size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print_info(f"Size: {size_mb:.2f} MB")

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print_success("Checkpoint loaded successfully")

        # Verify it can be loaded into model
        model = UNet()
        model.load_state_dict(state_dict)
        print_success("Checkpoint compatible with model architecture")

        return True

    except Exception as e:
        print_error("Checkpoint loading failed")
        print_info(f"Error: {e}")
        return False


def main():
    """Run all verification checks."""
    print_header("Phase 01 Verification Script")
    print_info(f"Project root: {project_root}")

    results = {
        "Imports": check_imports(),
        "CUDA/GPU": check_cuda(),
        "Model": check_model_instantiation(),
        "Dataset": check_dataset(),
        "Checkpoint": check_checkpoint(),
    }

    # Summary
    print_header("Verification Summary")

    all_passed = True
    for check_name, result in results.items():
        if result:
            print_success(f"{check_name}: PASS")
        else:
            print_error(f"{check_name}: FAIL")
            all_passed = False

    print()

    if all_passed:
        print_success("All checks passed! Ready for Phase 01 training.")
        print_info("Run: python src/training/train_segmentation.py")
        return 0
    else:
        print_error("Some checks failed. Fix errors before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
