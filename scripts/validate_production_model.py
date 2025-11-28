#!/usr/bin/env python3
"""
Production Model Validation Script

Validates the production model and data splitting infrastructure.
Run this before deploying or after any changes to ensure everything works.

Usage:
    python scripts/validate_production_model.py

Exit codes:
    0 - All validations passed
    1 - Validation failed
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_result(name: str, passed: bool, message: str = ""):
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if message:
        print(f"         {message}")


def validate_model_files() -> bool:
    """Check that model files exist."""
    print_header("Model Files Validation")

    model_dir = PROJECT_ROOT / "models" / "production"
    model_path = model_dir / "glaucoma_efficientnet_b0_v1.pt"

    # Check model weights
    model_exists = model_path.exists()
    if model_exists:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_result("Model weights exist", True, f"{size_mb:.1f} MB")
    else:
        print_result("Model weights exist", False, f"Not found: {model_path}")
        return False

    # Check model size is reasonable (EfficientNet-B0 should be ~40-50 MB)
    size_valid = 30 < size_mb < 70
    print_result("Model size reasonable", size_valid, f"Expected 40-50 MB, got {size_mb:.1f} MB")

    return model_exists and size_valid


def validate_model_loading() -> bool:
    """Test that model loads correctly."""
    print_header("Model Loading Validation")

    try:
        from src.inference import GlaucomaPredictor

        model_path = PROJECT_ROOT / "models" / "production" / "glaucoma_efficientnet_b0_v1.pt"

        if not model_path.exists():
            print_result("Model loading", False, "Model file not found")
            return False

        predictor = GlaucomaPredictor.from_checkpoint(str(model_path), device='cpu')
        print_result("Model loads successfully", True)

        # Verify model structure
        assert predictor.class_names == ['normal', 'glaucoma']
        print_result("Class names correct", True, str(predictor.class_names))

        return True

    except Exception as e:
        print_result("Model loading", False, str(e))
        return False


def validate_inference() -> bool:
    """Test inference on synthetic image."""
    print_header("Inference Validation")

    try:
        import numpy as np
        from PIL import Image
        from src.inference import GlaucomaPredictor

        model_path = PROJECT_ROOT / "models" / "production" / "glaucoma_efficientnet_b0_v1.pt"

        if not model_path.exists():
            print_result("Inference test", False, "Model file not found")
            return False

        predictor = GlaucomaPredictor.from_checkpoint(str(model_path), device='cpu')

        # Create synthetic fundus image
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = 180
        arr[:, :, 1] = 80
        arr[:, :, 2] = 50
        test_image = Image.fromarray(arr, mode='RGB')

        result = predictor.predict(test_image)

        print_result("Prediction returned", True, f"{result.prediction}")
        print_result("Confidence valid", 0 <= result.confidence <= 1,
                    f"{result.confidence:.3f}")

        prob_sum = sum(result.probabilities.values())
        print_result("Probabilities sum to 1", abs(prob_sum - 1.0) < 1e-5,
                    f"Sum: {prob_sum:.5f}")

        return True

    except Exception as e:
        print_result("Inference test", False, str(e))
        return False


def validate_hospital_splitter() -> bool:
    """Test hospital-based splitting."""
    print_header("Hospital Splitter Validation")

    try:
        from src.data.hospital_splitter import HospitalBasedSplitter

        # Create test metadata
        samples = []
        for i in range(30):
            hospital = ['r1', 'r2', 'r3'][i % 3]
            samples.append({
                'sample_id': i,
                'source_hospital': hospital,
                'label': i % 2
            })

        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=samples,
            test_institutions=['r1']
        )

        # Validate splits
        print_result("Splits created", True,
                    f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        # Validate no leakage
        no_leakage = splitter.validate_no_leakage(splits, samples)
        print_result("No data leakage", no_leakage)

        # Validate test set is all r1
        test_hospitals = set(samples[i]['source_hospital'] for i in splits['test'])
        correct_test = test_hospitals == {'r1'}
        print_result("Test set contains only r1", correct_test, str(test_hospitals))

        # Validate train/val has no r1
        train_val_indices = splits['train'] + splits['val']
        train_val_hospitals = set(samples[i]['source_hospital'] for i in train_val_indices)
        no_r1_in_train = 'r1' not in train_val_hospitals
        print_result("Train/val excludes r1", no_r1_in_train, str(train_val_hospitals))

        return no_leakage and correct_test and no_r1_in_train

    except Exception as e:
        print_result("Hospital splitter test", False, str(e))
        return False


def validate_rimone_metadata() -> bool:
    """Validate RIMONE dataset metadata if available."""
    print_header("RIMONE Dataset Validation")

    metadata_path = PROJECT_ROOT / "data" / "processed" / "rim_one" / "metadata.json"

    if not metadata_path.exists():
        print_result("RIMONE metadata", True, "Not found (optional)")
        return True

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

        samples = metadata.get('samples', [])
        print_result("Samples loaded", True, f"{len(samples)} samples")

        # Check for institution field
        has_hospital = all('source_hospital' in s for s in samples[:10])
        print_result("Hospital field present", has_hospital)

        # Check hospital distribution
        from src.data.institution_utils import group_samples_by_institution
        groups = group_samples_by_institution(samples)
        print_result("Hospital groups found", len(groups) >= 2, str(list(groups.keys())))

        return True

    except Exception as e:
        print_result("RIMONE validation", False, str(e))
        return False


def validate_tests() -> bool:
    """Run unit tests."""
    print_header("Unit Tests Validation")

    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             str(PROJECT_ROOT / "tests" / "unit"),
             "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=300
        )

        passed = result.returncode == 0

        # Count tests from output
        if "passed" in result.stdout:
            print_result("Unit tests", passed, result.stdout.split('\n')[-2])
        else:
            print_result("Unit tests", passed, result.stderr[:200] if not passed else "")

        return passed

    except subprocess.TimeoutExpired:
        print_result("Unit tests", False, "Timeout after 5 minutes")
        return False
    except Exception as e:
        print_result("Unit tests", False, str(e))
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print(" AcuVue Production Model Validation")
    print("="*60)

    results = []

    # Run validations
    results.append(("Model Files", validate_model_files()))
    results.append(("Model Loading", validate_model_loading()))
    results.append(("Inference", validate_inference()))
    results.append(("Hospital Splitter", validate_hospital_splitter()))
    results.append(("RIMONE Data", validate_rimone_metadata()))
    results.append(("Unit Tests", validate_tests()))

    # Summary
    print_header("Summary")

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All validations PASSED")
        return 0
    else:
        print("\n  Some validations FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
