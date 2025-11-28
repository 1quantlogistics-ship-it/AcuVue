#!/usr/bin/env python3
"""
Download Model Weights
======================

Downloads model weights from storage with checksum verification.

Usage:
    python scripts/download_weights.py --model glaucoma_efficientnet_b0_v1
    python scripts/download_weights.py --all
    python scripts/download_weights.py --list

This script manages weights for:
    - Expert heads (glaucoma classifiers)
    - Domain router (MobileNetV3-Small)
"""

import argparse
import hashlib
from pathlib import Path
import urllib.request
import sys


# Model weight registry with SHA256 hashes for verification
# URLs will be configured when cloud storage is set up
WEIGHTS_REGISTRY = {
    "glaucoma_efficientnet_b0_v1": {
        "filename": "glaucoma_efficientnet_b0_v1.pt",
        "destination": "models/production/",
        "sha256": None,  # TODO: Add hash after upload to cloud storage
        "url": None,     # TODO: Add URL after storage setup
        "size_mb": 46.4,
        "description": "EfficientNet-B0 glaucoma classifier (RIM-ONE domain)",
    },
    "domain_classifier_v1": {
        "filename": "domain_classifier_v1.pt",
        "destination": "models/routing/",
        "sha256": None,  # TODO: Add hash after training
        "url": None,     # TODO: Add URL after storage setup
        "size_mb": 5.0,  # Estimated - MobileNetV3-Small
        "description": "MobileNetV3-Small domain router",
    },
}


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """
    Verify file SHA256 checksum.

    Args:
        filepath: Path to file
        expected_sha256: Expected SHA256 hex digest

    Returns:
        True if checksum matches (or no checksum configured)
    """
    if expected_sha256 is None:
        print(f"  Warning: No checksum available for {filepath.name}")
        return True

    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual != expected_sha256:
        print(f"  ERROR: Checksum mismatch!")
        print(f"    Expected: {expected_sha256}")
        print(f"    Got:      {actual}")
        return False
    print(f"  Checksum verified")
    return True


def compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_weights(model_name: str, force: bool = False) -> bool:
    """
    Download weights for a specific model.

    Args:
        model_name: Name of model in registry
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    if model_name not in WEIGHTS_REGISTRY:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(WEIGHTS_REGISTRY.keys())}")
        return False

    info = WEIGHTS_REGISTRY[model_name]
    dest_dir = Path(info['destination'])
    dest_path = dest_dir / info['filename']

    # Check if already exists
    if dest_path.exists() and not force:
        print(f"✓ {model_name} already exists at {dest_path}")
        return verify_checksum(dest_path, info['sha256'])

    # Create directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if info['url'] is None:
        print(f"⚠ No URL configured for {model_name}")
        print(f"  Manual download required. Place file at: {dest_path}")
        print(f"  Expected size: {info['size_mb']:.1f} MB")
        return False

    print(f"Downloading {model_name} ({info['size_mb']:.1f} MB)...")
    try:
        urllib.request.urlretrieve(info['url'], dest_path)
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        return False

    # Verify
    if verify_checksum(dest_path, info['sha256']):
        print(f"✓ Downloaded and verified: {dest_path}")
        return True
    return False


def list_models(show_checksums: bool = False) -> None:
    """List all available models and their status."""
    print("Available models:")
    print("-" * 60)

    for name, info in WEIGHTS_REGISTRY.items():
        dest_path = Path(info['destination']) / info['filename']
        exists = dest_path.exists()
        status = "✓" if exists else "✗"

        print(f"  {status} {name}")
        print(f"      File: {info['filename']}")
        print(f"      Path: {info['destination']}")
        print(f"      Size: {info['size_mb']:.1f} MB")
        print(f"      Desc: {info['description']}")

        if exists and show_checksums:
            checksum = compute_checksum(dest_path)
            print(f"      SHA256: {checksum}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description='Download model weights for AcuVue glaucoma detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_weights.py --list
    python scripts/download_weights.py --model glaucoma_efficientnet_b0_v1
    python scripts/download_weights.py --all
    python scripts/download_weights.py --list --checksums
        """
    )
    parser.add_argument(
        '--model', type=str,
        help='Model name to download'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if file exists'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--checksums', action='store_true',
        help='Show SHA256 checksums for existing files (use with --list)'
    )

    args = parser.parse_args()

    if args.list:
        list_models(show_checksums=args.checksums)
        return

    if args.all:
        success = True
        for name in WEIGHTS_REGISTRY:
            if not download_weights(name, args.force):
                success = False
        sys.exit(0 if success else 1)

    elif args.model:
        success = download_weights(args.model, args.force)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
