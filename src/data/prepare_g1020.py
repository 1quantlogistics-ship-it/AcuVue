#!/usr/bin/env python3
"""
G1020 Dataset Preprocessor for AcuVue

Processes G1020 dataset (1020 fundus images with OD/OC masks and glaucoma labels)
into unified format matching RIM-ONE and REFUGE2 preprocessing.

Input format:
  data/raw/G1020/
    G1020.csv              # Labels (imageID, binaryLabels: 0=normal, 1=glaucoma)
    Images/                # JPG fundus images
    Images/*.json          # LabelMe-style polygon annotations (disc, cup, discLoc)
    Masks/                 # PNG segmentation masks (pre-rendered from polygons)

Output format:
  data/processed/g1020/
    images/                # PNG images (converted from JPG, resized to 512x512)
    masks/                 # PNG masks (converted/remapped to 0/1/2 encoding)
    metadata.json          # Labels, splits, sample info
    splits.json            # Train/val/test indices

Mask encoding (output):
  0 = Background
  1 = Optic Disc
  2 = Optic Cup

Usage:
  python src/data/prepare_g1020.py
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class G1020Preprocessor:
    """
    G1020 Dataset Preprocessor.

    Converts G1020 dataset with LabelMe annotations and CSV labels
    into unified AcuVue format.
    """

    def __init__(self,
                 raw_root: str = "data/raw/G1020",
                 processed_root: str = "data/processed/g1020",
                 target_size: int = 512,
                 seed: int = 42,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.10,
                 test_ratio: float = 0.20):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.target_size = target_size
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        np.random.seed(seed)

    def create_directories(self):
        """Create output directory structure."""
        self.processed_root.mkdir(parents=True, exist_ok=True)
        (self.processed_root / "images").mkdir(exist_ok=True)
        (self.processed_root / "masks").mkdir(exist_ok=True)
        print(f"✓ Created directories in {self.processed_root}")

    def convert_mask(self, mask_path: Path) -> np.ndarray:
        """
        Convert G1020 mask to AcuVue format.

        Assumes mask is already in PNG format with pixel values:
          0 = Background
          128 or 1 = Optic Disc
          255 or 2 = Optic Cup

        Output encoding: 0=background, 1=disc, 2=cup
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Create output mask
        output_mask = np.zeros_like(mask, dtype=np.uint8)

        # Map values to AcuVue encoding
        # Assume G1020 masks use: 0=bg, 128=disc, 255=cup OR 0=bg, 1=disc, 2=cup
        # Check unique values to determine encoding
        unique_vals = np.unique(mask)

        if set(unique_vals).issubset({0, 1, 2}):
            # Already in 0/1/2 format
            output_mask = mask.copy()
        elif set(unique_vals).issubset({0, 128, 255}):
            # Convert from 0/128/255 format
            output_mask[mask == 0] = 0      # Background
            output_mask[mask == 128] = 1    # Disc
            output_mask[mask == 255] = 2    # Cup
        else:
            # Try to infer from polygon rendering
            # Typically: background=0, disc=lower values, cup=higher values
            # For safety, use threshold-based approach
            output_mask[mask == 0] = 0
            output_mask[(mask > 0) & (mask < 200)] = 1  # Disc
            output_mask[mask >= 200] = 2                # Cup

        return output_mask

    def process(self):
        """Main preprocessing pipeline."""
        print("\n" + "="*60)
        print("G1020 Dataset Preprocessing")
        print("="*60)

        # Step 1: Create directories
        self.create_directories()

        # Step 2: Load labels from CSV
        csv_path = self.raw_root / "G1020.csv"
        labels_df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(labels_df)} samples from {csv_path}")
        print(f"  Label distribution: {labels_df['binaryLabels'].value_counts().to_dict()}")

        # Step 3: Create stratified train/val/test splits
        train_indices, temp_indices = train_test_split(
            np.arange(len(labels_df)),
            test_size=(self.val_ratio + self.test_ratio),
            stratify=labels_df['binaryLabels'],
            random_state=self.seed
        )

        val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_test_ratio),
            stratify=labels_df.iloc[temp_indices]['binaryLabels'],
            random_state=self.seed
        )

        splits = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }

        print(f"\n✓ Created stratified splits:")
        print(f"  Train: {len(train_indices)} samples ({self.train_ratio*100:.0f}%)")
        print(f"  Val: {len(val_indices)} samples ({self.val_ratio*100:.0f}%)")
        print(f"  Test: {len(test_indices)} samples ({self.test_ratio*100:.0f}%)")

        # Step 4: Process images and masks
        metadata = {
            "dataset": "g1020",
            "version": "G1020 fundus images with OD/OC masks",
            "source": "G1020 dataset (ORIGA + REFUGE + G1020 collection)",
            "num_samples": len(labels_df),
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_test": len(test_indices),
            "task": "classification",
            "has_masks": True,
            "has_labels": True,
            "label_distribution": labels_df['binaryLabels'].value_counts().to_dict(),
            "class_names": ["normal", "glaucoma"],
            "image_size": self.target_size,
            "seed": self.seed,
            "samples": []
        }

        # Determine split for each sample
        split_map = {}
        for idx in train_indices:
            split_map[idx] = 'train'
        for idx in val_indices:
            split_map[idx] = 'val'
        for idx in test_indices:
            split_map[idx] = 'test'

        # Process each sample
        images_dir = self.raw_root / "Images"
        masks_dir = self.raw_root / "Masks"

        processed_count = 0
        skipped_count = 0

        print(f"\n✓ Processing {len(labels_df)} samples...")

        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
            image_id = row['imageID']  # e.g., "image_0.jpg"
            label = int(row['binaryLabels'])

            # Determine source image and mask paths
            image_filename = image_id
            mask_filename = image_id.replace('.jpg', '.png')

            source_image_path = images_dir / image_filename
            source_mask_path = masks_dir / mask_filename

            # Skip if image doesn't exist
            if not source_image_path.exists():
                skipped_count += 1
                continue

            # Read and resize image
            image = cv2.imread(str(source_image_path))
            if image is None:
                skipped_count += 1
                continue

            image_resized = cv2.resize(image, (self.target_size, self.target_size),
                                      interpolation=cv2.INTER_LANCZOS4)

            # CLAHE removed (Phase 03e): Preprocessing audit determined CLAHE is incompatible
            # with ImageNet normalization and reduces transfer learning effectiveness

            # Process mask if available
            mask_resized = None
            if source_mask_path.exists():
                try:
                    mask = self.convert_mask(source_mask_path)
                    mask_resized = cv2.resize(mask, (self.target_size, self.target_size),
                                             interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    print(f"  Warning: Failed to process mask for {image_id}: {e}")

            # Save processed files
            output_image_filename = f"g1020_{idx:04d}.png"
            output_mask_filename = f"g1020_{idx:04d}_mask.png" if mask_resized is not None else None

            output_image_path = self.processed_root / "images" / output_image_filename
            cv2.imwrite(str(output_image_path), image_resized)

            if mask_resized is not None:
                output_mask_path = self.processed_root / "masks" / output_mask_filename
                cv2.imwrite(str(output_mask_path), mask_resized)

            # Add to metadata
            sample_info = {
                "sample_id": idx,
                "image_filename": output_image_filename,
                "mask_filename": output_mask_filename,
                "label": label,
                "label_name": "glaucoma" if label == 1 else "normal",
                "split": split_map[idx],
                "source_image_id": image_id
            }

            metadata['samples'].append(sample_info)
            processed_count += 1

        print(f"\n✓ Processed {processed_count} samples (skipped {skipped_count})")

        # Step 5: Save metadata and splits
        metadata_path = self.processed_root / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")

        splits_path = self.processed_root / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"✓ Saved splits: {splits_path}")

        # Print summary
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        print(f"Output directory: {self.processed_root}")
        print(f"Total samples: {processed_count}")
        print(f"  Train: {len(train_indices)}")
        print(f"  Val: {len(val_indices)}")
        print(f"  Test: {len(test_indices)}")
        print(f"\nLabel distribution:")
        for label_val, count in metadata['label_distribution'].items():
            label_name = "Glaucoma" if label_val == 1 else "Normal"
            pct = (count / processed_count) * 100
            print(f"  {label_name}: {count} ({pct:.1f}%)")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess G1020 dataset")
    parser.add_argument('--raw_root', type=str, default="data/raw/G1020",
                       help="Path to raw G1020 dataset")
    parser.add_argument('--processed_root', type=str, default="data/processed/g1020",
                       help="Path to output processed dataset")
    parser.add_argument('--target_size', type=int, default=512,
                       help="Target image size (default: 512)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for splits (default: 42)")
    parser.add_argument('--train_ratio', type=float, default=0.70,
                       help="Train split ratio (default: 0.70)")
    parser.add_argument('--val_ratio', type=float, default=0.10,
                       help="Val split ratio (default: 0.10)")
    parser.add_argument('--test_ratio', type=float, default=0.20,
                       help="Test split ratio (default: 0.20)")

    args = parser.parse_args()

    preprocessor = G1020Preprocessor(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        target_size=args.target_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    preprocessor.process()


if __name__ == "__main__":
    main()
