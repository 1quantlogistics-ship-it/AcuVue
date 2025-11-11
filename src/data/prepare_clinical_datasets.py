"""
Clinical Dataset Preprocessing for AcuVue

Processes RIM-ONE r3 and REFUGE 2 datasets into unified format matching
the synthetic dataset structure.

Output structure:
  data/processed/rim_one/
    images/          # PNG images
    metadata.json    # Labels, splits, sample info
    splits.json      # Train/val/test indices

  data/processed/refuge2/
    images/          # PNG images (converted from JPG)
    combined_masks/  # PNG masks (converted from BMP, remapped values)
    metadata.json    # Labels, splits, CDR, sample info
    splits.json      # Train/val/test indices

Usage:
  python src/data/prepare_clinical_datasets.py --dataset refuge2
  python src/data/prepare_clinical_datasets.py --dataset rim_one
  python src/data/prepare_clinical_datasets.py --dataset all
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

import cv2
import numpy as np


class DatasetPreprocessor:
    """Base class for dataset preprocessing."""

    def __init__(self, raw_root: str, processed_root: str, seed: int = 42):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.seed = seed
        np.random.seed(seed)

    def create_directories(self):
        """Create output directory structure."""
        self.processed_root.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: Dict):
        """Save metadata.json."""
        metadata_path = self.processed_root / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")

    def save_splits(self, splits: Dict[str, List[int]]):
        """Save splits.json."""
        splits_path = self.processed_root / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"✓ Saved splits: {splits_path}")


class REFUGE2Preprocessor(DatasetPreprocessor):
    """
    REFUGE 2 Dataset Preprocessor.

    Input format:
      REFUGE2/
        train/images/*.jpg, train/mask/*.bmp
        val/images/*.jpg, val/mask/*.bmp
        test/images/*.jpg, test/mask/*.bmp

    Mask encoding (BMP grayscale):
      255 = Background
      128 = Optic Disc
      0   = Optic Cup

    Output format:
      images/*.png (converted from JPG)
      combined_masks/*.png (converted from BMP, remapped to 0/1/2)
      metadata.json
      splits.json
    """

    def __init__(self, raw_root: str = "data/raw/REFUGE2",
                 processed_root: str = "data/processed/refuge2",
                 target_size: int = 512,
                 seed: int = 42):
        super().__init__(raw_root, processed_root, seed)
        self.target_size = target_size

    def convert_mask(self, mask_path: Path) -> np.ndarray:
        """
        Convert REFUGE mask (BMP or PNG) to AcuVue format.

        REFUGE encoding: 255=background, 128=disc, 0=cup
        AcuVue encoding: 0=background, 1=disc, 2=cup
        """
        # Load as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Create output mask
        output_mask = np.zeros_like(mask, dtype=np.uint8)

        # Remap values
        output_mask[mask == 255] = 0  # Background
        output_mask[mask == 128] = 1  # Disc
        output_mask[mask == 0] = 2    # Cup

        # Resize to target size
        if mask.shape[0] != self.target_size or mask.shape[1] != self.target_size:
            output_mask = cv2.resize(output_mask, (self.target_size, self.target_size),
                                    interpolation=cv2.INTER_NEAREST)

        return output_mask

    def extract_label_from_filename(self, filename: str) -> Tuple[int, str]:
        """
        Extract glaucoma label from filename.

        Naming: g####.jpg = glaucoma, n####.jpg = normal, T####.jpg = test (unknown)
        """
        if filename.startswith('g'):
            return 1, "glaucoma"
        elif filename.startswith('n'):
            return 0, "normal"
        elif filename.startswith('T'):
            return -1, "unknown"  # Test set labels not provided
        else:
            return -1, "unknown"

    def process_split(self, split_name: str, start_idx: int) -> Tuple[List[Dict], int]:
        """Process one split (train/val/test)."""
        split_dir = self.raw_root / split_name
        images_dir = split_dir / "images"
        masks_dir = split_dir / "mask"

        # Output directories
        out_images_dir = self.processed_root / "images"
        out_masks_dir = self.processed_root / "combined_masks"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_masks_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = sorted(images_dir.glob("*.jpg"))

        samples = []
        current_idx = start_idx

        print(f"\nProcessing {split_name} split ({len(image_files)} samples)...")

        for img_path in tqdm(image_files, desc=f"  {split_name}"):
            # Corresponding mask - try both .bmp and .png
            mask_path = masks_dir / (img_path.stem + ".bmp")
            if not mask_path.exists():
                mask_path = masks_dir / (img_path.stem + ".png")

            if not mask_path.exists():
                print(f"Warning: Missing mask for {img_path.name}, skipping...")
                continue

            # Output filenames (standardized)
            out_img_name = f"refuge2_{current_idx:04d}.png"
            out_mask_name = f"refuge2_{current_idx:04d}.png"

            # Load and convert image (JPG → PNG)
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load {img_path.name}, skipping...")
                continue

            # Resize image
            if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
                image = cv2.resize(image, (self.target_size, self.target_size))

            # CLAHE removed (Phase 03e): Preprocessing audit determined CLAHE is incompatible
            # with ImageNet normalization and reduces transfer learning effectiveness

            # Convert and save mask
            mask = self.convert_mask(mask_path)

            # Save processed files
            cv2.imwrite(str(out_images_dir / out_img_name), image)
            cv2.imwrite(str(out_masks_dir / out_mask_name), mask)

            # Extract label
            label, label_name = self.extract_label_from_filename(img_path.stem)

            # Create sample metadata
            sample = {
                "sample_id": current_idx,
                "image_filename": out_img_name,
                "mask_filename": out_mask_name,
                "original_image": str(img_path.relative_to(self.raw_root)),
                "original_mask": str(mask_path.relative_to(self.raw_root)),
                "label": label,
                "label_name": label_name,
                "split": split_name,
                "source_split": split_name
            }

            samples.append(sample)
            current_idx += 1

        return samples, current_idx

    def process(self):
        """Process entire REFUGE 2 dataset."""
        print("="*70)
        print("REFUGE 2 Dataset Preprocessing")
        print("="*70)

        self.create_directories()

        # Process all splits
        all_samples = []
        splits = {"train": [], "val": [], "test": []}

        current_idx = 0
        for split_name in ["train", "val", "test"]:
            samples, current_idx = self.process_split(split_name, current_idx)
            all_samples.extend(samples)
            splits[split_name] = [s["sample_id"] for s in samples if s["split"] == split_name]

        # Count labels (excluding test set with unknown labels)
        label_counts = {"glaucoma": 0, "normal": 0, "unknown": 0}
        for sample in all_samples:
            label_counts[sample["label_name"]] += 1

        # Create metadata
        metadata = {
            "dataset": "refuge2",
            "version": "REFUGE 2 Challenge",
            "num_samples": len(all_samples),
            "num_train": len(splits["train"]),
            "num_val": len(splits["val"]),
            "num_test": len(splits["test"]),
            "task": "segmentation",
            "has_masks": True,
            "has_labels": True,
            "label_distribution": label_counts,
            "class_names": ["normal", "glaucoma"],
            "mask_encoding": {
                "0": "background",
                "1": "optic_disc",
                "2": "optic_cup"
            },
            "image_size": self.target_size,
            "source_format": "JPG images + BMP masks",
            "processed_format": "PNG images + PNG masks",
            "seed": self.seed,
            "samples": all_samples
        }

        # Save files
        self.save_metadata(metadata)
        self.save_splits(splits)

        print(f"\n{'='*70}")
        print("REFUGE 2 Processing Complete!")
        print(f"{'='*70}")
        print(f"Total samples: {len(all_samples)}")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val:   {len(splits['val'])}")
        print(f"  Test:  {len(splits['test'])}")
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        print(f"\nOutput directory: {self.processed_root}")


class RIMONEPreprocessor(DatasetPreprocessor):
    """
    RIM-ONE r3 Dataset Preprocessor.

    Input format:
      RIM-ONE_DL_images/partitioned_by_hospital/
        training_set/glaucoma/*.png
        training_set/normal/*.png
        test_set/glaucoma/*.png
        test_set/normal/*.png

    Output format:
      images/*.png (copied/renamed)
      metadata.json (no masks, classification only)
      splits.json (train/val/test with new val split created)
    """

    def __init__(self, raw_root: str = "data/raw/RIM-ONE_DL_images",
                 processed_root: str = "data/processed/rim_one",
                 val_ratio: float = 0.1,
                 target_size: int = 512,
                 seed: int = 42):
        super().__init__(raw_root, processed_root, seed)
        self.val_ratio = val_ratio
        self.target_size = target_size
        self.partition = "partitioned_by_hospital"  # Use hospital partition

    def create_val_split(self, train_samples: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Create validation split from training set using stratified sampling.

        Returns:
            (train_indices, val_indices)
        """
        # Separate by class
        glaucoma_samples = [s for s in train_samples if s["label"] == 1]
        normal_samples = [s for s in train_samples if s["label"] == 0]

        # Shuffle within each class
        np.random.shuffle(glaucoma_samples)
        np.random.shuffle(normal_samples)

        # Split each class
        n_val_glaucoma = int(len(glaucoma_samples) * self.val_ratio)
        n_val_normal = int(len(normal_samples) * self.val_ratio)

        val_glaucoma = glaucoma_samples[:n_val_glaucoma]
        train_glaucoma = glaucoma_samples[n_val_glaucoma:]

        val_normal = normal_samples[:n_val_normal]
        train_normal = normal_samples[n_val_normal:]

        # Combine
        new_train = train_glaucoma + train_normal
        new_val = val_glaucoma + val_normal

        # Get indices
        train_indices = [s["sample_id"] for s in new_train]
        val_indices = [s["sample_id"] for s in new_val]

        print(f"\nValidation split created (stratified):")
        print(f"  Train: {len(train_glaucoma)} glaucoma + {len(train_normal)} normal = {len(new_train)}")
        print(f"  Val:   {len(val_glaucoma)} glaucoma + {len(val_normal)} normal = {len(new_val)}")

        return train_indices, val_indices

    def process_class_dir(self, class_dir: Path, label: int, label_name: str,
                         split_name: str, start_idx: int) -> Tuple[List[Dict], int]:
        """Process one class directory (glaucoma or normal)."""
        out_images_dir = self.processed_root / "images"
        out_images_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(class_dir.glob("*.png"))
        samples = []
        current_idx = start_idx

        for img_path in image_files:
            # Output filename (standardized)
            out_img_name = f"rimone_{current_idx:04d}.png"

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Failed to load {img_path.name}, skipping...")
                continue

            # Resize if needed
            if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
                image = cv2.resize(image, (self.target_size, self.target_size))

            # CLAHE removed (Phase 03e): Preprocessing audit determined CLAHE is incompatible
            # with ImageNet normalization and reduces transfer learning effectiveness

            # Save
            cv2.imwrite(str(out_images_dir / out_img_name), image)

            # Extract hospital code from filename (r1, r2, r3)
            hospital = img_path.stem.split('_')[0] if '_' in img_path.stem else "unknown"

            # Create sample metadata
            sample = {
                "sample_id": current_idx,
                "image_filename": out_img_name,
                "original_path": str(img_path.relative_to(self.raw_root)),
                "source_hospital": hospital,
                "label": label,
                "label_name": label_name,
                "split": split_name,  # Will be updated for val split
                "source_split": split_name
            }

            samples.append(sample)
            current_idx += 1

        return samples, current_idx

    def process(self):
        """Process entire RIM-ONE r3 dataset."""
        print("="*70)
        print("RIM-ONE r3 Dataset Preprocessing")
        print("="*70)

        self.create_directories()

        partition_root = self.raw_root / self.partition

        all_samples = []
        current_idx = 0

        # Process training set
        print(f"\nProcessing training set...")
        train_glaucoma_dir = partition_root / "training_set" / "glaucoma"
        train_normal_dir = partition_root / "training_set" / "normal"

        samples_tg, current_idx = self.process_class_dir(
            train_glaucoma_dir, 1, "glaucoma", "train", current_idx
        )
        samples_tn, current_idx = self.process_class_dir(
            train_normal_dir, 0, "normal", "train", current_idx
        )

        train_samples = samples_tg + samples_tn
        all_samples.extend(train_samples)

        print(f"  Train: {len(samples_tg)} glaucoma + {len(samples_tn)} normal = {len(train_samples)}")

        # Process test set
        print(f"\nProcessing test set...")
        test_glaucoma_dir = partition_root / "test_set" / "glaucoma"
        test_normal_dir = partition_root / "test_set" / "normal"

        samples_testg, current_idx = self.process_class_dir(
            test_glaucoma_dir, 1, "glaucoma", "test", current_idx
        )
        samples_testn, current_idx = self.process_class_dir(
            test_normal_dir, 0, "normal", "test", current_idx
        )

        test_samples = samples_testg + samples_testn
        all_samples.extend(test_samples)

        print(f"  Test: {len(samples_testg)} glaucoma + {len(samples_testn)} normal = {len(test_samples)}")

        # Create validation split from training
        train_indices, val_indices = self.create_val_split(train_samples)

        # Update split labels for val samples
        for sample in all_samples:
            if sample["sample_id"] in val_indices:
                sample["split"] = "val"

        # Create splits dict
        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": [s["sample_id"] for s in test_samples]
        }

        # Count labels
        label_counts = {"glaucoma": 0, "normal": 0}
        for sample in all_samples:
            label_counts[sample["label_name"]] += 1

        # Create metadata
        metadata = {
            "dataset": "rim_one_r3",
            "version": "RIM-ONE DL r3",
            "partition": self.partition,
            "num_samples": len(all_samples),
            "num_train": len(splits["train"]),
            "num_val": len(splits["val"]),
            "num_test": len(splits["test"]),
            "task": "classification",
            "has_masks": False,
            "has_labels": True,
            "label_distribution": label_counts,
            "class_names": ["normal", "glaucoma"],
            "image_size": self.target_size,
            "val_split_ratio": self.val_ratio,
            "val_split_method": "stratified",
            "seed": self.seed,
            "samples": all_samples
        }

        # Save files
        self.save_metadata(metadata)
        self.save_splits(splits)

        print(f"\n{'='*70}")
        print("RIM-ONE r3 Processing Complete!")
        print(f"{'='*70}")
        print(f"Total samples: {len(all_samples)}")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val:   {len(splits['val'])}")
        print(f"  Test:  {len(splits['test'])}")
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            pct = (count / len(all_samples)) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        print(f"\nOutput directory: {self.processed_root}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess clinical datasets for AcuVue")
    parser.add_argument("--dataset", type=str, choices=["refuge2", "rim_one", "all"],
                       default="all", help="Dataset to process")
    parser.add_argument("--target-size", type=int, default=512,
                       help="Target image size (default: 512)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if args.dataset in ["refuge2", "all"]:
        print("\n" + "="*70)
        print("Processing REFUGE 2...")
        print("="*70 + "\n")
        refuge_processor = REFUGE2Preprocessor(target_size=args.target_size, seed=args.seed)
        refuge_processor.process()

    if args.dataset in ["rim_one", "all"]:
        print("\n" + "="*70)
        print("Processing RIM-ONE r3...")
        print("="*70 + "\n")
        rimone_processor = RIMONEPreprocessor(target_size=args.target_size, seed=args.seed)
        rimone_processor.process()

    print("\n" + "="*70)
    print("All preprocessing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
