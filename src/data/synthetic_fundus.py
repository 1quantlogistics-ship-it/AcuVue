"""
Synthetic fundus image generator for Phase 02-Lite.

Generates anatomically plausible retinal fundus images with optic disc and cup
segmentation masks. This serves as a bridge solution until real RIM-ONE/REFUGE
datasets are integrated.

Features:
- Realistic circular fundus images with gradient backgrounds
- Anatomically plausible disc and cup regions
- Configurable CDR (Cup-to-Disc Ratio)
- Train/val/test splits with deterministic seeding
"""
import numpy as np
import cv2
from typing import Tuple, List, Dict
from pathlib import Path
import json


class SyntheticFundusGenerator:
    """Generate synthetic fundus images with disc/cup masks."""

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 512,
        seed: int = 42
    ):
        """
        Initialize generator.

        Args:
            num_samples: Total number of samples to generate
            image_size: Size of square images
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        np.random.seed(seed)

    def generate_fundus_background(self) -> np.ndarray:
        """Generate realistic fundus background with radial gradient."""
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Create circular mask
        center = self.image_size // 2
        Y, X = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)

        # Radial gradient (darker at edges)
        radius = self.image_size // 2 - 10
        circular_mask = dist_from_center <= radius

        # Generate orange/red fundus-like color
        for i in range(self.image_size):
            for j in range(self.image_size):
                if circular_mask[i, j]:
                    # Distance-based gradient
                    intensity = 1.0 - (dist_from_center[i, j] / radius) * 0.4

                    # Fundus-like RGB values (orange/reddish)
                    r = int(220 * intensity + np.random.randint(-15, 15))
                    g = int(140 * intensity + np.random.randint(-15, 15))
                    b = int(80 * intensity + np.random.randint(-15, 15))

                    img[i, j] = [
                        np.clip(b, 0, 255),  # OpenCV is BGR
                        np.clip(g, 0, 255),
                        np.clip(r, 0, 255)
                    ]

        # Add some texture noise
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def generate_disc_and_cup(
        self,
        cdr_vertical: float = 0.5,
        cdr_horizontal: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate optic disc and cup masks.

        Args:
            cdr_vertical: Cup-to-Disc Ratio (vertical)
            cdr_horizontal: Cup-to-Disc Ratio (horizontal)

        Returns:
            Tuple of (disc_mask, cup_mask)
        """
        # Disc parameters (off-center, nasal side)
        disc_center_x = int(self.image_size * 0.45)  # Slightly off-center
        disc_center_y = int(self.image_size * 0.5)
        disc_radius = int(self.image_size * 0.15)  # ~15% of image

        # Create disc mask
        Y, X = np.ogrid[:self.image_size, :self.image_size]
        disc_dist = np.sqrt(
            ((X - disc_center_x) / disc_radius)**2 +
            ((Y - disc_center_y) / disc_radius)**2
        )
        disc_mask = (disc_dist <= 1.0).astype(np.uint8) * 255

        # Create cup mask (inside disc)
        cup_radius_x = disc_radius * cdr_horizontal
        cup_radius_y = disc_radius * cdr_vertical
        cup_dist = np.sqrt(
            ((X - disc_center_x) / cup_radius_x)**2 +
            ((Y - disc_center_y) / cup_radius_y)**2
        )
        cup_mask = (cup_dist <= 1.0).astype(np.uint8) * 255

        # Apply smoothing to make boundaries more realistic
        disc_mask = cv2.GaussianBlur(disc_mask, (5, 5), 0)
        cup_mask = cv2.GaussianBlur(cup_mask, (5, 5), 0)

        # Threshold back to binary
        _, disc_mask = cv2.threshold(disc_mask, 127, 255, cv2.THRESH_BINARY)
        _, cup_mask = cv2.threshold(cup_mask, 127, 255, cv2.THRESH_BINARY)

        return disc_mask, cup_mask

    def generate_sample(self, sample_id: int) -> Dict:
        """
        Generate a single fundus image with masks.

        Args:
            sample_id: Unique identifier for reproducible generation

        Returns:
            Dictionary with 'image', 'disc_mask', 'cup_mask', 'metadata'
        """
        # Set seed for this specific sample
        np.random.seed(self.seed + sample_id)

        # Generate background
        image = self.generate_fundus_background()

        # Random CDR (healthy: 0.2-0.4, glaucoma: 0.5-0.8)
        is_glaucoma = np.random.random() > 0.5
        if is_glaucoma:
            cdr_v = np.random.uniform(0.5, 0.8)
            cdr_h = np.random.uniform(0.5, 0.8)
            label = 1
        else:
            cdr_v = np.random.uniform(0.2, 0.4)
            cdr_h = np.random.uniform(0.2, 0.4)
            label = 0

        # Generate disc and cup
        disc_mask, cup_mask = self.generate_disc_and_cup(cdr_v, cdr_h)

        # Combine masks for segmentation (disc=1, cup=2)
        combined_mask = np.zeros_like(disc_mask)
        combined_mask[disc_mask > 0] = 1
        combined_mask[cup_mask > 0] = 2

        metadata = {
            'sample_id': sample_id,
            'cdr_vertical': float(cdr_v),
            'cdr_horizontal': float(cdr_h),
            'label': int(label),
            'label_name': 'glaucoma' if label == 1 else 'healthy'
        }

        return {
            'image': image,
            'disc_mask': disc_mask,
            'cup_mask': cup_mask,
            'combined_mask': combined_mask,
            'metadata': metadata
        }

    def generate_dataset(
        self,
        save_dir: Path,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, List[int]]:
        """
        Generate complete dataset and save to disk.

        Args:
            save_dir: Directory to save generated data
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing

        Returns:
            Dictionary with split indices
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ['images', 'disc_masks', 'cup_masks', 'combined_masks']:
            (save_dir / subdir).mkdir(exist_ok=True)

        # Determine split sizes
        n_train = int(self.num_samples * train_split)
        n_val = int(self.num_samples * val_split)
        n_test = self.num_samples - n_train - n_val

        splits = {
            'train': list(range(0, n_train)),
            'val': list(range(n_train, n_train + n_val)),
            'test': list(range(n_train + n_val, self.num_samples))
        }

        # Generate and save all samples
        metadata_list = []

        print(f"Generating {self.num_samples} synthetic fundus images...")
        print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

        for i in range(self.num_samples):
            sample = self.generate_sample(i)

            # Determine split
            if i in splits['train']:
                split_name = 'train'
            elif i in splits['val']:
                split_name = 'val'
            else:
                split_name = 'test'

            sample['metadata']['split'] = split_name

            # Save image and masks
            cv2.imwrite(
                str(save_dir / 'images' / f'sample_{i:04d}.png'),
                sample['image']
            )
            cv2.imwrite(
                str(save_dir / 'disc_masks' / f'sample_{i:04d}.png'),
                sample['disc_mask']
            )
            cv2.imwrite(
                str(save_dir / 'cup_masks' / f'sample_{i:04d}.png'),
                sample['cup_mask']
            )
            cv2.imwrite(
                str(save_dir / 'combined_masks' / f'sample_{i:04d}.png'),
                sample['combined_mask']
            )

            metadata_list.append(sample['metadata'])

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{self.num_samples}")

        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump({
                'num_samples': self.num_samples,
                'image_size': self.image_size,
                'seed': self.seed,
                'splits': splits,
                'samples': metadata_list
            }, f, indent=2)

        # Save split indices
        with open(save_dir / 'splits.json', 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"✓ Dataset saved to {save_dir}")
        return splits


def generate_synthetic_dataset(
    output_dir: str = "data/synthetic",
    num_samples: int = 100,
    image_size: int = 512,
    seed: int = 42
) -> None:
    """
    Convenience function to generate synthetic dataset.

    Args:
        output_dir: Output directory path
        num_samples: Number of samples to generate
        image_size: Size of square images
        seed: Random seed
    """
    generator = SyntheticFundusGenerator(
        num_samples=num_samples,
        image_size=image_size,
        seed=seed
    )

    generator.generate_dataset(Path(output_dir))


if __name__ == "__main__":
    # Generate default dataset
    generate_synthetic_dataset()
