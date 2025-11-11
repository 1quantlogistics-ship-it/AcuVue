"""
Data splitting utilities for train/val/test splits.

Provides deterministic splitting with optional stratification for classification tasks.
"""
import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path


class DataSplitter:
    """Handle dataset splitting with reproducibility."""

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def split_indices(
        self,
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, List[int]]:
        """
        Split indices into train/val/test sets.

        Args:
            num_samples: Total number of samples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            shuffle: Whether to shuffle before splitting

        Returns:
            Dictionary with 'train', 'val', 'test' index lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # Create indices
        indices = np.arange(num_samples)

        if shuffle:
            np.random.shuffle(indices)

        # Calculate split sizes
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)

        # Split
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()

        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

    def stratified_split(
        self,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, List[int]]:
        """
        Stratified split maintaining class distribution.

        Args:
            labels: Array of labels for stratification
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            Dictionary with 'train', 'val', 'test' index lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        unique_labels = np.unique(labels)
        train_indices = []
        val_indices = []
        test_indices = []

        # Split each class separately
        for label in unique_labels:
            # Get indices for this class
            class_indices = np.where(labels == label)[0]
            np.random.shuffle(class_indices)

            # Calculate split sizes for this class
            n_class = len(class_indices)
            n_train = int(n_class * train_ratio)
            n_val = int(n_class * val_ratio)

            # Split
            train_indices.extend(class_indices[:n_train].tolist())
            val_indices.extend(class_indices[n_train:n_train + n_val].tolist())
            test_indices.extend(class_indices[n_train + n_val:].tolist())

        # Shuffle the combined splits
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

    def save_splits(
        self,
        splits: Dict[str, List[int]],
        save_path: Path
    ) -> None:
        """
        Save split indices to JSON file.

        Args:
            splits: Dictionary with split indices
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"✓ Splits saved to {save_path}")

    @staticmethod
    def load_splits(load_path: Path) -> Dict[str, List[int]]:
        """
        Load split indices from JSON file.

        Args:
            load_path: Path to JSON file

        Returns:
            Dictionary with split indices
        """
        with open(load_path, 'r') as f:
            splits = json.load(f)

        return splits

    def print_split_stats(
        self,
        splits: Dict[str, List[int]],
        labels: np.ndarray = None
    ) -> None:
        """
        Print statistics about the splits.

        Args:
            splits: Dictionary with split indices
            labels: Optional labels array for class distribution
        """
        print("\n" + "=" * 50)
        print("Dataset Split Statistics")
        print("=" * 50)

        total = sum(len(v) for v in splits.values())

        for split_name, indices in splits.items():
            n_samples = len(indices)
            percentage = (n_samples / total) * 100
            print(f"\n{split_name.upper()}:")
            print(f"  Samples: {n_samples} ({percentage:.1f}%)")

            if labels is not None:
                split_labels = labels[indices]
                unique, counts = np.unique(split_labels, return_counts=True)
                print(f"  Class distribution:")
                for label, count in zip(unique, counts):
                    pct = (count / n_samples) * 100
                    print(f"    Class {label}: {count} ({pct:.1f}%)")

        print("=" * 50 + "\n")


def create_splits(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    labels: np.ndarray = None,
    save_path: Path = None,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Convenience function to create and optionally save splits.

    Args:
        num_samples: Total number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        labels: Optional labels for stratified splitting
        save_path: Optional path to save splits
        seed: Random seed

    Returns:
        Dictionary with split indices
    """
    splitter = DataSplitter(seed=seed)

    # Choose splitting method
    if labels is not None:
        splits = splitter.stratified_split(
            labels, train_ratio, val_ratio, test_ratio
        )
        print("✓ Stratified split created")
    else:
        splits = splitter.split_indices(
            num_samples, train_ratio, val_ratio, test_ratio
        )
        print("✓ Random split created")

    # Print stats
    splitter.print_split_stats(splits, labels)

    # Save if requested
    if save_path is not None:
        splitter.save_splits(splits, save_path)

    return splits


if __name__ == "__main__":
    # Example usage
    splits = create_splits(
        num_samples=100,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        save_path=Path("data/synthetic/splits.json")
    )
