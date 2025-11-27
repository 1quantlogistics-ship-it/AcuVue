"""
Hospital-based data splitting for proper cross-institution evaluation.

Standard random splits on medical imaging datasets can leak information between
train and test sets when images from the same institution appear in both sets.
This module implements institution-based splitting to ensure zero overlap.

Key insight from AcuVue model validation:
- Random RIMONE splits: ~97% AUC (inflated due to data leakage)
- Hospital-based splits (train r2/r3, test r1): 93.7% AUC (realistic)

This module provides the HospitalBasedSplitter class as the standard evaluation
protocol for production models.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from .institution_utils import (
    group_samples_by_institution,
    get_institution_from_metadata,
    get_institution_statistics,
)


class HospitalBasedSplitter:
    """
    Splits dataset by institution to prevent data leakage.

    This splitter ensures that all samples from an institution go entirely
    into one split (train, val, or test), preventing any cross-contamination.

    Example usage:
        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=samples,
            test_institutions=['r1'],
            train_val_institutions=['r2', 'r3']
        )
        assert splitter.validate_no_leakage(splits)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the splitter.

        Args:
            seed: Random seed for reproducible train/val splitting
        """
        self.seed = seed
        np.random.seed(seed)

    def split_by_institution(
        self,
        metadata: List[Dict[str, Any]],
        test_institutions: List[str] = ['r1'],
        train_val_institutions: Optional[List[str]] = None,
        val_ratio: float = 0.1,
        stratified: bool = True
    ) -> Dict[str, List[int]]:
        """
        Split samples by institution with zero overlap between train/val and test.

        Args:
            metadata: List of sample metadata dictionaries. Each sample should
                contain either 'source_hospital', 'institution', or 'original_path'
                for institution extraction.
            test_institutions: Institutions to use exclusively for testing.
                Default: ['r1'] for RIMONE.
            train_val_institutions: Institutions for training/validation.
                If None, uses all institutions not in test_institutions.
            val_ratio: Fraction of train_val data to use for validation.
            stratified: Whether to maintain class balance in train/val split.

        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to lists of
            sample indices.
        """
        # Normalize institution names to lowercase
        test_institutions = [inst.lower() for inst in test_institutions]

        if train_val_institutions is not None:
            train_val_institutions = [inst.lower() for inst in train_val_institutions]

        # Group samples by institution
        institution_groups = group_samples_by_institution(metadata)

        # Separate test and train_val indices
        test_indices: List[int] = []
        train_val_indices: List[int] = []

        for institution, indices in institution_groups.items():
            if institution in test_institutions:
                test_indices.extend(indices)
            elif train_val_institutions is None or institution in train_val_institutions:
                train_val_indices.extend(indices)
            # Samples from other institutions are excluded

        # Split train_val into train and val
        train_indices, val_indices = self._split_train_val(
            train_val_indices,
            metadata,
            val_ratio,
            stratified
        )

        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

        return splits

    def _split_train_val(
        self,
        indices: List[int],
        metadata: List[Dict[str, Any]],
        val_ratio: float,
        stratified: bool
    ) -> Tuple[List[int], List[int]]:
        """
        Split indices into train and validation sets.

        Args:
            indices: List of sample indices to split
            metadata: Full metadata list for accessing labels
            val_ratio: Fraction to use for validation
            stratified: Whether to maintain class balance

        Returns:
            Tuple of (train_indices, val_indices)
        """
        if len(indices) == 0:
            return [], []

        indices = np.array(indices)
        np.random.shuffle(indices)

        if not stratified:
            # Simple random split
            n_val = int(len(indices) * val_ratio)
            val_indices = indices[:n_val].tolist()
            train_indices = indices[n_val:].tolist()
            return train_indices, val_indices

        # Stratified split by label
        labels = np.array([metadata[i].get('label', 0) for i in indices])
        unique_labels = np.unique(labels)

        train_indices: List[int] = []
        val_indices: List[int] = []

        for label in unique_labels:
            label_mask = labels == label
            label_indices = indices[label_mask]
            np.random.shuffle(label_indices)

            n_val = max(1, int(len(label_indices) * val_ratio))
            val_indices.extend(label_indices[:n_val].tolist())
            train_indices.extend(label_indices[n_val:].tolist())

        # Final shuffle
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        return train_indices, val_indices

    def validate_no_leakage(
        self,
        splits: Dict[str, List[int]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Verify zero institution overlap between train/val and test splits.

        Args:
            splits: Split indices dictionary from split_by_institution
            metadata: Sample metadata for institution lookup. Required to verify
                institutions, otherwise only checks index overlap.

        Returns:
            True if no leakage detected, False otherwise
        """
        train_set = set(splits.get('train', []))
        val_set = set(splits.get('val', []))
        test_set = set(splits.get('test', []))

        # Check index overlap (should never happen)
        train_val = train_set | val_set
        if train_val & test_set:
            return False

        if train_set & val_set:
            return False

        # If metadata provided, also verify institution separation
        if metadata is not None:
            train_val_institutions: set = set()
            test_institutions: set = set()

            for idx in train_val:
                inst = get_institution_from_metadata(metadata[idx])
                if inst:
                    train_val_institutions.add(inst)

            for idx in test_set:
                inst = get_institution_from_metadata(metadata[idx])
                if inst:
                    test_institutions.add(inst)

            # No institution should appear in both train/val and test
            if train_val_institutions & test_institutions:
                return False

        return True

    def get_split_statistics(
        self,
        splits: Dict[str, List[int]],
        metadata: List[Dict[str, Any]],
        label_key: str = 'label',
        label_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics for each split.

        Args:
            splits: Split indices dictionary
            metadata: Sample metadata list
            label_key: Key for accessing class labels
            label_names: Optional mapping from label values to names

        Returns:
            Dictionary with statistics for each split including:
            - count: number of samples
            - institutions: list of institutions in split
            - label_distribution: counts per label
            - label_percentages: percentages per label
        """
        stats: Dict[str, Any] = {
            'seed': self.seed,
            'splits': {}
        }

        for split_name, indices in splits.items():
            if not indices:
                stats['splits'][split_name] = {
                    'count': 0,
                    'institutions': [],
                    'label_distribution': {},
                    'label_percentages': {}
                }
                continue

            split_samples = [metadata[i] for i in indices]

            # Get institutions
            institutions = set()
            for sample in split_samples:
                inst = get_institution_from_metadata(sample)
                if inst:
                    institutions.add(inst)

            # Count labels
            label_counts: Dict[Any, int] = {}
            for sample in split_samples:
                label = sample.get(label_key, 'unknown')
                label_counts[label] = label_counts.get(label, 0) + 1

            # Calculate percentages
            total = len(indices)
            label_percentages = {
                k: round(v / total * 100, 1)
                for k, v in label_counts.items()
            }

            # Apply label names if provided
            if label_names:
                label_counts = {
                    label_names.get(k, str(k)): v
                    for k, v in label_counts.items()
                }
                label_percentages = {
                    label_names.get(k, str(k)): v
                    for k, v in label_percentages.items()
                }

            stats['splits'][split_name] = {
                'count': len(indices),
                'institutions': sorted(institutions),
                'label_distribution': label_counts,
                'label_percentages': label_percentages
            }

        return stats

    def save_splits(
        self,
        splits: Dict[str, List[int]],
        save_path: Path,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Save splits to a JSON file with optional statistics.

        Args:
            splits: Split indices dictionary
            save_path: Path to save the JSON file
            metadata: Optional metadata for including statistics
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'train': splits['train'],
            'val': splits['val'],
            'test': splits['test'],
            'splitting_method': 'hospital_based',
            'seed': self.seed
        }

        if metadata is not None:
            output['statistics'] = self.get_split_statistics(splits, metadata)

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def load_splits(load_path: Path) -> Dict[str, List[int]]:
        """
        Load splits from a JSON file.

        Args:
            load_path: Path to the JSON file

        Returns:
            Dictionary with 'train', 'val', 'test' index lists
        """
        with open(load_path, 'r') as f:
            data = json.load(f)

        return {
            'train': data['train'],
            'val': data['val'],
            'test': data['test']
        }

    def print_split_summary(
        self,
        splits: Dict[str, List[int]],
        metadata: List[Dict[str, Any]],
        label_names: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Print a human-readable summary of the splits.

        Args:
            splits: Split indices dictionary
            metadata: Sample metadata list
            label_names: Optional mapping from label values to names
        """
        stats = self.get_split_statistics(splits, metadata, label_names=label_names)

        print("\n" + "=" * 60)
        print("Hospital-Based Split Summary")
        print("=" * 60)
        print(f"Random seed: {self.seed}")
        print(f"Leakage-free: {self.validate_no_leakage(splits, metadata)}")
        print()

        total = sum(len(indices) for indices in splits.values())

        for split_name in ['train', 'val', 'test']:
            split_stats = stats['splits'][split_name]
            count = split_stats['count']
            pct = round(count / total * 100, 1) if total > 0 else 0

            print(f"{split_name.upper()}: {count} samples ({pct}%)")
            print(f"  Institutions: {split_stats['institutions']}")
            print(f"  Labels: {split_stats['label_distribution']}")
            print()

        print("=" * 60)


def create_hospital_based_splits(
    metadata: List[Dict[str, Any]],
    test_institutions: List[str] = ['r1'],
    train_val_institutions: Optional[List[str]] = None,
    val_ratio: float = 0.1,
    save_path: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Convenience function to create and optionally save hospital-based splits.

    This is the recommended way to create splits for production evaluation.

    Args:
        metadata: List of sample metadata dictionaries
        test_institutions: Institutions for test set (default: ['r1'])
        train_val_institutions: Institutions for train/val (default: all others)
        val_ratio: Fraction of train_val for validation
        save_path: Optional path to save splits JSON
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' index lists

    Example:
        >>> import json
        >>> with open('data/processed/rim_one/metadata.json') as f:
        ...     metadata = json.load(f)['samples']
        >>> splits = create_hospital_based_splits(metadata)
        >>> print(f"Train: {len(splits['train'])}, Test: {len(splits['test'])}")
    """
    splitter = HospitalBasedSplitter(seed=seed)

    splits = splitter.split_by_institution(
        metadata=metadata,
        test_institutions=test_institutions,
        train_val_institutions=train_val_institutions,
        val_ratio=val_ratio,
        stratified=True
    )

    # Validate no leakage
    if not splitter.validate_no_leakage(splits, metadata):
        raise ValueError("Data leakage detected in splits!")

    # Print summary
    splitter.print_split_summary(splits, metadata)

    # Save if requested
    if save_path is not None:
        splitter.save_splits(splits, save_path, metadata)
        print(f"Splits saved to {save_path}")

    return splits
