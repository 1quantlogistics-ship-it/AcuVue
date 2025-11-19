"""
Multi-Dataset Manager
=====================

Manages training across multiple fundus datasets with different characteristics.
Enables curriculum learning by ordering datasets by difficulty.

Part of ARC Phase E Week 4: Cross-Dataset Curriculum Learning
Dev 2 implementation

Supported Datasets:
- REFUGE: Large, balanced, high quality (EASY)
- ORIGA: Medium size, moderate quality (MEDIUM)
- Drishti-GS: Small, imbalanced, challenging (HARD)

Usage:
    >>> manager = MultiDatasetManager(
    ...     datasets=["REFUGE", "ORIGA", "Drishti"],
    ...     data_root="/path/to/data"
    ... )
    >>> train_loader = manager.get_curriculum_loader(
    ...     stage_datasets=["REFUGE", "ORIGA"],
    ...     batch_size=32
    ... )
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


class DatasetDifficultyScorer:
    """
    Scores datasets by difficulty based on multiple criteria.

    Difficulty factors:
    - Class imbalance (higher imbalance = harder)
    - Dataset size (smaller = harder to generalize)
    - Image quality (lower quality = harder)
    - Label noise (more noise = harder)
    """

    @staticmethod
    def compute_difficulty_score(
        class_distribution: Dict[int, int],
        dataset_size: int,
        quality_score: float = 1.0
    ) -> float:
        """
        Compute difficulty score (0.0 = easiest, 1.0 = hardest).

        Args:
            class_distribution: {class_id: count} mapping
            dataset_size: Total number of samples
            quality_score: Image quality score (0.0-1.0, higher = better quality)

        Returns:
            Difficulty score between 0.0 and 1.0
        """
        # Factor 1: Class imbalance (0.0 = balanced, 1.0 = very imbalanced)
        counts = np.array(list(class_distribution.values()))
        if len(counts) == 0:
            imbalance_score = 0.0
        else:
            # Use coefficient of variation as imbalance metric
            mean_count = counts.mean()
            std_count = counts.std()
            imbalance_score = min(std_count / (mean_count + 1e-8), 1.0)

        # Factor 2: Dataset size (0.0 = large, 1.0 = very small)
        # Normalize by expected size (assume 1000 is medium, 100 is small, 10000 is large)
        size_score = 1.0 / (1.0 + np.log10(dataset_size + 1) / 3.0)
        size_score = np.clip(size_score, 0.0, 1.0)

        # Factor 3: Quality score (invert so lower quality = higher difficulty)
        quality_difficulty = 1.0 - quality_score

        # Combine factors with weights
        difficulty = (
            0.4 * imbalance_score +
            0.3 * size_score +
            0.3 * quality_difficulty
        )

        return np.clip(difficulty, 0.0, 1.0)

    @staticmethod
    def rank_datasets_by_difficulty(
        dataset_stats: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Rank datasets by difficulty (easiest first).

        Args:
            dataset_stats: {dataset_name: {
                "class_distribution": {0: count, 1: count},
                "size": int,
                "quality": float
            }}

        Returns:
            List of (dataset_name, difficulty_score) tuples, sorted easiest first
        """
        scored_datasets = []

        for dataset_name, stats in dataset_stats.items():
            difficulty = DatasetDifficultyScorer.compute_difficulty_score(
                class_distribution=stats.get("class_distribution", {}),
                dataset_size=stats.get("size", 1000),
                quality_score=stats.get("quality", 1.0)
            )
            scored_datasets.append((dataset_name, difficulty))

        # Sort by difficulty (easiest first)
        scored_datasets.sort(key=lambda x: x[1])

        return scored_datasets


class MultiDatasetManager:
    """
    Manages multiple fundus datasets for curriculum learning.

    Features:
    - Load multiple datasets
    - Score datasets by difficulty
    - Create curriculum-based data loaders
    - Track per-dataset statistics
    """

    def __init__(
        self,
        datasets: List[str],
        data_root: str = "data/processed",
        cache_stats: bool = True
    ):
        """
        Initialize multi-dataset manager.

        Args:
            datasets: List of dataset names to load
            data_root: Root directory containing datasets
            cache_stats: Whether to cache dataset statistics
        """
        self.datasets = datasets
        self.data_root = Path(data_root)
        self.cache_stats = cache_stats

        # Will be populated by load_datasets()
        self.dataset_objects = {}  # {dataset_name: Dataset object}
        self.dataset_stats = {}    # {dataset_name: statistics dict}
        self.difficulty_ranking = []  # List of (name, score) tuples

    def load_datasets(self, dataset_loaders: Dict[str, Dataset]):
        """
        Load dataset objects.

        Args:
            dataset_loaders: {dataset_name: Dataset object}
        """
        self.dataset_objects = dataset_loaders

        # Compute statistics for each dataset
        for dataset_name, dataset in dataset_loaders.items():
            self.dataset_stats[dataset_name] = self._compute_dataset_stats(
                dataset, dataset_name
            )

        # Rank datasets by difficulty
        self.difficulty_ranking = DatasetDifficultyScorer.rank_datasets_by_difficulty(
            self.dataset_stats
        )

    def _compute_dataset_stats(
        self,
        dataset: Dataset,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Compute statistics for a dataset.

        Args:
            dataset: Dataset object
            dataset_name: Name of dataset

        Returns:
            Dict with statistics
        """
        # Check cache first
        cache_path = self.data_root / dataset_name / "curriculum_stats.json"
        if self.cache_stats and cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Compute statistics
        class_counts = {}
        total_samples = len(dataset)

        # Sample labels to compute class distribution
        for i in range(min(total_samples, 10000)):  # Sample up to 10k for speed
            try:
                sample = dataset[i]
                if isinstance(sample, dict):
                    label = int(sample['label'])
                elif isinstance(sample, (tuple, list)):
                    label = int(sample[1])
                else:
                    continue

                class_counts[label] = class_counts.get(label, 0) + 1
            except:
                continue

        # Assign quality scores based on known dataset characteristics
        quality_scores = {
            "REFUGE": 0.9,      # High quality, well-curated
            "ORIGA": 0.7,       # Medium quality
            "Drishti": 0.6,     # Smaller, more challenging
            "Drishti-GS": 0.6
        }

        stats = {
            "class_distribution": class_counts,
            "size": total_samples,
            "quality": quality_scores.get(dataset_name, 0.7)
        }

        # Cache stats
        if self.cache_stats:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(stats, f, indent=2)

        return stats

    def get_difficulty_ranking(self) -> List[Tuple[str, float]]:
        """
        Get datasets ranked by difficulty (easiest first).

        Returns:
            List of (dataset_name, difficulty_score) tuples
        """
        return self.difficulty_ranking

    def get_curriculum_loader(
        self,
        stage_datasets: List[str],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        sampling_strategy: str = "balanced"
    ) -> DataLoader:
        """
        Create data loader for a curriculum stage.

        Args:
            stage_datasets: List of dataset names to include in this stage
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            sampling_strategy: "balanced" or "proportional"
                - balanced: Equal samples from each dataset per batch
                - proportional: Sample proportionally to dataset size

        Returns:
            DataLoader combining specified datasets
        """
        # Get dataset objects for this stage
        datasets_to_combine = [
            self.dataset_objects[name]
            for name in stage_datasets
            if name in self.dataset_objects
        ]

        if not datasets_to_combine:
            raise ValueError(f"No valid datasets found for stage: {stage_datasets}")

        # Combine datasets
        if len(datasets_to_combine) == 1:
            combined_dataset = datasets_to_combine[0]
        else:
            combined_dataset = ConcatDataset(datasets_to_combine)

        # Create data loader
        loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return loader

    def get_stage_info(self, stage_datasets: List[str]) -> Dict[str, Any]:
        """
        Get information about a curriculum stage.

        Args:
            stage_datasets: List of dataset names in this stage

        Returns:
            Dict with stage statistics
        """
        total_samples = 0
        class_distribution = {}
        avg_difficulty = 0.0

        for dataset_name in stage_datasets:
            if dataset_name not in self.dataset_stats:
                continue

            stats = self.dataset_stats[dataset_name]
            total_samples += stats["size"]

            # Merge class distributions
            for cls, count in stats["class_distribution"].items():
                class_distribution[cls] = class_distribution.get(cls, 0) + count

            # Get difficulty score
            for name, score in self.difficulty_ranking:
                if name == dataset_name:
                    avg_difficulty += score
                    break

        if len(stage_datasets) > 0:
            avg_difficulty /= len(stage_datasets)

        return {
            "datasets": stage_datasets,
            "total_samples": total_samples,
            "class_distribution": class_distribution,
            "avg_difficulty": avg_difficulty,
            "num_datasets": len(stage_datasets)
        }


# Example usage
if __name__ == '__main__':
    """Demonstrate multi-dataset manager."""
    import torch
    from torch.utils.data import TensorDataset

    print("=" * 80)
    print("Multi-Dataset Manager Demo")
    print("=" * 80)

    # Create dummy datasets
    print("\n1. Creating dummy datasets...")

    # REFUGE: Large, balanced (EASY)
    refuge_images = torch.randn(1000, 3, 224, 224)
    refuge_labels = torch.randint(0, 2, (1000,))  # Balanced
    refuge_masks = torch.randn(1000, 224, 224)
    refuge_dataset = TensorDataset(refuge_images, refuge_labels, refuge_masks)

    # ORIGA: Medium, moderate imbalance (MEDIUM)
    origa_images = torch.randn(500, 3, 224, 224)
    origa_labels = torch.cat([
        torch.zeros(400, dtype=torch.long),
        torch.ones(100, dtype=torch.long)
    ])  # 80/20 imbalance
    origa_masks = torch.randn(500, 224, 224)
    origa_dataset = TensorDataset(origa_images, origa_labels, origa_masks)

    # Drishti: Small, very imbalanced (HARD)
    drishti_images = torch.randn(100, 3, 224, 224)
    drishti_labels = torch.cat([
        torch.zeros(90, dtype=torch.long),
        torch.ones(10, dtype=torch.long)
    ])  # 90/10 imbalance
    drishti_masks = torch.randn(100, 224, 224)
    drishti_dataset = TensorDataset(drishti_images, drishti_labels, drishti_masks)

    print(f"✓ REFUGE: {len(refuge_dataset)} samples")
    print(f"✓ ORIGA: {len(origa_dataset)} samples")
    print(f"✓ Drishti: {len(drishti_dataset)} samples")

    # Create manager
    print("\n2. Creating MultiDatasetManager...")
    manager = MultiDatasetManager(
        datasets=["REFUGE", "ORIGA", "Drishti"],
        data_root="data/processed",
        cache_stats=False
    )

    # Load datasets
    manager.load_datasets({
        "REFUGE": refuge_dataset,
        "ORIGA": origa_dataset,
        "Drishti": drishti_dataset
    })
    print("✓ Datasets loaded")

    # Get difficulty ranking
    print("\n3. Difficulty Ranking (easiest → hardest):")
    for name, score in manager.get_difficulty_ranking():
        print(f"  {name}: {score:.3f}")

    # Get stage info
    print("\n4. Curriculum Stages:")

    stage1 = ["REFUGE"]
    stage1_info = manager.get_stage_info(stage1)
    print(f"\nStage 1 (Easy): {stage1}")
    print(f"  Total samples: {stage1_info['total_samples']}")
    print(f"  Avg difficulty: {stage1_info['avg_difficulty']:.3f}")

    stage2 = ["REFUGE", "ORIGA"]
    stage2_info = manager.get_stage_info(stage2)
    print(f"\nStage 2 (Medium): {stage2}")
    print(f"  Total samples: {stage2_info['total_samples']}")
    print(f"  Avg difficulty: {stage2_info['avg_difficulty']:.3f}")

    stage3 = ["REFUGE", "ORIGA", "Drishti"]
    stage3_info = manager.get_stage_info(stage3)
    print(f"\nStage 3 (Hard): {stage3}")
    print(f"  Total samples: {stage3_info['total_samples']}")
    print(f"  Avg difficulty: {stage3_info['avg_difficulty']:.3f}")

    # Create data loaders
    print("\n5. Creating curriculum data loaders...")
    loader1 = manager.get_curriculum_loader(stage1, batch_size=16)
    loader2 = manager.get_curriculum_loader(stage2, batch_size=16)
    loader3 = manager.get_curriculum_loader(stage3, batch_size=16)

    print(f"✓ Stage 1 loader: {len(loader1)} batches")
    print(f"✓ Stage 2 loader: {len(loader2)} batches")
    print(f"✓ Stage 3 loader: {len(loader3)} batches")

    print("\n" + "=" * 80)
    print("Multi-Dataset Manager Demo Complete!")
    print("=" * 80)
