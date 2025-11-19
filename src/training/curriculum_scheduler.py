"""
Curriculum Scheduler
====================

Schedules curriculum learning progression across multiple datasets.
Implements various curriculum strategies for gradual difficulty increase.

Part of ARC Phase E Week 4: Cross-Dataset Curriculum Learning
Dev 2 implementation

Curriculum Strategies:
- pure_sequential: Train on one dataset at a time (easy → hard)
- gradual_mixing: Gradually introduce harder datasets
- adaptive: Adjust based on validation performance
- reverse: Start with hardest (anti-curriculum for robustness)

Usage:
    >>> scheduler = CurriculumScheduler(
    ...     strategy="gradual_mixing",
    ...     stages=[
    ...         {"datasets": ["REFUGE"], "epochs": 5},
    ...         {"datasets": ["REFUGE", "ORIGA"], "epochs": 5},
    ...         {"datasets": ["REFUGE", "ORIGA", "Drishti"], "epochs": 10}
    ...     ]
    ... )
    >>> current_stage = scheduler.get_current_stage(epoch=7)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """
    Defines a stage in the curriculum.

    Attributes:
        datasets: List of dataset names to include
        epochs: Number of epochs for this stage
        learning_rate_mult: Learning rate multiplier for this stage
        difficulty: Average difficulty of this stage
    """
    datasets: List[str]
    epochs: int
    learning_rate_mult: float = 1.0
    difficulty: float = 0.0


class CurriculumScheduler:
    """
    Schedules curriculum learning progression.

    Manages transitions between curriculum stages and provides
    stage-specific configuration.
    """

    VALID_STRATEGIES = [
        "pure_sequential",   # One dataset at a time
        "gradual_mixing",    # Gradually add datasets
        "adaptive",          # Adjust based on performance
        "reverse"            # Start with hardest (anti-curriculum)
    ]

    def __init__(
        self,
        strategy: str,
        stages: List[Dict[str, Any]],
        difficulty_ranking: Optional[List[Tuple[str, float]]] = None
    ):
        """
        Initialize curriculum scheduler.

        Args:
            strategy: Curriculum strategy name
            stages: List of stage definitions, each with:
                - datasets: List[str] - dataset names
                - epochs: int - number of epochs
                - learning_rate_mult: float (optional)
            difficulty_ranking: Optional list of (dataset_name, difficulty) tuples

        Raises:
            ValueError: If strategy is invalid or stages are malformed
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Valid strategies: {self.VALID_STRATEGIES}"
            )

        self.strategy = strategy
        self.stages = self._parse_stages(stages, difficulty_ranking)
        self.current_stage_idx = 0
        self.current_epoch = 0
        self.difficulty_ranking = difficulty_ranking or []

        # Validate stages
        self._validate_stages()

    def _parse_stages(
        self,
        stages: List[Dict[str, Any]],
        difficulty_ranking: Optional[List[Tuple[str, float]]]
    ) -> List[CurriculumStage]:
        """Parse stage dictionaries into CurriculumStage objects."""
        parsed_stages = []

        for stage_dict in stages:
            # Compute average difficulty
            datasets = stage_dict["datasets"]
            avg_difficulty = 0.0
            if difficulty_ranking:
                difficulties = [
                    score for name, score in difficulty_ranking
                    if name in datasets
                ]
                avg_difficulty = np.mean(difficulties) if difficulties else 0.0

            stage = CurriculumStage(
                datasets=datasets,
                epochs=stage_dict["epochs"],
                learning_rate_mult=stage_dict.get("learning_rate_mult", 1.0),
                difficulty=avg_difficulty
            )
            parsed_stages.append(stage)

        return parsed_stages

    def _validate_stages(self):
        """Validate curriculum stages."""
        if not self.stages:
            raise ValueError("At least one stage required")

        for i, stage in enumerate(self.stages):
            if not stage.datasets:
                raise ValueError(f"Stage {i}: No datasets specified")
            if stage.epochs <= 0:
                raise ValueError(f"Stage {i}: Epochs must be positive")

        # Check difficulty progression for non-reverse strategies
        if self.strategy != "reverse":
            for i in range(len(self.stages) - 1):
                if self.stages[i].difficulty > self.stages[i+1].difficulty:
                    print(
                        f"Warning: Stage {i} difficulty ({self.stages[i].difficulty:.3f}) "
                        f"> Stage {i+1} ({self.stages[i+1].difficulty:.3f}). "
                        f"Expected increasing difficulty."
                    )

    def get_current_stage(self, epoch: int) -> Tuple[int, CurriculumStage]:
        """
        Get current curriculum stage based on epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Tuple of (stage_index, CurriculumStage)
        """
        cumulative_epochs = 0

        for stage_idx, stage in enumerate(self.stages):
            if epoch < cumulative_epochs + stage.epochs:
                return stage_idx, stage
            cumulative_epochs += stage.epochs

        # If beyond all stages, return last stage
        return len(self.stages) - 1, self.stages[-1]

    def get_stage_by_index(self, stage_idx: int) -> CurriculumStage:
        """
        Get stage by index.

        Args:
            stage_idx: Stage index

        Returns:
            CurriculumStage

        Raises:
            IndexError: If stage_idx is out of range
        """
        return self.stages[stage_idx]

    def should_transition(self, epoch: int) -> bool:
        """
        Check if should transition to next stage.

        Args:
            epoch: Current training epoch

        Returns:
            True if transitioning to new stage
        """
        current_idx, _ = self.get_current_stage(epoch)
        next_idx, _ = self.get_current_stage(epoch + 1)

        return next_idx != current_idx

    def get_total_epochs(self) -> int:
        """Get total number of epochs across all stages."""
        return sum(stage.epochs for stage in self.stages)

    def get_stage_info(self, stage_idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a stage.

        Args:
            stage_idx: Stage index

        Returns:
            Dict with stage information
        """
        stage = self.stages[stage_idx]

        return {
            "stage_index": stage_idx,
            "datasets": stage.datasets,
            "epochs": stage.epochs,
            "learning_rate_mult": stage.learning_rate_mult,
            "difficulty": stage.difficulty,
            "num_datasets": len(stage.datasets)
        }

    def get_curriculum_summary(self) -> Dict[str, Any]:
        """
        Get summary of entire curriculum.

        Returns:
            Dict with curriculum summary
        """
        return {
            "strategy": self.strategy,
            "num_stages": len(self.stages),
            "total_epochs": self.get_total_epochs(),
            "stages": [
                self.get_stage_info(i)
                for i in range(len(self.stages))
            ]
        }


def create_automatic_curriculum(
    difficulty_ranking: List[Tuple[str, float]],
    strategy: str = "gradual_mixing",
    epochs_per_stage: int = 5
) -> CurriculumScheduler:
    """
    Create curriculum automatically from difficulty ranking.

    Args:
        difficulty_ranking: List of (dataset_name, difficulty) tuples (easiest first)
        strategy: Curriculum strategy
        epochs_per_stage: Epochs per stage

    Returns:
        CurriculumScheduler

    Example:
        >>> ranking = [("REFUGE", 0.2), ("ORIGA", 0.5), ("Drishti", 0.7)]
        >>> scheduler = create_automatic_curriculum(ranking, strategy="gradual_mixing")
    """
    if not difficulty_ranking:
        raise ValueError("Difficulty ranking required")

    dataset_names = [name for name, _ in difficulty_ranking]

    if strategy == "pure_sequential":
        # One dataset per stage
        stages = [
            {"datasets": [name], "epochs": epochs_per_stage}
            for name in dataset_names
        ]

    elif strategy == "gradual_mixing":
        # Gradually add datasets
        stages = []
        for i in range(len(dataset_names)):
            stages.append({
                "datasets": dataset_names[:i+1],
                "epochs": epochs_per_stage
            })

    elif strategy == "reverse":
        # Start with hardest
        reversed_names = list(reversed(dataset_names))
        stages = []
        for i in range(len(reversed_names)):
            stages.append({
                "datasets": reversed_names[:i+1],
                "epochs": epochs_per_stage
            })

    else:  # adaptive or default to gradual_mixing
        stages = []
        for i in range(len(dataset_names)):
            stages.append({
                "datasets": dataset_names[:i+1],
                "epochs": epochs_per_stage
            })

    return CurriculumScheduler(
        strategy=strategy,
        stages=stages,
        difficulty_ranking=difficulty_ranking
    )


# Example usage
if __name__ == '__main__':
    """Demonstrate curriculum scheduler."""

    print("=" * 80)
    print("Curriculum Scheduler Demo")
    print("=" * 80)

    # Example difficulty ranking (easiest → hardest)
    difficulty_ranking = [
        ("REFUGE", 0.2),
        ("ORIGA", 0.5),
        ("Drishti", 0.7)
    ]

    print("\n1. Difficulty Ranking:")
    for name, score in difficulty_ranking:
        print(f"  {name}: {score:.1f}")

    # Test 1: Manual curriculum
    print("\n2. Manual Gradual Mixing Curriculum:")
    stages = [
        {"datasets": ["REFUGE"], "epochs": 5},
        {"datasets": ["REFUGE", "ORIGA"], "epochs": 5},
        {"datasets": ["REFUGE", "ORIGA", "Drishti"], "epochs": 10}
    ]

    scheduler = CurriculumScheduler(
        strategy="gradual_mixing",
        stages=stages,
        difficulty_ranking=difficulty_ranking
    )

    summary = scheduler.get_curriculum_summary()
    print(f"Strategy: {summary['strategy']}")
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Num stages: {summary['num_stages']}")

    print("\nStages:")
    for stage_info in summary['stages']:
        print(f"  Stage {stage_info['stage_index']}: {stage_info['datasets']}")
        print(f"    Epochs: {stage_info['epochs']}, Difficulty: {stage_info['difficulty']:.2f}")

    # Test 2: Epoch-based stage lookup
    print("\n3. Epoch → Stage Mapping:")
    test_epochs = [0, 4, 5, 9, 10, 15, 19]
    for epoch in test_epochs:
        stage_idx, stage = scheduler.get_current_stage(epoch)
        print(f"  Epoch {epoch:2d} → Stage {stage_idx}: {stage.datasets}")

    # Test 3: Automatic curriculum generation
    print("\n4. Automatic Curriculum Generation:")

    for strategy in ["pure_sequential", "gradual_mixing", "reverse"]:
        auto_scheduler = create_automatic_curriculum(
            difficulty_ranking=difficulty_ranking,
            strategy=strategy,
            epochs_per_stage=5
        )

        print(f"\n  Strategy: {strategy}")
        auto_summary = auto_scheduler.get_curriculum_summary()
        for stage_info in auto_summary['stages']:
            print(f"    Stage {stage_info['stage_index']}: {stage_info['datasets']}")

    # Test 4: Transition detection
    print("\n5. Stage Transitions:")
    for epoch in range(scheduler.get_total_epochs()):
        if scheduler.should_transition(epoch):
            stage_idx, stage = scheduler.get_current_stage(epoch + 1)
            print(f"  Epoch {epoch} → {epoch+1}: Transition to Stage {stage_idx} ({stage.datasets})")

    print("\n" + "=" * 80)
    print("Curriculum Scheduler Demo Complete!")
    print("=" * 80)
