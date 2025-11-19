"""
Curriculum Learning Factory
============================

Central factory for building curriculum schedules from specifications.
Enables ARC's Explorer agent to propose curriculum learning strategies.

Part of ARC Phase E Week 4: Cross-Dataset Curriculum Learning
Dev 2 implementation

Usage:
    >>> spec = {
    ...     "strategy": "gradual_mixing",
    ...     "datasets": ["REFUGE", "ORIGA", "Drishti"],
    ...     "epochs_per_stage": 5,
    ...     "domain_adaptation": True,
    ...     "lambda_domain": 0.1
    ... }
    >>> scheduler, config = build_curriculum_from_spec(spec, dataset_manager)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

# Handle imports for both module and standalone execution
try:
    from .curriculum_scheduler import CurriculumScheduler, create_automatic_curriculum
    from ..data.multi_dataset_manager import MultiDatasetManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.curriculum_scheduler import CurriculumScheduler, create_automatic_curriculum
    from data.multi_dataset_manager import MultiDatasetManager


def validate_curriculum_spec(spec: Dict[str, Any]) -> bool:
    """
    Validate curriculum specification.

    Args:
        spec: Curriculum specification dict

    Returns:
        True if valid

    Raises:
        ValueError: If spec is invalid
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Curriculum spec must be dict, got {type(spec)}")

    # Required fields
    if "strategy" not in spec:
        raise ValueError("Curriculum spec missing 'strategy' field")

    if "datasets" not in spec:
        raise ValueError("Curriculum spec missing 'datasets' field")

    # Validate strategy
    valid_strategies = ["pure_sequential", "gradual_mixing", "adaptive", "reverse", "custom"]
    if spec["strategy"] not in valid_strategies:
        raise ValueError(
            f"Invalid strategy: {spec['strategy']}. "
            f"Valid strategies: {valid_strategies}"
        )

    # Validate datasets
    if not isinstance(spec["datasets"], list) or len(spec["datasets"]) == 0:
        raise ValueError("'datasets' must be non-empty list")

    # Validate custom stages if provided
    if spec["strategy"] == "custom":
        if "stages" not in spec:
            raise ValueError("Custom strategy requires 'stages' field")

        if not isinstance(spec["stages"], list):
            raise ValueError("'stages' must be a list")

        for i, stage in enumerate(spec["stages"]):
            if "datasets" not in stage:
                raise ValueError(f"Stage {i}: missing 'datasets'")
            if "epochs" not in stage:
                raise ValueError(f"Stage {i}: missing 'epochs'")

    # Validate epochs_per_stage for non-custom strategies
    if spec["strategy"] != "custom":
        if "epochs_per_stage" in spec and spec["epochs_per_stage"] <= 0:
            raise ValueError("'epochs_per_stage' must be positive")

    # Validate domain adaptation params
    if spec.get("domain_adaptation", False):
        if "lambda_domain" in spec and spec["lambda_domain"] < 0:
            raise ValueError("'lambda_domain' must be non-negative")
        if "grl_lambda" in spec and spec["grl_lambda"] < 0:
            raise ValueError("'grl_lambda' must be non-negative")

    return True


def build_curriculum_from_spec(
    spec: Dict[str, Any],
    dataset_manager: MultiDatasetManager,
    model: Optional[nn.Module] = None
) -> Tuple[CurriculumScheduler, Dict[str, Any]]:
    """
    Build curriculum scheduler from specification.

    Args:
        spec: Curriculum specification with keys:
            - strategy: "pure_sequential", "gradual_mixing", "adaptive", "reverse", "custom"
            - datasets: List[str] - dataset names
            - epochs_per_stage: int (optional, default 5)
            - stages: List[Dict] (required if strategy="custom")
            - domain_adaptation: bool (optional, default False)
            - lambda_domain: float (optional, default 0.1)
            - grl_lambda: float (optional, default 1.0)
        dataset_manager: MultiDatasetManager instance
        model: Model for domain adaptation (required if domain_adaptation=True)

    Returns:
        Tuple of (CurriculumScheduler, config_dict)

    Example:
        >>> spec = {
        ...     "strategy": "gradual_mixing",
        ...     "datasets": ["REFUGE", "ORIGA", "Drishti"],
        ...     "epochs_per_stage": 5,
        ...     "domain_adaptation": True,
        ...     "lambda_domain": 0.1
        ... }
        >>> scheduler, config = build_curriculum_from_spec(spec, dataset_manager, model)
    """
    # Validate spec
    validate_curriculum_spec(spec)

    strategy = spec["strategy"]
    datasets = spec["datasets"]
    use_domain_adaptation = spec.get("domain_adaptation", False)

    # Get difficulty ranking from dataset manager
    difficulty_ranking = dataset_manager.get_difficulty_ranking()

    # Filter to requested datasets
    filtered_ranking = [
        (name, score) for name, score in difficulty_ranking
        if name in datasets
    ]

    # Build curriculum scheduler
    if strategy == "custom":
        # Use custom stages with gradual_mixing strategy
        # (strategy name doesn't matter for custom stages, just use gradual_mixing)
        scheduler = CurriculumScheduler(
            strategy="gradual_mixing",  # Use valid strategy name
            stages=spec["stages"],
            difficulty_ranking=filtered_ranking
        )

    else:
        # Auto-generate curriculum
        epochs_per_stage = spec.get("epochs_per_stage", 5)

        scheduler = create_automatic_curriculum(
            difficulty_ranking=filtered_ranking,
            strategy=strategy,
            epochs_per_stage=epochs_per_stage
        )

    # Build configuration dict
    config = {
        "strategy": strategy,
        "datasets": datasets,
        "num_stages": len(scheduler.stages),
        "total_epochs": scheduler.get_total_epochs(),
        "domain_adaptation": use_domain_adaptation
    }

    # Add domain adaptation config if requested
    if use_domain_adaptation:
        if model is None:
            raise ValueError("Model required for domain adaptation")

        config["lambda_domain"] = spec.get("lambda_domain", 0.1)
        config["grl_lambda"] = spec.get("grl_lambda", 1.0)
        config["num_domains"] = len(datasets)
        config["dataset_to_id"] = {name: i for i, name in enumerate(datasets)}

    return scheduler, config


def get_curriculum_summary(
    scheduler: CurriculumScheduler,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get comprehensive curriculum summary.

    Args:
        scheduler: CurriculumScheduler instance
        config: Configuration dict from build_curriculum_from_spec

    Returns:
        Dict with curriculum summary
    """
    summary = scheduler.get_curriculum_summary()

    # Add config info
    summary["domain_adaptation"] = config.get("domain_adaptation", False)
    if summary["domain_adaptation"]:
        summary["lambda_domain"] = config["lambda_domain"]
        summary["grl_lambda"] = config["grl_lambda"]
        summary["num_domains"] = config["num_domains"]

    return summary


def get_stage_datasets(
    scheduler: CurriculumScheduler,
    epoch: int
) -> List[str]:
    """
    Get dataset names for current epoch.

    Args:
        scheduler: CurriculumScheduler instance
        epoch: Current training epoch

    Returns:
        List of dataset names for this epoch
    """
    _, stage = scheduler.get_current_stage(epoch)
    return stage.datasets


def get_stage_config(
    scheduler: CurriculumScheduler,
    config: Dict[str, Any],
    epoch: int
) -> Dict[str, Any]:
    """
    Get complete configuration for current epoch.

    Args:
        scheduler: CurriculumScheduler instance
        config: Base configuration dict
        epoch: Current training epoch

    Returns:
        Dict with stage-specific configuration
    """
    stage_idx, stage = scheduler.get_current_stage(epoch)

    stage_config = {
        "stage_index": stage_idx,
        "datasets": stage.datasets,
        "epochs": stage.epochs,
        "learning_rate_mult": stage.learning_rate_mult,
        "difficulty": stage.difficulty,
        "is_transition": scheduler.should_transition(epoch)
    }

    # Add domain adaptation config if enabled
    if config.get("domain_adaptation", False):
        stage_config["domain_adaptation"] = True
        stage_config["lambda_domain"] = config["lambda_domain"]
        stage_config["grl_lambda"] = config["grl_lambda"]
        stage_config["dataset_to_id"] = config["dataset_to_id"]

    return stage_config


# Example usage
if __name__ == '__main__':
    """Demonstrate curriculum factory."""
    import torch
    from torch.utils.data import TensorDataset

    print("=" * 80)
    print("Curriculum Factory Demo")
    print("=" * 80)

    # Create dummy datasets
    print("\n1. Creating dummy datasets...")

    # REFUGE: Large, balanced (EASY)
    refuge_dataset = TensorDataset(
        torch.randn(1000, 3, 224, 224),
        torch.randint(0, 2, (1000,)),
        torch.randn(1000, 224, 224)
    )

    # ORIGA: Medium, moderate imbalance (MEDIUM)
    origa_dataset = TensorDataset(
        torch.randn(500, 3, 224, 224),
        torch.cat([torch.zeros(400, dtype=torch.long), torch.ones(100, dtype=torch.long)]),
        torch.randn(500, 224, 224)
    )

    # Drishti: Small, very imbalanced (HARD)
    drishti_dataset = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)]),
        torch.randn(100, 224, 224)
    )

    print("✓ Datasets created")

    # Create dataset manager
    print("\n2. Creating MultiDatasetManager...")
    manager = MultiDatasetManager(
        datasets=["REFUGE", "ORIGA", "Drishti"],
        data_root="data/processed",
        cache_stats=False
    )
    manager.load_datasets({
        "REFUGE": refuge_dataset,
        "ORIGA": origa_dataset,
        "Drishti": drishti_dataset
    })
    print("✓ Manager created")

    # Test 1: Gradual mixing curriculum
    print("\n3. Gradual Mixing Curriculum")
    spec = {
        "strategy": "gradual_mixing",
        "datasets": ["REFUGE", "ORIGA", "Drishti"],
        "epochs_per_stage": 5
    }

    scheduler, config = build_curriculum_from_spec(spec, manager)
    summary = get_curriculum_summary(scheduler, config)

    print(f"Strategy: {summary['strategy']}")
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Num stages: {summary['num_stages']}")

    print("\nStages:")
    for stage_info in summary['stages']:
        print(f"  Stage {stage_info['stage_index']}: {stage_info['datasets']}")
        print(f"    Epochs: {stage_info['epochs']}, Difficulty: {stage_info['difficulty']:.2f}")

    # Test 2: Custom curriculum
    print("\n4. Custom Curriculum")
    spec = {
        "strategy": "custom",
        "datasets": ["REFUGE", "ORIGA", "Drishti"],
        "stages": [
            {"datasets": ["REFUGE"], "epochs": 10},
            {"datasets": ["REFUGE", "ORIGA"], "epochs": 10},
            {"datasets": ["ORIGA", "Drishti"], "epochs": 5}
        ]
    }

    scheduler, config = build_curriculum_from_spec(spec, manager)
    summary = get_curriculum_summary(scheduler, config)

    print(f"Strategy: {summary['strategy']}")
    print(f"Total epochs: {summary['total_epochs']}")

    print("\nStages:")
    for stage_info in summary['stages']:
        print(f"  Stage {stage_info['stage_index']}: {stage_info['datasets']}")

    # Test 3: With domain adaptation
    print("\n5. Curriculum with Domain Adaptation")
    spec = {
        "strategy": "gradual_mixing",
        "datasets": ["REFUGE", "ORIGA", "Drishti"],
        "epochs_per_stage": 5,
        "domain_adaptation": True,
        "lambda_domain": 0.1,
        "grl_lambda": 1.0
    }

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 10)

    model = DummyModel()

    scheduler, config = build_curriculum_from_spec(spec, manager, model)
    summary = get_curriculum_summary(scheduler, config)

    print(f"Strategy: {summary['strategy']}")
    print(f"Domain adaptation: {summary['domain_adaptation']}")
    print(f"Lambda domain: {summary['lambda_domain']}")
    print(f"GRL lambda: {summary['grl_lambda']}")
    print(f"Num domains: {summary['num_domains']}")

    # Test 4: Stage configuration
    print("\n6. Stage Configuration by Epoch")

    test_epochs = [0, 5, 10]
    for epoch in test_epochs:
        stage_config = get_stage_config(scheduler, config, epoch)
        print(f"\nEpoch {epoch}:")
        print(f"  Stage {stage_config['stage_index']}: {stage_config['datasets']}")
        print(f"  Difficulty: {stage_config['difficulty']:.2f}")
        print(f"  Transition: {stage_config['is_transition']}")

    # Test 5: Spec validation
    print("\n7. Spec Validation")

    try:
        invalid_spec = {"strategy": "invalid_strategy"}
        build_curriculum_from_spec(invalid_spec, manager)
    except ValueError as e:
        print(f"✓ Correctly rejected invalid spec: {e}")

    try:
        invalid_spec = {"strategy": "gradual_mixing", "datasets": []}
        build_curriculum_from_spec(invalid_spec, manager)
    except ValueError as e:
        print(f"✓ Correctly rejected empty datasets: {e}")

    print("\n" + "=" * 80)
    print("Curriculum Factory Demo Complete!")
    print("=" * 80)
