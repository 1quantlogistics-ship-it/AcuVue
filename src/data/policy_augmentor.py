"""
Policy-Based Augmentation for AutoAugment-style Search
=======================================================

Applies augmentation policies proposed by ARC's Explorer agent.
Each policy is a sequence of operations with probabilities and magnitudes.

Part of ARC Phase E Week 2: Augmentation Policy Search
Dev 2 implementation

Policy Format:
[
    {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
    {"operation": "brightness", "probability": 0.3, "magnitude": 0.15},
    {"operation": "gaussian_blur", "probability": 0.2, "magnitude": 2.0}
]

Constraint: Only safe operations allowed (no destructive augmentations)
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional, Tuple
import random

try:
    from .augmentation_ops import (
        get_operation,
        list_safe_operations,
        validate_operation_name,
        ForbiddenOperationError,
        SAFE_OPERATIONS
    )
except ImportError:
    from augmentation_ops import (
        get_operation,
        list_safe_operations,
        validate_operation_name,
        ForbiddenOperationError,
        SAFE_OPERATIONS
    )


class InvalidPolicyError(Exception):
    """Raised when a policy is malformed or contains forbidden operations."""
    pass


class PolicyAugmentor:
    """
    Applies augmentation policies to images.

    A policy is a list of augmentation sub-policies, where each sub-policy
    is applied stochastically based on its probability.

    Example:
        >>> policy = [
        ...     {"operation": "rotate", "probability": 0.5, "magnitude": 10.0},
        ...     {"operation": "brightness", "probability": 0.3, "magnitude": 0.1}
        ... ]
        >>> augmentor = PolicyAugmentor(policy)
        >>> augmented_image = augmentor(image)
    """

    def __init__(
        self,
        policy: List[Dict[str, Any]],
        seed: Optional[int] = None
    ):
        """
        Initialize PolicyAugmentor.

        Args:
            policy: List of sub-policies, each with:
                - operation: str (operation name)
                - probability: float (0-1, chance of applying)
                - magnitude: float (operation strength)
            seed: Random seed for reproducibility (optional)

        Raises:
            InvalidPolicyError: If policy is malformed or contains forbidden ops
        """
        self.policy = policy
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate policy on initialization
        self.validate_policy(policy)

        # Pre-load operations for efficiency
        self.operations = []
        for sub_policy in policy:
            op = get_operation(sub_policy["operation"])
            self.operations.append({
                "op": op,
                "probability": sub_policy["probability"],
                "magnitude": sub_policy["magnitude"]
            })

    @staticmethod
    def validate_policy(policy: List[Dict[str, Any]]) -> bool:
        """
        Validate that a policy is well-formed and safe.

        Args:
            policy: Policy to validate

        Returns:
            True if valid

        Raises:
            InvalidPolicyError: If policy is malformed or unsafe
        """
        if not isinstance(policy, list):
            raise InvalidPolicyError(f"Policy must be a list, got {type(policy)}")

        if len(policy) == 0:
            raise InvalidPolicyError("Policy must contain at least one sub-policy")

        if len(policy) > 10:
            raise InvalidPolicyError(
                f"Policy contains {len(policy)} operations (max 10 for efficiency)"
            )

        for i, sub_policy in enumerate(policy):
            # Check required fields
            required_fields = ["operation", "probability", "magnitude"]
            for field in required_fields:
                if field not in sub_policy:
                    raise InvalidPolicyError(
                        f"Sub-policy {i} missing required field: {field}"
                    )

            # Validate operation name (will raise ForbiddenOperationError if forbidden)
            operation = sub_policy["operation"]
            try:
                is_safe = validate_operation_name(operation)
                if not is_safe:
                    raise InvalidPolicyError(
                        f"Sub-policy {i} uses forbidden operation: {operation}"
                    )
            except ForbiddenOperationError as e:
                raise InvalidPolicyError(str(e))

            # Validate probability
            prob = sub_policy["probability"]
            if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
                raise InvalidPolicyError(
                    f"Sub-policy {i} probability must be in [0, 1], got {prob}"
                )

            # Validate magnitude (must be numeric, range checked by operation itself)
            mag = sub_policy["magnitude"]
            if not isinstance(mag, (int, float)):
                raise InvalidPolicyError(
                    f"Sub-policy {i} magnitude must be numeric, got {type(mag)}"
                )

        return True

    def apply_policy(
        self,
        image: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the augmentation policy to an image.

        Each sub-policy is applied stochastically based on its probability.

        Args:
            image: PIL Image or torch Tensor [C, H, W]

        Returns:
            Augmented image (same type as input)
        """
        augmented = image

        for sub_policy in self.operations:
            # Stochastic application based on probability
            if random.random() < sub_policy["probability"]:
                op = sub_policy["op"]
                magnitude = sub_policy["magnitude"]
                augmented = op.apply(augmented, magnitude)

        return augmented

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """Shorthand for apply_policy()."""
        return self.apply_policy(image)

    def apply_to_batch(
        self,
        images: List[Union[Image.Image, torch.Tensor]]
    ) -> List[Union[Image.Image, torch.Tensor]]:
        """
        Apply policy to a batch of images.

        Args:
            images: List of PIL Images or torch Tensors

        Returns:
            List of augmented images
        """
        return [self.apply_policy(img) for img in images]

    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the policy.

        Returns:
            Dict with policy statistics
        """
        return {
            "num_operations": len(self.policy),
            "operations": [sp["operation"] for sp in self.policy],
            "avg_probability": np.mean([sp["probability"] for sp in self.policy]),
            "operations_detail": [
                {
                    "operation": sp["operation"],
                    "probability": sp["probability"],
                    "magnitude": sp["magnitude"],
                    "magnitude_range": SAFE_OPERATIONS[sp["operation"]].magnitude_range
                }
                for sp in self.policy
            ]
        }

    def __repr__(self) -> str:
        return f"PolicyAugmentor(num_ops={len(self.policy)}, seed={self.seed})"


def create_random_policy(
    num_operations: int = 3,
    operation_pool: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create a random augmentation policy.

    Useful for initializing population-based search or testing.

    Args:
        num_operations: Number of operations in policy (default 3)
        operation_pool: List of operation names to sample from (default: all safe ops)
        seed: Random seed for reproducibility

    Returns:
        Random policy (list of sub-policies)

    Example:
        >>> policy = create_random_policy(num_operations=3, seed=42)
        >>> augmentor = PolicyAugmentor(policy)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if operation_pool is None:
        operation_pool = list_safe_operations()

    # Randomly sample operations
    selected_ops = random.sample(operation_pool, min(num_operations, len(operation_pool)))

    policy = []
    for op_name in selected_ops:
        op = get_operation(op_name)

        # Random probability (favor higher probabilities)
        probability = np.random.beta(2, 2)  # Beta distribution centered at 0.5

        # Random magnitude within operation's range
        mag_min, mag_max = op.magnitude_range
        magnitude = np.random.uniform(mag_min, mag_max)

        policy.append({
            "operation": op_name,
            "probability": float(probability),
            "magnitude": float(magnitude)
        })

    return policy


def mutate_policy(
    policy: List[Dict[str, Any]],
    mutation_rate: float = 0.2,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mutate an augmentation policy for evolutionary search.

    Args:
        policy: Original policy
        mutation_rate: Probability of mutating each component (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        Mutated policy

    Example:
        >>> mutated = mutate_policy(original_policy, mutation_rate=0.3)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    mutated = []

    for sub_policy in policy:
        new_sub = sub_policy.copy()

        # Mutate probability
        if random.random() < mutation_rate:
            new_sub["probability"] = np.clip(
                new_sub["probability"] + np.random.normal(0, 0.1),
                0.0, 1.0
            )

        # Mutate magnitude
        if random.random() < mutation_rate:
            op = get_operation(sub_policy["operation"])
            mag_min, mag_max = op.magnitude_range
            mag_range = mag_max - mag_min

            new_sub["magnitude"] = np.clip(
                new_sub["magnitude"] + np.random.normal(0, mag_range * 0.1),
                mag_min, mag_max
            )

        # Mutate operation (replace with different operation)
        if random.random() < mutation_rate * 0.5:  # Lower chance
            all_ops = list_safe_operations()
            new_sub["operation"] = random.choice(all_ops)

            # Reset magnitude to middle of new operation's range
            new_op = get_operation(new_sub["operation"])
            mag_min, mag_max = new_op.magnitude_range
            new_sub["magnitude"] = (mag_min + mag_max) / 2.0

        mutated.append(new_sub)

    return mutated


def crossover_policies(
    policy1: List[Dict[str, Any]],
    policy2: List[Dict[str, Any]],
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform crossover between two policies for evolutionary search.

    Args:
        policy1: First parent policy
        policy2: Second parent policy
        seed: Random seed for reproducibility

    Returns:
        Tuple of (child1, child2) policies

    Example:
        >>> child1, child2 = crossover_policies(parent1, parent2)
    """
    if seed is not None:
        random.seed(seed)

    # Single-point crossover
    min_len = min(len(policy1), len(policy2))
    if min_len <= 1:
        # Can't crossover, just return copies
        return policy1.copy(), policy2.copy()

    crossover_point = random.randint(1, min_len - 1)

    child1 = policy1[:crossover_point] + policy2[crossover_point:]
    child2 = policy2[:crossover_point] + policy1[crossover_point:]

    return child1, child2


# Example usage
if __name__ == '__main__':
    """Demonstrate PolicyAugmentor usage."""
    from PIL import Image

    print("=" * 80)
    print("PolicyAugmentor - Augmentation Policy Application")
    print("=" * 80)

    # Create a sample policy
    policy = [
        {"operation": "rotate", "probability": 0.7, "magnitude": 10.0},
        {"operation": "brightness", "probability": 0.5, "magnitude": 0.1},
        {"operation": "gaussian_blur", "probability": 0.3, "magnitude": 2.0}
    ]

    print("\nPolicy Definition:")
    for i, sp in enumerate(policy):
        print(f"  {i+1}. {sp['operation']:20s} - prob={sp['probability']:.2f}, mag={sp['magnitude']:.2f}")

    # Create augmentor
    augmentor = PolicyAugmentor(policy, seed=42)
    print(f"\n✓ Created: {augmentor}")

    # Get policy summary
    summary = augmentor.get_policy_summary()
    print(f"\nPolicy Summary:")
    print(f"  • Num operations: {summary['num_operations']}")
    print(f"  • Avg probability: {summary['avg_probability']:.3f}")

    # Test with dummy image
    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    augmented = augmentor(image)
    print(f"\n✓ Applied policy to image: {image.size} -> {augmented.size}")

    # Test random policy generation
    print("\n" + "=" * 80)
    print("Random Policy Generation")
    print("=" * 80)

    random_policy = create_random_policy(num_operations=4, seed=123)
    print("\nGenerated Random Policy:")
    for i, sp in enumerate(random_policy):
        print(f"  {i+1}. {sp['operation']:20s} - prob={sp['probability']:.2f}, mag={sp['magnitude']:.2f}")

    # Test mutation
    print("\n" + "=" * 80)
    print("Policy Mutation")
    print("=" * 80)

    mutated_policy = mutate_policy(policy, mutation_rate=0.5, seed=456)
    print("\nMutated Policy:")
    for i, sp in enumerate(mutated_policy):
        print(f"  {i+1}. {sp['operation']:20s} - prob={sp['probability']:.2f}, mag={sp['magnitude']:.2f}")

    # Test crossover
    print("\n" + "=" * 80)
    print("Policy Crossover")
    print("=" * 80)

    policy1 = create_random_policy(num_operations=3, seed=111)
    policy2 = create_random_policy(num_operations=3, seed=222)
    child1, child2 = crossover_policies(policy1, policy2, seed=333)

    print("\nParent 1:")
    for sp in policy1:
        print(f"  • {sp['operation']}")

    print("\nParent 2:")
    for sp in policy2:
        print(f"  • {sp['operation']}")

    print("\nChild 1:")
    for sp in child1:
        print(f"  • {sp['operation']}")

    print("\nChild 2:")
    for sp in child2:
        print(f"  • {sp['operation']}")

    # Test forbidden operation detection
    print("\n" + "=" * 80)
    print("Forbidden Operation Detection")
    print("=" * 80)

    try:
        forbidden_policy = [
            {"operation": "cutout", "probability": 0.5, "magnitude": 0.2}
        ]
        augmentor = PolicyAugmentor(forbidden_policy)
    except InvalidPolicyError as e:
        print(f"✓ Correctly rejected forbidden operation:\n  {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
