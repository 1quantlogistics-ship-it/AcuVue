"""
Data Samplers for Handling Class Imbalance
==========================================
"""

import torch
from torch.utils.data import WeightedRandomSampler, Dataset
from typing import Optional, List
import numpy as np


def get_sampler(
    dataset: Dataset,
    labels: Optional[List[int]] = None,
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling class imbalance.
    
    Args:
        dataset: PyTorch dataset
        labels: Optional list of labels (if not provided, tries to get from dataset)
        replacement: Whether to sample with replacement
        
    Returns:
        WeightedRandomSampler instance
    """
    if labels is None:
        # Try to get labels from dataset
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        elif hasattr(dataset, "targets"):
            labels = dataset.targets
        else:
            # Extract labels by iterating (slow but works)
            labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Convert to numpy array
    labels = np.array(labels)
    
    # Calculate class weights
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts
    
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).float()
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement
    )
