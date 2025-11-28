"""
Domain Label Extraction for Multi-Source Fundus Datasets

This module extracts domain labels from fundus image metadata based on
filename patterns, folder structure, and explicit metadata fields.

The domain classifier learns WHERE an image came from (acquisition device/hospital),
NOT what disease it has. This enables routing to the appropriate expert head.

Supported Domains:
- rimone: RIM-ONE dataset (r1, r2, r3 hospitals)
- refuge: REFUGE/REFUGE2 dataset
- g1020: G1020 dataset
- unknown: Cannot determine source

Usage:
    >>> from src.data.domain_labels import extract_domain, DOMAIN_CLASSES
    >>> domain = extract_domain(sample_metadata)
    >>> print(f"Domain: {domain}")  # 'rimone'
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json


# Domain class definitions
DOMAIN_CLASSES = ['rimone', 'refuge', 'g1020', 'unknown']
DOMAIN_TO_IDX = {domain: idx for idx, domain in enumerate(DOMAIN_CLASSES)}
IDX_TO_DOMAIN = {idx: domain for idx, domain in enumerate(DOMAIN_CLASSES)}
NUM_DOMAINS = len(DOMAIN_CLASSES) - 1  # Exclude 'unknown' from classification


# Filename patterns for domain detection
DOMAIN_PATTERNS = {
    'rimone': [
        r'r[123]_Im\d+',           # r1_Im001, r2_Im347, etc.
        r'rimone_\d+',             # rimone_0000, rimone_0001
        r'rim.?one',               # Any reference to RIM-ONE
    ],
    'refuge': [
        r'refuge',                 # REFUGE dataset
        r'REFUGE',
        r'T\d{4}',                 # T0001, T0002 (REFUGE test format)
        r'V\d{4}',                 # V0001 (REFUGE validation)
        r'g\d{4}\.jpg',            # g0001.jpg (REFUGE glaucoma)
        r'n\d{4}\.jpg',            # n0001.jpg (REFUGE normal)
    ],
    'g1020': [
        r'g1020',                  # G1020 dataset
        r'G1020',
        r'BinRushed',              # G1020 source hospital
        r'Magrabi',                # G1020 source hospital
    ],
}


def extract_domain_from_filename(filename: str) -> Optional[str]:
    """
    Extract domain from filename using pattern matching.

    Args:
        filename: Image filename (e.g., 'r1_Im001.png', 'T0001.jpg')

    Returns:
        Domain string ('rimone', 'refuge', 'g1020') or None if not detected
    """
    filename_lower = filename.lower()

    for domain, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return domain

    return None


def extract_domain_from_path(filepath: str) -> Optional[str]:
    """
    Extract domain from full file path.

    Checks both the filename and directory structure.

    Args:
        filepath: Full or partial path to image

    Returns:
        Domain string or None if not detected
    """
    path_str = str(filepath).lower()

    # Check directory names
    if 'rim_one' in path_str or 'rimone' in path_str or 'rim-one' in path_str:
        return 'rimone'
    if 'refuge' in path_str:
        return 'refuge'
    if 'g1020' in path_str:
        return 'g1020'

    # Fall back to filename extraction
    filename = Path(filepath).name
    return extract_domain_from_filename(filename)


def extract_domain_from_metadata(sample: Dict[str, Any]) -> str:
    """
    Extract domain label from a sample's metadata.

    Checks multiple fields in order of preference:
    1. 'domain' - explicit domain field
    2. 'dataset' - dataset identifier
    3. 'source_dataset' - source dataset field
    4. 'original_path' - extract from path
    5. 'image_filename' - extract from filename

    Args:
        sample: Sample metadata dictionary

    Returns:
        Domain string ('rimone', 'refuge', 'g1020', 'unknown')
    """
    # Check explicit domain field
    if 'domain' in sample and sample['domain']:
        domain = sample['domain'].lower()
        if domain in DOMAIN_CLASSES:
            return domain

    # Check dataset field
    for field in ['dataset', 'source_dataset', 'dataset_name']:
        if field in sample and sample[field]:
            value = str(sample[field]).lower()
            if 'rim' in value:
                return 'rimone'
            if 'refuge' in value:
                return 'refuge'
            if 'g1020' in value:
                return 'g1020'

    # Extract from original path
    if 'original_path' in sample and sample['original_path']:
        domain = extract_domain_from_path(sample['original_path'])
        if domain:
            return domain

    # Extract from image filename
    for field in ['image_filename', 'filename', 'image']:
        if field in sample and sample[field]:
            domain = extract_domain_from_filename(str(sample[field]))
            if domain:
                return domain

    return 'unknown'


def extract_domain(sample: Dict[str, Any]) -> str:
    """
    Main function to extract domain from any sample format.

    Convenience wrapper around extract_domain_from_metadata.

    Args:
        sample: Sample metadata dictionary

    Returns:
        Domain string
    """
    return extract_domain_from_metadata(sample)


def generate_domain_labels(
    samples: List[Dict[str, Any]],
    exclude_unknown: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Generate domain labels for a list of samples.

    Args:
        samples: List of sample metadata dictionaries
        exclude_unknown: If True, filter out samples with unknown domain

    Returns:
        Tuple of (indices, labels) where:
        - indices: List of sample indices with valid domain
        - labels: Corresponding domain labels (integers)
    """
    indices = []
    labels = []

    for idx, sample in enumerate(samples):
        domain = extract_domain(sample)

        if exclude_unknown and domain == 'unknown':
            continue

        indices.append(idx)
        labels.append(DOMAIN_TO_IDX[domain])

    return indices, labels


def get_domain_distribution(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get the distribution of domains in a dataset.

    Args:
        samples: List of sample metadata dictionaries

    Returns:
        Dictionary mapping domain names to counts
    """
    distribution = {domain: 0 for domain in DOMAIN_CLASSES}

    for sample in samples:
        domain = extract_domain(sample)
        distribution[domain] += 1

    return distribution


def create_domain_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    stratified: bool = True,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Create train/val splits for domain classification.

    Args:
        samples: List of sample metadata dictionaries
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        stratified: Whether to maintain domain balance
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train' and 'val' keys mapping to sample indices
    """
    import numpy as np
    np.random.seed(seed)

    indices, labels = generate_domain_labels(samples, exclude_unknown=True)
    indices = np.array(indices)
    labels = np.array(labels)

    if not stratified:
        # Simple random split
        np.random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        return {
            'train': indices[:n_train].tolist(),
            'val': indices[n_train:].tolist()
        }

    # Stratified split by domain
    train_indices = []
    val_indices = []

    for domain_idx in range(NUM_DOMAINS):
        domain_mask = labels == domain_idx
        domain_indices = indices[domain_mask]
        np.random.shuffle(domain_indices)

        n_train = int(len(domain_indices) * train_ratio)
        train_indices.extend(domain_indices[:n_train].tolist())
        val_indices.extend(domain_indices[n_train:].tolist())

    # Final shuffle
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return {
        'train': train_indices,
        'val': val_indices
    }


def save_domain_labels(
    samples: List[Dict[str, Any]],
    output_path: str,
    include_stats: bool = True
) -> None:
    """
    Save domain labels to a JSON file.

    Args:
        samples: List of sample metadata dictionaries
        output_path: Path to save JSON file
        include_stats: Whether to include distribution statistics
    """
    indices, labels = generate_domain_labels(samples, exclude_unknown=False)

    output = {
        'indices': indices,
        'labels': labels,
        'domain_names': DOMAIN_CLASSES,
        'domain_to_idx': DOMAIN_TO_IDX,
    }

    if include_stats:
        output['distribution'] = get_domain_distribution(samples)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_domain_labels(input_path: str) -> Dict[str, Any]:
    """
    Load domain labels from a JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary with indices, labels, and metadata
    """
    with open(input_path, 'r') as f:
        return json.load(f)
