"""
Institution/hospital extraction utilities for medical imaging datasets.

Provides functions to extract institution identifiers from filenames and metadata,
primarily designed for the RIMONE dataset where hospital codes (r1, r2, r3) are
embedded in the original file paths.
"""
import re
from pathlib import Path
from typing import Optional, Dict, List, Any


# Known institution patterns for different datasets
INSTITUTION_PATTERNS = {
    'rimone': r'r(\d+)',  # Matches r1, r2, r3 in RIMONE filenames
    'refuge': r'refuge(\d*)',  # REFUGE dataset identifier
}


def extract_institution_from_filename(
    filename: str,
    dataset_type: str = 'rimone'
) -> Optional[str]:
    """
    Extract institution code from a filename.

    Args:
        filename: Filename to parse (e.g., 'r2_Im347.png')
        dataset_type: Type of dataset for pattern matching

    Returns:
        Institution code (e.g., 'r2') or None if not found
    """
    pattern = INSTITUTION_PATTERNS.get(dataset_type, INSTITUTION_PATTERNS['rimone'])

    # For RIMONE, look for r1_, r2_, r3_ pattern
    if dataset_type == 'rimone':
        match = re.search(r'(r\d+)_', filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Generic pattern matching
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(0).lower()

    return None


def extract_institution_from_path(
    filepath: str,
    dataset_type: str = 'rimone'
) -> Optional[str]:
    """
    Extract institution code from a file path.

    Searches the full path for institution identifiers, useful when
    the directory structure contains institution information.

    Args:
        filepath: Full or partial file path
        dataset_type: Type of dataset for pattern matching

    Returns:
        Institution code or None if not found
    """
    path_str = str(filepath)

    # For RIMONE, the original path format is:
    # partitioned_by_hospital/training_set/glaucoma/r2_Im347.png
    match = re.search(r'(r\d+)_Im\d+\.png', path_str, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback to filename extraction
    filename = Path(filepath).name
    return extract_institution_from_filename(filename, dataset_type)


def get_institution_from_metadata(
    sample: Dict[str, Any],
    fallback_to_path: bool = True
) -> Optional[str]:
    """
    Get institution from a sample's metadata.

    Checks multiple metadata fields in order of preference:
    1. 'source_hospital' - explicit hospital code
    2. 'institution' - explicit institution code
    3. 'original_path' - extract from original filename (if fallback enabled)

    Args:
        sample: Sample metadata dictionary
        fallback_to_path: Whether to extract from original_path if not explicit

    Returns:
        Institution code or None if not found
    """
    # Check explicit fields
    if 'source_hospital' in sample and sample['source_hospital']:
        return sample['source_hospital'].lower()

    if 'institution' in sample and sample['institution']:
        return sample['institution'].lower()

    # Fallback to extraction from path
    if fallback_to_path and 'original_path' in sample:
        return extract_institution_from_path(sample['original_path'])

    return None


def group_samples_by_institution(
    samples: List[Dict[str, Any]],
    fallback_to_path: bool = True
) -> Dict[str, List[int]]:
    """
    Group sample indices by their institution.

    Args:
        samples: List of sample metadata dictionaries
        fallback_to_path: Whether to extract from path if not explicit

    Returns:
        Dictionary mapping institution codes to lists of sample indices
    """
    institution_groups: Dict[str, List[int]] = {}

    for idx, sample in enumerate(samples):
        institution = get_institution_from_metadata(sample, fallback_to_path)

        if institution is None:
            institution = 'unknown'

        if institution not in institution_groups:
            institution_groups[institution] = []

        institution_groups[institution].append(idx)

    return institution_groups


def validate_institution_coverage(
    samples: List[Dict[str, Any]],
    required_institutions: List[str]
) -> Dict[str, bool]:
    """
    Validate that all required institutions are present in the dataset.

    Args:
        samples: List of sample metadata dictionaries
        required_institutions: List of institution codes that must be present

    Returns:
        Dictionary mapping each required institution to whether it's present
    """
    groups = group_samples_by_institution(samples)
    present_institutions = set(groups.keys())

    return {
        inst: inst.lower() in present_institutions
        for inst in required_institutions
    }


def get_institution_statistics(
    samples: List[Dict[str, Any]],
    label_key: str = 'label'
) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for each institution.

    Args:
        samples: List of sample metadata dictionaries
        label_key: Key for accessing class labels

    Returns:
        Dictionary with institution statistics including:
        - count: number of samples
        - label_distribution: count per label
    """
    groups = group_samples_by_institution(samples)
    stats: Dict[str, Dict[str, Any]] = {}

    for institution, indices in groups.items():
        institution_samples = [samples[i] for i in indices]

        # Count labels
        label_counts: Dict[Any, int] = {}
        for sample in institution_samples:
            label = sample.get(label_key, 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

        stats[institution] = {
            'count': len(indices),
            'label_distribution': label_counts,
            'indices': indices
        }

    return stats
