"""
Unit tests for hospital-based data splitting.

Tests the HospitalBasedSplitter and institution_utils modules to ensure
proper institution extraction and data leakage prevention.
"""

import pytest
import numpy as np
from typing import List, Dict

from src.data.hospital_splitter import (
    HospitalBasedSplitter,
    create_hospital_based_splits,
)
from src.data.institution_utils import (
    extract_institution_from_filename,
    extract_institution_from_path,
    get_institution_from_metadata,
    group_samples_by_institution,
    get_institution_statistics,
)


@pytest.fixture
def sample_rimone_metadata() -> List[Dict]:
    """Create sample metadata resembling RIMONE dataset."""
    samples = []
    sample_id = 0

    # r1 hospital (test set in production)
    for i in range(20):
        samples.append({
            'sample_id': sample_id,
            'image_filename': f'rimone_{sample_id:04d}.png',
            'original_path': f'partitioned_by_hospital/test_set/{"glaucoma" if i % 2 == 0 else "normal"}/r1_Im{i:03d}.png',
            'source_hospital': 'r1',
            'label': i % 2,
            'label_name': 'glaucoma' if i % 2 == 0 else 'normal',
        })
        sample_id += 1

    # r2 hospital (train set)
    for i in range(40):
        samples.append({
            'sample_id': sample_id,
            'image_filename': f'rimone_{sample_id:04d}.png',
            'original_path': f'partitioned_by_hospital/training_set/{"glaucoma" if i % 3 == 0 else "normal"}/r2_Im{i:03d}.png',
            'source_hospital': 'r2',
            'label': 1 if i % 3 == 0 else 0,
            'label_name': 'glaucoma' if i % 3 == 0 else 'normal',
        })
        sample_id += 1

    # r3 hospital (train set)
    for i in range(30):
        samples.append({
            'sample_id': sample_id,
            'image_filename': f'rimone_{sample_id:04d}.png',
            'original_path': f'partitioned_by_hospital/training_set/{"glaucoma" if i % 4 == 0 else "normal"}/r3_Im{i:03d}.png',
            'source_hospital': 'r3',
            'label': 1 if i % 4 == 0 else 0,
            'label_name': 'glaucoma' if i % 4 == 0 else 'normal',
        })
        sample_id += 1

    return samples


class TestInstitutionUtils:
    """Tests for institution extraction utilities."""

    def test_extract_from_filename_rimone(self):
        """Test extracting hospital code from RIMONE filename."""
        assert extract_institution_from_filename('r1_Im347.png') == 'r1'
        assert extract_institution_from_filename('r2_Im001.png') == 'r2'
        assert extract_institution_from_filename('r3_Im999.png') == 'r3'

    def test_extract_from_filename_no_match(self):
        """Test returns None for non-matching filenames."""
        assert extract_institution_from_filename('image_001.png') is None

    def test_extract_from_path_rimone(self):
        """Test extracting hospital from full path."""
        path = 'partitioned_by_hospital/training_set/glaucoma/r2_Im347.png'
        assert extract_institution_from_path(path) == 'r2'

    def test_get_institution_from_metadata_explicit(self):
        """Test getting institution from explicit field."""
        sample = {'source_hospital': 'r1', 'original_path': 'r2_Im001.png'}
        assert get_institution_from_metadata(sample) == 'r1'

    def test_group_samples_by_institution(self, sample_rimone_metadata):
        """Test grouping samples by institution."""
        groups = group_samples_by_institution(sample_rimone_metadata)

        assert 'r1' in groups
        assert 'r2' in groups
        assert 'r3' in groups
        assert len(groups['r1']) == 20
        assert len(groups['r2']) == 40
        assert len(groups['r3']) == 30


class TestHospitalBasedSplitter:
    """Tests for HospitalBasedSplitter class."""

    def test_split_by_institution_basic(self, sample_rimone_metadata):
        """Test basic institution-based splitting."""
        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=sample_rimone_metadata,
            test_institutions=['r1'],
            train_val_institutions=['r2', 'r3'],
            val_ratio=0.1
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        assert len(splits['test']) == 20
        assert len(splits['train']) + len(splits['val']) == 70

    def test_validate_no_leakage_success(self, sample_rimone_metadata):
        """Test leakage validation passes for proper splits."""
        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=sample_rimone_metadata,
            test_institutions=['r1'],
            train_val_institutions=['r2', 'r3']
        )

        assert splitter.validate_no_leakage(splits, sample_rimone_metadata) is True

    def test_validate_no_leakage_index_overlap(self):
        """Test leakage detection for index overlap."""
        splitter = HospitalBasedSplitter()
        bad_splits = {
            'train': [0, 1, 2, 3, 4],
            'val': [5, 6],
            'test': [3, 7, 8]  # Index 3 overlaps
        }
        assert splitter.validate_no_leakage(bad_splits) is False

    def test_get_split_statistics(self, sample_rimone_metadata):
        """Test split statistics computation."""
        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=sample_rimone_metadata,
            test_institutions=['r1']
        )

        stats = splitter.get_split_statistics(splits, sample_rimone_metadata)

        assert 'splits' in stats
        assert 'r1' in stats['splits']['test']['institutions']
        assert 'r1' not in stats['splits']['train']['institutions']


class TestCreateHospitalBasedSplits:
    """Tests for the convenience function."""

    def test_create_splits_basic(self, sample_rimone_metadata):
        """Test basic usage of create_hospital_based_splits."""
        splits = create_hospital_based_splits(
            metadata=sample_rimone_metadata,
            test_institutions=['r1'],
            seed=42
        )

        assert len(splits['test']) == 20
        assert len(splits['train']) + len(splits['val']) == 70
