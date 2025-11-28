"""
Integration tests for the production model pipeline.

These tests verify end-to-end functionality with actual model weights
and real dataset samples when available.
"""

import pytest
import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np

from src.inference.predictor import GlaucomaPredictor, GlaucomaClassifier, PredictionResult
from src.inference.preprocessing import preprocess_image, load_and_preprocess
from src.inference.config import InferenceConfig, PRODUCTION_V1_METADATA
from src.data.hospital_splitter import HospitalBasedSplitter, create_hospital_based_splits


# ============================================================================
# Path Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "glaucoma_efficientnet_b0_v1.pt"
RIMONE_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "rim_one" / "metadata.json"
RIMONE_IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "rim_one" / "images"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def production_model_exists() -> bool:
    """Check if production model weights exist."""
    return PRODUCTION_MODEL_PATH.exists()


@pytest.fixture
def rimone_data_exists() -> bool:
    """Check if RIMONE dataset exists."""
    return RIMONE_METADATA_PATH.exists() and RIMONE_IMAGES_PATH.exists()


@pytest.fixture
def rimone_metadata():
    """Load RIMONE metadata if available."""
    if not RIMONE_METADATA_PATH.exists():
        pytest.skip("RIMONE metadata not found")
    with open(RIMONE_METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture
def sample_fundus_image():
    """Create a synthetic fundus-like image for testing."""
    # Create a 512x512 RGB image with fundus-like appearance
    arr = np.zeros((512, 512, 3), dtype=np.uint8)

    # Dark reddish-orange background (typical fundus color)
    arr[:, :, 0] = 180  # Red
    arr[:, :, 1] = 80   # Green
    arr[:, :, 2] = 50   # Blue

    # Add optic disc region (lighter circular area)
    y, x = np.ogrid[:512, :512]
    center_x, center_y = 350, 256
    r = 60
    disc_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2
    arr[disc_mask, 0] = 255
    arr[disc_mask, 1] = 220
    arr[disc_mask, 2] = 180

    # Add some vessel-like darker regions
    for i in range(5):
        angle = np.pi * i / 5
        for t in range(200):
            px = int(center_x + t * np.cos(angle))
            py = int(center_y + t * np.sin(angle))
            if 0 <= px < 512 and 0 <= py < 512:
                # Darken for vessels
                arr[py, max(0, px - 1):min(512, px + 2), :] = (
                    arr[py, max(0, px - 1):min(512, px + 2), :] * 0.7
                ).astype(np.uint8)

    return Image.fromarray(arr, mode='RGB')


# ============================================================================
# Production Model Integration Tests
# ============================================================================

class TestProductionModelIntegration:
    """Integration tests for production model inference."""

    @pytest.mark.skipif(
        not PRODUCTION_MODEL_PATH.exists(),
        reason="Production model not found"
    )
    def test_load_production_model(self):
        """Test loading the production model weights."""
        predictor = GlaucomaPredictor.from_checkpoint(
            str(PRODUCTION_MODEL_PATH),
            device='cpu'
        )

        assert isinstance(predictor, GlaucomaPredictor)
        assert predictor.class_names == ['normal', 'glaucoma']

    @pytest.mark.skipif(
        not PRODUCTION_MODEL_PATH.exists(),
        reason="Production model not found"
    )
    def test_production_model_inference(self, sample_fundus_image):
        """Test running inference with production model."""
        predictor = GlaucomaPredictor.from_checkpoint(
            str(PRODUCTION_MODEL_PATH),
            device='cpu'
        )

        result = predictor.predict(sample_fundus_image)

        assert isinstance(result, PredictionResult)
        assert result.prediction in ['normal', 'glaucoma']
        assert 0 <= result.confidence <= 1
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5

    @pytest.mark.skipif(
        not PRODUCTION_MODEL_PATH.exists(),
        reason="Production model not found"
    )
    def test_production_model_batch_inference(self, sample_fundus_image):
        """Test batch inference with production model."""
        predictor = GlaucomaPredictor.from_checkpoint(
            str(PRODUCTION_MODEL_PATH),
            device='cpu'
        )

        # Create batch of images
        images = [sample_fundus_image] * 5
        results = predictor.predict_batch(images)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, PredictionResult)

    @pytest.mark.skipif(
        not PRODUCTION_MODEL_PATH.exists() or not RIMONE_IMAGES_PATH.exists(),
        reason="Production model or RIMONE data not found"
    )
    def test_production_model_on_real_data(self, rimone_metadata):
        """Test production model on actual RIMONE images."""
        predictor = GlaucomaPredictor.from_checkpoint(
            str(PRODUCTION_MODEL_PATH),
            device='cpu'
        )

        # Get a few sample images
        samples = rimone_metadata.get('samples', [])[:5]

        for sample in samples:
            image_path = RIMONE_IMAGES_PATH / sample['image_filename']
            if image_path.exists():
                result = predictor.predict(str(image_path))

                assert isinstance(result, PredictionResult)
                assert result.prediction in ['normal', 'glaucoma']


class TestModelMetadata:
    """Tests for model metadata and configuration."""

    def test_production_v1_metadata(self):
        """Test production v1 metadata values."""
        assert PRODUCTION_V1_METADATA.version == "v1"
        assert PRODUCTION_V1_METADATA.architecture == "efficientnet_b0"
        assert PRODUCTION_V1_METADATA.library == "timm"
        assert PRODUCTION_V1_METADATA.input_size == 224
        assert PRODUCTION_V1_METADATA.num_classes == 2

    def test_production_v1_metrics(self):
        """Test that expected metrics are present."""
        metrics = PRODUCTION_V1_METADATA.metrics

        assert 'test_auc' in metrics
        assert metrics['test_auc'] > 0.9  # Should be ~93.7%
        assert 'test_accuracy' in metrics
        assert 'test_sensitivity' in metrics
        assert 'test_specificity' in metrics

    def test_production_v1_training_config(self):
        """Test training configuration metadata."""
        config = PRODUCTION_V1_METADATA.training_config

        assert config['splitting_strategy'] == 'hospital_based'
        assert config['test_hospitals'] == ['r1']
        assert 'r2' in config['train_val_hospitals']
        assert 'r3' in config['train_val_hospitals']


# ============================================================================
# Hospital-Based Splitting Integration Tests
# ============================================================================

class TestHospitalSplittingIntegration:
    """Integration tests for hospital-based data splitting."""

    @pytest.mark.skipif(
        not RIMONE_METADATA_PATH.exists(),
        reason="RIMONE metadata not found"
    )
    def test_split_actual_rimone_data(self, rimone_metadata):
        """Test splitting actual RIMONE dataset."""
        samples = rimone_metadata.get('samples', [])

        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=samples,
            test_institutions=['r1'],
            train_val_institutions=['r2', 'r3']
        )

        # Verify no leakage
        assert splitter.validate_no_leakage(splits, samples)

        # Verify split sizes
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        # May not equal len(samples) if some samples have unknown institutions

        # Test set should contain only r1
        for idx in splits['test']:
            assert samples[idx].get('source_hospital', '').lower() == 'r1'

        # Train/val should not contain r1
        for idx in splits['train'] + splits['val']:
            assert samples[idx].get('source_hospital', '').lower() != 'r1'

    @pytest.mark.skipif(
        not RIMONE_METADATA_PATH.exists(),
        reason="RIMONE metadata not found"
    )
    def test_split_statistics_match_expected(self, rimone_metadata):
        """Test that split statistics match expected distribution."""
        samples = rimone_metadata.get('samples', [])

        splitter = HospitalBasedSplitter(seed=42)
        splits = splitter.split_by_institution(
            metadata=samples,
            test_institutions=['r1']
        )

        stats = splitter.get_split_statistics(splits, samples)

        # Test set should have r1 only
        test_institutions = stats['splits']['test']['institutions']
        assert test_institutions == ['r1']

        # Train should not have r1
        train_institutions = stats['splits']['train']['institutions']
        assert 'r1' not in train_institutions


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:
    """End-to-end tests for the complete inference pipeline."""

    def test_preprocessing_to_inference_pipeline(self, sample_fundus_image):
        """Test the full preprocessing -> inference pipeline."""
        # Create a model (with random weights for this test)
        model = GlaucomaClassifier(num_classes=2, dropout=0.3)
        model.eval()

        # Preprocess image
        tensor = preprocess_image(sample_fundus_image, input_size=224, device='cpu')

        assert tensor.shape == (1, 3, 224, 224)

        # Run through model
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)

        assert logits.shape == (1, 2)
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_hospital_splitting_to_evaluation_pipeline(self):
        """Test hospital splitting to evaluation setup."""
        # Create synthetic metadata
        samples = []
        for i in range(30):
            hospital = ['r1', 'r2', 'r3'][i % 3]
            samples.append({
                'sample_id': i,
                'source_hospital': hospital,
                'label': i % 2,
                'image_filename': f'test_{i:03d}.png'
            })

        # Create hospital-based splits
        splits = create_hospital_based_splits(
            metadata=samples,
            test_institutions=['r1'],
            seed=42
        )

        # Verify we can use splits for evaluation
        test_samples = [samples[i] for i in splits['test']]
        train_samples = [samples[i] for i in splits['train']]

        # All test samples should be from r1
        for sample in test_samples:
            assert sample['source_hospital'] == 'r1'

        # No train samples should be from r1
        for sample in train_samples:
            assert sample['source_hospital'] != 'r1'

    @pytest.mark.skipif(
        not PRODUCTION_MODEL_PATH.exists(),
        reason="Production model not found"
    )
    def test_full_production_pipeline(self, sample_fundus_image):
        """Test the complete production inference pipeline."""
        # 1. Load production model
        predictor = GlaucomaPredictor.from_checkpoint(
            str(PRODUCTION_MODEL_PATH),
            device='cpu'
        )

        # 2. Run inference
        result = predictor.predict(sample_fundus_image)

        # 3. Verify result structure
        assert isinstance(result, PredictionResult)
        assert result.prediction in ['normal', 'glaucoma']
        assert 0 <= result.confidence <= 1

        # 4. Verify probabilities
        assert 'normal' in result.probabilities
        assert 'glaucoma' in result.probabilities
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5

        # 5. Verify dictionary export works
        result_dict = result.to_dict()
        assert 'prediction' in result_dict
        assert 'confidence' in result_dict
        assert 'probabilities' in result_dict


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_inference_batch_size_scaling(self, sample_fundus_image):
        """Test that batch inference scales reasonably."""
        import time

        # Create model with random weights
        model = GlaucomaClassifier()
        model.eval()

        # Preprocess images
        tensor = preprocess_image(sample_fundus_image)

        # Time single inference
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(tensor)
        single_time = time.time() - start

        # Time batch inference
        batch_tensor = tensor.repeat(10, 1, 1, 1)
        start = time.time()
        with torch.no_grad():
            _ = model(batch_tensor)
        batch_time = time.time() - start

        # Batch should be more efficient (or at least not much slower)
        assert batch_time < single_time * 1.5
