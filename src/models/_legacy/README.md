# Legacy Models (v1)

## Why Capped

The legacy classifier implementation has been moved here because:

1. **Architecture Difference**: v1 uses `global_pool=""` (manual pooling), while the production model uses `global_pool='avg'` (integrated pooling). This causes state_dict incompatibility when loading production weights.

2. **Hospital-Based Splitting**: The production model (v2) was trained with hospital-based data splitting to prevent data leakage, achieving realistic 93.7% AUC instead of inflated metrics from random splitting.

3. **Cleaner Interface**: The new `GlaucomaPredictor` class provides a simpler, production-ready API.

## Files

| File | Description |
|------|-------------|
| `efficientnet_classifier_v1.py` | Original timm-based classifier with manual pooling |
| `__init__.py` | Module with deprecation warnings |

## How to Revert

If you need to use v1 for any reason:

```python
# Option 1: Direct import (will show deprecation warning)
from src.models._legacy.efficientnet_classifier_v1 import EfficientNetClassifier, create_classifier

# Option 2: Silence warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from src.models._legacy import EfficientNetClassifier
```

## Production Usage

For all new code, use the production inference pipeline:

```python
from src.inference import GlaucomaPredictor

# Load model
predictor = GlaucomaPredictor.from_checkpoint(
    "models/production/glaucoma_efficientnet_b0_v1.pt"
)

# Predict
result = predictor.predict("path/to/fundus_image.png")
print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
```

## Migration Date

November 2024
