# AcuVue Inference Pipeline v2

Production-ready glaucoma detection from fundus images.

## Quick Start

```python
from src.inference import GlaucomaPredictor

# Load production model
predictor = GlaucomaPredictor.from_checkpoint(
    "models/production/glaucoma_efficientnet_b0_v1.pt"
)

# Single image prediction
result = predictor.predict("path/to/fundus.png")
print(f"{result.prediction}: {result.confidence:.1%}")

# Batch prediction
results = predictor.predict_batch(["img1.png", "img2.png", "img3.png"])
```

## Model Architecture

| Component | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (timm) |
| Input Size | 224x224 RGB |
| Normalization | ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| Output | 2 classes: ['normal', 'glaucoma'] |
| Dropout | 0.3 |

## Performance Metrics

Evaluated using **hospital-based splitting** (train on r2/r3, test on r1):

| Metric | Value |
|--------|-------|
| AUC | 93.7% |
| Accuracy | 76.5% |
| Sensitivity | 74.4% |
| Specificity | 91.7% |

> **Note**: Standard random splits show ~97% AUC due to data leakage. The 93.7% figure represents realistic cross-institution performance.

## Files

```
models/production/
├── glaucoma_efficientnet_b0_v1.pt     # Model weights (47MB)
├── training_history_v1.json            # Training metrics
└── dataset_metadata_v1.json            # Dataset info

src/inference/
├── __init__.py         # Public API exports
├── config.py           # InferenceConfig, ModelMetadata
├── preprocessing.py    # Image transforms
├── predictor.py        # GlaucomaPredictor, GlaucomaClassifier
├── model_registry.py   # Version tracking (optional)
└── batch_processor.py  # Folder-based processing (optional)
```

## API Reference

### GlaucomaPredictor

```python
class GlaucomaPredictor:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto)
        config: Optional[InferenceConfig] = None
    ) -> 'GlaucomaPredictor'

    def predict(
        self,
        image: Union[str, Path, PIL.Image.Image]
    ) -> PredictionResult

    def predict_batch(
        self,
        images: List[Union[str, Path, PIL.Image.Image]],
        batch_size: int = 32
    ) -> List[PredictionResult]
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    prediction: str           # 'normal' or 'glaucoma'
    confidence: float         # 0.0 to 1.0
    probabilities: Dict[str, float]  # {'normal': 0.3, 'glaucoma': 0.7}
    image_path: Optional[str] # Path if provided

    def to_dict(self) -> Dict[str, Any]  # JSON-serializable
```

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    model_path: str = "models/production/glaucoma_efficientnet_b0_v1.pt"
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'
    input_size: Tuple[int, int] = (224, 224)
    class_names: List[str] = ['normal', 'glaucoma']
    confidence_threshold: float = 0.5
```

## Usage Examples

### Basic Inference

```python
from src.inference import GlaucomaPredictor

predictor = GlaucomaPredictor.from_checkpoint(
    "models/production/glaucoma_efficientnet_b0_v1.pt"
)

result = predictor.predict("patient_001.png")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Probabilities: {result.probabilities}")
```

### Batch Processing

```python
from pathlib import Path

# Get all images in folder
image_folder = Path("data/new_patients/")
images = list(image_folder.glob("*.png"))

# Run batch inference
results = predictor.predict_batch(images, batch_size=16)

# Process results
for result in results:
    if result.prediction == 'glaucoma' and result.confidence > 0.7:
        print(f"High-confidence glaucoma: {result.image_path}")
```

### From PIL Image

```python
from PIL import Image

# Load image directly
img = Image.open("fundus.png")

# Predict
result = predictor.predict(img)
```

### Export Results to JSON

```python
import json

results = predictor.predict_batch(images)
output = [r.to_dict() for r in results]

with open("predictions.json", "w") as f:
    json.dump(output, f, indent=2)
```

## Integration with Evaluation

For proper model evaluation, use hospital-based splitting:

```python
from src.data.hospital_splitter import create_hospital_based_splits
import json

# Load dataset metadata
with open("data/processed/rim_one/metadata.json") as f:
    metadata = json.load(f)

samples = metadata['samples']

# Create splits
splits = create_hospital_based_splits(
    metadata=samples,
    test_institutions=['r1'],  # Held-out hospital
    seed=42
)

# Evaluate on test set
test_images = [f"data/processed/rim_one/images/{samples[i]['image_filename']}"
               for i in splits['test']]
results = predictor.predict_batch(test_images)
```

## Legacy Pipeline

The previous torchvision-based pipeline is preserved in:
- `src/models/_legacy/efficientnet_classifier_v1.py`
- See `src/models/_legacy/README.md` for revert instructions

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or switch to CPU:
```python
predictor = GlaucomaPredictor.from_checkpoint(path, device='cpu')
```

### Model Loading Issues
Ensure the checkpoint format matches. The loader handles:
- Direct state_dict
- `{'model_state_dict': ...}` format
- `{'state_dict': ...}` format

### Wrong Predictions
Ensure input images are:
- RGB fundus photographs
- Properly oriented (optic disc visible)
- Similar quality to training data (RIMONE dataset)
