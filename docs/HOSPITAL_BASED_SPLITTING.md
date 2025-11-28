# Hospital-Based Data Splitting

Why standard evaluation metrics are misleading and how to evaluate properly.

## The Problem: Data Leakage

### What Happens with Random Splits

When you randomly split a medical imaging dataset:
- Images from the same hospital appear in both train and test sets
- The model learns hospital-specific artifacts (scanner type, lighting, positioning)
- Test performance is inflated because the model recognizes "where" an image came from

### Real Example: RIMONE Dataset

| Split Strategy | Test AUC | Reality |
|----------------|----------|---------|
| Random 80/20 | ~97% | Inflated (data leakage) |
| Hospital-based (r1 held out) | 93.7% | **Realistic** |

The 3-4% difference represents the model's false confidence from learning hospital signatures rather than actual glaucoma features.

## The Solution: Hospital-Based Splitting

Assign entire hospitals to either train/val OR test, never both.

```
RIMONE Dataset Structure:
├── r1 (Hospital 1) → TEST SET
├── r2 (Hospital 2) → TRAIN/VAL
└── r3 (Hospital 3) → TRAIN/VAL
```

This ensures the model is evaluated on truly unseen data from an institution it has never encountered.

## How to Use

### Quick Start

```python
from src.data.hospital_splitter import create_hospital_based_splits
import json

# Load your dataset metadata
with open("data/processed/rim_one/metadata.json") as f:
    metadata = json.load(f)

samples = metadata['samples']

# Create hospital-based splits
splits = create_hospital_based_splits(
    metadata=samples,
    test_institutions=['r1'],      # Hospital 1 for testing
    train_val_institutions=['r2', 'r3'],  # Hospitals 2,3 for training
    val_ratio=0.1,
    seed=42
)

print(f"Train: {len(splits['train'])} samples")
print(f"Val: {len(splits['val'])} samples")
print(f"Test: {len(splits['test'])} samples")
```

### Validate No Leakage

```python
from src.data.hospital_splitter import HospitalBasedSplitter

splitter = HospitalBasedSplitter(seed=42)
splits = splitter.split_by_institution(samples, test_institutions=['r1'])

# This MUST return True
is_valid = splitter.validate_no_leakage(splits, samples)
print(f"No leakage: {is_valid}")
```

### Get Split Statistics

```python
stats = splitter.get_split_statistics(splits, samples)

for split_name, split_stats in stats['splits'].items():
    print(f"\n{split_name.upper()}:")
    print(f"  Samples: {split_stats['count']}")
    print(f"  Hospitals: {split_stats['institutions']}")
    print(f"  Labels: {split_stats['label_distribution']}")
```

## API Reference

### HospitalBasedSplitter

```python
class HospitalBasedSplitter:
    def __init__(self, seed: int = 42)

    def split_by_institution(
        self,
        metadata: List[Dict],
        test_institutions: List[str] = ['r1'],
        train_val_institutions: Optional[List[str]] = None,
        val_ratio: float = 0.1,
        stratified: bool = True
    ) -> Dict[str, List[int]]

    def validate_no_leakage(
        self,
        splits: Dict[str, List[int]],
        metadata: Optional[List[Dict]] = None
    ) -> bool

    def get_split_statistics(
        self,
        splits: Dict[str, List[int]],
        metadata: List[Dict]
    ) -> Dict[str, Any]

    def save_splits(
        self,
        splits: Dict[str, List[int]],
        save_path: Path,
        metadata: Optional[List[Dict]] = None
    ) -> None

    @staticmethod
    def load_splits(load_path: Path) -> Dict[str, List[int]]
```

### Institution Utilities

```python
from src.data.institution_utils import (
    extract_institution_from_filename,  # 'r1_Im001.png' → 'r1'
    extract_institution_from_path,      # Full path parsing
    get_institution_from_metadata,      # Check 'source_hospital' field
    group_samples_by_institution,       # Group indices by hospital
    get_institution_statistics,         # Count per institution
)
```

## Metadata Requirements

Your dataset metadata must have one of these fields for each sample:

1. **Explicit field** (preferred):
   ```json
   {"sample_id": 0, "source_hospital": "r1", ...}
   ```

2. **Fallback to path** (automatic):
   ```json
   {"sample_id": 0, "original_path": "data/r1_Im001.png", ...}
   ```

The RIMONE dataset already has `source_hospital` in its metadata.

## Why This Matters

### Clinical Deployment

A model deployed in a new hospital will encounter:
- Different fundus cameras
- Different lighting conditions
- Different patient demographics
- Different imaging protocols

Hospital-based evaluation simulates this real-world scenario.

### Model Selection

When comparing models, always use hospital-based metrics:

| Model | Random Split AUC | Hospital Split AUC |
|-------|------------------|-------------------|
| Model A | 98.2% | 91.5% |
| Model B | 97.1% | **94.2%** |

Model B is actually better for clinical use, despite lower random-split performance.

### Publication Standards

For medical imaging papers, reviewers increasingly expect:
- Clear statement of train/test institution separation
- Performance metrics on held-out institutions
- Discussion of generalization limitations

## Best Practices

1. **Always validate no leakage** before training:
   ```python
   assert splitter.validate_no_leakage(splits, metadata)
   ```

2. **Report both metrics** when available:
   ```
   Within-hospital AUC: 97.2%
   Cross-hospital AUC: 93.7%
   ```

3. **Use multiple test institutions** if possible:
   ```python
   splits_r1 = create_splits(metadata, test_institutions=['r1'])
   splits_r2 = create_splits(metadata, test_institutions=['r2'])
   # Average performance across held-out hospitals
   ```

4. **Document your split strategy** in any publication or deployment.

## Related

- [INFERENCE_PIPELINE_V2.md](INFERENCE_PIPELINE_V2.md) - Using the production model
- `src/models/_legacy/README.md` - Legacy pipeline information
