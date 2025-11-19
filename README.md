# AcuVue - GPU-Based Medical Image Segmentation

AI-powered optic disc and cup segmentation system for glaucoma detection using deep learning.

## Overview

AcuVue is a medical imaging pipeline that uses U-Net architecture to segment optic disc and cup regions from retinal fundus images. The system is designed for GPU-accelerated training on RunPod (A40) with development managed through GitHub Codespaces.

**Technology Stack:**
- **GPU Compute:** RunPod (PyTorch 2.8 + CUDA 12.8)
- **Development:** GitHub Codespaces + VSCode
- **Deep Learning:** PyTorch, U-Net architecture
- **Configuration:** Hydra
- **Data Versioning:** DVC (planned)
- **Experiment Tracking:** WandB (optional)

## Project Structure

```
AcuVue/
├── src/
│   ├── data/
│   │   ├── synthetic_fundus.py      # Synthetic fundus generator (Phase 02)
│   │   ├── fundus_dataset.py        # Unified dataset loader (Phase 02)
│   │   ├── data_splitter.py         # Train/val/test splitting (Phase 02)
│   │   ├── preprocess.py            # Image preprocessing (CLAHE, cropping)
│   │   └── segmentation_dataset.py  # PyTorch Dataset (Phase 01)
│   ├── models/
│   │   ├── unet_disc_cup.py         # U-Net model + Dice loss
│   │   ├── fusion_modules.py        # Multi-modal fusion strategies (Phase E)
│   │   ├── backbones.py             # Backbone architectures (Phase E)
│   │   ├── model_factory.py         # Architecture builder from specs (Phase E)
│   │   └── tests/                   # Model architecture unit tests
│   ├── training/
│   │   ├── train_phase02.py         # Phase 02 training with validation
│   │   └── train_segmentation.py    # Phase 01 smoke test
│   ├── evaluation/
│   │   └── metrics.py               # Segmentation metrics (Phase 02)
│   └── utils/
│       └── checkpoint.py            # Checkpoint management (Phase 02)
├── configs/
│   ├── phase02_baseline.yaml        # Phase 02 baseline config
│   ├── phase02_production.yaml      # Phase 02 production (GPU-only)
│   └── phase01_smoke_test.yaml      # Phase 01 config
├── scripts/
│   ├── setup_runpod.sh              # RunPod environment bootstrap
│   ├── run_on_gpu.sh                # Remote GPU training launcher
│   ├── sync_results.sh              # Artifact sync from RunPod
│   └── verify_phase01.py            # Environment verification
├── docs/
│   └── REMOTE_COMPUTE.md            # Remote GPU infrastructure guide
├── .runpod/
│   └── connection.env.template      # RunPod connection config template
├── data/                            # Datasets (synthetic/real)
├── models/                          # Saved model checkpoints
├── reports/                         # Generated reports
├── requirements.txt                 # Python dependencies
├── PHASE02_QUICKSTART.md           # Phase 02 execution guide
└── README.md                        # This file
```

## Phase 01: Smoke Test (Current)

**Goal:** Verify the segmentation pipeline executes on GPU and produces a model checkpoint.

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, but CPU works for testing)
- Git

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/1quantlogistics/AcuVue.git
cd AcuVue
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Verification

Before running training, verify your environment:

```bash
python scripts/verify_phase01.py
```

**Expected output:**
- ✓ All imports resolve
- ✓ CUDA available (on GPU systems)
- ✓ Model instantiates correctly
- ✓ Forward pass works
- ✓ Dataset loads properly

### Running Phase 01 Training

#### Basic Usage

```bash
python src/training/train_segmentation.py
```

This runs a 1-epoch smoke test with dummy data using the default configuration ([configs/phase01_smoke_test.yaml](configs/phase01_smoke_test.yaml)).

#### Custom Configuration (Hydra Override)

```bash
# Override specific parameters
python src/training/train_segmentation.py training.epochs=2 training.batch_size=4

# Use different learning rate
python src/training/train_segmentation.py training.learning_rate=0.0001

# Change number of dummy samples
python src/training/train_segmentation.py training.num_dummy_samples=20
```

### Expected Output

```
============================================================
         Phase 01: Smoke Test - Segmentation Training
============================================================

Configuration:
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  num_dummy_samples: 10
...

Using device: cuda
GPU: NVIDIA A40
GPU Memory: 46.00 GB

Generating 10 dummy image-mask pairs...
Dataset size: 10 samples

UNet model with 7,761,729 trainable parameters

============================================================
        Starting training for 1 epoch(s)
============================================================

Epoch 1/1: 100%|████████████| 5/5 [00:02<00:00, loss=1.8234, avg_loss=1.8145]

Epoch 1/1 - Average Loss: 1.8145

Saving checkpoint to: models/unet_disc_cup.pt
✓ Checkpoint saved successfully (29.63 MB)
Testing checkpoint load...
✓ Checkpoint loads successfully

============================================================
              Phase 01 Smoke Test: COMPLETE
============================================================
```

### Verify Checkpoint

After training completes:

```bash
python -c "import torch; model = torch.load('models/unet_disc_cup.pt'); print('✓ Checkpoint OK')"
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit/test_smoke.py -v

# Run specific test
pytest tests/unit/test_smoke.py::TestModel::test_model_forward_pass -v
```

## RunPod Deployment

### Setup on RunPod GPU Pod

```bash
# SSH into pod
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519

# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/1quantlogistics/AcuVue.git
cd AcuVue

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify environment
python scripts/verify_phase01.py

# Run training
python src/training/train_segmentation.py
```

## Development Roadmap

- [x] **Phase 01:** Dry Run (1-epoch smoke test)
- [x] **Phase 02:** Baseline Training (10 epochs, validation, metrics)
- [x] **Phase 02.5:** Remote GPU Infrastructure (RunPod integration)
- [x] **Phase 03:** Multi-dataset training with domain normalization
- [x] **Phase E Week 1:** Architecture Grammar System ← **Current**
- [ ] **Phase E Week 2:** Augmentation Policy Search
- [ ] **Phase E Week 3:** Loss Function Engineering
- [ ] **Phase E Week 4:** Cross-Dataset Curriculum Learning
- [ ] **Phase 04:** CI/CD + Deployment Pipeline

See [claude_plan/](claude_plan/) for detailed phase documentation.

## Phase 02: Baseline Training with Validation

**Goal:** Complete 10-epoch training pipeline with train/val/test splits, comprehensive metrics tracking, and checkpoint management.

### Quick Start

```bash
# 1. Generate synthetic dataset (for testing infrastructure)
python src/data/synthetic_fundus.py

# 2. Run 10-epoch training
python src/training/train_phase02.py

# 3. Check results
cat models/test_results.json
```

**See [PHASE02_QUICKSTART.md](PHASE02_QUICKSTART.md) for complete guide.**

### Phase 02 Features

- ✅ Synthetic fundus image generator (100 samples with anatomically plausible disc/cup masks)
- ✅ Unified dataset loader supporting synthetic and real data (RIM-ONE/REFUGE)
- ✅ Train/val/test splits (70/20/10) with deterministic seeding
- ✅ Data augmentation (flip, rotation, brightness, contrast)
- ✅ Comprehensive metrics (Dice, IoU, accuracy, sensitivity, specificity)
- ✅ Best model checkpointing based on validation Dice
- ✅ Training history logging (JSON format)
- ✅ Test set evaluation after training
- ✅ Hydra configuration with CLI overrides

### Switching to Real Data

To use real datasets (RIM-ONE or REFUGE), change ONE line in config:

```yaml
# configs/phase02_baseline.yaml
data:
  source: rim_one  # Changed from "synthetic"
  data_root: data/raw/RIM-ONE
```

Then run the same command:
```bash
python src/training/train_phase02.py
```

No other code changes required!

## Remote GPU Training (Phase 02.5)

**🎯 Goal:** Ensure all training runs on RunPod GPU, preventing accidental local CPU execution.

### Quick Start: Run Training on RunPod

```bash
# 1. Configure connection (one-time setup)
cp .runpod/connection.env.template .runpod/connection.env
# Edit with your RunPod pod details

# 2. Launch training on GPU
bash scripts/run_on_gpu.sh

# 3. Download results
bash scripts/sync_results.sh
```

### GPU Fail-Safe System

The training scripts include a fail-safe mechanism to prevent accidental local CPU training:

```yaml
# configs/phase02_production.yaml
system:
  require_gpu: true  # Abort if CUDA not available
```

When `require_gpu: true`, training will immediately fail if no GPU is detected, with a clear error message directing you to use `scripts/run_on_gpu.sh`.

### Infrastructure Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_runpod.sh` | Bootstrap fresh RunPod pod environment |
| `scripts/run_on_gpu.sh` | Launch training on RunPod from local machine |
| `scripts/sync_results.sh` | Download training artifacts (models, logs, reports) |

### Example Workflows

**Development / Testing:**
```bash
# Local testing with synthetic data (CPU allowed)
python src/training/train_phase02.py system.require_gpu=false
```

**Production Training:**
```bash
# Always use RunPod GPU
bash scripts/run_on_gpu.sh phase02_production

# Monitor via WandB (if enabled)
# Download results when complete
bash scripts/sync_results.sh
```

**See [docs/REMOTE_COMPUTE.md](docs/REMOTE_COMPUTE.md) for complete remote compute guide.**

## Phase E: Architecture Grammar System (Week 1)

**Goal:** Enable ARC's autonomous multi-agent system to explore fusion architectures and backbone alternatives beyond fixed templates, transforming from hyperparameter tuning to true architecture search.

### What's New

**Architecture Grammar System** allows ARC agents to propose and evaluate different model architectures:

- **4 Fusion Strategies**: FiLM, Cross-Attention, Gated, Late fusion
- **3 Backbone Alternatives**: EfficientNet-B3, ConvNeXt-Tiny, DeiT-Small
- **Model Factory**: Build complete models from architecture specs
- **12 Validated Combinations**: All backbone × fusion pairs tested

### Quick Start

```python
from src.models.model_factory import build_model_from_spec

# Define architecture
spec = {
    "backbone": "efficientnet-b3",
    "fusion_type": "film",
    "clinical_dim": 4,
    "head_config": {"dropout": 0.3}
}

# Build model
model = build_model_from_spec(spec, num_classes=2)

# Use model
logits = model(images, clinical_indicators)
```

### Architecture Spec Format

ARC's Architect agent can now propose architecture specs:

```python
{
    "backbone": "efficientnet-b3" | "convnext-tiny" | "deit-small",
    "fusion_type": "film" | "cross_attention" | "gated" | "late",
    "clinical_dim": 4,  # Number of clinical indicators
    "head_config": {
        "hidden_dim": 256,
        "dropout": 0.3
    }
}
```

### Fusion Strategies

| Fusion Type | Description | Use Case |
|-------------|-------------|----------|
| **FiLM** | Feature-wise Linear Modulation | Strong clinical indicators |
| **CrossAttention** | Multi-head attention | Spatial localization |
| **Gated** | Learnable per-sample gates | Variable clinical quality |
| **Late** | Concatenation + MLP | Baseline comparison |

### Backbone Options

| Backbone | Parameters | Description |
|----------|------------|-------------|
| **efficientnet-b3** | 10.7M | Upgraded capacity from B0 |
| **convnext-tiny** | 27.8M | Modern CNN architecture |
| **deit-small** | 21.7M | Vision Transformer (ViT) |

### Running Tests

```bash
# Unit tests for all architecture components
cd src/models
pytest tests/test_architectures.py -v

# Integration test (validates all 12 combinations)
cd src
python3 -m models.model_factory
```

### Integration with ARC

The architecture grammar integrates with ARC's multi-agent system:

| Agent | Function | Usage |
|-------|----------|-------|
| **Architect** | Propose architectures | Generates `architecture_spec` dicts |
| **Critic** | Validate specs | `validate_architecture_spec(spec)` |
| **Executor** | Build models | `build_model_from_spec(spec)` |
| **Historian** | Track families | `get_model_summary(model)` |

### Documentation

- **[src/models/README_PHASE_E_WEEK1.md](src/models/README_PHASE_E_WEEK1.md)** - Complete implementation details
- **[ARCHITECTURE_SPEC_FORMAT.md](ARCHITECTURE_SPEC_FORMAT.md)** - Quick reference for ARC agents

**See Phase E documentation for complete details on architecture search capabilities.**

## Configuration

Phase 01 configuration is in [configs/phase01_smoke_test.yaml](configs/phase01_smoke_test.yaml):

```yaml
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  num_dummy_samples: 10

model:
  in_channels: 3
  out_channels: 1

data:
  image_size: 512
  use_augmentation: false

system:
  device: auto  # cuda if available, else cpu
  seed: 42
  log_level: INFO
```

## Troubleshooting

### ImportError: No module named 'src'

Make sure you're running from the project root directory and the virtual environment is activated.

### CUDA out of memory

Reduce batch size:
```bash
python src/training/train_segmentation.py training.batch_size=1
```

### Checkpoint doesn't load

Verify the checkpoint file exists and isn't corrupted:
```bash
ls -lh models/unet_disc_cup.pt
```

## Contributing

This project follows a phased development approach. See [claude_plan/](claude_plan/) for current phase requirements.

## License

[Add license information]

## Contact

[Add contact information]

---

**Current Status:** Phase E Week 1 - Architecture Grammar System Complete ✓
