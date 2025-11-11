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
│   │   ├── preprocess.py          # Image preprocessing (CLAHE, cropping)
│   │   └── segmentation_dataset.py # PyTorch Dataset for segmentation
│   ├── models/
│   │   └── unet_disc_cup.py       # U-Net model + Dice loss
│   └── training/
│       └── train_segmentation.py  # Main training script (Hydra)
├── configs/
│   └── phase01_smoke_test.yaml    # Phase 01 configuration
├── claude_plan/
│   └── phase_01_dryrun.yaml       # Phase documentation
├── tests/
│   └── unit/
│       └── test_smoke.py          # Unit tests
├── scripts/
│   └── verify_phase01.py          # Environment verification
├── models/                         # Saved model checkpoints
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

- [x] **Phase 01:** Dry Run (1-epoch smoke test) ← **Current**
- [ ] **Phase 02:** Baseline Training (10 epochs, DVC integration)
- [ ] **Phase 03:** Feature Extractor + Fusion Model
- [ ] **Phase 04:** CI/CD + Deployment Pipeline

See [claude_plan/](claude_plan/) for detailed phase documentation.

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

**Current Status:** Phase 01 - Smoke Test Implementation Complete
