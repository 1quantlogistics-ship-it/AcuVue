# AcuVue

Deep learning system for glaucoma detection via optic disc and cup segmentation from retinal fundus images.

## Overview

AcuVue segments optic disc and cup regions from fundus images using U-Net-based architectures, enabling automated cup-to-disc ratio (CDR) calculation for glaucoma screening. The system supports multiple backbone architectures and has been validated on clinical datasets.

## Results

**Best AUC: 0.93** on combined clinical dataset (RIM-ONE + REFUGE + G1020)

Achieved using EfficientNet-B0 backbone with domain adaptation across 1,905 fundus images.

## Architecture

The system supports multiple configurations:

**Backbones:**
- EfficientNet-B0/B3 (default)
- ConvNeXt-Tiny
- DeiT-Small (Vision Transformer)

**Fusion strategies** for combining image features with clinical indicators:
- FiLM (Feature-wise Linear Modulation)
- Cross-Attention
- Gated fusion
- Late fusion (concatenation)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/1quantlogistics-ship-it/AcuVue.git
cd AcuVue
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train on synthetic data (for testing)
python src/data/synthetic_fundus.py
python src/training/train_segmentation.py

# Train on real data
python src/training/train_segmentation.py data.source=rim_one data.data_root=data/raw/RIM-ONE
```

## Datasets

Supports three clinical fundus datasets:
- **RIM-ONE**: 485 images
- **REFUGE**: 400 training images
- **G1020**: 1020 images

Data preprocessing includes resizing to 512x512 and optional ImageNet normalization for transfer learning.

## Project Structure

```
AcuVue/
├── src/
│   ├── data/           # Dataset loaders, preprocessing, augmentation
│   ├── models/         # U-Net, backbones, fusion modules
│   ├── training/       # Training scripts, loss functions
│   └── evaluation/     # Metrics (Dice, IoU, AUC)
├── configs/            # Hydra configuration files
├── scripts/            # Utility scripts for GPU training
└── tests/              # Unit tests
```

## Configuration

Training is configured via Hydra YAML files:

```yaml
training:
  epochs: 10
  batch_size: 4
  learning_rate: 0.001

model:
  backbone: efficientnet-b0
  fusion_type: late

data:
  source: synthetic
  image_size: 512
```

Override from command line:
```bash
python src/training/train_segmentation.py training.epochs=20 training.batch_size=8
```

## Key Features

- Multiple backbone architectures with pretrained ImageNet weights
- Four fusion strategies for multimodal learning
- Domain adaptation for cross-dataset generalization
- Curriculum learning for training on multiple datasets
- Custom loss functions (Focal, AUC surrogate, weighted BCE)
- Augmentation policy search with medical imaging constraints

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended for training)

See `requirements.txt` for full dependencies.

## Testing

```bash
pytest tests/ -v
```

## License

MIT
