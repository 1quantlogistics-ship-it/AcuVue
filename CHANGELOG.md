# Changelog

## v0.4.0 - 2025-01-14

### Added
- Cross-dataset curriculum learning with progressive difficulty
- Domain adaptation via gradient reversal
- Cross-dataset evaluation metrics

### Changed
- Improved multi-dataset training pipeline

## v0.3.0 - 2025-01-14

### Added
- Custom loss functions: Focal Loss, AUC surrogate, weighted BCE
- DRI (Disc Relevance Index) regularization for attention constraints
- Loss function factory for flexible configuration

## v0.2.0 - 2025-01-13

### Added
- Augmentation policy search with medical imaging constraints
- Safe/forbidden operation lists for fundus images
- Policy evaluation with fast proxy training

### Changed
- Augmentation pipeline now supports policy-based configuration

## v0.1.0 - 2025-01-12

### Added
- Architecture grammar system with multiple backbones
  - EfficientNet-B0/B3
  - ConvNeXt-Tiny
  - DeiT-Small
- Four fusion strategies: FiLM, Cross-Attention, Gated, Late
- Model factory for building architectures from specs

## v0.0.2 - 2025-01-11

### Added
- Multi-dataset support (RIM-ONE, REFUGE, G1020)
- ImageNet normalization for transfer learning
- Validation metrics: Dice, IoU, AUC, sensitivity, specificity

### Changed
- Removed CLAHE preprocessing (incompatible with pretrained models)

## v0.0.1 - 2025-01-10

### Added
- Initial U-Net segmentation pipeline
- Synthetic fundus data generator
- Basic training loop with checkpointing
- Hydra configuration system
