# Changelog

All notable changes to the AcuVue glaucoma classification project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Phase 03f - FAILED] - 2025-01-14

### Summary
**Status**: ❌ **EXPERIMENT FAILED - HYPOTHESIS REJECTED**

Phase 03f smoke test (5 epochs) revealed catastrophic regression on RIM-ONE dataset and severe overfitting. ImageNet normalization hypothesis was rejected. Full 30-epoch training was **NOT AUTHORIZED** per manager decision gate.

### Results
- **G1020 AUC**: 0.5614 (baseline: 0.5323) → +5.5% (marginal, insufficient)
- **RIM-ONE AUC**: 0.4685 (baseline: 0.7763) → **–39.7% CATASTROPHIC FAILURE** ❌❌❌
- **Combined AUC**: 0.5096 (baseline: 0.5323) → –4.3% regression ❌
- **Calibration (ECE)**: 0.35 (target: ≤0.05) → 7x worse than acceptable ❌
- **Validation Overfitting**: Training-reported AUC 0.5906 vs cross-eval AUC 0.5096 (–13.7%) ❌

### Root Cause Analysis
1. **Domain Shift Sensitivity**: ImageNet normalization disrupted RIM-ONE-specific learned patterns
2. **Insufficient Training**: 5 epochs not enough to learn robust ImageNet-aligned features
3. **Dataset Heterogeneity**: RIM-ONE and G1020 imaging characteristics incompatible with single normalization scheme
4. **Over-Correction**: CLAHE removal + ImageNet norm was too aggressive - lost critical domain knowledge

### Deliverables
- [configs/phase03f_smoketest.yaml](configs/phase03f_smoketest.yaml): 5-epoch smoke test configuration
- [configs/phase03f.yaml](configs/phase03f.yaml): 30-epoch config (NOT EXECUTED)
- [models/phase03f_smoketest/best_model.pt](models/phase03f_smoketest/best_model.pt): Failed model checkpoint
- [reports/phase03f_smoketest_analysis.md](reports/phase03f_smoketest_analysis.md): Complete failure analysis
- [reports/phase03f_smoketest_cross_eval.json](reports/phase03f_smoketest_cross_eval.json): Per-dataset cross-evaluation
- [reports/phase03f_calibration.json](reports/phase03f_calibration.json): Calibration assessment

### Decision
**Manager Decision Gate**: 0/5 success criteria met
- ❌ G1020 AUC < 0.58 (target: ≥0.58)
- ❌❌❌ RIM-ONE AUC catastrophic regression (0.4685 vs target ≥0.75)
- ❌ Severe miscalibration
- ❌ Validation fold overfitting
- ❌ Combined AUC regression

**Action**: HALT Phase 03f. Revert to Phase 03d baseline. Plan Phase 03g with domain adaptation focus.

### Lessons Learned
1. ✓ Smoke test protocol worked as intended - caught failure early (saved 2.5 hours GPU time)
2. ❌ ImageNet normalization + CLAHE removal is fundamentally flawed for this task
3. → Phase 03d (despite poor G1020 AUC 0.53) is superior due to balanced cross-dataset performance
4. → Future work requires domain adaptation techniques, not simple normalization changes

## [Phase 03e] - 2025-01-14

### Added
- **ImageNet Normalization Support**:
  - Added optional `use_imagenet_norm` parameter to `FundusDataset` ([src/data/fundus_dataset.py](src/data/fundus_dataset.py))
  - Apply standard ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
  - Configurable via `data.use_imagenet_norm` in Hydra configs
  - Improves transfer learning with pretrained EfficientNet-B0 models

- **Preprocessing Validation Scripts**:
  - `scripts/validate_normalization.py`: Validates ImageNet normalization implementation
  - `scripts/log_batch_stats.py`: Computes baseline batch statistics for datasets
  - `scripts/verify_preprocessing.py`: Detects CLAHE application via histogram analysis

- **Phase 03e Configuration**:
  - [configs/phase03e.yaml](configs/phase03e.yaml): Config with ImageNet normalization enabled
  - Inherits from phase03d_baseline with preprocessing audit changes

- **Documentation**:
  - [docs/preprocessing_pipeline.md](docs/preprocessing_pipeline.md): Complete preprocessing pipeline documentation
  - [data/processed/combined_v2/MANIFEST.md](data/processed/combined_v2/MANIFEST.md): Dataset composition and metadata
  - [reports/imagenet_norm_validation.txt](reports/imagenet_norm_validation.txt): Validation test results
  - [reports/batch_statistics.json](reports/batch_statistics.json): Baseline statistics for clean datasets

### Changed
- **BREAKING: Removed CLAHE from Preprocessing**:
  - Removed CLAHE application from [src/data/prepare_clinical_datasets.py](src/data/prepare_clinical_datasets.py) (lines 182-184, 369-371)
  - Removed CLAHE application from [src/data/prepare_g1020.py](src/data/prepare_g1020.py) (lines 227-229)
  - **Rationale**: CLAHE was incompatible with ImageNet normalization and reduced transfer learning effectiveness

- **Dataset Regeneration** (combined_v2):
  - Regenerated all datasets from scratch without CLAHE
  - RIM-ONE: 485 samples
  - REFUGE2: 400 training samples
  - G1020: 1020 samples
  - **combined_v2**: 1905 samples (1394 train, 132 val, 379 test)
  - New timestamp: 2025-01-14

- **Preprocessing Version**:
  - Updated from `v1_clahe` (Phase 03c/03d) to `v3_no_clahe_imagenet_norm`
  - Clean datasets without CLAHE, with optional ImageNet normalization

- **Training Script Updates**:
  - Updated [src/training/train_classification.py](src/training/train_classification.py) (lines 320-349)
  - Pass `use_imagenet_norm` parameter to dataset loaders
  - Extract normalization flag from Hydra config

### Fixed
- Phase 03d failure root cause identified: CLAHE + missing ImageNet normalization
- All Phase 03d experiments (Rev A, B, C) affected by preprocessing issues
- G1020 test AUC stuck at ~0.53 due to suboptimal transfer learning

### Validated
- **Normalization Tests** (all passed):
  - ✅ WITHOUT normalization: values in [0.0157, 0.9843]
  - ✅ WITH normalization: values in [-1.8256, 2.1804]
  - ✅ Formula verified: (img/255 - mean) / std (max diff: 0.000000)
  - ✅ Labels unchanged across normalization

- **Batch Statistics** (1394 train samples, BEFORE ImageNet normalization):
  - Mean (R,G,B): (0.541, 0.258, 0.129)
  - Std (R,G,B): (0.312, 0.156, 0.088)
  - Range: [0.00, 1.00]

### Deprecated
- Phase 03c/03d datasets with CLAHE (`v1_clahe` preprocessing)
- Direct use of [0, 1] normalized inputs with pretrained models (now uses ImageNet norm)

---

## [Phase 03d Rev C] - 2025-01-13

### Added
- Conditional Batch Normalization (CBN) in classifier head
- Dual-input architecture: image + label for feature modulation
- Ablation study: CBN vs. standard BatchNorm

### Changed
- Increased feature extraction capacity with CBN layers
- Updated classifier architecture in [src/models/classifier.py](src/models/classifier.py)

### Results
- G1020 test AUC: **0.53** (no improvement over baseline)
- **Root cause**: CLAHE + missing ImageNet normalization (identified in Phase 03e audit)

---

## [Phase 03d Rev B] - 2025-01-13

### Added
- Focal Loss with gamma=2.0
- Balanced batch sampling
- Class weight tuning experiments

### Changed
- Loss function from standard CrossEntropy to Focal Loss
- Sampling strategy to address 73:27 class imbalance

### Results
- G1020 test AUC: **0.53** (no improvement over baseline)
- **Root cause**: CLAHE + missing ImageNet normalization (identified in Phase 03e audit)

---

## [Phase 03d Rev A / Baseline] - 2025-01-12

### Added
- Baseline experiment with standard training configuration
- combined_v2 dataset (RIM-ONE + REFUGE2 + G1020)
- EfficientNet-B0 backbone (pretrained ImageNet)

### Changed
- Switched from synthetic data to multi-dataset clinical data
- 1905 total samples (1394 train, 132 val, 379 test)

### Results
- G1020 test AUC: **0.53** (baseline performance)
- **Root cause**: CLAHE + missing ImageNet normalization (identified in Phase 03e audit)

---

## [Phase 03c.C] - 2025-01-12

### Added
- Dataset fusion: RIM-ONE + REFUGE2 + G1020
- Multi-center data diversity
- [src/data/combine_three_datasets.py](src/data/combine_three_datasets.py): Dataset fusion script
- [data/processed/combined_v2/](data/processed/combined_v2/): Combined dataset

### Changed
- Increased training data from 681 (Phase 03c.A/B) to 1394 samples
- Class balance: 73.3% Normal, 26.7% Glaucoma
- Recommended class weights: [0.2525, 0.7475]

### Note
- **CLAHE applied during preprocessing** (later identified as problematic in Phase 03e)

---

## [Earlier Phases]

*(To be documented)*

- Phase 01: Synthetic data generation
- Phase 02: Initial training experiments
- Phase 03a/b: Clinical dataset experiments (RIM-ONE, REFUGE2)

---

## Preprocessing Version History

| Version | CLAHE | ImageNet Norm | Phase | Status |
|---------|-------|---------------|-------|--------|
| v1_clahe | ✅ Yes | ❌ No | 03c, 03d | **Deprecated** |
| v2_no_clahe | ❌ No | ❌ No | - | Testing only |
| v3_no_clahe_imagenet_norm | ❌ No | ✅ Yes | **03e+** | **Current** |

---

## References

For detailed preprocessing information, see:
- [docs/preprocessing_pipeline.md](docs/preprocessing_pipeline.md)
- [data/processed/combined_v2/MANIFEST.md](data/processed/combined_v2/MANIFEST.md)
