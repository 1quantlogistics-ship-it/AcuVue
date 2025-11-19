# Phase E Week 3: Loss Function Engineering

**ARC Training Improvements DevNote v1.0 - Feature 3**
**Dev 2 Implementation**

## Overview

This phase implements specialized loss functions for medical imaging that address common challenges in glaucoma detection:

1. **Class Imbalance**: Medical datasets are often heavily imbalanced (few positive cases)
2. **Asymmetric Costs**: False negatives (missed disease) are more costly than false positives
3. **AUC Optimization**: Clinical utility measured by AUC rather than accuracy
4. **Attention Constraint**: Model must focus on optic disc region (DRI)

## Architecture

### Loss Functions

#### 1. WeightedBCELoss
**Purpose**: Handle class imbalance with automatic weight computation

```python
from training.custom_losses import WeightedBCELoss

# Manual weights
loss_fn = WeightedBCELoss(pos_weight=2.0, neg_weight=1.0)

# Auto-compute from training data
pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(train_labels)
loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
```

**When to use:**
- Imbalanced datasets (10% positive, 90% negative)
- Need balanced optimization across both classes
- Standard baseline for medical imaging

**Key features:**
- Automatic weight computation: `n_samples / (n_classes * class_count)`
- Works with multi-class classification
- Reduction modes: mean, sum, none

#### 2. AsymmetricFocalLoss
**Purpose**: Reduce false negatives by focusing on hard examples

```python
from training.custom_losses import AsymmetricFocalLoss

loss_fn = AsymmetricFocalLoss(
    gamma_pos=2.0,  # Focus on hard positive examples
    gamma_neg=1.0,  # Standard penalty for negatives
    clip=0.05       # Probability clipping for stability
)
```

**When to use:**
- False negatives are more costly than false positives
- Need to reduce missed disease cases
- Hard examples require more attention

**Key features:**
- `gamma_pos`: Controls focus on hard positive examples (higher = more focus)
- `gamma_neg`: Controls penalty for easy negative examples
- `clip`: Prevents numerical instability with extreme probabilities

**Parameters guide:**
- `gamma_pos=2.0, gamma_neg=1.0`: Standard asymmetric (reduce FN)
- `gamma_pos=3.0, gamma_neg=0.5`: Strong FN reduction (aggressive)
- `gamma_pos=2.0, gamma_neg=2.0`: Symmetric focal (hard example focus)

#### 3. AUCSurrogateLoss
**Purpose**: Directly optimize AUC via pairwise ranking

```python
from training.custom_losses import AUCSurrogateLoss

loss_fn = AUCSurrogateLoss(margin=1.0)
```

**When to use:**
- Clinical evaluation uses AUC/ROC
- Need to optimize ranking of positive vs negative scores
- Want to improve discriminative power

**How it works:**
- Computes pairwise differences between positive and negative predictions
- Penalizes cases where positive score ≤ negative score + margin
- Smooth hinge loss: `ReLU(margin - (pos_score - neg_score))`

**Key features:**
- `margin`: Required separation between positive and negative scores
- Returns 0 if all positive/negative (no ranking possible)
- Differentiable surrogate for AUC optimization

#### 4. DRIRegularizer
**Purpose**: Constrain model attention to optic disc region

```python
from training.custom_losses import DRIRegularizer

reg = DRIRegularizer(
    model=model,
    lambda_dri=0.1,      # Regularization strength
    dri_threshold=0.6    # Target DRI value
)

penalty = reg(images, disc_masks, clinical)
```

**When to use:**
- Model attention drifts away from optic disc
- Need to enforce medical constraints
- Combine with any base loss via CombinedLoss

**How it works:**
- Computes Grad-CAM attention maps for each image
- Calculates IoU with disc mask (DRI)
- Applies penalty if `avg_dri < threshold`: `lambda_dri * (threshold - avg_dri)`

**Key features:**
- `lambda_dri`: Regularization strength (0.1 = 10% of base loss)
- `dri_threshold`: Target DRI value (0.6 = 60% overlap)
- Differentiable through Grad-CAM computation

#### 5. CombinedLoss
**Purpose**: Combine any base loss with DRI regularization

```python
from training.custom_losses import WeightedBCELoss, CombinedLoss

base_loss = WeightedBCELoss(pos_weight=2.0)
combined = CombinedLoss(
    base_loss=base_loss,
    model=model,
    lambda_dri=0.1,
    dri_threshold=0.6
)

result = combined(logits, labels, images, disc_masks)
# Returns: {'total': total_loss, 'base': base_loss, 'dri_penalty': penalty}
```

**When to use:**
- Want to add DRI constraint to any loss function
- Need detailed loss component tracking
- Training with attention constraints

**Key features:**
- Works with any base loss (BCE, Focal, AUC, CrossEntropy)
- Returns dict with loss components for logging
- `total = base + dri_penalty`

## Loss Factory (ARC Integration)

The loss factory enables ARC's Explorer agent to propose different loss functions via specifications.

### Building Losses from Specs

```python
from training.loss_factory import build_loss_from_spec

# Example 1: Weighted BCE
spec = {
    "loss_type": "weighted_bce",
    "pos_weight": 2.0,
    "neg_weight": 1.0
}
loss_fn = build_loss_from_spec(spec)

# Example 2: Asymmetric Focal with DRI
spec = {
    "loss_type": "asymmetric_focal",
    "gamma_pos": 2.0,
    "gamma_neg": 0.5,
    "dri_regularization": True,
    "lambda_dri": 0.1,
    "dri_threshold": 0.6
}
loss_fn = build_loss_from_spec(spec, model=model)

# Example 3: AUC Surrogate
spec = {
    "loss_type": "auc_surrogate",
    "margin": 1.0
}
loss_fn = build_loss_from_spec(spec)

# Example 4: Auto-compute weights
spec = {
    "loss_type": "weighted_bce"
}
loss_fn = build_loss_from_spec(spec, train_labels=train_labels)
```

### Spec Validation

```python
from training.loss_factory import validate_loss_spec

spec = {
    "loss_type": "weighted_bce",
    "pos_weight": 2.0,
    "neg_weight": 1.0
}

try:
    validate_loss_spec(spec)  # Returns True if valid
except ValueError as e:
    print(f"Invalid spec: {e}")
```

**Valid loss types:**
- `weighted_bce`: Weighted binary cross-entropy
- `asymmetric_focal`: Asymmetric focal loss
- `auc_surrogate`: AUC surrogate loss
- `cross_entropy`: Standard PyTorch CrossEntropyLoss

**Validation checks:**
- Required fields present (`loss_type`)
- Valid loss_type
- Parameter ranges (e.g., `pos_weight > 0`, `gamma >= 0`)
- DRI regularization requirements (model required if `dri_regularization=True`)

### Loss Summary

```python
from training.loss_factory import get_loss_summary

summary = get_loss_summary(loss_fn)
print(summary)
# {'loss_class': 'WeightedBCELoss', 'pos_weight': 2.0, 'neg_weight': 1.0}
```

## ARC Integration Example

ARC's Explorer agent can now propose loss functions:

```python
# ARC proposes experiment
experiment = {
    "architecture_spec": {...},  # From Phase E Week 1
    "augmentation_policy": [...],  # From Phase E Week 2
    "loss_spec": {  # NEW: From Phase E Week 3
        "loss_type": "asymmetric_focal",
        "gamma_pos": 2.0,
        "gamma_neg": 0.5,
        "dri_regularization": True,
        "lambda_dri": 0.1
    }
}

# Build loss function
loss_fn = build_loss_from_spec(
    experiment["loss_spec"],
    model=model,
    train_labels=train_labels
)

# Train model
for batch in train_loader:
    images, labels, disc_masks, clinical = batch
    logits = model(images, clinical)

    if isinstance(loss_fn, CombinedLoss):
        result = loss_fn(logits, labels, images, disc_masks, clinical)
        loss = result['total']
        log_metrics(result)  # Log all components
    else:
        loss = loss_fn(logits, labels)

    loss.backward()
    optimizer.step()
```

## Usage Examples

### Example 1: Class Imbalance Handling

```python
from training.custom_losses import WeightedBCELoss

# Scenario: 5% positive, 95% negative
train_labels = torch.tensor([...])  # Your training labels

# Auto-compute optimal weights
pos_weight, neg_weight = WeightedBCELoss.compute_weights_from_labels(train_labels)
print(f"Weights: pos={pos_weight:.2f}, neg={neg_weight:.2f}")
# Output: Weights: pos=9.50, neg=0.50

loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)

# Training loop
for batch in train_loader:
    logits = model(images, clinical)
    loss = loss_fn(logits, labels)
    loss.backward()
```

### Example 2: Reduce False Negatives

```python
from training.custom_losses import AsymmetricFocalLoss

# Focus heavily on reducing false negatives
loss_fn = AsymmetricFocalLoss(
    gamma_pos=3.0,  # Strong focus on hard positive examples
    gamma_neg=0.5,  # Reduce penalty on easy negatives
    clip=0.05
)

# Training loop
for batch in train_loader:
    logits = model(images, clinical)
    loss = loss_fn(logits, labels)
    loss.backward()
```

### Example 3: AUC Optimization

```python
from training.custom_losses import AUCSurrogateLoss

# Optimize for ranking quality
loss_fn = AUCSurrogateLoss(margin=1.0)

# Training loop
for batch in train_loader:
    logits = model(images, clinical)
    loss = loss_fn(logits, labels)

    # Note: Loss will be 0 if batch is all positive or all negative
    if loss > 0:
        loss.backward()
```

### Example 4: Attention Constraint

```python
from training.custom_losses import WeightedBCELoss, CombinedLoss

# Combine weighted BCE with DRI constraint
base_loss = WeightedBCELoss(pos_weight=2.0, neg_weight=1.0)
loss_fn = CombinedLoss(
    base_loss=base_loss,
    model=model,
    lambda_dri=0.1,  # 10% regularization
    dri_threshold=0.6  # Require 60% overlap
)

# Training loop
for batch in train_loader:
    images, labels, disc_masks, clinical = batch
    logits = model(images, clinical)

    result = loss_fn(logits, labels, images, disc_masks, clinical)

    # Log all components
    print(f"Total: {result['total']:.4f}")
    print(f"Base: {result['base']:.4f}")
    print(f"DRI Penalty: {result['dri_penalty']:.4f}")

    result['total'].backward()
```

## Performance Characteristics

### Computational Cost

| Loss Function | Relative Cost | Memory | Notes |
|---------------|---------------|---------|-------|
| WeightedBCELoss | 1.0x | Low | Same as standard BCE |
| AsymmetricFocalLoss | 1.2x | Low | Extra focal term computation |
| AUCSurrogateLoss | 2.0x | Medium | Pairwise comparisons (N² complexity) |
| DRIRegularizer | 3.0x | High | Grad-CAM computation per image |
| CombinedLoss | Base + 3.0x | High | Base loss + DRI computation |

**Recommendations:**
- Use WeightedBCELoss or AsymmetricFocalLoss for fast training
- Use AUCSurrogateLoss for final fine-tuning
- Use DRIRegularizer sparingly (every 5-10 iterations)

### When to Use Each Loss

| Goal | Recommended Loss | Configuration |
|------|------------------|---------------|
| Handle class imbalance | WeightedBCELoss | Auto-compute weights |
| Reduce false negatives | AsymmetricFocalLoss | `gamma_pos=3.0, gamma_neg=0.5` |
| Optimize AUC | AUCSurrogateLoss | `margin=1.0` |
| Enforce attention | CombinedLoss | Any base + `lambda_dri=0.1` |
| General purpose | AsymmetricFocalLoss | `gamma_pos=2.0, gamma_neg=1.0` |

## Testing

All loss functions have comprehensive unit tests:

```bash
# Run all tests
cd /path/to/AcuVue
python3 -m pytest tests/unit/test_custom_losses.py -v

# Run specific test class
python3 -m pytest tests/unit/test_custom_losses.py::TestWeightedBCELoss -v

# Run with coverage
python3 -m pytest tests/unit/test_custom_losses.py --cov=src/training/custom_losses
```

**Test coverage:**
- 52 unit tests
- All loss functions tested
- Gradient flow verification
- Edge case handling (all positive/negative batches)
- Loss factory validation

## Files

```
src/training/
├── custom_losses.py           # All loss function implementations
├── loss_factory.py            # Loss factory for ARC integration
└── README_PHASE_E_WEEK3.md    # This file

tests/unit/
└── test_custom_losses.py      # 52 unit tests (all passing)
```

## Integration with Existing Pipeline

### Training Loop Integration

```python
from training.architecture_factory import build_model_from_spec
from training.loss_factory import build_loss_from_spec
from data.policy_augmentor import PolicyAugmentor

# Build model (Phase E Week 1)
model = build_model_from_spec(experiment["architecture_spec"])

# Build augmentation policy (Phase E Week 2)
augmentor = PolicyAugmentor(experiment["augmentation_policy"])

# Build loss function (Phase E Week 3)
loss_fn = build_loss_from_spec(
    experiment["loss_spec"],
    model=model,
    train_labels=train_labels
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels, disc_masks, clinical = batch

        # Apply augmentation
        images = augmentor.apply_to_batch(images)

        # Forward pass
        logits = model(images, clinical)

        # Compute loss
        if isinstance(loss_fn, CombinedLoss):
            result = loss_fn(logits, labels, images, disc_masks, clinical)
            loss = result['total']

            # Log components
            wandb.log({
                'loss/total': result['total'].item(),
                'loss/base': result['base'].item(),
                'loss/dri_penalty': result['dri_penalty'].item()
            })
        else:
            loss = loss_fn(logits, labels)
            wandb.log({'loss': loss.item()})

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Future Work (Phase E Week 4)

Next phase will implement **Cross-Dataset Curriculum Learning**:
1. Multi-dataset training (ORIGA + REFUGE + Drishti-GS)
2. Curriculum scheduling (easy → hard datasets)
3. Domain adaptation techniques
4. Cross-dataset evaluation

The loss functions from this phase will be used across all datasets.

## References

1. **Asymmetric Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **AUC Optimization**: Yan et al., "Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic", ICML 2003
3. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017

## License

Part of AcuVue Training Pipeline - ARC Phase E Week 3 Implementation
