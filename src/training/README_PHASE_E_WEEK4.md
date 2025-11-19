# Phase E Week 4: Cross-Dataset Curriculum Learning

**ARC Training Improvements DevNote v1.0 - Feature 4**
**Dev 2 Implementation**

## Overview

This phase implements curriculum learning across multiple fundus datasets (REFUGE, ORIGA, Drishti-GS) to improve model generalization and robustness. Instead of training on a single dataset, models progressively learn from easier to harder datasets, similar to how students progress through increasingly difficult material.

**Key Benefits:**
1. **Better Generalization**: Models learn from diverse data distributions
2. **Improved Robustness**: Exposure to varied difficulty levels
3. **Reduced Overfitting**: Progressive difficulty prevents memorization
4. **Domain Adaptation**: Models learn domain-invariant features

## Architecture

### Core Components

#### 1. MultiDatasetManager (`src/data/multi_dataset_manager.py`)
**Purpose**: Manage multiple fundus datasets with automatic difficulty scoring

**Classes:**
- **DatasetDifficultyScorer**: Scores datasets by difficulty (0.0 = easiest, 1.0 = hardest)
- **MultiDatasetManager**: Loads, manages, and combines datasets for curriculum learning

**Difficulty Factors:**
- **Class Imbalance** (40% weight): Higher imbalance = harder
- **Dataset Size** (30% weight): Smaller dataset = harder to generalize
- **Image Quality** (30% weight): Lower quality = harder

**Usage:**
```python
from data.multi_dataset_manager import MultiDatasetManager

# Create manager
manager = MultiDatasetManager(
    datasets=["REFUGE", "ORIGA", "Drishti"],
    data_root="data/processed"
)

# Load datasets
manager.load_datasets({
    "REFUGE": refuge_dataset,
    "ORIGA": origa_dataset,
    "Drishti": drishti_dataset
})

# Get difficulty ranking (easiest → hardest)
ranking = manager.get_difficulty_ranking()
# Returns: [("REFUGE", 0.2), ("ORIGA", 0.5), ("Drishti", 0.7)]

# Create curriculum data loader
loader = manager.get_curriculum_loader(
    stage_datasets=["REFUGE", "ORIGA"],
    batch_size=32
)
```

**Key Methods:**
- `get_difficulty_ranking()`: Returns datasets sorted by difficulty
- `get_curriculum_loader(stage_datasets, ...)`: Creates DataLoader for curriculum stage
- `get_stage_info(stage_datasets)`: Returns stage statistics

#### 2. CurriculumScheduler (`src/training/curriculum_scheduler.py`)
**Purpose**: Schedule curriculum progression across training epochs

**Strategies:**
- **pure_sequential**: Train on one dataset at a time (REFUGE → ORIGA → Drishti)
- **gradual_mixing**: Gradually add harder datasets ([REFUGE] → [REFUGE, ORIGA] → [REFUGE, ORIGA, Drishti])
- **adaptive**: Adjust based on validation performance (future work)
- **reverse**: Start with hardest (anti-curriculum for robustness testing)

**Usage:**
```python
from training.curriculum_scheduler import CurriculumScheduler, create_automatic_curriculum

# Manual curriculum
stages = [
    {"datasets": ["REFUGE"], "epochs": 5},
    {"datasets": ["REFUGE", "ORIGA"], "epochs": 5},
    {"datasets": ["REFUGE", "ORIGA", "Drishti"], "epochs": 10}
]

scheduler = CurriculumScheduler(
    strategy="gradual_mixing",
    stages=stages,
    difficulty_ranking=ranking
)

# Automatic curriculum generation
scheduler = create_automatic_curriculum(
    difficulty_ranking=ranking,
    strategy="gradual_mixing",
    epochs_per_stage=5
)

# During training
for epoch in range(total_epochs):
    stage_idx, stage = scheduler.get_current_stage(epoch)
    datasets_for_epoch = stage.datasets

    # Load data for current stage
    loader = manager.get_curriculum_loader(datasets_for_epoch, batch_size=32)
```

**Key Methods:**
- `get_current_stage(epoch)`: Returns (stage_idx, CurriculumStage) for epoch
- `should_transition(epoch)`: Returns True if transitioning to new stage
- `get_total_epochs()`: Total epochs across all stages
- `get_curriculum_summary()`: Complete curriculum information

#### 3. Domain Adaptation (`src/training/domain_adaptation.py`)
**Purpose**: Learn domain-invariant features across datasets

**Components:**
- **GradientReversalLayer**: Reverses gradients for adversarial training
- **DomainClassifier**: Discriminates between datasets
- **DomainAdversarialLoss**: Combined task + domain adversarial loss

**How It Works:**
1. Feature extractor learns task (glaucoma detection)
2. Domain classifier tries to identify which dataset an image came from
3. Gradient reversal makes feature extractor confuse the domain classifier
4. Result: Features that work well across all datasets

**Usage:**
```python
from training.domain_adaptation import GradientReversalLayer, DomainClassifier, DomainAdversarialLoss

# Add to model architecture
class GlaucomaModelWithDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...  # Feature extractor
        self.classifier = ...  # Task classifier

        # Domain adaptation components
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_classifier = DomainClassifier(
            feature_dim=512,
            num_domains=3  # REFUGE, ORIGA, Drishti
        )

    def forward(self, x, return_domain_logits=False):
        features = self.encoder(x)
        task_logits = self.classifier(features)

        if return_domain_logits:
            reversed_features = self.grl(features)
            domain_logits = self.domain_classifier(reversed_features)
            return task_logits, domain_logits

        return task_logits

# Training with domain adaptation
task_loss_fn = nn.BCEWithLogitsLoss()
da_loss = DomainAdversarialLoss(
    task_loss_fn=task_loss_fn,
    lambda_domain=0.1,  # Weight for domain loss
    schedule_lambda=True  # Gradually increase lambda
)

for batch in train_loader:
    images, labels, domain_labels = batch

    task_logits, domain_logits = model(images, return_domain_logits=True)

    result = da_loss(
        task_logits, labels,
        domain_logits, domain_labels,
        epoch=current_epoch,
        max_epochs=total_epochs
    )

    loss = result['total']  # task_loss + lambda * domain_loss
    loss.backward()
```

**Lambda Scheduling:**
Lambda increases from 0 to `lambda_domain` over training using formula:
```
lambda(p) = lambda_domain * (2 / (1 + exp(-10*p)) - 1)
where p = epoch / max_epochs
```

This ensures model first learns task, then gradually enforces domain-invariance.

#### 4. Curriculum Factory (`src/training/curriculum_factory.py`)
**Purpose**: Build curriculum schedules from ARC specifications

**Usage:**
```python
from training.curriculum_factory import build_curriculum_from_spec

# ARC proposes curriculum spec
spec = {
    "strategy": "gradual_mixing",
    "datasets": ["REFUGE", "ORIGA", "Drishti"],
    "epochs_per_stage": 5,
    "domain_adaptation": True,
    "lambda_domain": 0.1,
    "grl_lambda": 1.0
}

# Build scheduler and config
scheduler, config = build_curriculum_from_spec(
    spec,
    dataset_manager=manager,
    model=model  # Required if domain_adaptation=True
)

# Get stage configuration for current epoch
stage_config = get_stage_config(scheduler, config, epoch=7)
# {
#     "stage_index": 1,
#     "datasets": ["REFUGE", "ORIGA"],
#     "difficulty": 0.35,
#     "is_transition": False,
#     "domain_adaptation": True,
#     "lambda_domain": 0.1,
#     ...
# }
```

**Spec Validation:**
```python
from training.curriculum_factory import validate_curriculum_spec

spec = {
    "strategy": "custom",
    "datasets": ["REFUGE", "ORIGA"],
    "stages": [
        {"datasets": ["REFUGE"], "epochs": 10},
        {"datasets": ["ORIGA"], "epochs": 10}
    ]
}

try:
    validate_curriculum_spec(spec)  # Raises ValueError if invalid
except ValueError as e:
    print(f"Invalid spec: {e}")
```

#### 5. Cross-Dataset Evaluator (`src/evaluation/cross_dataset_evaluator.py`)
**Purpose**: Evaluate model generalization across datasets

**Usage:**
```python
from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator

# Create evaluator
evaluator = CrossDatasetEvaluator(
    model=trained_model,
    datasets={
        "REFUGE": refuge_test_loader,
        "ORIGA": origa_test_loader,
        "Drishti": drishti_test_loader
    },
    device="cuda"
)

# Evaluate all datasets
results = evaluator.evaluate_all()

# Per-dataset metrics
for dataset_name, metrics in results['per_dataset'].items():
    print(f"{dataset_name}: AUC={metrics['auc']:.3f}, Sensitivity={metrics['sensitivity']:.3f}")

# Overall statistics
print(f"Mean AUC: {results['overall']['mean_auc']:.3f}")
print(f"Std AUC: {results['overall']['std_auc']:.3f}")

# Domain shift analysis
shift = evaluator.compute_domain_shift("REFUGE", "Drishti")
print(f"REFUGE → Drishti shift: {shift:.3f}")
```

**Metrics Computed:**
- AUC (Area Under ROC Curve)
- PR-AUC (Precision-Recall AUC)
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Accuracy

## Complete Training Example

Here's how all components work together:

```python
from data.multi_dataset_manager import MultiDatasetManager
from training.curriculum_factory import build_curriculum_from_spec, get_stage_config
from training.domain_adaptation import DomainAdversarialLoss, compute_domain_labels
from evaluation.cross_dataset_evaluator import CrossDatasetEvaluator

# 1. Setup datasets
manager = MultiDatasetManager(
    datasets=["REFUGE", "ORIGA", "Drishti"],
    data_root="data/processed"
)
manager.load_datasets({
    "REFUGE": refuge_train_dataset,
    "ORIGA": origa_train_dataset,
    "Drishti": drishti_train_dataset
})

# 2. Build curriculum from ARC spec
spec = {
    "strategy": "gradual_mixing",
    "datasets": ["REFUGE", "ORIGA", "Drishti"],
    "epochs_per_stage": 5,
    "domain_adaptation": True,
    "lambda_domain": 0.1
}

scheduler, config = build_curriculum_from_spec(spec, manager, model)
total_epochs = scheduler.get_total_epochs()

# 3. Setup loss function
task_loss_fn = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=0.5)
da_loss = DomainAdversarialLoss(
    task_loss_fn=task_loss_fn,
    lambda_domain=config['lambda_domain'],
    schedule_lambda=True
)

# 4. Training loop
for epoch in range(total_epochs):
    # Get current stage configuration
    stage_config = get_stage_config(scheduler, config, epoch)

    # Log stage transition
    if stage_config['is_transition']:
        print(f"Transitioning to Stage {stage_config['stage_index']}")
        print(f"Datasets: {stage_config['datasets']}")

    # Create data loader for current stage
    train_loader = manager.get_curriculum_loader(
        stage_datasets=stage_config['datasets'],
        batch_size=32,
        shuffle=True
    )

    # Training epoch
    model.train()
    for batch in train_loader:
        images, labels, disc_masks, batch_datasets = batch

        # Forward pass with domain adaptation
        if config['domain_adaptation']:
            task_logits, domain_logits = model(images, return_domain_logits=True)

            # Convert dataset names to domain labels
            domain_labels = compute_domain_labels(
                batch_datasets,
                config['dataset_to_id']
            )

            # Compute combined loss
            result = da_loss(
                task_logits, labels,
                domain_logits, domain_labels,
                epoch=epoch, max_epochs=total_epochs
            )

            loss = result['total']

            # Log components
            wandb.log({
                'loss/task': result['task'].item(),
                'loss/domain': result['domain'].item(),
                'loss/lambda': result['lambda']
            })
        else:
            task_logits = model(images)
            loss = task_loss_fn(task_logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. Evaluate across datasets
evaluator = CrossDatasetEvaluator(
    model=model,
    datasets={
        "REFUGE": refuge_test_loader,
        "ORIGA": origa_test_loader,
        "Drishti": drishti_test_loader
    }
)

results = evaluator.evaluate_all()
print(f"Mean AUC across datasets: {results['overall']['mean_auc']:.3f}")
```

## Curriculum Strategies Comparison

| Strategy | Description | Use Case | Pros | Cons |
|----------|-------------|----------|------|------|
| **pure_sequential** | One dataset per stage | Maximum control over progression | Clear difficulty ordering | Longer training |
| **gradual_mixing** | Gradually add datasets | Smooth difficulty increase | Best generalization | May plateau early |
| **adaptive** | Adjust based on performance | Dynamic difficulty | Optimal pacing | Complex implementation |
| **reverse** | Start with hardest | Robustness testing | Tests anti-curriculum | May hurt final performance |

**Recommendation**: Start with **gradual_mixing** for best generalization.

## Expected Difficulty Ranking

Based on dataset characteristics:

| Dataset | Size | Balance | Quality | Difficulty Score | Rank |
|---------|------|---------|---------|------------------|------|
| **REFUGE** | 1200 | High (50/50) | High | **0.2** (Easiest) | 1 |
| **ORIGA** | 650 | Medium (168/482) | Medium | **0.5** (Medium) | 2 |
| **Drishti-GS** | 101 | Low (70/31) | Medium | **0.7** (Hardest) | 3 |

## Performance Expectations

### Without Curriculum Learning
```
REFUGE:  AUC = 0.92
ORIGA:   AUC = 0.78
Drishti: AUC = 0.72
Mean:    AUC = 0.81
```

### With Curriculum Learning (Gradual Mixing)
```
REFUGE:  AUC = 0.93  (+0.01)
ORIGA:   AUC = 0.83  (+0.05)
Drishti: AUC = 0.79  (+0.07)
Mean:    AUC = 0.85  (+0.04)
```

**Key Improvement**: Better performance on harder datasets (ORIGA, Drishti).

### With Curriculum + Domain Adaptation
```
REFUGE:  AUC = 0.94  (+0.02)
ORIGA:   AUC = 0.86  (+0.08)
Drishti: AUC = 0.82  (+0.10)
Mean:    AUC = 0.87  (+0.06)
```

**Key Improvement**: Even better cross-dataset generalization.

## Integration with Previous Phases

This phase integrates seamlessly with:

### Phase E Week 1: Architecture Factory
```python
from training.architecture_factory import build_model_from_spec

# Build model with domain adaptation support
arch_spec = {
    "model_type": "hybrid_attention",
    "clinical_branch": True,
    "attention_type": "cbam"
}
model = build_model_from_spec(arch_spec)

# Add domain adaptation components
from training.domain_adaptation import GradientReversalLayer, DomainClassifier
model.grl = GradientReversalLayer(lambda_=1.0)
model.domain_classifier = DomainClassifier(feature_dim=512, num_domains=3)
```

### Phase E Week 2: Augmentation Policy
```python
from data.policy_augmentor import PolicyAugmentor

# Apply augmentation to curriculum batches
augmentor = PolicyAugmentor(augmentation_policy)

for batch in curriculum_loader:
    images = augmentor.apply_to_batch(images)
    # Continue training...
```

### Phase E Week 3: Loss Functions
```python
from training.loss_factory import build_loss_from_spec
from training.domain_adaptation import DomainAdversarialLoss

# Build base loss
base_loss = build_loss_from_spec({
    "loss_type": "asymmetric_focal",
    "gamma_pos": 2.0,
    "gamma_neg": 0.5
})

# Wrap with domain adaptation
da_loss = DomainAdversarialLoss(
    task_loss_fn=base_loss,
    lambda_domain=0.1
)
```

## Files

```
src/
├── data/
│   └── multi_dataset_manager.py       # Dataset management & difficulty scoring
├── training/
│   ├── curriculum_scheduler.py        # Curriculum progression scheduling
│   ├── curriculum_factory.py          # Factory for ARC integration
│   ├── domain_adaptation.py           # Domain-adversarial training
│   └── README_PHASE_E_WEEK4.md        # This file
└── evaluation/
    └── cross_dataset_evaluator.py     # Cross-dataset evaluation

tests/unit/
└── test_curriculum_learning.py        # Unit tests (to be implemented)
```

## Testing

```bash
# Run individual demos
python3 src/data/multi_dataset_manager.py
python3 src/training/curriculum_scheduler.py
python3 src/training/curriculum_factory.py
python3 src/training/domain_adaptation.py
python3 src/evaluation/cross_dataset_evaluator.py

# Run unit tests (after implementation)
python3 -m pytest tests/unit/test_curriculum_learning.py -v
```

## Future Work

1. **Adaptive Curriculum**: Adjust difficulty based on validation performance
2. **Meta-Learning**: Learn optimal curriculum schedules across experiments
3. **Active Learning**: Select most informative samples from each dataset
4. **Multi-Task Learning**: Joint training with segmentation + classification
5. **Self-Paced Learning**: Let model choose its own curriculum

## References

1. **Curriculum Learning**: Bengio et al., "Curriculum Learning", ICML 2009
2. **Domain Adaptation**: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
3. **Anti-Curriculum**: Hacohen & Weinshall, "On The Power of Curriculum Learning in Training Deep Networks", ICML 2019
4. **Self-Paced Learning**: Kumar et al., "Self-Paced Learning for Latent Variable Models", NeurIPS 2010

## License

Part of AcuVue Training Pipeline - ARC Phase E Week 4 Implementation
