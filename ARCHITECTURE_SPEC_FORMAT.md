# Architecture Spec Format - Quick Reference

## For ARC's Architect Agent

This document defines the exact format for architecture specifications that can be built by the model factory.

---

## Complete Spec Template

```python
architecture_spec = {
    # REQUIRED FIELDS
    "backbone": str,              # See "Valid Backbones" below
    "fusion_type": str,           # See "Valid Fusion Types" below
    "clinical_dim": int,          # Number of clinical indicators (typically 4-8)

    # OPTIONAL FIELDS
    "head_config": {
        "hidden_dim": int,        # Fusion output dimension (default: 256)
        "dropout": float          # Dropout probability (default: 0.3)
    },
    "backbone_config": {
        "pretrained": bool,       # Load ImageNet weights (default: True)
        "freeze_backbone": bool   # Freeze backbone during training (default: False)
    },
    "fusion_config": {
        # Fusion-specific kwargs (see below)
    }
}
```

---

## Valid Backbones

| Backbone Name | Feature Dim | Parameters | Description |
|---------------|-------------|------------|-------------|
| `"efficientnet-b0"` | 1280 | 4.0M | Current baseline |
| `"efficientnet-b3"` | 1536 | 10.7M | **Recommended** upgrade |
| `"convnext-tiny"` | 768 | 27.8M | Modern CNN |
| `"convnext-small"` | 768 | 49.5M | Larger CNN |
| `"deit-tiny"` | 192 | 5.7M | Small ViT |
| `"deit-small"` | 384 | 21.7M | **Recommended** ViT |
| `"deit-base"` | 768 | 86.6M | Large ViT |

---

## Valid Fusion Types

| Fusion Type | Description | Recommended For |
|-------------|-------------|-----------------|
| `"film"` | Feature-wise Linear Modulation | Strong clinical indicators |
| `"cross_attention"` | Multi-head cross-attention | Spatial localization tasks |
| `"gated"` | Learnable per-sample gates | Variable clinical quality |
| `"late"` | Concatenation + MLP (baseline) | Baseline comparison |

---

## Fusion-Specific Configs

### FiLM (`fusion_type: "film"`)
```python
"fusion_config": {
    "hidden_dim": 128,         # Default: 128
    "use_global_pool": True    # Default: True
}
```

### CrossAttention (`fusion_type: "cross_attention"`)
```python
"fusion_config": {
    "num_heads": 4,            # Default: 4 (must divide hidden_dim)
    "hidden_dim": 64,          # Default: 64
    "dropout": 0.1             # Default: 0.1
}
```

### Gated (`fusion_type: "gated"`)
```python
"fusion_config": {
    "hidden_dim": 128,         # Default: 128
    "gate_activation": "sigmoid"  # "sigmoid" or "softmax"
}
```

### Late (`fusion_type: "late"`)
```python
"fusion_config": {
    "hidden_dim": 256,         # Default: 256
    "dropout": 0.3             # Default: 0.3
}
```

---

## Example Specs for Common Use Cases

### 1. Baseline (Current AcuVue)
```python
{
    "backbone": "efficientnet-b0",
    "fusion_type": "late",
    "clinical_dim": 4,
    "head_config": {"dropout": 0.3},
    "backbone_config": {"pretrained": True}
}
```

### 2. Recommended Upgrade (EfficientNet-B3 + FiLM)
```python
{
    "backbone": "efficientnet-b3",
    "fusion_type": "film",
    "clinical_dim": 4,
    "head_config": {"hidden_dim": 256, "dropout": 0.3},
    "backbone_config": {"pretrained": True, "freeze_backbone": False}
}
```

### 3. Vision Transformer (DeiT-Small + Cross-Attention)
```python
{
    "backbone": "deit-small",
    "fusion_type": "cross_attention",
    "clinical_dim": 4,
    "head_config": {"hidden_dim": 256, "dropout": 0.3},
    "backbone_config": {"pretrained": True},
    "fusion_config": {"num_heads": 4, "dropout": 0.1}
}
```

### 4. Large Capacity (ConvNeXt-Tiny + Gated)
```python
{
    "backbone": "convnext-tiny",
    "fusion_type": "gated",
    "clinical_dim": 4,
    "head_config": {"hidden_dim": 256, "dropout": 0.3},
    "backbone_config": {"pretrained": True},
    "fusion_config": {"gate_activation": "softmax"}
}
```

### 5. Transfer Learning (Frozen Backbone)
```python
{
    "backbone": "efficientnet-b3",
    "fusion_type": "late",
    "clinical_dim": 4,
    "head_config": {"hidden_dim": 256, "dropout": 0.5},
    "backbone_config": {
        "pretrained": True,
        "freeze_backbone": True  # Only train fusion + classifier
    }
}
```

---

## Validation Rules

The `validate_architecture_spec()` function checks:

1. **Required fields present**: `backbone`, `fusion_type`, `clinical_dim`
2. **Valid backbone**: Must be in supported list
3. **Valid fusion type**: Must be `film`, `cross_attention`, `gated`, or `late`
4. **Clinical dim > 0**: Must be positive integer
5. **Dropout in [0, 1)**: If specified
6. **Hidden dim > 0**: If specified

**Example:**
```python
from models.model_factory import validate_architecture_spec

spec = {...}

try:
    validate_architecture_spec(spec)
    # Proceed with building
except ValueError as e:
    # Reject proposal with reason
    print(f"Invalid spec: {e}")
```

---

## Build and Use

```python
from models.model_factory import build_model_from_spec
import torch

# 1. Validate spec
validate_architecture_spec(spec)

# 2. Build model
model = build_model_from_spec(spec, num_classes=2)

# 3. Use model
images = torch.randn(batch_size, 3, 224, 224)
clinical = torch.randn(batch_size, 4)

logits = model(images, clinical)  # [batch_size, 2]
```

---

## Parameter Counts by Architecture

| Backbone | Fusion | Total Params | Trainable (frozen=False) |
|----------|--------|--------------|--------------------------|
| efficientnet-b3 | film | 11.5M | 11.5M |
| efficientnet-b3 | cross_attention | 10.9M | 10.9M |
| efficientnet-b3 | gated | 11.0M | 11.0M |
| efficientnet-b3 | late | 11.2M | 11.2M |
| convnext-tiny | film | 28.2M | 28.2M |
| convnext-tiny | cross_attention | 27.9M | 27.9M |
| convnext-tiny | gated | 28.0M | 28.0M |
| convnext-tiny | late | 28.1M | 28.1M |
| deit-small | film | 21.9M | 21.9M |
| deit-small | cross_attention | 21.7M | 21.7M |
| deit-small | gated | 21.8M | 21.8M |
| deit-small | late | 21.8M | 21.8M |

---

## Tips for ARC Agents

### For Architect Agent:
- Start with `efficientnet-b3` + `film` (good balance)
- Use `cross_attention` when spatial localization is important
- Use `gated` when clinical indicator quality varies per sample
- Use Vision Transformers (`deit-small`) for small datasets

### For Parameter Scientist:
- `dropout`: 0.3-0.5 for small datasets, 0.1-0.3 for large
- `hidden_dim`: 256 is default, try 128/512 for capacity tuning
- `freeze_backbone=True`: For very small datasets (<1000 samples)
- `pretrained=True`: Always recommended unless training from scratch

### For Critic Agent:
- Reject if `clinical_dim` doesn't match dataset (typically 4-8)
- Reject if `dropout >= 1.0` or `dropout < 0.0`
- Warn if `freeze_backbone=True` with small learning rate (will barely train)
- Suggest `late` fusion as baseline if exploring new fusion types

### For Historian Agent:
Track architecture families:
```python
from models.model_factory import get_model_summary

summary = get_model_summary(model)
# Log: backbone_type, fusion_type, total_parameters, trainable_parameters
```

---

## Common Pitfalls

1. **Missing `timm` for DeiT**: Install with `pip install timm`
2. **`num_heads` not dividing `hidden_dim`**: Use compatible values (e.g., 4 heads with 64/128/256 hidden)
3. **Large models on small GPUs**: ConvNeXt-Small may OOM on 8GB GPU with batch_size > 8
4. **Frozen backbone with high dropout**: Leads to poor performance

---

**For full implementation details, see [README_PHASE_E_WEEK1.md](README_PHASE_E_WEEK1.md)**
