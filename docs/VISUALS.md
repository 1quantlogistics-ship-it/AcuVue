# Documentation visuals — provenance

Every image used in the README is listed here with its source, generation
method, and integrity notes. The goal is that a reviewer can trace each pixel
back to committed code or a committed artifact.

Two of the README diagrams (the **system overview** and the **routing
architecture**) are authored as **Mermaid** directly in the README and render
natively on GitHub; they have no image file and no generation script.

> **Environment note.** This documentation pass was prepared in an environment
> where the project code could not be executed (it targets a separate VM). As a
> result, the committed SVG charts were **authored by hand from the exact values
> in the cited committed artifacts**, and the companion scripts under
> `scripts/visualization/` were **authored but not executed here**. Running a
> script regenerates an equivalent chart (Matplotlib) at the same path from the
> same source values. The one raster figure that requires executing repository
> code (`segmentation-demo.png`) is deliberately **not committed** — regenerate
> it locally.

---

## `docs/images/cdr-schematic.svg`

| Field | Value |
|---|---|
| Type | Manually authored diagram (SVG) |
| Script | None (hand-authored) |
| Data | Illustrative geometry only |
| Synthetic or clinical | **Neither** — a labelled schematic |
| Checkpoint | None |
| Shows | Definition of vertical CDR: cup vertical diameter ÷ disc vertical diameter, healthy (≈0.3) vs glaucomatous (≈0.7) |
| Limitation | Not a model output, not a real image; CDR values are illustrative |

---

## `docs/images/dataset-distribution.svg`

| Field | Value |
|---|---|
| Type | Analytical bar chart (SVG) |
| Script | [`scripts/visualization/generate_dataset_summary.py`](../scripts/visualization/generate_dataset_summary.py) |
| Data source | [`models/production/dataset_metadata_v1.json`](../models/production/dataset_metadata_v1.json) (RIM-ONE = 485, cross-checked in-script) and [`data/processed/combined_v2/MANIFEST.md`](../data/processed/combined_v2/MANIFEST.md) "Source Datasets" section (REFUGE2 = 400, G1020 = 1,020) |
| Synthetic or clinical | Metadata only (no images) |
| Checkpoint | None |
| Shows | Documented "samples used" per public source dataset; RIM-ONE highlighted as the only dataset behind the recorded 0.937 result |
| Limitation | Bars are documented *samples-used* counts, not full published dataset sizes; only RIM-ONE underlies the recorded metric |

The committed SVG was hand-authored so it renders without running the script.
The script's assertion fails loudly if the RIM-ONE constant ever drifts from
`dataset_metadata_v1.json`.

---

## `docs/images/evaluation-summary.svg`

| Field | Value |
|---|---|
| Type | Analytical bar chart (SVG) |
| Script | [`scripts/visualization/generate_evaluation_summary.py`](../scripts/visualization/generate_evaluation_summary.py) |
| Data source | [`models/production/training_history_v1.json`](../models/production/training_history_v1.json) → `test_metrics`, `best_auc`, `best_epoch` |
| Synthetic or clinical | Recorded metrics from a RIM-ONE (public clinical dataset) experiment |
| Checkpoint | Refers to `glaucoma_efficientnet_b0_v1.pt` (**not committed**) |
| Shows | Held-out test AUC 0.937, specificity 0.917, accuracy 0.765, sensitivity 0.744; caption notes best val AUC 0.9875 @ epoch 29 |
| Limitation | Single run; small test set (98 images, 12 glaucoma). **No ROC curve** — per-sample scores are not stored, so one cannot be drawn honestly |

Exact values (transcribed from the artifact):

```json
"test_metrics": {
  "loss": 0.46263614296913147,
  "accuracy": 0.7653061223708871,
  "sensitivity": 0.7441860464250947,
  "specificity": 0.9166666659027777,
  "auc_roc": 0.937015503875969
},
"best_auc": 0.9875346260387812,
"best_epoch": 29
```

---

## `docs/images/segmentation-demo.png` (not committed — regenerate locally)

| Field | Value |
|---|---|
| Type | Raster demonstration figure |
| Script | [`scripts/visualization/generate_segmentation_demo.py`](../scripts/visualization/generate_segmentation_demo.py) |
| Data source | [`src/data/synthetic_fundus.py`](../src/data/synthetic_fundus.py) (procedural generator, `seed=42`) |
| Synthetic or clinical | **Synthetic** — labelled as such in the figure title |
| Checkpoint | None — masks are **generated ground truth**, not predictions |
| Shows | Healthy and glaucomatous synthetic fundus + disc mask + cup mask + overlay + vertical CDR |
| Limitation | Synthetic demonstration of the task and CDR only; not a clinical result and not a model prediction |

Not committed because it requires executing repository code. Generate with:

```bash
python scripts/visualization/generate_segmentation_demo.py --seed 42
```

---

## Integrity checklist

- [x] No fabricated ROC curves or confusion matrices.
- [x] No interpolated or invented metrics; every charted number is transcribed
      from a committed artifact and cited above.
- [x] Synthetic content is labelled synthetic; the CDR diagram is labelled a
      schematic.
- [x] No idealized mask is presented as a model prediction.
- [x] No private or improperly licensed medical images are used or committed.
- [x] Difficult framing (single-run, small test set, blocked reproducibility) is
      stated, not hidden.
