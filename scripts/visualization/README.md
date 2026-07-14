# Documentation visualization scripts

Reproducible scripts that (re)generate the analytical figures used in the root
`README.md`. Every script:

- runs from the repository root;
- reads only committed artifacts or the repository's own synthetic generator
  (no silent downloads, no absolute paths);
- uses a non-interactive Matplotlib backend and a fixed seed where sampling is
  involved;
- creates `docs/images/` if it is missing;
- fails loudly if a required input artifact is absent.

## Dependencies

```bash
pip install matplotlib numpy opencv-python   # subset of requirements.txt
```

`generate_dataset_summary.py` and `generate_evaluation_summary.py` need only
`matplotlib`. `generate_segmentation_demo.py` also imports the repository's
`src/data/synthetic_fundus.py` (numpy + opencv-python); it does **not** need
PyTorch, a GPU, clinical data, or trained weights.

## Commands

```bash
# Bar chart of documented source-dataset sizes (reads dataset_metadata_v1.json)
python scripts/visualization/generate_dataset_summary.py
# -> docs/images/dataset-distribution.svg

# Recorded RIM-ONE test metrics (reads training_history_v1.json)
python scripts/visualization/generate_evaluation_summary.py
# -> docs/images/evaluation-summary.svg

# Synthetic disc/cup + CDR demonstration (uses the procedural generator)
python scripts/visualization/generate_segmentation_demo.py --seed 42
# -> docs/images/segmentation-demo.png  (not committed; regenerate locally)
```

See [`docs/VISUALS.md`](../../docs/VISUALS.md) for the provenance of each
committed image, including which assets are hand-authored diagrams versus
script-generated charts.
