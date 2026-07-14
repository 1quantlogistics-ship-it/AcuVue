#!/usr/bin/env python3
"""
Generate the dataset-summary chart used in the README.

Source of truth (committed artifacts, no downloads required):
  - models/production/dataset_metadata_v1.json  (RIM-ONE, hospital-based split)
  - data/processed/combined_v2/MANIFEST.md      (multi-dataset preprocessing set)

The chart shows the documented sizes of the three public source datasets and
highlights RIM-ONE, which is the only dataset behind the recorded classification
result (test AUC 0.937). It does NOT invent numbers: every value is read from,
or explicitly cited to, a committed file.

Output:
  docs/images/dataset-distribution.svg

Run from the repository root:
  python scripts/visualization/generate_dataset_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = REPO_ROOT / "models/production/dataset_metadata_v1.json"
OUT_PATH = REPO_ROOT / "docs/images/dataset-distribution.svg"

# Documented "samples used" for each public source dataset.
# Values are transcribed from data/processed/combined_v2/MANIFEST.md
# ("Source Datasets" section) and confirmed for RIM-ONE by the JSON loaded below.
SOURCE_DATASETS = [
    ("RIM-ONE r3", 485, "Spanish clinical centers"),
    ("REFUGE2", 400, "REFUGE 2018 training split"),
    ("G1020", 1020, "Chinese multi-center"),
]


def load_rimone_total() -> int:
    """Read the RIM-ONE total from the committed metadata to cross-check the constant."""
    if not METADATA.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {METADATA}. Run from the repository root."
        )
    with open(METADATA) as f:
        meta = json.load(f)
    return int(meta["statistics"]["total_images"])


def main() -> None:
    rimone_total = load_rimone_total()
    # Fail loudly if the cited constant drifts from the committed artifact.
    assert SOURCE_DATASETS[0][1] == rimone_total, (
        f"RIM-ONE constant ({SOURCE_DATASETS[0][1]}) does not match "
        f"dataset_metadata_v1.json ({rimone_total})."
    )

    labels = [f"{name}\n({note})" for name, _, note in SOURCE_DATASETS]
    counts = [count for _, count, _ in SOURCE_DATASETS]
    # RIM-ONE is the only dataset behind the recorded 0.937 result -> highlight it.
    colors = ["#1f6feb", "#8b949e", "#8b949e"]

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    bars = ax.barh(labels, counts, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("Documented samples used")
    ax.set_title(
        "AcuVue public source datasets\n"
        "RIM-ONE (blue) = the only dataset behind the recorded test AUC 0.937",
        fontsize=11,
    )
    ax.set_xlim(0, max(counts) * 1.18)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
