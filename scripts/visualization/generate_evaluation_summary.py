#!/usr/bin/env python3
"""
Generate the evaluation-summary chart used in the README.

Source of truth (committed artifact, no downloads required):
  - models/production/training_history_v1.json

This is the recorded result for the production glaucoma *classifier*
(EfficientNet-B0) on RIM-ONE with a hospital-based (institution-level) split.
The chart plots the held-out test metrics exactly as recorded. It deliberately
does NOT draw a ROC curve: the artifact stores summary metrics, not per-sample
scores, so a ROC curve cannot be reproduced honestly.

Output:
  docs/images/evaluation-summary.svg

Run from the repository root:
  python scripts/visualization/generate_evaluation_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY = REPO_ROOT / "models/production/training_history_v1.json"
OUT_PATH = REPO_ROOT / "docs/images/evaluation-summary.svg"

METRIC_ORDER = [
    ("auc_roc", "Test AUC"),
    ("specificity", "Specificity"),
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity"),
]


def main() -> None:
    if not HISTORY.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {HISTORY}. Run from the repository root."
        )
    with open(HISTORY) as f:
        history = json.load(f)

    test = history["test_metrics"]
    best_auc = history["best_auc"]
    best_epoch = history["best_epoch"]

    labels = [label for key, label in METRIC_ORDER]
    values = [test[key] for key, _ in METRIC_ORDER]
    colors = ["#1f6feb" if key == "auc_roc" else "#57606a" for key, _ in METRIC_ORDER]

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score (held-out test set, RIM-ONE)")
    ax.set_title(
        "Recorded classification result — EfficientNet-B0, RIM-ONE\n"
        f"Hospital-based split | best val AUC {best_auc:.3f} @ epoch {best_epoch}",
        fontsize=11,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
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
