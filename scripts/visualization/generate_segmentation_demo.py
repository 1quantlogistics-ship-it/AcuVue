#!/usr/bin/env python3
"""
Generate a SYNTHETIC segmentation demonstration figure.

This uses the repository's own procedural generator
(src/data/synthetic_fundus.py) to produce a labelled example of the optic
disc / optic cup segmentation target and the cup-to-disc ratio (CDR) it
defines. The masks are *generated ground truth*, not model predictions.

    ┌──────────┬───────────┬──────────┬─────────────┐
    │ fundus   │ disc mask │ cup mask │ disc+cup    │
    │ (synth)  │ (GT)      │ (GT)     │ overlay+CDR │
    └──────────┴───────────┴──────────┴─────────────┘

Why synthetic: the trained checkpoints and clinical images are not committed
(see docs/VISUALS.md), so a real prediction figure cannot be produced from a
clean clone. This figure documents the task and data shape, clearly labelled
as a synthetic demonstration — not a clinical-performance result.

Requires: numpy, opencv-python, matplotlib (NOT torch).

Output:
  docs/images/segmentation-demo.png

Run from the repository root:
  python scripts/visualization/generate_segmentation_demo.py --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic_fundus import SyntheticFundusGenerator  # noqa: E402

OUT_PATH = REPO_ROOT / "docs/images/segmentation-demo.png"


def vertical_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
    """Vertical cup-to-disc ratio = cup vertical extent / disc vertical extent."""

    def v_extent(mask: np.ndarray) -> int:
        rows = np.where(mask.max(axis=1) > 0)[0]
        return int(rows.max() - rows.min() + 1) if rows.size else 0

    disc_v = v_extent(disc_mask)
    cup_v = v_extent(cup_mask)
    return cup_v / disc_v if disc_v else 0.0


def pick_sample(gen: SyntheticFundusGenerator, want_label: int, scan: int = 40):
    """Return the first generated sample whose label matches want_label."""
    for i in range(scan):
        sample = gen.generate_sample(i)
        if sample["metadata"]["label"] == want_label:
            return sample
    return gen.generate_sample(0)


def render_row(axes, sample: dict, title_prefix: str) -> None:
    image_bgr = sample["image"]
    image_rgb = image_bgr[:, :, ::-1]  # generator writes BGR
    disc = sample["disc_mask"]
    cup = sample["cup_mask"]
    cdr = vertical_cdr(disc, cup)
    label = sample["metadata"]["label_name"]

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"{title_prefix}: synthetic fundus\n(label: {label})", fontsize=9)

    axes[1].imshow(disc, cmap="gray")
    axes[1].set_title("optic disc mask (GT)", fontsize=9)

    axes[2].imshow(cup, cmap="gray")
    axes[2].set_title("optic cup mask (GT)", fontsize=9)

    overlay = image_rgb.copy()
    overlay[disc > 0] = (0.4 * overlay[disc > 0] + 0.6 * np.array([60, 160, 255])).astype(
        np.uint8
    )
    overlay[cup > 0] = (0.4 * overlay[cup > 0] + 0.6 * np.array([255, 90, 60])).astype(
        np.uint8
    )
    axes[3].imshow(overlay)
    axes[3].set_title(f"disc+cup overlay\nvertical CDR = {cdr:.2f}", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    gen = SyntheticFundusGenerator(num_samples=40, image_size=args.image_size, seed=args.seed)
    healthy = pick_sample(gen, want_label=0)
    glaucoma = pick_sample(gen, want_label=1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    render_row(axes[0], healthy, "Healthy")
    render_row(axes[1], glaucoma, "Glaucoma")
    fig.suptitle(
        "Synthetic demonstration — NOT a clinical-performance result\n"
        "Masks are generated ground truth from src/data/synthetic_fundus.py "
        f"(seed={args.seed}); larger vertical CDR indicates glaucoma.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
