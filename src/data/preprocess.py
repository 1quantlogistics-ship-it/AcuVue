"""
Preprocessing functions for retinal fundus images.

Includes illumination normalization and center cropping.
"""
import cv2
import numpy as np
from typing import Union


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the green channel
    of an RGB fundus image for illumination correction.

    The green channel typically has the best contrast for retinal structures.

    Args:
        img: Input image as numpy array. Can be RGB (H, W, 3) or grayscale (H, W).

    Returns:
        Image with normalized illumination (same shape as input).
    """
    if img.ndim == 3:
        g = img[:, :, 1]  # Extract green channel
    else:
        g = img

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)

    # Copy original image and replace green channel
    img_out = img.copy()
    if img.ndim == 3:
        img_out[:, :, 1] = g_eq
    else:
        img_out = g_eq

    return img_out


def center_crop(img: np.ndarray, margin_ratio: float = 0.1) -> np.ndarray:
    """
    Crop the center region of the image, removing a percentage margin from all sides.

    Useful for removing black borders or focusing on the central retinal region.

    Args:
        img: Input image as numpy array (H, W) or (H, W, C).
        margin_ratio: Fraction of image dimension to remove from each side (default: 0.1 = 10%).

    Returns:
        Center-cropped image.
    """
    h, w = img.shape[:2]
    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)
    y0, y1 = margin_h, h - margin_h
    x0, x1 = margin_w, w - margin_w
    return img[y0:y1, x0:x1]
