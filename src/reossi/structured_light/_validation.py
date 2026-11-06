"""Shared shape checks for the source Gray-sequence contract."""

import numpy as np


def gray_sequence(images: np.ndarray) -> np.ndarray:
    """Validate without promoting dtype or scanning/copying full frame buffers."""
    images = np.asarray(images)
    if images.ndim != 3 or min(images.shape[1:]) < 1:
        raise ValueError(
            "A Gray sequence must have shape (frames,H,W), with H,W positive"
        )
    if len(images) < 14 or len(images) > 66 or (len(images) - 2) % 4:
        raise ValueError("Gray sequences require 4*bits+2 frames, with 3..16 bits")
    if images.dtype.kind not in "uif":
        raise ValueError("Gray captures must have real numeric intensity dtype")
    return images
