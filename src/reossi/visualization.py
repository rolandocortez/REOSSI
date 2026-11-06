# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND MIT
# Copyright 2026 Rolando Cortez Garcia
# Copyright (c) 2023 Guillaume Charron
# Original additions and inherited portions: see docs/licensing.md.
"""Optional scientific views; plotting never filters or modifies saved data.

The identifier color interpolation follows Charron's reference convention.
Matplotlib is imported only by plotting functions. Point export and numerical
filtering belong to reconstruction, independently of visualization.
"""

from pathlib import Path

import numpy as np


def require_plotting():
    """Load optional plotting dependencies with an actionable installation hint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Visualization requires Matplotlib; install it with pip install 'reossi[plots]'"
        ) from error
    return plt


def _gradient_channel(upper_left, upper_right, lower_left, lower_right, width, height):
    """Interpolate one color channel with the reference corner ordering."""
    channel = np.linspace(
        np.linspace(lower_left, lower_right, width),
        np.linspace(upper_left, upper_right, width),
        height,
        dtype=int,
    )
    return channel[:, :, None]


def create_color_map(width, height):
    """Return the reference integer RGB map with shape ``(height, width, 3)``."""
    red = _gradient_channel(0, 255, 0, 255, width, height)
    green = _gradient_channel(0, 0, 255, 0, width, height)
    blue = _gradient_channel(255, 255, 0, 0, width, height)
    return np.concatenate([red, green, blue], axis=2)


def colorize_identifiers(column_ids, row_ids, color_map, *, reject_negative=True):
    """Return RGB identifiers with the reference lookup, clipping and sentinel.

    Inputs have camera shape ``(H,W)`` and integer dtype; color_map is
    projector-shaped ``(h,w,3)``. The local rule rejects <= -1; the optional
    upstream rule rejects exactly -1 and retains its negative-index behavior.
    """
    column_ids, row_ids = np.asarray(column_ids), np.asarray(row_ids)
    if column_ids.shape != row_ids.shape or column_ids.ndim != 2:
        raise ValueError("Identifier maps must have matching (H,W) shapes")
    if column_ids.dtype.kind not in "iu" or row_ids.dtype.kind not in "iu":
        raise ValueError("Color lookup requires integer identifier maps")
    if color_map.ndim != 3 or color_map.shape[2] != 3:
        raise ValueError("color_map must have shape (height,width,3)")
    width = color_map.shape[1]
    height = color_map.shape[0]
    invalid = (
        (column_ids <= -1) | (row_ids <= -1)
        if reject_negative
        else (column_ids == -1) | (row_ids == -1)
    )
    columns = np.minimum(column_ids, width - 1).copy()
    rows = np.minimum(row_ids, height - 1).copy()
    columns[invalid], rows[invalid] = 0, 0
    result = color_map[rows, columns].copy()
    result[invalid] = 0
    return result


def _save_figure(figure, save_path):
    if save_path is not None:
        path = Path(save_path)
        if path.exists():
            raise FileExistsError(f"Figure already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", dpi=180)


def plot_decoded_identifiers(
    column_ids, row_ids, color_map, save_path=None, *, reject_negative=True, show=True
):
    """Plot camera-indexed projector IDs, returning the figure and axes."""
    plt = require_plotting()

    figure, axes = plt.subplots(figsize=(9, 6))
    axes.imshow(
        colorize_identifiers(
            column_ids, row_ids, color_map, reject_negative=reject_negative
        )
    )
    axes.set_title("Decoded projector coordinates")
    axes.set_axis_off()
    _save_figure(figure, save_path)
    if show:
        plt.show()
    return figure, axes


def plot_point_cloud(
    points,
    colors,
    save_path=None,
    *,
    length_unit="m",
    coordinate_frame="camera",
    show=True,
):
    """Plot finite ``(3,N)`` points and ``(N,3)`` RGB colors without filtering.

    Saving writes a figure only; point export is a separate workflow operation.
    The returned figure can be closed explicitly by headless callers.
    """
    plt = require_plotting()

    points, colors = np.asarray(points), np.asarray(colors)
    if points.ndim != 2 or points.shape[0] != 3 or colors.shape != (points.shape[1], 3):
        raise ValueError("Expected points (3,N) and aligned RGB colors (N,3)")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("Point-cloud view requires finite points and colors")
    if np.any((colors < 0) | (colors > 1)):
        raise ValueError("RGB colors must lie in 0..1")
    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(*points, c=colors, s=2)
    for name, setter in [
        ("X", axes.set_xlabel),
        ("Y", axes.set_ylabel),
        ("Z", axes.set_zlabel),
    ]:
        setter(f"{name} ({length_unit})")
    axes.set_title(
        f"Reconstruction in {coordinate_frame} coordinates · {points.shape[1]} points"
    )
    if points.shape[1]:
        extent = np.ptp(points, axis=1)
        axes.set_box_aspect(np.maximum(extent, max(float(extent.max()), 1e-9) * 1e-3))
    _save_figure(figure, save_path)
    if show:
        plt.show()
    return figure, axes
