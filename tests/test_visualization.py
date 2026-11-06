"""Headless figures preserve the numerical data and source color lookup."""

import numpy as np
import pytest

from reossi.visualization import colorize_identifiers, create_color_map


def test_identifier_colors_preserve_order_clipping_and_invalid_black():
    colors = np.arange(4 * 3 * 3).reshape(3, 4, 3)
    columns = np.array([[0, 3, -1], [99, 1, -2]])
    rows = np.array([[0, 2, 1], [99, 0, 1]])
    expected = np.array(
        [
            [colors[0, 0], colors[2, 3], [0, 0, 0]],
            [colors[2, 3], colors[0, 1], [0, 0, 0]],
        ]
    )
    np.testing.assert_array_equal(colorize_identifiers(columns, rows, colors), expected)
    upstream = colorize_identifiers(columns, rows, colors, reject_negative=False)
    np.testing.assert_array_equal(upstream[1, 2], colors[1, -2])


def test_plot_cloud_saves_figure_without_modifying_or_exporting_points(
    tmp_path, monkeypatch
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reossi.visualization import plot_point_cloud

    points = np.array([[0.0, 1.0], [0.1, 0.2], [1.0, 2.0]])
    colors = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
    expected = points.copy()
    figure, _ = plot_point_cloud(points, colors, tmp_path / "view.png", show=False)
    np.testing.assert_array_equal(points, expected)
    assert (tmp_path / "view.png").is_file()
    assert not (tmp_path / "cloud.ply").exists()
    plt.close(figure)


def test_empty_point_cloud_is_a_valid_view():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reossi.visualization import plot_point_cloud

    figure, _ = plot_point_cloud(np.empty((3, 0)), np.empty((0, 3)), show=False)
    plt.close(figure)


def test_color_gradient_has_requested_shape_and_rgb_bounds():
    result = create_color_map(8, 5)
    assert result.shape == (5, 8, 3)
    assert result.min() >= 0 and result.max() <= 255
