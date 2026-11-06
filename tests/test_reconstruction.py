"""Known identifier transforms and analytically projected 3D geometry."""

import cv2
import numpy as np
import pytest

from reossi.reconstruction.geometry import StereoTriangulator
from reossi.reconstruction.identifiers import fuse_identifier_maps, scale_identifiers


def test_identifier_transform_is_in_place_with_exact_sentinel_policy():
    identifiers = np.array([-2, -1, 0, 1, 2])
    result = scale_identifiers(identifiers, 3, 2)
    assert result is identifiers
    np.testing.assert_array_equal(result, [-6, -1, 0, 3, 6])


@pytest.mark.parametrize(
    "invalid_values,expected",
    [
        ((-1,), [-8, -1, -2, 1, 4]),
        ((), [-8, -5, -2, 1, 4]),
        ((-1, 0), [-8, -1, 0, 1, 4]),
    ],
)
def test_explicit_invalid_policies_keep_distinct_source_contracts(
    invalid_values, expected
):
    # Factor3, displacement0, step1; compare known affine coordinates directly.
    values = np.array([-2, -1, 0, 1, 2])
    scale_identifiers(values, 3, 0, invalid_values=invalid_values)
    np.testing.assert_array_equal(values, expected)


def test_fusion_selects_per_pixel_maximum_for_both_layer_axes():
    first = np.array([[1, -1], [4, 2]])
    second = np.array([[2, 3], [1, -1]])
    columns, rows = fuse_identifier_maps([first, second], [first.T, second.T])
    np.testing.assert_array_equal(columns, [[2, 3], [4, 2]])
    np.testing.assert_array_equal(rows, [[2, 4], [3, 2]])


@pytest.mark.parametrize("angle", [0.0, 0.1, -0.08])
def test_sine_rule_recovers_analytically_projected_points(angle):
    matrix = np.array([[100.0, 0, 20], [0, 105, 15], [0, 0, 1]])
    rotation, _ = cv2.Rodrigues(np.array([0.0, angle, 0.0]))
    translation = np.array([[-0.12], [0.01], [0.005]])
    points = np.array([[-0.1, -0.1, 1.0], [0.05, 0.08, 1.5], [0.13, -0.04, 2.0]])
    camera_points, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), matrix, np.zeros(5)
    )
    projector_points, _ = cv2.projectPoints(
        points, cv2.Rodrigues(rotation)[0], translation, matrix, np.zeros(5)
    )
    triangulator = StereoTriangulator(
        camera_matrix=matrix,
        camera_distortion=np.zeros(5),
        projector_matrix=matrix.copy(),
        projector_distortion=np.zeros(5),
        rotation=rotation,
        translation=translation,
    )
    legacy_points = triangulator.triangulate(
        camera_points[:, 0], projector_points[:, 0]
    )
    np.testing.assert_allclose(
        triangulator.to_camera_frame(legacy_points), points.T, rtol=1e-10, atol=1e-10
    )


def test_correspondences_keep_pixel_order_clipping_and_color_association():
    triangulator = StereoTriangulator(camera_size=(3, 2), projector_size=(8, 6))
    columns = np.array([[0, -1, 99], [1, -2, 4]])
    rows = np.array([[0, 1, 99], [5, 2, -1]])
    colors = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    camera, projector, selected = triangulator.collect_correspondences(
        columns, rows, colors
    )
    np.testing.assert_array_equal(camera, [[0, 0], [0, 1], [2, 0]])
    np.testing.assert_array_equal(projector, [[0, 0], [1, 5], [7, 5]])
    np.testing.assert_array_equal(
        selected, colors[[0, 1, 0], [0, 0, 2]].astype(float) / 255
    )


def test_box_filter_uses_strict_boundaries_and_shared_color_mask():
    triangulator = StereoTriangulator()
    points = np.array([[0, 0.5, -0.5, 0.1], [0, 0, 0, 0.2], [0.1, 0, 0, -0.3]])
    colors = np.arange(12).reshape(4, 3)
    kept_points, kept_colors = triangulator.filter_points(points, colors, 0.5)
    np.testing.assert_array_equal(kept_points, points[:, [0, 3]])
    np.testing.assert_array_equal(kept_colors, colors[[0, 3]])
