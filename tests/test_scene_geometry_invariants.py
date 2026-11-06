"""Analytic and rigid-world invariants for the small geometric forward model.

The tests establish covered geometry contracts, not photometric realism or
physical scanner accuracy. All scenes use two triangles and at most16x12 rays.
"""

import cv2
import numpy as np
import pytest

from reossi.simulation.geometry import (
    PinholeDevice,
    TriangleMesh,
    make_device,
    make_surface,
    relative_calibration,
)
from reossi.simulation.raytrace import intersect_mesh, scene_correspondences


def _device(center, rotation_vector, matrix):
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float))
    center = np.asarray(center, dtype=float)
    return PinholeDevice((16, 12), matrix, rotation, -rotation @ center)


def _rig():
    matrix = np.array([[18.0, 0, 7.5], [0, 17.0, 5.5], [0, 0, 1]])
    camera = _device([0.1, -0.05, 0.02], [0.03, -0.06, 0.025], matrix)
    projector = _device([0.28, 0.015, 0.04], [-0.015, 0.04, -0.02], matrix)
    # The surface is deliberately smaller than the camera field, so NaN
    # background and visibility masks participate in the invariant check.
    mesh = make_surface(width=1.1, height=0.75, depth=2.0)
    return mesh, camera, projector


def _transform_device(device, rotation, translation):
    # X'=QX+t implies R'=RQ.T and T'=T-R't for world-to-device extrinsics.
    new_rotation = device.rotation @ rotation.T
    new_translation = device.translation - new_rotation @ translation
    return PinholeDevice(
        device.image_size, device.matrix.copy(), new_rotation, new_translation
    )


@pytest.mark.parametrize("seed", [7, 23, 81])
def test_world_rigid_transform_preserves_correspondences_and_relative_calibration(seed):
    mesh, camera, projector = _rig()
    original = scene_correspondences(mesh, camera, projector)
    assert original.illuminated.any()
    assert original.camera_visible.any() and not original.camera_visible.all()
    rng = np.random.default_rng(seed)
    world_rotation, _ = cv2.Rodrigues(rng.uniform(-0.7, 0.7, 3))
    world_translation = rng.uniform(-0.5, 0.5, 3)
    moved_mesh = TriangleMesh(
        mesh.vertices @ world_rotation.T + world_translation, mesh.faces.copy()
    )
    moved_camera = _transform_device(camera, world_rotation, world_translation)
    moved_projector = _transform_device(projector, world_rotation, world_translation)
    moved = scene_correspondences(moved_mesh, moved_camera, moved_projector)

    np.testing.assert_array_equal(moved.camera_visible, original.camera_visible)
    np.testing.assert_array_equal(moved.illuminated, original.illuminated)
    np.testing.assert_allclose(
        moved.projector_pixels,
        original.projector_pixels,
        atol=2e-12,
        rtol=2e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        moved.points_camera,
        original.points_camera,
        atol=2e-12,
        rtol=2e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        moved.points_world,
        original.points_world @ world_rotation.T + world_translation,
        atol=2e-12,
        rtol=2e-12,
        equal_nan=True,
    )
    original_rotation, original_translation = relative_calibration(camera, projector)
    moved_rotation, moved_translation = relative_calibration(
        moved_camera, moved_projector
    )
    np.testing.assert_allclose(
        moved_rotation, original_rotation, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        moved_translation, original_translation, atol=1e-14, rtol=1e-14
    )


@pytest.mark.parametrize("tilt", [-0.25, 0.0, 0.35])
def test_tilted_plane_matches_closed_form_ray_plane_solution(tilt):
    matrix = np.array([[18.0, 0, 7.5], [0, 17.0, 5.5], [0, 0, 1]])
    center = np.array([0.1, -0.15, 0.05])
    camera = _device(center, [0.02, 0.05, -0.01], matrix)
    projector = _device([0.3, 0.02, 0.0], [-0.015, -0.04, 0.025], matrix)
    depth = 2.0
    mesh = make_surface(width=4, height=3, depth=depth, tilt=tilt)

    # Compute rays directly from scalar intrinsics, independently of rays().
    u, v = np.meshgrid(np.arange(16), np.arange(12))
    camera_directions = np.stack(
        [(u - 7.5) / 18.0, (v - 5.5) / 17.0, np.ones_like(u)], axis=-1
    )
    world_directions = camera_directions @ camera.rotation
    world_directions /= np.linalg.norm(world_directions, axis=-1, keepdims=True)
    # Plane equation: (-tilt,0,1).X=depth.
    normal = np.array([-tilt, 0.0, 1.0])
    distances = (depth - center @ normal) / (world_directions @ normal)
    expected_world = center + distances[..., None] * world_directions
    expected_camera = expected_world @ camera.rotation.T + camera.translation
    projector_points = expected_world @ projector.rotation.T + projector.translation
    expected_u = 18.0 * projector_points[..., 0] / projector_points[..., 2] + 7.5
    expected_v = 17.0 * projector_points[..., 1] / projector_points[..., 2] + 5.5
    expected_projector = np.stack([expected_u, expected_v], axis=-1)
    expected_illuminated = (
        (projector_points[..., 2] > 0)
        & (expected_u >= -0.5)
        & (expected_u < 15.5)
        & (expected_v >= -0.5)
        & (expected_v < 11.5)
    )

    actual = scene_correspondences(mesh, camera, projector)
    assert actual.camera_visible.all()
    np.testing.assert_allclose(
        actual.points_world, expected_world, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        actual.points_camera, expected_camera, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        actual.projector_pixels, expected_projector, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_array_equal(actual.illuminated, expected_illuminated)


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_scene_correspondences_are_invariant_to_length_unit_rescaling(scale):
    # The baseline gives0.6pixel disparity, deliberately away from half-pixel
    # viewport ties whose binary classification is numerically ill-conditioned.
    plane = make_surface(width=4, height=3, depth=2)
    original = scene_correspondences(
        plane, make_device((8, 5)), make_device((8, 5), center=(0.15, 0, 0))
    )
    scaled_mesh = TriangleMesh(plane.vertices * scale, plane.faces.copy())
    scaled = scene_correspondences(
        scaled_mesh,
        make_device((8, 5)),
        make_device((8, 5), center=(0.15 * scale, 0, 0)),
    )
    assert int(original.camera_visible.sum()) == 40
    assert int(original.illuminated.sum()) == 35
    np.testing.assert_array_equal(scaled.camera_visible, original.camera_visible)
    np.testing.assert_array_equal(scaled.illuminated, original.illuminated)
    np.testing.assert_allclose(
        scaled.points_camera / scale, original.points_camera, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        scaled.projector_pixels, original.projector_pixels, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_small_valid_triangle_is_not_rejected_by_an_absolute_area_threshold(scale):
    mesh = TriangleMesh(
        scale * np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]),
        np.array([[0, 1, 2]]),
    )
    hits = intersect_mesh(
        mesh, scale * np.array([[0.25, 0.25, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    )
    assert hits.face_indices[0] == 0
    assert hits.distances[0] / scale == pytest.approx(2)


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
@pytest.mark.parametrize(
    "vertices",
    [
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [1, 1, 0], [2, 2, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 1e-18, 0]],
    ],
)
def test_triangle_degeneracy_uses_relative_shape_not_world_units(scale, vertices):
    with pytest.raises(ValueError, match="degenerate"):
        TriangleMesh(scale * np.array(vertices, dtype=float), np.array([[0, 1, 2]]))


def test_standalone_minimum_ray_distance_is_still_an_explicit_world_unit_value():
    mesh = TriangleMesh(
        1e-6 * np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]),
        np.array([[0, 1, 2]]),
    )
    origins = np.array([[0.25e-6, 0.25e-6, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0]])
    accepted = intersect_mesh(mesh, origins, directions, minimum_distance=1e-8)
    rejected = intersect_mesh(mesh, origins, directions, minimum_distance=1e-5)
    assert accepted.face_indices[0] == 0
    assert rejected.face_indices[0] == -1
