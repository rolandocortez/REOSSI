"""3D ground truth, ray visibility and replay through the shared decoder."""

import numpy as np
import pytest

from reossi.acquisition.recording import iter_recorded_sequences
from reossi.pipeline import reconstruct_identifier_maps
from reossi.plan import ScanPlan
from reossi.simulation.geometry import (
    PinholeDevice,
    TriangleMesh,
    load_mesh,
    make_device,
    make_surface,
    relative_calibration,
)
from reossi.simulation.raytrace import intersect_mesh, scene_correspondences
from reossi.simulation.scene import SensorModel, save_scene_result, simulate_scene_scan


def plane_rig():
    return (
        make_surface(width=4, height=3, depth=2),
        make_device((16, 12)),
        make_device((16, 12), center=(0.125, 0, 0)),
    )


def test_camera_rays_hit_analytic_plane_at_expected_depth():
    mesh, camera, _ = plane_rig()
    origins, rays = camera.rays()
    hits = intersect_mesh(mesh, origins, rays)
    np.testing.assert_allclose(hits.points[:, 2], 2)
    pixels, _ = camera.project(hits.points)
    x, y = np.meshgrid(np.arange(16), np.arange(12))
    np.testing.assert_allclose(
        pixels, np.stack([x, y], axis=-1).reshape(-1, 2), atol=1e-12
    )


def test_projector_calibration_and_visibility_have_known_one_pixel_disparity():
    mesh, camera, projector = plane_rig()
    geometry = scene_correspondences(mesh, camera, projector)
    x, y = np.meshgrid(np.arange(16), np.arange(12))
    np.testing.assert_allclose(geometry.projector_pixels[..., 0], x - 1, atol=1e-12)
    np.testing.assert_allclose(geometry.projector_pixels[..., 1], y, atol=1e-12)
    assert geometry.camera_visible.all()
    assert not geometry.illuminated[:, 0].any()
    assert geometry.illuminated[:, 1:].all()
    rotation, translation = relative_calibration(camera, projector)
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_array_equal(translation, [[-0.125], [0], [0]])


def test_occluding_triangle_wins_nearest_intersection():
    mesh = TriangleMesh(
        np.array(
            [[-1, -1, 2], [1, -1, 2], [0, 1, 2], [-1, -1, 1], [1, -1, 1], [0, 1, 1]]
        ),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    hits = intersect_mesh(
        mesh,
        np.array([[0, 0, 0], [4, 0, 0]], dtype=float),
        np.array([[0, 0, 1], [0, 0, 1]], dtype=float),
    )
    np.testing.assert_array_equal(hits.face_indices, [1, -1])
    assert hits.distances[0] == 1
    assert np.isinf(hits.distances[1])
    assert np.isnan(hits.points[1]).all()


def test_noiseless_virtual_conventional_scan_matches_known_projector_coordinates():
    mesh, camera, projector = plane_rig()
    plan = ScanPlan(mode="conventional", projector_size=(16, 12), repeats=2)
    result = simulate_scene_scan(mesh, camera, projector, plan)
    mask = result.geometry.illuminated
    np.testing.assert_array_equal(result.maps.columns[mask], result.true_columns[mask])
    np.testing.assert_array_equal(result.maps.rows[mask], result.true_rows[mask])
    assert result.metrics["valid_fraction"] == 1
    assert result.metrics["joint_exact_fraction_of_illuminated"] == 1
    assert result.metrics["decoded_outside_illuminated_pixels"] == 0


def test_unilluminated_counter_includes_out_of_viewport_without_claiming_shadows():
    mesh, camera, projector = plane_rig()
    plan = ScanPlan(mode="conventional", projector_size=(16, 12), repeats=1)
    result = simulate_scene_scan(mesh, camera, projector, plan)
    pixels = result.geometry.projector_pixels
    outside_viewport = (
        (pixels[..., 0] < -0.5)
        | (pixels[..., 0] >= 15.5)
        | (pixels[..., 1] < -0.5)
        | (pixels[..., 1] >= 11.5)
    )
    # This single plane has no occluder. All twelve excluded pixels lie
    # outside the projector viewport rather than in a projected shadow.
    assert result.geometry.camera_visible.all()
    assert int(outside_viewport.sum()) == 12
    np.testing.assert_array_equal(result.geometry.illuminated, ~outside_viewport)
    assert result.model["camera_visible_unilluminated_pixels"] == 12
    assert "projector_shadow_pixels" not in result.model


def test_virtual_recording_replays_identically_and_keeps_calibration(tmp_path):
    mesh, camera, projector = plane_rig()
    plan = ScanPlan(projector_size=(16, 12), emulated_size=(5, 4), repeats=2)
    result = simulate_scene_scan(
        mesh,
        camera,
        projector,
        plan,
        sensor=SensorModel(noise_std=35),
        seed=12,
        recording_folder=tmp_path / "captures",
    )
    restored = reconstruct_identifier_maps(
        iter_recorded_sequences(tmp_path / "captures", plan), plan
    )
    np.testing.assert_array_equal(restored.columns, result.maps.columns)
    np.testing.assert_array_equal(restored.rows, result.maps.rows)
    save_scene_result(tmp_path / "scan", result, plan)
    from reossi.calibration.workflow import load_stereo_calibration

    calibration = load_stereo_calibration(tmp_path / "scan/calibration.npz")
    assert calibration.camera_size == (16, 12)
    np.testing.assert_array_equal(calibration.translation, [[-0.125], [0], [0]])
    assert (tmp_path / "scan/true_points_camera.npy").is_file()


def test_scene_guard_rejects_expensive_work_before_intersection(monkeypatch):
    from reossi.simulation import scene

    monkeypatch.setattr(
        scene,
        "scene_correspondences",
        lambda *a, **kw: pytest.fail("Memory guard should reject before tracing"),
    )
    mesh, camera, projector = plane_rig()
    with pytest.raises(ValueError, match="max_sequence_mib"):
        simulate_scene_scan(
            mesh,
            camera,
            projector,
            ScanPlan(mode="conventional", projector_size=(16, 12)),
            max_sequence_mib=0.00001,
        )


def test_ray_test_budget_is_checked():
    mesh, camera, projector = plane_rig()
    with pytest.raises(ValueError, match="budget"):
        scene_correspondences(mesh, camera, projector, max_triangle_tests=100)


def test_mesh_npz_and_pose_validation(tmp_path):
    mesh = make_surface(columns=3, rows=4, tilt=0.1, curvature=0.2)
    path = tmp_path / "surface.npz"
    np.savez(path, vertices=mesh.vertices, faces=mesh.faces)
    restored = load_mesh(path)
    np.testing.assert_array_equal(restored.vertices, mesh.vertices)
    np.testing.assert_array_equal(restored.faces, mesh.faces)
    with pytest.raises(ValueError, match="orthonormal"):
        PinholeDevice((8, 6), np.eye(3), rotation=np.ones((3, 3)))
    with pytest.raises(ValueError, match="degenerate"):
        TriangleMesh(np.zeros((3, 3)), np.array([[0, 1, 2]]))
