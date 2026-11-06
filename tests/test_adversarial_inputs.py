"""Small adversarial cases for public simulation and decoding contracts.

These checks use analytic inputs and mocks only; none depend on historical
source trees, optional rendering packages, device access or large allocation.
"""

import json

import numpy as np
import pytest

from reossi.cli.main import main
from reossi.pipeline import decode_displacement
from reossi.plan import ScanPlan
from reossi.simulation.geometry import (
    PinholeDevice,
    TriangleMesh,
    look_at_rotation,
    make_device,
    make_surface,
)
from reossi.simulation.raster import sample_pattern_sequence, simulate_scan
from reossi.simulation.raytrace import intersect_mesh, scene_correspondences
from reossi.simulation.scene import simulate_scene_scan
from reossi.structured_light.decoding import decode_sequence


@pytest.fixture
def conventional_plan():
    return ScanPlan(mode="conventional", projector_size=(8, 5), repeats=1)


@pytest.mark.parametrize("limit", [np.nan, np.inf, -np.inf])
def test_raster_budget_requires_a_finite_positive_limit(conventional_plan, limit):
    with pytest.raises(ValueError):
        simulate_scan(conventional_plan, camera_size=(8, 5), max_sequence_mib=limit)


@pytest.mark.parametrize(
    "settings",
    [
        {"noise_std": np.nan},
        {"noise_std": np.inf},
        {"black_level": np.inf},
        {"direct_level": np.nan},
    ],
)
def test_raster_sensor_rejects_nonfinite_values(conventional_plan, settings):
    with pytest.raises(ValueError):
        simulate_scan(conventional_plan, camera_size=(8, 5), **settings)


@pytest.mark.parametrize("limit", [np.nan, np.inf, -np.inf])
def test_scene_budget_requires_a_finite_positive_limit(conventional_plan, limit):
    with pytest.raises(ValueError):
        simulate_scene_scan(
            make_surface(),
            make_device((8, 5)),
            make_device((8, 5), center=(0.125, 0, 0)),
            conventional_plan,
            max_sequence_mib=limit,
        )


@pytest.mark.parametrize(
    "settings",
    [{"epsilon": np.nan}, {"minimum_distance": np.nan}, {"max_triangle_tests": np.nan}],
)
def test_ray_tolerances_and_budget_reject_nonfinite_values(settings):
    with pytest.raises(ValueError):
        intersect_mesh(
            make_surface(), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), **settings
        )


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -np.inf])
def test_visibility_tolerance_must_be_finite(tolerance):
    with pytest.raises(ValueError):
        scene_correspondences(
            make_surface(),
            make_device((8, 5)),
            make_device((8, 5), center=(0.125, 0, 0)),
            visibility_tolerance=tolerance,
        )


def test_scene_triangle_guard_precedes_camera_ray_allocation(monkeypatch):
    def forbidden_rays(self):
        raise AssertionError("A rejected ray budget must not allocate camera rays")

    monkeypatch.setattr(PinholeDevice, "rays", forbidden_rays)
    with pytest.raises(ValueError):
        scene_correspondences(
            make_surface(),
            make_device((16, 12)),
            make_device((16, 12)),
            max_triangle_tests=1,
        )


def test_singular_intrinsics_are_rejected_at_device_construction():
    singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        PinholeDevice((8, 5), singular)


@pytest.mark.parametrize(
    "center,target",
    [([0, 0, 0], [0, 0, 0]), ([np.nan, 0, 0], [0, 0, 1]), ([0, 0, 0], [0, np.inf, 1])],
)
def test_look_at_rejects_undefined_or_nonfinite_pose(center, target):
    with pytest.raises(ValueError):
        look_at_rotation(np.array(center, dtype=float), np.array(target, dtype=float))


def test_ray_visibility_masks_distinguish_camera_background_from_projector_shadow():
    # A small plane covers only the central image; a shifted projector sees a
    # subset. Misses stay NaN and never become illuminated points.
    geometry = scene_correspondences(
        make_surface(width=0.75, height=0.5, depth=2),
        make_device((16, 12)),
        make_device((16, 12), center=(0.75, 0, 0)),
    )
    assert not geometry.camera_visible.all()
    assert geometry.camera_visible.any()
    assert not geometry.illuminated[~geometry.camera_visible].any()
    assert np.isnan(geometry.points_world[~geometry.camera_visible]).all()
    assert np.isnan(geometry.projector_pixels[~geometry.camera_visible]).all()
    assert not geometry.illuminated[geometry.camera_visible].all()


def test_projector_occluder_blocks_light_without_hiding_camera_surface():
    plane = make_surface(width=4, height=3, depth=2)
    blocker = make_surface(width=0.2, height=0.8, depth=1)
    vertices = blocker.vertices.copy()
    vertices[:, 0] += 0.40625
    mesh = TriangleMesh(
        np.concatenate((plane.vertices, vertices)),
        np.concatenate((plane.faces, blocker.faces + len(plane.vertices))),
    )
    camera = make_device((16, 12))
    projector = make_device((16, 12), center=(0.75, 0, 0))
    unblocked = scene_correspondences(plane, camera, projector)
    blocked = scene_correspondences(mesh, camera, projector)
    assert unblocked.illuminated[5, 8]
    assert blocked.camera_visible[5, 8]
    assert blocked.points_world[5, 8, 2] == pytest.approx(2)
    assert blocked.projector_pixels[5, 8, 0] == pytest.approx(2)
    assert not blocked.illuminated[5, 8]


@pytest.mark.parametrize("displacement", [-1, 1])
def test_decode_displacement_rejects_position_outside_plan(
    conventional_plan, displacement
):
    captures = np.zeros((conventional_plan.sequence_length, 1, 1), dtype=float)
    with pytest.raises(ValueError):
        decode_displacement([captures], conventional_plan, displacement)


def test_decode_rejects_empty_camera_images(conventional_plan):
    captures = np.zeros((conventional_plan.sequence_length, 0, 1), dtype=float)
    with pytest.raises(ValueError):
        decode_displacement([captures], conventional_plan, 0)


def test_decode_rejects_a_sequence_with_unmatched_extra_frame():
    with pytest.raises(ValueError):
        decode_sequence(np.zeros((15, 1, 1), dtype=float))


def test_decode_stops_extra_repetitions_before_decoding_them(
    conventional_plan, monkeypatch
):
    calls = []

    def classify(images, **kwargs):
        calls.append(None)
        if len(calls) > conventional_plan.repeats:
            raise AssertionError(
                "Unexpected repetition must be rejected before decoding"
            )
        return np.zeros((3, 1, 1), dtype=np.int8), np.zeros((3, 1, 1), dtype=np.int8)

    monkeypatch.setattr("reossi.pipeline.decode_sequence", classify)
    sequence = np.zeros((conventional_plan.sequence_length, 1, 1), dtype=float)
    with pytest.raises(ValueError):
        decode_displacement(iter([sequence, sequence]), conventional_plan, 0)
    assert len(calls) == conventional_plan.repeats


def test_cli_invalid_config_object_reports_input_error_without_traceback(tmp_path):
    config = tmp_path / "list.json"
    config.write_text(json.dumps([]))
    with pytest.raises(SystemExit) as error:
        main(["plan", "--config", str(config)])
    assert error.value.code == 2


def test_cli_unknown_plan_field_reports_input_error_without_traceback(tmp_path):
    config = tmp_path / "unknown.json"
    config.write_text(json.dumps({"mode": "conventional", "unexpected": 1}))
    with pytest.raises(SystemExit) as error:
        main(["plan", "--config", str(config)])
    assert error.value.code == 2


def test_simulation_does_not_modify_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing research")
    with pytest.raises(SystemExit) as error:
        main(["simulate", "--output", str(output), "--camera-size", "8x5"])
    assert error.value.code == 2
    assert sentinel.read_text() == "existing research"


def test_cli_generated_mesh_budget_precedes_surface_allocation(tmp_path, monkeypatch):
    config = tmp_path / "plan.json"
    config.write_text(
        json.dumps(
            {
                "mode": "conventional",
                "projector_size": [8, 5],
                "repeats": 1,
            }
        )
    )
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "camera": {"image_size": [8, 5]},
                "projector": {"image_size": [8, 5], "center": [0.125, 0, 0]},
                "surface": {"columns": 3, "rows": 3},
            }
        )
    )

    def forbidden_surface(**kwargs):
        raise AssertionError("Rejected mesh budget must not construct the surface")

    monkeypatch.setattr("reossi.simulation.geometry.make_surface", forbidden_surface)
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "simulate-scene",
                "--config",
                str(config),
                "--scene-config",
                str(scene),
                "--output",
                str(output),
                "--max-triangle-tests",
                "1",
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "settings", [{"columns": 2.5}, {"rows": True}, {"columns": 1}, {"rows": np.nan}]
)
def test_surface_counts_are_integers_validated_before_meshgrid(settings, monkeypatch):
    def forbidden_grid(*args, **kwargs):
        raise AssertionError("Invalid surface counts must fail before grid allocation")

    monkeypatch.setattr("reossi.simulation.geometry.np.meshgrid", forbidden_grid)
    with pytest.raises(ValueError):
        make_surface(**settings)


@pytest.mark.parametrize("settings", [{"columns": 2.5}, {"rows": True}])
def test_cli_invalid_surface_counts_report_input_error(tmp_path, settings):
    config = tmp_path / "plan.json"
    config.write_text(
        json.dumps(
            {
                "mode": "conventional",
                "projector_size": [8, 5],
                "repeats": 1,
            }
        )
    )
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "camera": {"image_size": [8, 5]},
                "projector": {"image_size": [8, 5], "center": [0.125, 0, 0]},
                "surface": settings,
            }
        )
    )
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "simulate-scene",
                "--config",
                str(config),
                "--scene-config",
                str(scene),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("column", [-1.0, 8.0, np.nan])
def test_raster_sampling_does_not_wrap_invalid_coordinates(column):
    with pytest.raises(ValueError):
        sample_pattern_sequence(
            np.array([[[0, 255]]], dtype=np.uint8),
            np.array([[column]]),
            np.zeros((1, 1)),
            (8, 5),
        )


def test_raster_sensor_uses_float_arithmetic_and_clips_saturation():
    patterns = np.array([[[0, 255]]], dtype=np.uint8)
    result = sample_pattern_sequence(
        patterns,
        np.array([[0, 7]]),
        np.array([[0, 0]]),
        (8, 5),
        black_level=250,
        direct_level=20,
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[[250.0, 255.0]]])


@pytest.mark.parametrize(
    "matrix", [np.full((3, 3), np.nan), np.diag([np.inf, 1.0, 1.0])]
)
def test_nonfinite_virtual_intrinsics_reject_at_construction(matrix):
    with pytest.raises(ValueError):
        PinholeDevice((8, 5), matrix)
