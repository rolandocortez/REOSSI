"""Run connected workflow commands on tiny known geometry without hardware."""

import json

import numpy as np
import pytest

from reossi.cli.main import main


@pytest.fixture
def scene_cli(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "mode": "conventional",
                "projector_size": [16, 12],
                "emulated_size": [16, 12],
                "repeats": 1,
            }
        )
    )
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "camera": {"image_size": [16, 12], "focal_length": 16},
                "projector": {
                    "image_size": [16, 12],
                    "focal_length": 16,
                    "center": [0.125, 0, 0],
                },
                "surface": {"width": 4, "height": 3, "depth": 2},
            }
        )
    )
    output = tmp_path / "scene"
    assert (
        main(
            [
                "simulate-scene",
                "--config",
                str(plan),
                "--scene-config",
                str(scene),
                "--output",
                str(output),
                "--save-captures",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["joint_exact_fraction_of_illuminated"] == 1
    )
    return plan, output


def test_virtual_capture_replay_and_reconstruction_commands(
    scene_cli, tmp_path, capsys
):
    plan, scene = scene_cli
    replay = tmp_path / "replayed"
    assert (
        main(
            [
                "replay",
                "--config",
                str(plan),
                "--input",
                str(scene / "captures"),
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    assert (replay / "white.png").is_file()
    np.testing.assert_array_equal(
        np.load(replay / "h_pixels.npy"), np.load(scene / "h_pixels.npy")
    )
    cloud = tmp_path / "cloud"
    assert (
        main(
            [
                "reconstruct",
                "--input",
                str(replay),
                "--calibration",
                str(scene / "calibration.npz"),
                "--output",
                str(cloud),
                "--identifier-origin",
                "zero",
            ]
        )
        == 0
    )
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["retained_points"] == 180
    points = np.load(cloud / "pts_3d.npy")
    camera_pixels = np.load(cloud / "camera_pixels.npy").astype(int)
    truth = np.load(scene / "true_points_camera.npy")
    # The reference correspondence collector supplies float32 image points;
    # this closed loop therefore uses a geometric float32 tolerance in metres.
    np.testing.assert_allclose(
        points.T, truth[camera_pixels[:, 1], camera_pixels[:, 0]], atol=1e-5, rtol=0
    )
    assert (cloud / "cloud.ply").is_file()


@pytest.mark.parametrize("placement_source", ["flag", "configuration"])
def test_scene_command_uses_explicit_native_pixel_placement(
    tmp_path, capsys, placement_source
):
    plan = tmp_path / "plan.json"
    plan_settings = {
        "mode": "reossi",
        "projector_size": [24, 16],
        "emulated_size": [12, 8],
        "repeats": 1,
    }
    if placement_source == "configuration":
        plan_settings = {
            "plan": plan_settings,
            "acquisition": {"pattern_placement": "native_pixels"},
        }
    plan.write_text(json.dumps(plan_settings))
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "camera": {"image_size": [24, 16], "focal_length": 24},
                "projector": {
                    "image_size": [24, 16],
                    "focal_length": 24,
                    "center": [0.25, 0, 0],
                },
                "surface": {"width": 4, "height": 3, "depth": 2},
            }
        )
    )
    output = tmp_path / "native-scan"
    arguments = [
        "simulate-scene",
        "--config",
        str(plan),
        "--scene-config",
        str(scene),
        "--output",
        str(output),
    ]
    if placement_source == "flag":
        arguments += ["--pattern-placement", "native_pixels"]
    assert main(arguments) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["joint_exact_fraction_of_illuminated"] == 1
    metadata = json.loads((output / "simulation.json").read_text())
    assert metadata["model"]["pattern_placement"] == "native_pixels"
    saved_plan = json.loads((output / "scan.json").read_text())
    assert saved_plan["acquisition"]["pattern_placement"] == "native_pixels"


def test_saved_cloud_and_identifier_view_commands(scene_cli, tmp_path, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    _, scene = scene_cli
    cloud = tmp_path / "cloud"
    main(
        [
            "reconstruct",
            "--input",
            str(scene),
            "--calibration",
            str(scene / "calibration.npz"),
            "--output",
            str(cloud),
        ]
    )
    capsys.readouterr()
    assert (
        main(
            [
                "view",
                "--input",
                str(cloud),
                "--output",
                str(tmp_path / "cloud.png"),
                "--no-show",
                "--max-points",
                "50",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["stored_points"] == 180 and summary["displayed_points"] == 50
    assert (
        main(
            [
                "view-identifiers",
                "--input",
                str(scene),
                "--output",
                str(tmp_path / "ids.png"),
                "--no-show",
            ]
        )
        == 0
    )
    assert (tmp_path / "cloud.png").is_file() and (tmp_path / "ids.png").is_file()


def test_identifier_demo_cli_is_independent_of_capture(tmp_path, capsys):
    config = tmp_path / "identifiers.json"
    config.write_text(json.dumps({"curvature": 0}))
    output = tmp_path / "identifiers"
    assert (
        main(["simulate-identifiers", "--config", str(config), "--output", str(output)])
        == 0
    )
    values = json.loads(capsys.readouterr().out)
    assert values["shape"] == [10, 24]
    np.testing.assert_array_equal(
        np.load(output / "fused.npy"), np.tile(np.repeat(np.arange(1, 13), 2), (10, 1))
    )


def test_projector_collection_invalid_grid_never_opens_camera(tmp_path, monkeypatch):
    from reossi.calibration.workflow import CameraCalibration
    from reossi.cli import calibrate

    calibration = tmp_path / "camera.npz"
    CameraCalibration(
        np.array([[10.0, 0, 8], [0, 10.0, 6], [0, 0, 1]]), np.zeros(5), (16, 12)
    ).save(calibration)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"camera_size": [16, 12], "projector_size": [16, 12]}))
    monkeypatch.setattr(
        calibrate,
        "open_camera",
        lambda **kwargs: pytest.fail("Invalid setup opened a camera"),
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "collect-projector-calibration",
                "--config",
                str(config),
                "--camera-calibration",
                str(calibration),
                "--output",
                str(tmp_path / "observations"),
            ]
        )
    assert error.value.code == 2


def test_explicit_solver_settings_use_both_termination_conditions():
    import cv2

    from reossi.cli.calibrate import solver_criteria

    assert solver_criteria({}, "projector_criteria") is None
    assert solver_criteria(
        {"projector_criteria": {"max_iterations": 100, "epsilon": 1e-12}},
        "projector_criteria",
    ) == (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-12)


@pytest.mark.parametrize("view_kind", ["cloud", "identifiers"])
def test_optional_plotting_has_an_actionable_installation_error(monkeypatch, view_kind):
    import builtins

    from reossi import visualization

    original_import = builtins.__import__

    def without_plotting(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError(
                "Matplotlib is intentionally unavailable", name="matplotlib"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotting)
    with pytest.raises(ImportError, match=r"reossi\[plots\]"):
        if view_kind == "cloud":
            visualization.plot_point_cloud(
                np.empty((3, 0)), np.empty((0, 3)), show=False
            )
        else:
            identifiers = np.zeros((1, 1), dtype=int)
            visualization.plot_decoded_identifiers(
                identifiers, identifiers, np.zeros((1, 1, 3)), show=False
            )
