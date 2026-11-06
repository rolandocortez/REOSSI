"""Check the walkthrough against independently known plane correspondences."""

import json

import numpy as np
import pytest

from examples import compare_modes


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    output = tmp_path_factory.mktemp("mode-comparison") / "native"
    values = compare_modes.run_comparison(output)
    return output, values


def test_native_walkthrough_recovers_known_geometry_and_accounts_for_acquisition(
    comparison,
):
    output, values = comparison
    assert values["pattern_placement"] == "native_pixels"
    assert values["sensor"]["noise_std"] == 0
    assert values["camera_size"] == values["projector_size"] == [48, 32]
    expected_frames = {"conventional": 26, "ld": 18, "reossi": 72}
    for mode, result in values["modes"].items():
        assert result["projected_frames"] == expected_frames[mode]
        assert result["model"]["pattern_placement"] == "native_pixels"
        assert result["identifier_metrics"]["illuminated_pixels"] == 45 * 32
        assert result["identifier_metrics"]["valid_fraction"] == 1
        assert result["identifier_metrics"]["out_of_range_valid_pixels"] == 0
        cloud_folder = output / mode / "cloud"
        scan_folder = output / mode / "scan"
        pixels = np.load(cloud_folder / "camera_pixels.npy").astype(int)
        points = np.load(cloud_folder / "pts_3d.npy")
        colors = np.load(cloud_folder / "colors.npy")
        assert pixels.shape == (points.shape[1], 2)
        assert colors.shape == (points.shape[1], 3)
        assert result["geometry_metrics"]["retained_points"] == points.shape[1]
        assert result["geometry_metrics"]["evaluated_illuminated_points"] == 45 * 32
        truth = np.load(scan_folder / "true_points_camera.npy")
        if mode in {"conventional", "reossi"}:
            assert (
                result["identifier_metrics"]["joint_exact_fraction_of_illuminated"] == 1
            )
            np.testing.assert_allclose(
                points.T, truth[pixels[:, 1], pixels[:, 0]], atol=1e-5, rtol=0
            )
            assert result["geometry_metrics"]["maximum_euclidean_error"] < 1e-5
        else:
            assert (
                result["identifier_metrics"][
                    "columns_mean_absolute_error_native_pixels"
                ]
                > 1
            )
            assert result["geometry_metrics"]["mean_euclidean_error"] > 0.1
        assert (cloud_folder / "cloud.ply").is_file()
    assert json.loads((output / "comparison.json").read_text()) == json.loads(
        json.dumps(values)
    )


def test_preserved_cell_walkthrough_keeps_its_actual_quantized_result(tmp_path):
    output = tmp_path / "cells"
    values = compare_modes.run_comparison(output, pattern_placement="emulated_cells")
    assert values["pattern_placement"] == "emulated_cells"
    for name in ("h_pixels.npy", "v_pixels.npy"):
        np.testing.assert_array_equal(
            np.load(output / "ld" / "scan" / name),
            np.load(output / "reossi" / "scan" / name),
        )
    assert (
        values["modes"]["reossi"]["identifier_metrics"][
            "columns_mean_absolute_error_native_pixels"
        ]
        > 1
    )


def test_existing_output_is_preserved_before_any_simulation(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("Existing observation")

    def unexpected_simulation(*args, **kwargs):
        raise AssertionError("Existing output must be rejected before rendering")

    monkeypatch.setattr(compare_modes, "simulate_scene_scan", unexpected_simulation)
    with pytest.raises(FileExistsError):
        compare_modes.run_comparison(output)
    assert marker.read_text() == "Existing observation"
    assert list(output.iterdir()) == [marker]


def test_plot_is_optional_and_closed_after_export(comparison, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = tmp_path / "plotted"
    before = plt.get_fignums()
    compare_modes.run_comparison(output, plot=True)
    assert (output / "comparison.png").stat().st_size > 0
    assert plt.get_fignums() == before


def test_invalid_native_dimensions_fail_before_creating_output(tmp_path):
    output = tmp_path / "invalid-ratio"
    with pytest.raises(ValueError):
        compare_modes.run_comparison(output, projector_size=(49, 32))
    assert not output.exists()
