"""Digital diagnostics with independent known coordinates and legacy parity."""

import json

import numpy as np
import pytest

from reossi.cli.main import main
from reossi.plan import ScanPlan
from reossi.simulation.raster import simulate_scan


def test_noiseless_conventional_scan_recovers_known_native_coordinates():
    plan = ScanPlan(mode="conventional", projector_size=(12, 8), repeats=2)
    result = simulate_scan(plan, camera_size=(9, 6))
    expected_columns = np.tile([0, 2, 3, 4, 6, 7, 8, 10, 11], (6, 1))
    expected_rows = np.tile(np.array([0, 2, 3, 4, 6, 7])[:, None], (1, 9))
    np.testing.assert_array_equal(result.true_columns, expected_columns)
    np.testing.assert_array_equal(result.true_rows, expected_rows)
    np.testing.assert_array_equal(result.maps.columns, expected_columns)
    np.testing.assert_array_equal(result.maps.rows, expected_rows)
    assert result.metrics["valid_fraction"] == 1
    for axis in ("columns", "rows"):
        assert result.metrics[f"{axis}_mean_absolute_error_native_pixels"] == 0
        assert result.metrics[f"{axis}_exact_fraction_of_valid"] == 1


def test_seeded_sensor_noise_is_reproducible():
    plan = ScanPlan(
        mode="reossi", projector_size=(24, 16), emulated_size=(6, 4), repeats=2
    )
    first = simulate_scan(plan, camera_size=(12, 8), noise_std=100, seed=81)
    second = simulate_scan(plan, camera_size=(12, 8), noise_std=100, seed=81)
    np.testing.assert_array_equal(first.maps.columns, second.maps.columns)
    np.testing.assert_array_equal(first.maps.rows, second.maps.rows)
    assert first.metrics == second.metrics
    assert first.model == second.model
    assert first.model["noise_std"] == 100
    assert first.model["seed"] == 81


def test_memory_guard_runs_before_pattern_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Allocation guard must run before pattern generation")

    monkeypatch.setattr("reossi.simulation.raster.generate_pattern_sequence", forbidden)
    with pytest.raises(ValueError, match="max_sequence_mib"):
        simulate_scan(ScanPlan(), max_sequence_mib=0.00001)


def test_noiseless_reossi_retains_error_instead_of_changing_ground_truth():
    parameters = {"projector_size": (24, 16), "emulated_size": (6, 4), "repeats": 2}
    reossi = simulate_scan(ScanPlan(mode="reossi", **parameters), camera_size=(12, 8))
    ld = simulate_scan(ScanPlan(mode="ld", **parameters), camera_size=(12, 8))
    # The source shifts whole LD cells; initial-position maxima dominate here.
    # These equal maps are an observed limitation under this forward model.
    np.testing.assert_array_equal(reossi.maps.columns, ld.maps.columns)
    np.testing.assert_array_equal(reossi.maps.rows, ld.maps.rows)
    np.testing.assert_array_equal(reossi.true_columns[0], np.arange(2, 25, 2))
    np.testing.assert_array_equal(reossi.true_rows[:, 0], np.arange(2, 17, 2))
    assert reossi.metrics["valid_fraction"] == 1
    for axis in ("columns", "rows"):
        assert reossi.metrics[f"{axis}_mean_absolute_error_native_pixels"] == 2
        assert reossi.metrics[f"{axis}_exact_fraction_of_valid"] == 0


def test_cli_simulate_saves_raw_maps_and_model(tmp_path, capsys, monkeypatch):
    # Explicit native baseline keeps the expected image coordinates independent
    # of REOSSI's historical identifier scaling convention.
    config = tmp_path / "plan.json"
    config.write_text(
        json.dumps(
            {
                "mode": "conventional",
                "projector_size": [8, 5],
                "emulated_size": [8, 5],
                "repeats": 1,
            }
        )
    )

    def simulator_with_explicit_limit(*args, **kwargs):
        assert kwargs["max_sequence_mib"] == 0.02
        return simulate_scan(*args, **kwargs)

    monkeypatch.setattr(
        "reossi.simulation.raster.simulate_scan", simulator_with_explicit_limit
    )
    output = tmp_path / "simulation"
    assert (
        main(
            [
                "simulate",
                "--config",
                str(config),
                "--output",
                str(output),
                "--camera-size",
                "8x5",
                "--noise-std",
                "0",
                "--seed",
                "11",
                "--max-sequence-mib",
                "0.02",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid_fraction"] == 1
    record = json.loads((output / "simulation.json").read_text())
    assert record["model"]["seed"] == 11
    assert record["model"]["camera_size"] == [8, 5]
    np.testing.assert_array_equal(
        np.load(output / "h_pixels.npy"), np.load(output / "true_columns.npy")
    )
    np.testing.assert_array_equal(
        np.load(output / "v_pixels.npy"), np.load(output / "true_rows.npy")
    )
    assert (output / "scan.json").is_file()
