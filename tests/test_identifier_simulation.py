"""Analytical direct-ID examples, independent of capture and private models."""

import numpy as np
import pytest

import reossi.simulation.identifiers as identifiers
from reossi.simulation.identifiers import (
    generate_stripe_map,
    shift_identifier_map,
    simulate_identifier_map,
)


def test_flat_layers_recover_each_target_identifier_twice():
    result = simulate_identifier_map(height=2, curvature=0)
    expected_base = np.tile(np.repeat(np.arange(1, 5), 6), (2, 1))
    expected_fused = np.tile(np.repeat(np.arange(1, 13), 2), (2, 1))
    np.testing.assert_array_equal(result.base, expected_base)
    np.testing.assert_array_equal(result.fused, expected_fused)
    assert len(result.layers) == len(result.scaled_layers) == 3
    assert result.config["positions"] == 3
    assert result.config["shift_pixels"] == 2


def test_curved_profile_matches_known_integer_row_shifts():
    result = generate_stripe_map(4, 6, 10, curvature=5)
    row = np.repeat(np.arange(1, 5), 6)
    expected = np.zeros((10, 24), dtype=int)
    # Radius5 and curvature5 give these analytically known floor displacements.
    for index, shift in enumerate([0, 3, 4, 4, 4, 5, 4, 4, 4, 3]):
        expected[index, shift:] = row[: 24 - shift]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "axis,expected",
    [
        (0, [[-1, -1, -1], [-1, -1, -1], [1, 2, 3]]),
        (1, [[-1, -1, 1], [-1, -1, 4], [-1, -1, 7]]),
    ],
)
def test_shift_removes_wraparound_and_preserves_original(axis, expected):
    original = np.arange(1, 10).reshape(3, 3)
    shifted = shift_identifier_map(
        original, 1, shift_pixels=2, axis=axis, fill_value=-1
    )
    np.testing.assert_array_equal(shifted, expected)
    np.testing.assert_array_equal(original, np.arange(1, 10).reshape(3, 3))
    assert not np.shares_memory(original, shifted)
    np.testing.assert_array_equal(
        shift_identifier_map(original, 9, axis=axis), np.zeros((3, 3))
    )


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_scale_copies_and_exact_minus_one_policy(dtype):
    base = np.array([[-2, -1, 0, 1, 2, 3]], dtype=dtype)
    untouched = base.copy()
    result = simulate_identifier_map(base, scale_factor=3, positions=1, shift_pixels=0)
    np.testing.assert_array_equal(result.scaled_layers[0], [[-8, -1, -2, 1, 4, 7]])
    np.testing.assert_array_equal(result.fused, result.scaled_layers[0])
    np.testing.assert_array_equal(base, untouched)
    np.testing.assert_array_equal(result.base, untouched)
    np.testing.assert_array_equal(result.layers[0], untouched)
    assert result.fused.dtype == dtype
    for array in (result.base, result.layers[0], result.scaled_layers[0], result.fused):
        assert not np.shares_memory(base, array)
    result.scaled_layers[0][0, 0] = 999
    assert result.layers[0][0, 0] == -2
    assert result.fused[0, 0] == -8


def test_vertical_axis_simulation_uses_the_same_shared_transform():
    base = np.repeat(np.arange(1, 4), 2).reshape(6, 1)
    result = simulate_identifier_map(
        base, scale_factor=2, positions=2, shift_pixels=1, axis=0
    )
    np.testing.assert_array_equal(result.fused[:, 0], np.arange(1, 7))


def test_displacement_step_is_explicit_and_does_not_mutate_layers():
    base = np.array([[1, 1, 2, 2]])
    result = simulate_identifier_map(
        base, scale_factor=2, positions=2, shift_pixels=1, step=2
    )
    np.testing.assert_array_equal(result.layers[1], [[0, 1, 1, 2]])
    np.testing.assert_array_equal(result.scaled_layers[1], [[1, 3, 3, 5]])
    np.testing.assert_array_equal(result.fused, [[1, 3, 3, 5]])


def test_generation_budget_is_checked_before_numpy_allocation(monkeypatch):
    monkeypatch.setattr(
        identifiers.np,
        "arange",
        lambda *a, **k: pytest.fail("Allocation preceded guard"),
    )
    with pytest.raises(ValueError, match="max_pixels"):
        generate_stripe_map(10000, 10000, 10000, max_pixels=100)


def test_layer_budget_is_checked_before_generation_or_shifting(monkeypatch):
    monkeypatch.setattr(
        identifiers,
        "generate_stripe_map",
        lambda *a, **k: pytest.fail("Generated before budget"),
    )
    monkeypatch.setattr(
        identifiers,
        "shift_identifier_map",
        lambda *a, **k: pytest.fail("Shifted before budget"),
    )
    with pytest.raises(ValueError, match="max_layer_pixels"):
        simulate_identifier_map(positions=1_000_000, max_layer_pixels=100)
    with pytest.raises(ValueError, match="max_layer_pixels"):
        simulate_identifier_map(np.ones((2, 3)), positions=100, max_layer_pixels=100)


@pytest.mark.parametrize(
    "parameters",
    [
        {"stripe_count": 0},
        {"stripe_width": 1.5},
        {"height": 0},
        {"positions": 0},
        {"positions": True},
        {"positions": float("inf")},
        {"scale_factor": 0},
        {"scale_factor": 10**100},
        {"step": -1},
        {"shift_pixels": float("nan")},
        {"curvature": float("inf")},
        {"curvature": -1},
        {"axis": 2},
        {"fill_value": float("nan")},
        {"max_pixels": 0},
        {"max_layer_pixels": float("inf")},
        {"height": 1, "curvature": 1},
    ],
)
def test_invalid_parameters_fail_without_generating(parameters, monkeypatch):
    monkeypatch.setattr(
        identifiers,
        "generate_stripe_map",
        lambda *a, **k: pytest.fail("Generated invalid configuration"),
    )
    with pytest.raises(ValueError):
        simulate_identifier_map(**parameters)


@pytest.mark.parametrize(
    "base",
    [
        np.ones(3),
        np.empty((0, 3)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([["1"]]),
        np.array([[True]]),
    ],
)
def test_direct_input_requires_finite_nonempty_numeric_maps(base):
    with pytest.raises(ValueError):
        simulate_identifier_map(base)


def test_supplied_input_has_its_own_configuration_and_no_generator_dependency():
    result = simulate_identifier_map(
        np.ones((2, 3)), positions=1, fill_value=np.int64(-1)
    )
    assert result.config["input"] == "provided_identifiers"
    assert result.config["shape"] == [2, 3]
    assert "curvature" not in result.config
    assert result.config["fill_value"] == -1
