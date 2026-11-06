"""Analytical Gray-code and photometric contracts, independent of old sources."""

import numpy as np
import pytest

from reossi.pipeline import decode_displacement, generate_pattern_sequence
from reossi.plan import ScanPlan
from reossi.structured_light.decoding import (
    aggregate_repeated_bits,
    classify_gray_bits,
    decode_gray_word,
    decode_identifier_maps,
)
from reossi.structured_light.illumination import estimate_illumination
from reossi.structured_light.patterns import generate_gray_codewords

GRAY_3 = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ],
    dtype=np.uint8,
)


def test_standard_three_bit_gray_sequence():
    np.testing.assert_array_equal(generate_gray_codewords(8, 5), GRAY_3)
    for identifier, word in enumerate(GRAY_3):
        assert decode_gray_word(word) == identifier
    assert decode_gray_word([0, -1, 1]) == -1


def test_one_based_cell_shift_and_power_of_two_edge_are_explicit():
    shifted = generate_gray_codewords(7, 4, identifier_origin=1, displacement=2)
    expected = np.vstack([GRAY_3[0], GRAY_3[0], GRAY_3[1:6]])
    np.testing.assert_array_equal(shifted, expected)
    words = generate_gray_codewords(8, 4, identifier_origin=1)
    np.testing.assert_array_equal(words[-1], [1, 0, 0])
    assert decode_gray_word(words[-1]) == 7  # Retained final-identifier limitation.


def test_illumination_formula_on_known_extrema():
    images = np.full((14, 1, 1), 20.0)
    images[0], images[1] = 2, 100
    images[2::2] = 50
    images[3::2] = 5
    direct, global_component = estimate_illumination(images)
    inverse = 100 / 102
    np.testing.assert_allclose(direct, [[45 * inverse]])
    np.testing.assert_allclose(global_component, [[2 * (50 - 45 * inverse) * inverse]])


def test_uint8_normalization_keeps_stored_arithmetic():
    images = np.full((14, 1, 1), 20, dtype=np.uint8)
    images[1] = 250
    images[2::2] = 50
    images[3::2] = 5
    direct, _ = estimate_illumination(images)
    np.testing.assert_allclose(direct, [[45 * (250 / 14)]])  # 270 modulo 256.
    promoted, _ = estimate_illumination(images.astype(float))
    np.testing.assert_allclose(promoted, [[45 * (250 / 270)]])


def test_later_classifier_rules_can_overwrite_low_direct_assignment():
    images = np.ones((14, 1, 1), dtype=float)
    images[2:8] = 20
    columns, rows = classify_gray_bits(images, np.array([[5.0]]), np.array([[0.0]]))
    np.testing.assert_array_equal(columns, np.ones((3, 1, 1), dtype=np.int8))
    np.testing.assert_array_equal(rows, np.ones((3, 1, 1), dtype=np.int8))


def test_repetitions_combine_bits_before_integer_decode():
    first = np.array([[[0]], [[1]], [[-1]]], dtype=np.int8)
    second = np.array([[[1]], [[0]], [[0]]], dtype=np.int8)
    columns, rows = aggregate_repeated_bits([first, second], [first, second])
    np.testing.assert_array_equal(columns[:, 0, 0], [1, 1, 0])
    assert decode_gray_word(columns[:, 0, 0]) == 4
    np.testing.assert_array_equal(rows, columns)


def test_identifier_maps_use_reversed_row_words():
    columns = GRAY_3[[1, 2, 3, 4]].T.reshape(3, 2, 2)
    rows = GRAY_3[[4, 3, 2, 1]].T[::-1].reshape(3, 2, 2)
    actual_columns, actual_rows = decode_identifier_maps(columns, rows)
    np.testing.assert_array_equal(actual_columns, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(actual_rows, [[4, 3], [2, 1]])


def test_configured_margin_controls_pipeline_classification():
    plan = ScanPlan(
        mode="conventional", projector_size=(8, 5), repeats=1, contrast_margin=300
    )
    images = generate_pattern_sequence(plan).astype(float)
    maps = decode_displacement([images], plan, 0)
    np.testing.assert_array_equal(maps.columns, np.full((5, 8), -1))
    np.testing.assert_array_equal(maps.rows, np.full((5, 8), -1))


def test_incomplete_repeated_capture_is_rejected():
    plan = ScanPlan(mode="conventional", projector_size=(8, 5), repeats=2)
    with pytest.raises(ValueError, match="Expected 2 repeated sequences"):
        decode_displacement([generate_pattern_sequence(plan).astype(float)], plan, 0)
