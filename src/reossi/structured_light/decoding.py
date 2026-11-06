"""Gray-bit classification and identifier decoding."""

# Gray-code decoding derived from Guillaume Charron's implementation (MIT).
# Its cited algorithm sources and attribution appear in THIRD_PARTY_NOTICES.md.

from collections.abc import Sequence

import numpy as np

from ._validation import gray_sequence
from .illumination import IlluminationEstimator
from .illumination import estimate_illumination as estimate_illumination


def classify_gray_bits(
    images: np.ndarray,
    direct: np.ndarray,
    global_component: np.ndarray,
    *,
    contrast_margin: int | float = 1,
    minimum_direct: int | float = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference column/row ``int8`` bits, each shaped ``(B, H, W)``.

    Unknown bits are ``-1``. Rule assignments intentionally retain the
    original order, including overwrites of earlier assignments.
    """
    images = gray_sequence(images)
    if direct.shape != images.shape[1:] or global_component.shape != images.shape[1:]:
        raise ValueError("Illumination components must match (H,W)")
    pattern_length = int((len(images) - 2) / 4)
    pattern_images = images[2:]
    normal_patterns = pattern_images[: pattern_length * 2]
    inverse_patterns = pattern_images[pattern_length * 2 :]

    column_indices = [2 * index for index in range(pattern_length)]
    row_indices = [2 * index + 1 for index in range(pattern_length)]
    column_normal = normal_patterns[column_indices]
    row_normal = normal_patterns[row_indices]
    column_inverse = inverse_patterns[column_indices]
    row_inverse = inverse_patterns[row_indices]

    column_bits = np.zeros(column_normal.shape, dtype=np.int8) - 1
    row_bits = np.zeros(row_normal.shape, dtype=np.int8) - 1
    direct = np.repeat(direct[np.newaxis, :, :], pattern_length, axis=0)
    global_component = np.repeat(
        global_component[np.newaxis, :, :], pattern_length, axis=0
    )

    column_bits[np.where(direct < minimum_direct)] = -1
    row_bits[np.where(direct < minimum_direct)] = -1

    column_bits[
        np.where(
            (direct > (global_component + contrast_margin))
            & (column_normal > (column_inverse + contrast_margin))
        )
    ] = 1
    row_bits[
        np.where(
            (direct > (global_component + contrast_margin))
            & (row_normal > (row_inverse + contrast_margin))
        )
    ] = 1

    column_bits[
        np.where(
            (direct > (global_component + contrast_margin))
            & ((column_normal + contrast_margin) < column_inverse)
        )
    ] = 0
    row_bits[
        np.where(
            (direct > (global_component + contrast_margin))
            & ((row_normal + contrast_margin) < row_inverse)
        )
    ] = 0

    column_bits[
        np.where(
            ((column_normal + contrast_margin) < direct)
            & (column_inverse > (global_component + contrast_margin))
        )
    ] = 0
    row_bits[
        np.where(
            ((row_normal + contrast_margin) < direct)
            & (row_inverse > (global_component + contrast_margin))
        )
    ] = 0

    column_bits[
        np.where(
            (column_normal > (global_component + contrast_margin))
            & ((column_inverse + contrast_margin) < direct)
        )
    ] = 1
    row_bits[
        np.where(
            (row_normal > (global_component + contrast_margin))
            & ((row_inverse + contrast_margin) < direct)
        )
    ] = 1
    return column_bits, row_bits


def decode_sequence(
    images: np.ndarray,
    *,
    normalization_epsilon: float = 0.0,
    contrast_margin: int | float = 1,
    minimum_direct: int | float = 10,
    illumination_estimator: IlluminationEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate illumination and classify one grayscale capture sequence.

    A supplied estimator receives ``(N,H,W)`` captures and returns direct and
    global components of shape ``(H,W)``. Its parameters belong to the caller;
    normalization_epsilon configures only the built-in estimator. This plain
    function boundary also supports externally prepared illumination maps.
    """
    images = gray_sequence(images)
    if illumination_estimator is None:
        direct, global_component = estimate_illumination(
            images, normalization_epsilon=normalization_epsilon
        )
    else:
        direct, global_component = illumination_estimator(images)
    if direct.shape != images.shape[1:] or global_component.shape != images.shape[1:]:
        raise ValueError(
            "Illumination components must match the grayscale camera shape (H,W)"
        )
    return classify_gray_bits(
        images,
        direct,
        global_component,
        contrast_margin=contrast_margin,
        minimum_direct=minimum_direct,
    )


def aggregate_repeated_bits(
    column_runs: Sequence[np.ndarray], row_runs: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Combine repeated captures by the original maximum per classified bit."""
    column_bits = np.max(column_runs, axis=0)
    row_bits = np.max(row_runs, axis=0)
    return column_bits, row_bits


def decode_gray_word(word: Sequence[int] | np.ndarray) -> int:
    """Decode one bit sequence, returning ``-1`` if a bit is unknown.

    String conversion and integer Gray decoding follow the reference, rather
    than introducing a vectorized conversion or changing invalid-word rules.
    """
    bit_strings = [str(word[index]) for index in range(0, len(word))]
    gray_string = "".join(bit_strings)
    if "-1" in gray_string:
        return -1

    value = int(gray_string, 2)
    shifted = value >> 1
    while shifted:
        value ^= shifted
        shifted >>= 1
    return value


def decode_identifier_maps(
    column_bits: np.ndarray, row_bits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``(B, H, W)`` bits into two ``(H, W)`` identifier maps.

    The nested pixel traversal and NumPy-inferred integer dtype are retained.
    Only each row-axis word is reversed, as in the original capture scripts.
    """
    column_identifiers = np.array(
        [
            decode_gray_word(column_bits[:, y, x])
            for y in range(0, column_bits.shape[1])
            for x in range(0, column_bits.shape[2])
        ]
    ).reshape((column_bits.shape[1], column_bits.shape[2]))
    row_identifiers = np.array(
        [
            decode_gray_word(np.flip(row_bits[:, y, x]))
            for y in range(0, row_bits.shape[1])
            for x in range(0, row_bits.shape[2])
        ]
    ).reshape((row_bits.shape[1], row_bits.shape[2]))
    return column_identifiers, row_identifiers
