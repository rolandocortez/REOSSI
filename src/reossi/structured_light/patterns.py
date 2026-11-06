# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND MIT
# Copyright 2026 Rolando Cortez Garcia
# Copyright (c) 2023 Guillaume Charron
# Original additions and inherited portions: see docs/licensing.md.
"""Generate and rasterize the reference Gray-code pattern sequences."""

# Gray-code primitives derived from Guillaume Charron's implementation (MIT).
# Source and attribution details: THIRD_PARTY_NOTICES.md.

import numpy as np


def generate_gray_codewords(
    width: int,
    height: int,
    *,
    identifier_origin: int = 0,
    displacement: int = 0,
) -> np.ndarray:
    """Return reference Gray words as a ``uint8`` bit matrix.

    The shape is ``(max(width, height), ceil(log2(max(width, height))))``.
    Origin zero reproduces Charron's identifiers; origin one reproduces the
    legacy REOSSI identifiers. Displacement shifts identifier cells before
    rasterization, with the leading cells filled with zero. The reference
    bit count, integer dtype, and byte swapping are retained.
    """
    max_size = max(width, height)
    bit_count = int(np.ceil(np.log2(max_size)))

    identifiers = np.arange(max_size, dtype=np.uint16) + identifier_origin
    shifted_identifiers = np.roll(identifiers, displacement)
    shifted_identifiers[:displacement] = 0

    gray_words = (shifted_identifiers >> 1) ^ shifted_identifiers
    gray_words.byteswap(inplace=True)
    return np.unpackbits(gray_words.view(dtype=np.uint8)).reshape((-1, 16))[
        :, 16 - bit_count :
    ]


def render_gray_patterns(codewords: np.ndarray, width: int, height: int) -> np.ndarray:
    """Render words into the reference ``uint8`` sequence of shape ``(N, H, W)``.

    Black and white reference frames precede interleaved column and row
    patterns, followed by their inverses. Frame order, stripe sizing, and row
    indexing follow the original renderer, including its edge behavior.
    """
    images = np.zeros((4 * len(codewords[0]) + 2, height, width), dtype=np.uint8)
    images[0, :, :] = 0
    images[1, :, :] = 255
    stripe_size = width // len(codewords)

    for identifier, word in enumerate(codewords):
        for bit_index, bit in enumerate(word):
            start = identifier * stripe_size
            end = (identifier + 1) * stripe_size

            column_frame = 2 * bit_index + 2
            row_frame = 2 * (len(word) - (bit_index + 1)) + 3
            inverse_column_frame = 2 * bit_index + 2 * len(word) + 2
            inverse_row_frame = 2 * (len(word) - (bit_index + 1)) + 3 + 2 * len(word)

            images[column_frame, :, start:end] = 255 if bit == 1 else 0
            images[inverse_column_frame, :, start:end] = (
                255 - images[column_frame, :, start:end]
            )
            if identifier < height:
                images[row_frame, start:end, :] = 255 if bit == 1 else 0
                images[inverse_row_frame, start:end, :] = (
                    255 - images[row_frame, start:end, :]
                )

    return images


def shift_native_patterns(
    patterns: np.ndarray,
    projector_size: tuple[int, int],
    displacement: int = 0,
) -> np.ndarray:
    """Enlarge a Gray sequence and translate bit frames by native pixels.

    Nearest-neighbor enlargement precedes translation. Column patterns move
    right and row patterns move down, without wraparound. The new prefix
    represents Gray identifier zero: normal bits are black, inverses white.
    Black/white reference frames remain unchanged. The input is not modified.
    """
    width, height = projector_size
    columns = np.arange(width) * patterns.shape[2] // width
    rows = np.arange(height) * patterns.shape[1] // height
    native = patterns[:, rows[:, None], columns[None, :]]
    if displacement == 0:
        return native
    inverse_start = (len(patterns) - 2) // 2 + 2
    for frame_index in range(2, len(patterns)):
        frame = native[frame_index]
        prefix = 0 if frame_index < inverse_start else 255
        if frame_index % 2 == 0:
            frame[:, displacement:] = frame[:, :-displacement].copy()
            frame[:, :displacement] = prefix
        else:
            frame[displacement:, :] = frame[:-displacement, :].copy()
            frame[:displacement, :] = prefix
    return native
