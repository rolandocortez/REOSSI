"""Reference direct/global illumination estimation for Gray-code decoding."""

# Derived from Guillaume Charron's implementation (MIT).
# See THIRD_PARTY_NOTICES.md for attribution and cited algorithm sources.

from collections.abc import Callable

import numpy as np

from ._validation import gray_sequence

IlluminationEstimator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def estimate_illumination(
    images: np.ndarray, *, normalization_epsilon: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate direct and global illumination from ``(N, H, W)`` images.

    An epsilon of zero preserves Charron's normalization expression; ``1e-5``
    reproduces the local legacy expression. Input arithmetic and the selected
    reference frames retain their original dtype and operation order.
    """
    images = gray_sequence(images)
    if not np.isfinite(normalization_epsilon) or normalization_epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and nonnegative")
    black = images[0]
    white = images[1]
    pattern_length = (len(images) - 2) / 4
    horizontal_indices = np.array(
        [
            2 * pattern_length - 2,
            2 * pattern_length - 4,
            2 * pattern_length - 6,
            4 * pattern_length - 2,
            4 * pattern_length - 4,
            4 * pattern_length - 6,
        ],
        dtype=np.uint8,
    )
    vertical_indices = np.array(
        [
            1,
            3,
            5,
            2 * pattern_length + 1,
            2 * pattern_length + 3,
            2 * pattern_length + 5,
        ],
        dtype=np.uint8,
    )
    if normalization_epsilon == 0.0:
        inverse_normalization = white / (white + black)
    else:
        inverse_normalization = white / (white + black + normalization_epsilon)

    remaining_images = images[2:]
    maximum = np.max(remaining_images[horizontal_indices], axis=0)
    minimum = np.min(remaining_images[vertical_indices], axis=0)
    direct = (maximum - minimum) * inverse_normalization
    global_component = 2.0 * (maximum - direct) * inverse_normalization
    return direct, global_component
