"""Shared scalar response for raster and geometric virtual captures."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorModel:
    """Float64 response levels; Gaussian noise uses 0..255 intensity units."""

    black_level: float = 2.0
    direct_level: float = 180.0
    noise_std: float = 0.0

    def __post_init__(self):
        values = (self.black_level, self.direct_level, self.noise_std)
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(
                "Sensor levels and noise_std must be finite and nonnegative"
            )
