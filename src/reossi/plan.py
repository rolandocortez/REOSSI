"""Explicit parameters for the original scanner and historical REOSSI paths."""

from dataclasses import asdict, dataclass, fields
from math import ceil, isfinite, log2
from numbers import Integral, Real
from typing import Literal

ScanMode = Literal["conventional", "ld", "reossi"]


@dataclass(frozen=True)
class ScanPlan:
    """Describe one scan without opening a camera or generating image stacks.

    Sizes are ``(width, height)``. Conventional mode reproduces Charron's
    zero-based generator and illumination normalization. LD and REOSSI use
    the local one-based generator and ``1e-5`` normalization epsilon.

    The factors deliberately use the legacy maximum-dimension integer ratios.
    ``target_size`` preserves the historical intermediate-resolution variant,
    including its unresolved displacement-unit convention.
    """

    mode: ScanMode = "reossi"
    projector_size: tuple[int, int] = (1920, 1080)
    emulated_size: tuple[int, int] = (320, 180)
    target_size: tuple[int, int] | None = None
    repeats: int = 2
    contrast_margin: float = 1
    minimum_direct: float = 10
    illumination_epsilon: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {
            "conventional",
            "ld",
            "reossi",
        }:
            raise ValueError(f"Unknown scan mode: {self.mode}")
        sizes = [self.projector_size, self.emulated_size]
        if self.target_size is not None:
            sizes.append(self.target_size)
        if any(
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < 2
                for value in size
            )
            for size in sizes
        ):
            raise ValueError("Each resolution must contain width and height >= 2")
        for name in ("projector_size", "emulated_size", "target_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(int(v) for v in value))
        if (
            not isinstance(self.repeats, Integral)
            or isinstance(self.repeats, bool)
            or self.repeats < 1
        ):
            raise ValueError("repeats must be positive")
        if any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in (self.contrast_margin, self.minimum_direct)
        ):
            raise ValueError("Classification thresholds must be nonnegative")
        if self.illumination_epsilon is not None and (
            not isinstance(self.illumination_epsilon, Real)
            or isinstance(self.illumination_epsilon, bool)
            or not isfinite(self.illumination_epsilon)
            or self.illumination_epsilon < 0
        ):
            raise ValueError("illumination_epsilon must be nonnegative")
        if self.mode == "conventional" and self.target_size is not None:
            raise ValueError("Conventional mode does not use an intermediate target")
        if self.scale_factor < 1 or self.native_step < 1:
            raise ValueError("The legacy scale and native step must be at least one")
        if self.pattern_size[0] < self.pattern_size[1]:
            raise ValueError(
                "The historical renderer requires landscape or square patterns"
            )
        if not 3 <= ceil(log2(max(self.pattern_size))) <= 16:
            raise ValueError(
                "The historical generator/selectors require 3..16 pattern bits"
            )

    @property
    def pattern_size(self) -> tuple[int, int]:
        """Dimensions passed to the historical pattern renderer."""
        return (
            self.projector_size if self.mode == "conventional" else self.emulated_size
        )

    @property
    def scale_factor(self) -> int:
        """Legacy ratio between the target/native grid and the LD grid."""
        target = self.target_size or self.projector_size
        return int(max(target) / max(self.pattern_size))

    @property
    def native_step(self) -> int:
        """Historical native/target ratio; this is not a measured physical shift."""
        if self.target_size is None:
            return 1
        return int(max(self.projector_size) / max(self.target_size))

    @property
    def displacement_count(self) -> int:
        """Total acquisitions including the initial position, not extra shifts."""
        return self.scale_factor if self.mode == "reossi" else 1

    @property
    def identifier_origin(self) -> int:
        return 0 if self.mode == "conventional" else 1

    @property
    def normalization_epsilon(self) -> float:
        if self.illumination_epsilon is not None:
            return self.illumination_epsilon
        return 0.0 if self.mode == "conventional" else 1e-5

    @property
    def sequence_length(self) -> int:
        return 4 * ceil(log2(max(self.pattern_size))) + 2

    @property
    def compatibility_notes(self) -> list[str]:
        """Explain retained source limitations without changing their formulas."""
        notes = []
        target = self.target_size or self.projector_size
        for name, larger, smaller in (
            ("target/pattern", target, self.pattern_size),
            ("native/target", self.projector_size, target),
        ):
            if larger[0] * smaller[1] != larger[1] * smaller[0]:
                notes.append(
                    f"{name} aspect ratios differ; the source uses the maximum dimension"
                )
            if max(larger) % max(smaller):
                notes.append(f"{name} ratio is truncated to an integer by the source")
        maximum = max(self.pattern_size)
        if self.identifier_origin == 1 and maximum & (maximum - 1) == 0:
            notes.append(
                "The final one-based identifier exceeds the historical Gray bit count"
            )
        if self.mode == "reossi":
            notes.append(
                "Displacements roll whole LD identifier cells before rendering"
            )
        if self.target_size is not None:
            notes.append(
                "Intermediate target displacement units retain the unresolved legacy convention"
            )
        return notes

    def to_dict(self) -> dict:
        """Return JSON-compatible scan parameters, independent of hardware."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ScanPlan":
        if not isinstance(values, dict):
            raise ValueError("Scan plan must be a JSON object")
        parameters = dict(values)
        unknown = set(parameters) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown scan plan fields: {', '.join(sorted(unknown))}")
        for name in ("projector_size", "emulated_size", "target_size"):
            if parameters.get(name) is not None:
                if not isinstance(parameters[name], (tuple, list)):
                    raise ValueError(f"{name} must contain width and height")
                parameters[name] = tuple(parameters[name])
        return cls(**parameters)
