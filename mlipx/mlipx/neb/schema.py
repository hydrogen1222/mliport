"""Typed, calculator-independent options for fixed-cell NEB."""

from __future__ import annotations

from dataclasses import dataclass
import math


class NEBError(ValueError):
    """Base class for invalid or unsupported NEB inputs."""


class NEBPreparationError(NEBError):
    """The band cannot be constructed without changing its physical meaning."""


class NEBEndpointNotConvergedError(NEBError):
    """An endpoint requested for relaxation did not converge."""


def _finite_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool or float")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass(frozen=True, slots=True)
class NEBOptions:
    """Validated options for the first fixed-cell NEB implementation.

    ``n_intermediate_images`` follows the project contract: total images are
    this value plus the two endpoints. Only ASE's FIRE optimizer and the
    improved tangent are supported in this first implementation.
    """

    n_intermediate_images: int = 7
    climb: bool = False
    method: str = "improvedtangent"
    interpolation: str = "linear"
    path_convention: str = "mic"
    spring_eV_A2: float = 0.1
    optimizer: str = "FIRE"
    fmax_eV_A: float = 0.03
    max_steps: int = 1000
    pre_fmax_eV_A: float = 0.10
    pre_max_steps: int = 300
    maxstep_A: float = 0.10
    endpoint_policy: str = "validate"
    endpoint_fmax_eV_A: float = 0.02
    endpoint_steps: int = 500
    idpp_fmax_eV_A: float = 0.10
    idpp_steps: int = 200
    idpp_mic: bool = True
    min_distance_A: float = 0.5
    force_abort_eV_A: float = 20.0

    def __post_init__(self) -> None:
        _positive_int(self.n_intermediate_images, "n_intermediate_images")
        _positive_int(self.max_steps, "max_steps", minimum=0)
        _positive_int(self.pre_max_steps, "pre_max_steps", minimum=0)
        _positive_int(self.endpoint_steps, "endpoint_steps")
        _positive_int(self.idpp_steps, "idpp_steps")
        if self.method != "improvedtangent":
            raise ValueError("Only NEB method='improvedtangent' is supported")
        if self.optimizer.upper() != "FIRE":
            raise ValueError("Only NEB optimizer='FIRE' is supported")
        if self.interpolation not in {"linear", "idpp"}:
            raise ValueError("interpolation must be linear or idpp")
        if self.path_convention not in {"mic", "unwrapped"}:
            raise ValueError("path_convention must be mic or unwrapped")
        if self.endpoint_policy not in {"validate", "relax"}:
            raise ValueError("endpoint_policy must be validate or relax")
        for value, name in (
            (self.spring_eV_A2, "spring_eV_A2"),
            (self.fmax_eV_A, "fmax_eV_A"),
            (self.pre_fmax_eV_A, "pre_fmax_eV_A"),
            (self.maxstep_A, "maxstep_A"),
            (self.endpoint_fmax_eV_A, "endpoint_fmax_eV_A"),
            (self.idpp_fmax_eV_A, "idpp_fmax_eV_A"),
            (self.min_distance_A, "min_distance_A"),
            (self.force_abort_eV_A, "force_abort_eV_A"),
        ):
            _finite_positive(value, name)
        if not isinstance(self.climb, bool) or not isinstance(self.idpp_mic, bool):
            raise TypeError("climb and idpp_mic must be booleans")
