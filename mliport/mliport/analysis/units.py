"""Explicit unit conversions used by Analysis v2.

Analysis results never expose an unqualified scalar such as ``D`` or
``sigma``.  Keeping the conversions here makes the unit contract executable
and easy to cross-check in tests.
"""

from __future__ import annotations

ANGSTROM_TO_M = 1.0e-10
FS_TO_S = 1.0e-15
PS_TO_S = 1.0e-12
M2_S_TO_CM2_S = 1.0e4
CM2_S_TO_M2_S = 1.0e-4
S_M_TO_S_CM = 1.0e-2
S_M_TO_MS_CM = 10.0
EV_TO_J = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


def diffusion_A2_fs_to_m2_s(value: float) -> float:
    """Convert a diffusion coefficient from Angstrom^2/fs to m^2/s."""

    return float(value) * ANGSTROM_TO_M**2 / FS_TO_S


def diffusion_m2_s_to_cm2_s(value: float) -> float:
    """Convert a diffusion coefficient from m^2/s to cm^2/s."""

    return float(value) * M2_S_TO_CM2_S
