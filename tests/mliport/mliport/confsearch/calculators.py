"""Test-only synthetic calculators (spec section 13.1).

No backend imports: every potential here is analytic, CPU-only and
deterministic, so the whole search engine is testable in CI.

* :class:`EggCrateCalculator` -- a rigid periodic slab plus a 3-atom bonded
  molecule whose anchor atom feels a piecewise-harmonic site term.  The minima
  sit exactly at known sites with exactly known energies.
* :class:`MorseBreakCalculator` -- the same slab/site term with a shallow Morse
  bond so that hot MD breaks the molecule reproducibly.
* :class:`DirectForceCalculator` -- advertises ``direct_forces=True``.
* :class:`NonFiniteCalculator` -- returns NaN for the energy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Anchor-atom index inside the adsorbate block.
ANCHOR_LOCAL_INDEX = 0


def _mic(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = np.asarray(delta, dtype=float) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


class EggCrateCalculator(Calculator):
    """Rigid slab + 3-atom molecule with a piecewise-harmonic site term.

    Sites (fractional, in units of the in-plane cell) and their exact minimum
    energies (eV):

    ``hollow`` (1/2, 1/2) -> -0.60, ``bridge`` (1/2, 0) -> -0.35,
    ``top`` (0, 0) -> -0.20.

    The internal terms vanish at their minimum geometry (bonds ``bond_A``,
    angle ``angle_deg``), so a relaxed molecule sitting on a site has exactly
    the site energy.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        *,
        sites: Sequence[tuple[float, float, float]] | None = None,
        bond_A: float = 1.5,
        angle_deg: float = 109.47,
        bond_k: float = 20.0,
        angle_k: float = 5.0,
        site_k: float = 4.0,
        n_slab: int = 4,
        spacing_A: float = 3.0,
    ) -> None:
        super().__init__()
        self.sites = list(
            sites
            if sites is not None
            else [
                (0.5, 0.5, -0.60),  # hollow
                (0.5, 0.0, -0.35),  # bridge
                (0.0, 0.0, -0.20),  # top
            ]
        )
        self.bond_A = float(bond_A)
        self.angle = np.deg2rad(angle_deg)
        self.bond_k = float(bond_k)
        self.angle_k = float(angle_k)
        self.site_k = float(site_k)
        self.n_slab = int(n_slab)
        self.spacing_A = float(spacing_A)

    # -- helpers ---------------------------------------------------------
    @property
    def cell(self) -> np.ndarray:
        return np.diag(
            [self.n_slab * self.spacing_A, self.n_slab * self.spacing_A, 20.0]
        )

    def _site_energy_forces(self, position: np.ndarray) -> tuple[float, np.ndarray]:
        """Nearest-site piecewise harmonic term (exact minima at the sites)."""
        cell = self.cell
        in_plane = cell[:2, :2]
        best: tuple[float, np.ndarray] | None = None
        for u, v, depth in self.sites:
            target = np.array([u * cell[0, 0], v * cell[1, 1], position[2]])
            delta = _mic(position - target, cell)
            delta[2] = 0.0
            value = float(depth) + 0.5 * self.site_k * float(np.dot(delta, delta))
            if best is None or value < best[0]:
                best = (value, delta)
        assert best is not None
        return best[0], -self.site_k * best[1]

    def _internal(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """Harmonic bonds 0-1, 1-2 and a harmonic angle 0-1-2."""
        energy = 0.0
        forces = np.zeros_like(positions)
        for first, second in ((0, 1), (1, 2)):
            delta = positions[second] - positions[first]
            distance = float(np.linalg.norm(delta))
            stretch = distance - self.bond_A
            energy += 0.5 * self.bond_k * stretch**2
            # Force ON the second atom; the first feels the opposite one.
            force_on_second = -self.bond_k * stretch * delta / distance
            forces[second] += force_on_second
            forces[first] -= force_on_second
        first, middle, last = positions[0], positions[1], positions[2]
        left = first - middle
        right = last - middle
        cos_theta = float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        )
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        energy += 0.5 * self.angle_k * (theta - self.angle) ** 2
        # d(theta)/dr for a three-body angle; standard chain-rule form.
        sin_theta = float(np.sin(theta)) or 1e-12
        gradient_first = (
            cos_theta * left / np.dot(left, left)
            - right / (np.linalg.norm(left) * np.linalg.norm(right))
        ) / sin_theta
        gradient_last = (
            cos_theta * right / np.dot(right, right)
            - left / (np.linalg.norm(left) * np.linalg.norm(right))
        ) / sin_theta
        dtheta = self.angle_k * (theta - self.angle)
        forces[0] -= dtheta * gradient_first
        forces[2] -= dtheta * gradient_last
        forces[1] -= -dtheta * (gradient_first + gradient_last)
        return energy, forces

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        n_slab = len(atoms) - 3
        energy = 0.0
        forces = np.zeros_like(positions)
        molecule = positions[n_slab:]
        site_energy, site_force = self._site_energy_forces(molecule[ANCHOR_LOCAL_INDEX])
        energy += site_energy
        forces[n_slab + ANCHOR_LOCAL_INDEX] += site_force
        internal_energy, internal_forces = self._internal(molecule)
        energy += internal_energy
        forces[n_slab:] += internal_forces
        self.results = {"energy": float(energy), "forces": forces}


class MorseBreakCalculator(EggCrateCalculator):
    """Egg-crate model whose first bond is a shallow Morse potential.

    ``D_e`` is small enough that a 1000 K Langevin segment breaks the bond
    reproducibly with a fixed seed.
    """

    def __init__(self, *, D_e: float = 0.04, alpha: float = 1.0, **kwargs) -> None:
        kwargs.setdefault("bond_k", 20.0)
        super().__init__(**kwargs)
        self.D_e = float(D_e)
        self.alpha = float(alpha)

    def _internal(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        energy, forces = super()._internal(positions)
        # Replace the harmonic 0-1 bond by a Morse bond (subtract the harmonic
        # term, add the Morse term).
        delta = positions[1] - positions[0]
        distance = float(np.linalg.norm(delta))
        stretch = distance - self.bond_A
        # Remove the harmonic 0-1 term (energy and both forces).
        energy -= 0.5 * self.bond_k * stretch**2
        harmonic_on_second = -self.bond_k * stretch * delta / distance
        forces[1] -= harmonic_on_second
        forces[0] += harmonic_on_second
        # Add the Morse term.
        exponential = np.exp(-self.alpha * stretch)
        morse = self.D_e * (1.0 - exponential) ** 2
        d_morse = 2.0 * self.D_e * self.alpha * exponential * (1.0 - exponential)
        energy += morse
        morse_on_second = -d_morse * delta / distance
        forces[1] += morse_on_second
        forces[0] -= morse_on_second
        return energy, forces


class DirectForceCalculator(EggCrateCalculator):
    """Advertises ``direct_forces`` so the identity gate can be tested."""

    def info(self) -> dict:
        info = dict(super().info())
        info["direct_forces"] = True
        return info


class NonFiniteCalculator(EggCrateCalculator):
    """Well-behaved for ``fail_after`` evaluations, then NaN energy/forces.

    The start quench must succeed, so the NaN has to appear during the MD
    segment; the counter makes that deterministic.
    """

    def __init__(self, *, fail_after: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_after = int(fail_after)
        self.calls = 0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        self.calls += 1
        if self.calls <= self.fail_after:
            super().calculate(atoms, properties, system_changes)
            return
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float("nan"),
            "forces": np.full_like(atoms.positions, float("nan")),
        }


__all__ = [
    "ANCHOR_LOCAL_INDEX",
    "DirectForceCalculator",
    "EggCrateCalculator",
    "MorseBreakCalculator",
    "NonFiniteCalculator",
]
