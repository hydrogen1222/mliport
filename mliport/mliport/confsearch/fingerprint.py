"""Focused Oganov-Valle fingerprint (spec section 9.2 revision 5).

ASE's ``OFPComparator`` is deliberately not called: with ``n_top`` it
fingerprints the adsorbate in isolation (slab atoms are excluded, so adsorption
sites are invisible) and it imports matplotlib at module level.  This module
implements the descriptor in NumPy instead.

Species structure: centres are the adsorbate atoms ``A``, neighbours are all
atoms, and every **ordered** species pair ``(alpha, beta)`` with
``alpha in species(A)`` and ``beta in species(all)`` gets its own fingerprint

    F_ab(r_b) = sum_{i in A, a} sum_{j in b} g_bin(d_ij)
                / (4 pi r_b^2 (N_b / V) Delta n_a) - 1

with the weights ``w_ab = n_a N_b / sum(n_a' N_b')`` and
``D(1,2) = 0.5 (1 - sum w F1.F2 / (|F1| |F2|))``, ``|F| = sqrt(sum w F.F)``.
With ``A`` = all atoms the ordered matrix is symmetric (``F_ab == F_ba``), which
is what makes the strict known-answer comparison below possible.

Discretisation: **ASE's, exactly** (spec section 9.2 revision 3).  The
pre-revision-3 text described a point-sampled Gaussian on bins
``r_k = (k + 1/2) Delta`` with an extra ``1/Delta`` and could not satisfy the
known-answer requirement; that formulation is withdrawn.  This module uses
``nbins = ceil(rcut / Delta)``, bin radius ``r_k = (k + 1) Delta``, a pair in
bin ``k = floor(d / Delta)`` spread over ``k + i`` for ``i = -m..m`` with
``m = ceil(4 sigma / Delta)`` and the analytic bin-integrated weights
``0.5 [erf(c(2i+1)) - erf(c(2i-1))] / erf(c(2m+1))``, ``c = sqrt(2) Delta / (4 sigma)``,
renormalised to sum to 1, with out-of-range bins dropped exactly as ASE does.

Known answer (strict, always runs in CI): for every unordered key the per-bin
arrays equal ``OFPComparator(rcut, binwidth, sigma, nsigma=4)._take_fingerprints``
to **1e-10 absolute**, ``F_ab == F_ba`` to 1e-10, and collapsing to unordered
keys with ASE's weights reproduces ASE's ``_cosine_distance`` to 1e-10 on
several structure pairs.  ASE 3.29 turned ``ase.ga.ofp_comparator`` into a
deprecation shim, so the test falls back to a verbatim copy of the ASE 3.26
algorithm (MIT) and therefore never skips; ``scipy`` and ``matplotlib`` are in
the ``dev`` extra so the real class is used when it is importable.

Self-image pairs (``i == j`` with a non-zero lattice offset) are enumerated
explicitly instead of being taken from ``neighbor_list``.  ``neighbor_list``'s
self-image shell depends on where an image falls relative to the cutoff: on a
4x4x20 A slab with rcut=8 (images at exactly 8.0 A) it returned 48 self pairs
before and 72 after a rigid rotation of the whole cell, while the 350 cross pairs
were identical.  That is a cutoff-boundary effect, not a claim about ASE's
correctness, and it is not acceptable here because it would make the descriptor
depend on the cell's orientation in space.  A self-image distance is
``|n . cell|``, so the explicit enumeration is complete and rotation-covariant;
ASE's ``NeighborList`` (which ``OFPComparator`` uses) does return the complete
shell, so the known-answer comparison is unaffected.

The 1e-9 A rounding in :func:`_bin_weights` keeps the bin index stable when the
same structure is rigidly rotated: without it, a distance at a decimal multiple
of ``Delta`` (a 0.96 A O-H bond is 47.99999999999999 bins) flips its whole
21-bin smearing window.  It is the only deviation from a verbatim port of ASE's
discretisation and cannot change a per-bin value unless a distance lies within
5e-10 A of a bin boundary without lying exactly on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase.neighborlist import neighbor_list

from mliport.confsearch.errors import ConfsearchError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ase import Atoms

    from mliport.confsearch.options import ConfsearchOptions

#: Truncation of the Gaussian in units of sigma (spec section 9.2).
NSIGMA = 4.0
#: Shortcut: energy gaps above this skip the (expensive) fingerprint distance.
DEDUP_ENERGY_SHORTCUT_EV = 0.2

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class FingerprintGrid:
    """The discretisation shared by every fingerprint of one search."""

    rcut: float
    binwidth: float
    sigma: float
    n_bins: int
    centres: np.ndarray
    #: Per-bin Gaussian weights for offsets ``-m..m`` (normalised to sum 1).
    smearing: np.ndarray

    @classmethod
    def build(cls, *, rcut: float, binwidth: float, sigma: float) -> FingerprintGrid:
        n_bins = int(math.ceil(rcut / binwidth))
        if n_bins < 1:
            raise ConfsearchError(
                f"cs_fp_rcut ({rcut}) must be at least one bin "
                f"(cs_fp_binwidth={binwidth})"
            )
        centres = binwidth * np.arange(1, n_bins + 1, dtype=float)
        m = int(math.ceil(NSIGMA * sigma / binwidth))
        c = 0.25 * _SQRT2 * binwidth / sigma
        smearing = np.array(
            [
                0.5 * (math.erf(c * (2 * i + 1)) - math.erf(c * (2 * i - 1)))
                for i in range(-m, m + 1)
            ],
            dtype=float,
        )
        smearing = smearing / math.erf(c * (2 * m + 1))
        return cls(
            rcut=float(rcut),
            binwidth=float(binwidth),
            sigma=float(sigma),
            n_bins=n_bins,
            centres=centres,
            smearing=smearing,
        )

    @property
    def window(self) -> int:
        return (self.smearing.size - 1) // 2


@dataclass(frozen=True, slots=True)
class FocusedFingerprint:
    """The focused descriptor of one structure."""

    species_centres: tuple[str, ...]
    species_all: tuple[str, ...]
    values: Mapping[tuple[str, str], np.ndarray]
    weights: Mapping[tuple[str, str], float]

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.values)

    def norm(self) -> float:
        total = 0.0
        for key, value in self.values.items():
            total += self.weights[key] * float(np.dot(value, value))
        return math.sqrt(total)

    def distance(self, other: FocusedFingerprint) -> float:
        """Cosine-style distance ``D`` from spec section 9.2."""
        if self.species_centres != other.species_centres or (
            self.species_all != other.species_all
        ):
            raise ConfsearchError(
                "Fingerprints are only comparable when the species sets match; "
                f"got {self.species_centres}/{self.species_all} and "
                f"{other.species_centres}/{other.species_all}"
            )
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self == 0.0 and norm_other == 0.0:
            return 0.0
        if norm_self == 0.0 or norm_other == 0.0:
            return 1.0
        dot = 0.0
        for key, value in self.values.items():
            dot += self.weights[key] * float(np.dot(value, other.values[key]))
        return 0.5 * (1.0 - dot / (norm_self * norm_other))

    def to_dict(self) -> dict:
        return {
            "species_centres": list(self.species_centres),
            "species_all": list(self.species_all),
            "weights": {f"{a}-{b}": w for (a, b), w in self.weights.items()},
            "values": {
                f"{a}-{b}": [float(x) for x in value]
                for (a, b), value in self.values.items()
            },
        }


def lattice_image_distances(
    cell: np.ndarray, rcut: float, *, max_images: int = 100_000
) -> np.ndarray:
    """Distances ``|n . cell|`` in ``(0, rcut]`` for all non-zero integer ``n``.

    The search range is rigorous: ``|n . cell| >= |n_i| d_i`` where ``d_i`` is
    the spacing between the lattice planes perpendicular to cell vector ``i``,
    so only ``|n_i| <= rcut / d_i`` can contribute.  Lengths are rounded to
    1e-9 A before the ``< rcut`` test for the same reason as
    :func:`_bin_weights`: an image at exactly ``rcut`` must be excluded
    consistently, whatever the cell's orientation in space.
    """
    matrix = np.asarray(cell, dtype=float)
    volume = abs(float(np.linalg.det(matrix)))
    if volume <= 0.0:
        raise ConfsearchError("Self-image distances need a full-rank cell")
    ranges = []
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        area = float(np.linalg.norm(np.cross(matrix[others[0]], matrix[others[1]])))
        spacing = volume / area if area > 0 else 0.0
        if spacing <= 0.0:
            raise ConfsearchError("Cell vectors must be non-degenerate")
        limit = int(math.floor(rcut / spacing)) + 1
        ranges.append(range(-limit, limit + 1))
    combinations = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T
    if combinations.shape[0] > max_images:
        raise ConfsearchError(
            f"cs_fp_rcut={rcut} needs {combinations.shape[0]} lattice images for "
            "the self-image check; reduce cs_fp_rcut or use a larger cell"
        )
    nonzero = combinations[np.any(combinations != 0, axis=1)]
    if nonzero.size == 0:
        return np.zeros(0, dtype=float)
    lengths = np.linalg.norm(nonzero @ matrix, axis=1)
    keep = (lengths > 0.0) & (np.round(lengths, 9) < rcut)
    return np.sort(lengths[keep])


def _bin_weights(distances: np.ndarray, grid: FingerprintGrid) -> np.ndarray:
    """Accumulate the smeared pair distances into the radial bins.

    The binning is ASE's: ``k = floor(d / Delta)`` and the pair is spread over
    ``k + i`` for ``i = -m..m`` with the out-of-range bins dropped.  The only
    deviation is that ``d`` is rounded to 1e-9 A before the division, because a
    distance that is a decimal multiple of Delta (a textbook bond length such as
    0.96 A) sits systematically ~1e-16 below the bin boundary and a rigid
    rotation of the same structure would flip its whole 21-bin smearing window,
    breaking the rotation invariance section 9.2 requires.  1e-9 A is 1e-3 fm,
    far below any physical distance difference; for a distance that is exactly
    on a boundary both implementations still agree, and elsewhere they agree
    unless a distance lies within 5e-10 A of a boundary without being on it.
    """
    values = np.zeros(grid.n_bins, dtype=float)
    if distances.size == 0:
        return values
    base = np.floor(np.round(distances, 9) / grid.binwidth).astype(int)
    window = grid.window
    for offset_index, offset in enumerate(range(-window, window + 1)):
        targets = base + offset
        valid = (targets >= 0) & (targets < grid.n_bins)
        if not np.any(valid):
            continue
        np.add.at(values, targets[valid], grid.smearing[offset_index])
    return values


def build_fingerprint(
    atoms: Atoms,
    adsorbate: np.ndarray,
    *,
    options: ConfsearchOptions,
    grid: FingerprintGrid | None = None,
) -> FocusedFingerprint:
    """Build the focused fingerprint of ``atoms`` around the ``adsorbate`` set."""
    centres = np.asarray(adsorbate, dtype=int)
    if centres.size == 0:
        raise ConfsearchError("A fingerprint needs a non-empty adsorbate set A")
    if grid is None:
        grid = FingerprintGrid.build(
            rcut=options.cs_fp_rcut,
            binwidth=options.cs_fp_binwidth,
            sigma=options.cs_fp_sigma,
        )

    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    volume = float(abs(np.linalg.det(np.asarray(atoms.cell, dtype=float))))
    if volume <= 0.0:
        raise ConfsearchError(
            "A focused fingerprint needs a full-rank cell; got a zero volume"
        )

    species_centres = tuple(sorted(set(symbols[centres])))
    species_all = tuple(sorted(set(symbols)))
    counts_centres = {
        species: int(np.count_nonzero(symbols[centres] == species))
        for species in species_centres
    }
    counts_all = {
        species: int(np.count_nonzero(symbols == species)) for species in species_all
    }

    i, j, distances = neighbor_list("ijd", atoms, grid.rcut, self_interaction=False)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    distances = np.asarray(distances, dtype=float)
    # Cross pairs come from ASE's neighbour list; self-image pairs (i == j) are
    # enumerated explicitly below because that function returns an
    # orientation-dependent subset of them.
    keep = (i != j) & np.isin(i, centres)
    i, j, distances = i[keep], j[keep], distances[keep]
    self_image_distances = lattice_image_distances(atoms.cell, grid.rcut)

    total_weight = 0.0
    weights: dict[tuple[str, str], float] = {}
    for alpha in species_centres:
        for beta in species_all:
            weight = counts_centres[alpha] * counts_all[beta]
            weights[(alpha, beta)] = float(weight)
            total_weight += weight

    values: dict[tuple[str, str], np.ndarray] = {}
    for alpha in species_centres:
        for beta in species_all:
            selected = (symbols[i] == alpha) & (symbols[j] == beta)
            bins = _bin_weights(distances[selected], grid)
            if alpha == beta and self_image_distances.size:
                # Every centre of species alpha sees the same image shell.
                bins = bins + _bin_weights(
                    np.tile(self_image_distances, counts_centres[alpha]), grid
                )
            density = counts_all[beta] / volume
            denominator = (
                4.0
                * math.pi
                * grid.centres**2
                * density
                * grid.binwidth
                * counts_centres[alpha]
            )
            values[(alpha, beta)] = bins / denominator - 1.0
            weights[(alpha, beta)] /= total_weight
    return FocusedFingerprint(
        species_centres=species_centres,
        species_all=species_all,
        values=values,
        weights=weights,
    )


def fingerprint_distance(
    first: FocusedFingerprint, second: FocusedFingerprint
) -> float:
    """Distance between two fingerprints (spec section 9.2)."""
    return first.distance(second)


__all__ = [
    "DEDUP_ENERGY_SHORTCUT_EV",
    "NSIGMA",
    "FingerprintGrid",
    "FocusedFingerprint",
    "build_fingerprint",
    "fingerprint_distance",
    "lattice_image_distances",
]
