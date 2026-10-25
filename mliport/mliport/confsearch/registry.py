"""Unique-minimum registry, deduplication and consolidation (spec section 9).

Matching rule (section 9.3 revision 4): a new minimum ``m`` joins unique minimum
``u`` iff

    |E_m - E_u| <= cs_dedup_energy_tol   AND   ( G(m, u) OR D(m, u) <= cs_dedup_fp_tol )

where ``G`` is the *direct geometric identity*: with a non-empty fixed set F and
a fixed atom order, the largest minimum-image displacement over the adsorbate
atoms is at most ``cs_dedup_geom_tol``.  ``G`` absorbs quench noise in flat
basins; ``D`` is only needed for what ``G`` cannot see (symmetry-equivalent
sites and permutation-equivalent conformers).  If F is empty, ``G`` is disabled
for the whole search and recorded in the run warnings.

Only representatives are compared; when several match, the smallest
``(not G, D)`` wins (G first, then smallest D; ties: lowest id).  ``|dE| > 0.2
eV`` skips G and D entirely.

Every status class has its own registry and classes never merge.  Registration
order is execution order, so the ids are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mliport.confsearch.fingerprint import (
    DEDUP_ENERGY_SHORTCUT_EV,
    FingerprintGrid,
    FocusedFingerprint,
    build_fingerprint,
)
from mliport.confsearch.geometry import mic_vectors

if TYPE_CHECKING:
    from ase import Atoms

    from mliport.confsearch.graphs import MinimumClassification
    from mliport.confsearch.options import ConfsearchOptions


@dataclass(frozen=True, slots=True)
class MinimumOrigin:
    """Where a quench came from: ``(replica, cycle, source)``."""

    replica: int
    cycle: int
    source: str
    snapshot_index: int | None = None

    def to_dict(self) -> dict:
        record = {
            "replica": self.replica,
            "cycle": self.cycle,
            "source": self.source,
        }
        if self.snapshot_index is not None:
            record["snapshot_index"] = self.snapshot_index
        return record

    def label(self) -> str:
        if self.source == "start":
            return f"replica {self.replica} start quench"
        if self.source == "snapshot":
            return (
                f"replica {self.replica} cycle {self.cycle} "
                f"snapshot {self.snapshot_index}"
            )
        return f"replica {self.replica} cycle {self.cycle} end"


@dataclass(slots=True)
class MinimumRecord:
    """One converged quench (a candidate minimum)."""

    record_id: int
    energy_eV: float
    atoms: Atoms
    classification: MinimumClassification
    origin: MinimumOrigin
    quench_index: int
    cumulative_md_ps: float
    fmax_eV_A: float
    steps: int
    fingerprint: FocusedFingerprint | None = None

    @property
    def status(self) -> str:
        return self.classification.status

    @property
    def flags(self) -> tuple[str, ...]:
        return self.classification.flags


@dataclass(slots=True)
class UniqueMinimum:
    """A cluster of minima judged identical by section 9.3."""

    unique_id: int
    status: str
    members: list[MinimumRecord] = field(default_factory=list)
    replicas: set[int] = field(default_factory=set)
    first_found_md_ps: float = 0.0
    first_found_quench_index: int = 0

    @property
    def representative(self) -> MinimumRecord:
        """Lowest-energy member (spec section 9.4)."""
        return min(
            self.members, key=lambda member: (member.energy_eV, member.record_id)
        )

    @property
    def hit_count(self) -> int:
        return len(self.members)

    @property
    def energy_eV(self) -> float:
        return self.representative.energy_eV

    @property
    def flags(self) -> tuple[str, ...]:
        return self.representative.flags

    def add(self, member: MinimumRecord) -> None:
        self.members.append(member)
        self.replicas.add(member.origin.replica)

    def merge(self, other: UniqueMinimum) -> None:
        """Absorb ``other`` (same status class) into this unique minimum."""
        for member in other.members:
            self.add(member)
        self.first_found_md_ps = min(self.first_found_md_ps, other.first_found_md_ps)
        self.first_found_quench_index = min(
            self.first_found_quench_index, other.first_found_quench_index
        )

    def to_dict(self) -> dict:
        representative = self.representative
        positions = np.asarray(representative.atoms.get_positions(), dtype=float)
        return {
            "id": self.unique_id,
            "status": self.status,
            "energy_eV": float(representative.energy_eV),
            "flags": list(self.flags),
            "hit_count": self.hit_count,
            "replicas": sorted(self.replicas),
            "first_found_md_ps": float(self.first_found_md_ps),
            "first_found_quench_index": int(self.first_found_quench_index),
            "representative": {
                "record_id": representative.record_id,
                "energy_eV": float(representative.energy_eV),
                "fmax_eV_A": float(representative.fmax_eV_A),
                "steps": int(representative.steps),
                "origin": representative.origin.to_dict(),
                "numbers": [int(z) for z in representative.atoms.numbers],
                "symbols": representative.atoms.get_chemical_symbols(),
                "cell_A": np.asarray(representative.atoms.cell, dtype=float).tolist(),
                "pbc": [bool(x) for x in representative.atoms.pbc],
                "positions_A": positions.tolist(),
                "classification": representative.classification.to_dict(),
            },
            "members": [
                {
                    "record_id": member.record_id,
                    "energy_eV": float(member.energy_eV),
                    "fmax_eV_A": float(member.fmax_eV_A),
                    "steps": int(member.steps),
                    "status": member.status,
                    "flags": list(member.flags),
                    "origin": member.origin.to_dict(),
                    "cumulative_md_ps": float(member.cumulative_md_ps),
                }
                for member in self.members
            ],
        }


class MinimumRegistry:
    """Deduplicating registry of quench results, per status class."""

    def __init__(
        self,
        *,
        options: ConfsearchOptions,
        grid: FingerprintGrid | None = None,
        adsorbate: np.ndarray | None = None,
        fixed: np.ndarray | None = None,
    ) -> None:
        if adsorbate is None:
            raise ValueError("MinimumRegistry needs the adsorbate indices A")
        self.options = options
        self.adsorbate = np.asarray(adsorbate, dtype=int)
        #: G is only meaningful when fixed atoms anchor the slab (spec 9.3).
        self.geometric_identity_enabled = fixed is not None and len(fixed) > 0
        self.grid = grid or FingerprintGrid.build(
            rcut=options.cs_fp_rcut,
            binwidth=options.cs_fp_binwidth,
            sigma=options.cs_fp_sigma,
        )
        self._classes: dict[str, list[UniqueMinimum]] = {}
        self._unique_by_id: dict[int, UniqueMinimum] = {}
        self._next_unique_id = 1
        self._next_record_id = 1
        self.merges: list[dict] = []

    # -- registration ------------------------------------------------------

    def new_record_id(self) -> int:
        record_id = self._next_record_id
        self._next_record_id += 1
        return record_id

    def fingerprint(self, atoms: Atoms) -> FocusedFingerprint:
        return build_fingerprint(
            atoms, self.adsorbate, options=self.options, grid=self.grid
        )

    def register(self, record: MinimumRecord) -> UniqueMinimum:
        """Register one quench result and return its unique minimum."""
        if record.fingerprint is None:
            record.fingerprint = self.fingerprint(record.atoms)
        candidates = self._classes.setdefault(record.status, [])
        best: UniqueMinimum | None = None
        best_key: tuple[bool, float] | None = None
        for unique in candidates:
            energy_gap = abs(record.energy_eV - unique.energy_eV)
            if energy_gap > DEDUP_ENERGY_SHORTCUT_EV:
                continue
            if energy_gap > self.options.cs_dedup_energy_tol:
                continue
            representative = unique.representative
            geometric = self.geometric_match(record, representative)
            distance = float("inf")
            if not geometric:
                assert record.fingerprint is not None
                assert representative.fingerprint is not None
                distance = record.fingerprint.distance(representative.fingerprint)
                if distance > self.options.cs_dedup_fp_tol:
                    continue
            key = (not geometric, distance)
            if (
                best_key is None
                or key < best_key
                or (key == best_key and unique.unique_id < best.unique_id)
            ):
                best = unique
                best_key = key
        if best is None:
            unique = UniqueMinimum(
                unique_id=self._next_unique_id,
                status=record.status,
                first_found_md_ps=record.cumulative_md_ps,
                first_found_quench_index=record.quench_index,
            )
            self._next_unique_id += 1
            unique.add(record)
            candidates.append(unique)
            self._unique_by_id[unique.unique_id] = unique
            return unique
        best.add(record)
        return best

    # -- queries -----------------------------------------------------------

    def status_classes(self) -> dict[str, list[UniqueMinimum]]:
        return {status: list(values) for status, values in self._classes.items()}

    def uniques(self, status: str | None = None) -> list[UniqueMinimum]:
        if status is None:
            result: list[UniqueMinimum] = []
            for values in self._classes.values():
                result.extend(values)
            return result
        return list(self._classes.get(status, []))

    def total_records(self) -> int:
        return self._next_record_id - 1

    # -- consolidation -----------------------------------------------------

    def consolidate(self) -> list[dict]:
        """Merge representatives that match (spec section 9.5).

        Repeats until stable; every merge is recorded.
        """
        merges: list[dict] = []
        for status, uniques in self._classes.items():
            changed = True
            while changed:
                changed = False
                ordered = sorted(
                    uniques, key=lambda item: (item.energy_eV, item.unique_id)
                )
                for position, first in enumerate(ordered):
                    for second in ordered[position + 1 :]:
                        if not self._matches(first, second):
                            continue
                        first.merge(second)
                        uniques.remove(second)
                        del self._unique_by_id[second.unique_id]
                        merges.append(
                            {
                                "status": status,
                                "kept": first.unique_id,
                                "merged": second.unique_id,
                                "merged_members": second.hit_count,
                            }
                        )
                        changed = True
                        break
                    if changed:
                        break
        self.merges.extend(merges)
        return merges

    def geometric_match(self, first: MinimumRecord, second: MinimumRecord) -> bool:
        """``G``: direct geometric identity under a fixed slab and atom order.

        Disabled (always False) when the fixed set F is empty: without anchors a
        whole-minimum rigid translation would look like the same structure.
        """
        if not self.geometric_identity_enabled:
            return False
        left = np.asarray(first.atoms.get_positions(), dtype=float)[self.adsorbate]
        right = np.asarray(second.atoms.get_positions(), dtype=float)[self.adsorbate]
        deltas = left - right
        folded = mic_vectors(np.asarray(first.atoms.cell, dtype=float), deltas)
        maximum = float(np.max(np.linalg.norm(folded, axis=1))) if deltas.size else 0.0
        return maximum <= self.options.cs_dedup_geom_tol

    def _matches(self, first: UniqueMinimum, second: UniqueMinimum) -> bool:
        energy_gap = abs(first.energy_eV - second.energy_eV)
        if energy_gap > DEDUP_ENERGY_SHORTCUT_EV:
            return False
        if energy_gap > self.options.cs_dedup_energy_tol:
            return False
        left_rep = first.representative
        right_rep = second.representative
        if self.geometric_match(left_rep, right_rep):
            return True
        left = left_rep.fingerprint
        right = right_rep.fingerprint
        if left is None or right is None:
            return False
        return left.distance(right) <= self.options.cs_dedup_fp_tol

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": "mliport.confsearch-registry/1",
            "n_records": self.total_records(),
            "classes": {
                status: [
                    unique.to_dict()
                    for unique in sorted(
                        values, key=lambda item: (item.energy_eV, item.unique_id)
                    )
                ]
                for status, values in sorted(self._classes.items())
            },
            "merges": list(self.merges),
        }

    def displacement_between(
        self, first: MinimumRecord, second: MinimumRecord
    ) -> float:
        """MIC RMS displacement between two records (diagnostic helper)."""
        deltas = np.asarray(first.atoms.get_positions()) - np.asarray(
            second.atoms.get_positions()
        )
        folded = mic_vectors(first.atoms.cell, deltas)
        return float(np.sqrt(np.mean(np.sum(folded**2, axis=1))))


__all__ = [
    "MinimumOrigin",
    "MinimumRecord",
    "MinimumRegistry",
    "UniqueMinimum",
]
