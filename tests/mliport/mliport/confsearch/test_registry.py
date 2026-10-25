"""Deduplication, representative bookkeeping and consolidation (spec section 9)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mliport.confsearch.registry import (
    MinimumOrigin,
    MinimumRecord,
    MinimumRegistry,
)
from tests.mliport.mliport.confsearch.structures import (
    adsorbate_indices,
    combine,
    fake_classification,
    mode_a_options,
    oh_molecule,
    slab_atoms,
)


def _registry(*, fp_tol=0.05, energy_tol=0.010, spacing=3.0, fixed=None):
    slab = slab_atoms(spacing=spacing)
    molecule = oh_molecule()
    options = mode_a_options(cs_dedup_fp_tol=fp_tol, cs_dedup_energy_tol=energy_tol)
    return (
        slab,
        molecule,
        options,
        MinimumRegistry(
            options=options,
            adsorbate=adsorbate_indices(slab, molecule),
            fixed=np.asarray([], dtype=int) if fixed is None else fixed,
        ),
    )


def _record(
    registry,
    atoms,
    *,
    energy,
    replica=0,
    cycle=0,
    source="end",
    md_ps=1.0,
    status="valid",
):
    return MinimumRecord(
        record_id=registry.new_record_id(),
        energy_eV=energy,
        atoms=atoms,
        classification=fake_classification(status=status),
        origin=MinimumOrigin(replica=replica, cycle=cycle, source=source),
        quench_index=registry.total_records(),
        cumulative_md_ps=md_ps,
        fmax_eV_A=0.04,
        steps=10,
    )


def test_identical_minima_join_one_unique_minimum():
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))

    first = registry.register(_record(registry, atoms, energy=-10.0))
    second = registry.register(
        _record(registry, atoms.copy(), energy=-10.001, replica=1, cycle=1, md_ps=2.0)
    )

    assert first is second
    assert first.hit_count == 2
    assert first.replicas == {0, 1}
    assert first.first_found_md_ps == pytest.approx(1.0)
    assert first.first_found_quench_index == 1
    assert first.energy_eV == pytest.approx(-10.001)  # representative updated


def test_energy_gap_beyond_tolerance_creates_a_new_unique_minimum():
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    registry.register(_record(registry, atoms, energy=-10.0))
    other = registry.register(_record(registry, atoms.copy(), energy=-9.5))
    assert other.hit_count == 1
    assert len(registry.uniques("valid")) == 2


def test_fingerprint_gap_beyond_tolerance_creates_a_new_unique_minimum():
    slab, molecule, _options, registry = _registry(fp_tol=0.001)
    on_site = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    other_site = combine(slab, molecule, shift=(2.5, 2.5, 2.5))
    registry.register(_record(registry, on_site, energy=-10.0))
    other = registry.register(_record(registry, other_site, energy=-10.0005))
    assert other.hit_count == 1
    assert len(registry.uniques("valid")) == 2


def test_energy_shortcut_skips_the_fingerprint(monkeypatch):
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    calls = {"n": 0}
    original = registry.fingerprint

    def counting(atoms_arg):
        calls["n"] += 1
        return original(atoms_arg)

    monkeypatch.setattr(registry, "fingerprint", counting)
    registry.register(_record(registry, atoms, energy=-10.0))
    calls_after_first = calls["n"]
    registry.register(_record(registry, atoms.copy(), energy=-9.5))  # > 0.2 eV
    assert calls["n"] == calls_after_first + 1  # only the new record is fingerprinted


def test_status_classes_never_merge():
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    valid = registry.register(_record(registry, atoms, energy=-10.0, status="valid"))
    reacted = registry.register(
        _record(registry, atoms.copy(), energy=-10.0, status="reacted")
    )
    assert valid is not reacted
    assert len(registry.uniques("valid")) == 1
    assert len(registry.uniques("reacted")) == 1


def test_consolidation_merges_matching_representatives():
    slab, molecule, options, registry = _registry(fp_tol=0.001)
    on_site = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    other_site = combine(slab, molecule, shift=(2.5, 2.5, 2.5))

    registry.register(_record(registry, on_site, energy=-10.0))
    registry.register(_record(registry, other_site, energy=-10.0))
    assert len(registry.uniques("valid")) == 2

    # Widen the fingerprint tolerance (a scientific revision change) and run
    # the final consolidation pass: the two representatives now match.
    registry.options = replace(options, cs_dedup_fp_tol=0.5)
    merges = registry.consolidate()
    assert len(merges) == 1
    assert merges[0]["kept"] == 1 and merges[0]["merged"] == 2
    assert len(registry.uniques("valid")) == 1
    assert registry.uniques("valid")[0].hit_count == 2


def test_registry_serializes_everything_the_exporter_needs():
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    registry.register(_record(registry, atoms, energy=-10.0))
    payload = registry.to_dict()
    assert payload["schema"] == "mliport.confsearch-registry/1"
    entry = payload["classes"]["valid"][0]
    assert entry["id"] == 1
    assert entry["hit_count"] == 1
    assert entry["replicas"] == [0]
    assert len(entry["representative"]["positions_A"]) == len(atoms)
    assert entry["representative"]["numbers"] == [int(z) for z in atoms.numbers]
    assert entry["representative"]["classification"]["status"] == "valid"
    assert entry["members"][0]["origin"] == {
        "replica": 0,
        "cycle": 0,
        "source": "end",
    }


def test_geometric_identity_merges_a_moved_adsorbate_when_f_is_non_empty():
    slab, molecule, _options, registry = _registry(fixed=np.array([0, 1]))
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    moved = atoms.copy()
    moved.positions[4] += [0.05, 0.0, 0.0]  # 0.05 A < cs_dedup_geom_tol
    moved.positions[5] += [0.0, -0.05, 0.0]
    first = registry.register(_record(registry, atoms, energy=-10.0))
    second = registry.register(_record(registry, moved, energy=-10.0))
    assert first is second
    assert first.hit_count == 2


def test_geometric_identity_is_disabled_when_f_is_empty():
    slab, molecule, _options, registry = _registry(fixed=None)
    assert registry.geometric_identity_enabled is False
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    moved = atoms.copy()
    moved.positions[4] += [0.05, 0.0, 0.0]
    first = _record(registry, atoms, energy=-10.0)
    second = _record(registry, moved, energy=-10.0)
    # Without anchors a whole-minimum rigid translation must not count as the
    # same structure: G is disabled, so the fingerprint decides.
    assert registry.geometric_match(second, first) is False


def test_geometric_identity_rejects_a_large_movement():
    slab, molecule, _options, registry = _registry(fixed=np.array([0, 1]))
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    moved = atoms.copy()
    moved.positions[4] += [0.5, 0.0, 0.0]  # > cs_dedup_geom_tol
    first = registry.register(_record(registry, atoms, energy=-10.0))
    second = registry.register(_record(registry, moved, energy=-10.0))
    assert first is not second


def test_registration_order_is_deterministic():
    slab, molecule, _options, registry = _registry()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    other = combine(slab, molecule, shift=(2.5, 2.5, 2.5))
    registry.register(_record(registry, atoms, energy=-10.0))
    registry.register(_record(registry, other, energy=-9.0))
    assert [unique.unique_id for unique in registry.uniques("valid")] == [1, 2]
    assert registry.total_records() == 2
