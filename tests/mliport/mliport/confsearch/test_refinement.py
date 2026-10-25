"""Refinement, ranking and diagnostics contracts (spec 10.1/10.3/10.4, PR #64 review).

The refinement engine is stubbed so the ranking rules can be tested directly:

* a refinement that lowers a candidate's order must move it to the top and into
  ``top/rank_001``;
* two refined minima that deduplicate into one entry must merge (summed hit
  counts), not be exported twice;
* an unconverged refinement keeps the quench result and is flagged;
* a refinement that changes to a non-rankable status leaves the ranking with a
  recorded reason;
* ``minima/all_minima.extxyz`` keeps every quench result of every status;
* the baseline and saturation diagnostics count *members*, not representatives.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read

from mliport.confsearch import results as results_module
from mliport.confsearch.engine import MinimumOrigin, MinimumRecord, MinimumRegistry
from mliport.confsearch.regions import resolve_regions
from tests.mliport.mliport.confsearch.structures import fake_classification
from tests.mliport.mliport.confsearch.test_engine import SLAB_ATOMS, _options, _start


def _setup():
    options = _options()
    atoms = _start((0.5, 0.5))
    regions = resolve_regions(
        atoms, options, adsorbate_indices=np.arange(SLAB_ATOMS, len(atoms))
    )
    registry = MinimumRegistry(
        options=options, adsorbate=regions.adsorbate, fixed=regions.fixed
    )
    return options, regions, registry, atoms


def _register(
    registry, atoms, *, energy, replica, source="end", status="valid", cycle=1, quench=1
):
    record = MinimumRecord(
        record_id=registry.new_record_id(),
        energy_eV=float(energy),
        atoms=atoms,
        classification=fake_classification(status=status),
        origin=MinimumOrigin(replica=replica, cycle=cycle, source=source),
        quench_index=int(quench),
        cumulative_md_ps=float(quench),
        fmax_eV_A=0.04,
        steps=10,
    )
    return registry.register(record)


def _outcome(
    unique_id, *, energy, status="valid", converged=True, atoms=None, fmax=0.02
):
    return results_module.RefinementOutcome(
        unique_id=int(unique_id),
        status=status,
        energy_eV=float(energy),
        fmax_eV_A=float(fmax),
        steps=5,
        atoms=atoms,
        classification=fake_classification(status=status),
        converged=bool(converged),
    )


def _build(options, regions, registry, refinements, atoms, *, tmp_path):
    monkeypatch_target = results_module._refine_top_k
    results_module._refine_top_k = lambda *a, **k: refinements
    try:
        return results_module.build_results(
            registry=registry,
            options=options,
            replicas=[SimpleNamespace(status="active")],
            discovery=[],
            baseline={"stage": "start_quench", "best_energy_eV": -0.60},
            warnings=[],
            output_dir=tmp_path,
            model_record={"model_type": "synthetic"},
            gradient_preflight={"status": "passed"},
            cumulative_md_ps=1.0,
            seed=7,
            starts=[],
            regions=regions,
            references=[None],
            calculator=object(),
        )
    finally:
        results_module._refine_top_k = monkeypatch_target


def test_refinement_reorders_the_ranking_and_the_exports(tmp_path):
    """A candidate whose refined energy is lower must become rank_001."""
    options, regions, registry, atoms = _setup()
    high = _register(registry, atoms, energy=-0.50, replica=0)
    low = _register(registry, atoms, energy=-0.45, replica=1, quench=2)
    refined_high = atoms.copy()
    refined_high.positions[SLAB_ATOMS] += 0.02
    refined_low = atoms.copy()
    refined_low.positions[SLAB_ATOMS] += 0.05
    refinements = {
        high.unique_id: _outcome(high.unique_id, energy=-0.20, atoms=refined_high),
        low.unique_id: _outcome(low.unique_id, energy=-0.90, atoms=refined_low),
    }
    results = _build(options, regions, registry, refinements, atoms, tmp_path=tmp_path)

    refined = results["ranking"]["refined"]
    assert [entry["id"] for entry in refined] == [low.unique_id, high.unique_id]
    assert refined[0]["delta_energy_eV"] == pytest.approx(0.0)
    assert results["ranking"]["best_energy_source"] == "refined"
    assert results["ranking"]["unrefined"] == []
    exported = read(tmp_path / "top" / "rank_001" / "structure.extxyz")
    # rank_001 is the refined geometry of the *lowest refined* candidate.
    assert exported.info["confsearch_id"] == low.unique_id


def test_refined_duplicates_merge_into_one_entry(tmp_path):
    """Spec 10.3: re-deduplicate the refined set, keeping the hit counts."""
    options, regions, registry, atoms = _setup()
    first = _register(registry, atoms, energy=-0.50, replica=0)
    _register(registry, atoms, energy=-0.50, replica=1, quench=2)  # same basin
    assert first.hit_count == 2
    second = _register(registry, atoms, energy=-0.30, replica=2, quench=3)
    refinements = {
        first.unique_id: _outcome(first.unique_id, energy=-0.80, atoms=atoms),
        second.unique_id: _outcome(second.unique_id, energy=-0.80, atoms=atoms.copy()),
    }
    results = _build(options, regions, registry, refinements, atoms, tmp_path=tmp_path)

    refined = results["ranking"]["refined"]
    assert len(refined) == 1
    assert refined[0]["hit_count"] == 3  # 2 + 1, summed over the merged sources
    assert sorted(refined[0]["merged_ids"]) == sorted(
        [first.unique_id, second.unique_id]
    )
    assert results["refinement"]["n_merged_into_existing"] == 1
    assert len(results["exports"]) == 1


def test_unconverged_refinement_keeps_the_quench_result_and_flags_it(tmp_path):
    options, regions, registry, atoms = _setup()
    unique = _register(registry, atoms, energy=-0.50, replica=0)
    refinements = {
        unique.unique_id: _outcome(
            unique.unique_id,
            energy=-0.40,
            status="quench_not_converged",
            converged=False,
        )
    }
    results = _build(options, regions, registry, refinements, atoms, tmp_path=tmp_path)

    assert results["ranking"]["refined"] == []
    unrefined = results["ranking"]["unrefined"]
    assert len(unrefined) == 1
    assert unrefined[0]["energy_eV"] == pytest.approx(-0.50)  # quench energy kept
    assert unrefined[0]["refinement_failed"] is True
    notes = results["refinement"]["notes"]
    assert notes and notes[0]["reason"] == "refinement_not_converged"
    assert notes[0]["kept"] == "quench_result"


def test_refinement_status_change_removes_the_entry_with_a_reason(tmp_path):
    options, regions, registry, atoms = _setup()
    unique = _register(registry, atoms, energy=-0.50, replica=0)
    refinements = {
        unique.unique_id: _outcome(unique.unique_id, energy=-0.40, status="desorbed")
    }
    results = _build(options, regions, registry, refinements, atoms, tmp_path=tmp_path)

    assert results["ranking"]["refined"] == []
    assert results["ranking"]["unrefined"] == []
    notes = results["refinement"]["notes"]
    assert notes and notes[0]["reason"] == "refinement_changed_status"
    assert notes[0]["refinement_status"] == "desorbed"


def test_minima_directory_keeps_every_quench_result(tmp_path):
    options, regions, registry, atoms = _setup()
    _register(registry, atoms, energy=-0.50, replica=0, source="start", quench=1)
    _register(registry, atoms, energy=-0.40, replica=0, quench=2)
    _register(registry, atoms, energy=-0.20, replica=1, status="reacted", quench=3)
    _register(
        registry,
        atoms,
        energy=float("nan"),
        replica=1,
        status="quench_failed",
        quench=4,
    )

    results = _build(options, regions, registry, {}, atoms, tmp_path=tmp_path)

    frames = read(tmp_path / "minima" / "all_minima.extxyz", index=":")
    assert len(frames) == 4
    statuses = {frame.info["confsearch_status"] for frame in frames}
    assert {"valid", "reacted", "quench_failed"} <= statuses
    # A failed quench has a non-finite energy: the frame is kept without it.
    failed = [f for f in frames if f.info["confsearch_status"] == "quench_failed"]
    assert "energy_eV" not in failed[0].info
    payload = json.loads((tmp_path / "minima" / "registry.json").read_text())
    assert payload["schema"] == "mliport.confsearch-registry/1"
    assert results["diagnostics"]["registry"] == payload


def test_baseline_and_saturation_use_members_and_first_found_index(tmp_path):
    options, regions, registry, atoms = _setup()
    # One unique reached by a start quench (member origin) and one only by MD.
    reached = _register(
        registry, atoms, energy=-0.60, replica=0, source="start", quench=1
    )
    _register(registry, atoms, energy=-0.60, replica=1, quench=4)  # same basin, MD
    never = _register(registry, atoms, energy=-0.30, replica=0, quench=2)
    assert reached.hit_count == 2

    results = _build(options, regions, registry, {}, atoms, tmp_path=tmp_path)

    baseline = results["baseline"]
    assert baseline["top_k_never_reached_by_start_quench"] == [never.unique_id]
    best = results["ranking"]["unrefined"][0]
    # first_found_fraction must use the *first-found* quench index (1), not the
    # representative's own quench index.
    assert best["first_found_quench_index"] == 1
    assert best["first_found_fraction"] == pytest.approx(1 / registry.total_records())
    saturation = results["diagnostics"]["saturation"]
    assert saturation["best_hit_count"] == 2
    assert saturation["best_first_found_fraction"] == pytest.approx(
        1 / registry.total_records()
    )
