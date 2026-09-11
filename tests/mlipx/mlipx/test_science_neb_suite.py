"""Known-answer and end-to-end tests for the NEB validation tier (t5).

Everything here runs on CPU with the ASE EMT calculator, which is an exact
documented reference for the Cu/FCC geometry used by the validation fixtures.
The GPU engine sweeps execute the same workflows through the full product
``mlipx.neb.workflow.run_neb_workflow`` path; these tests exercise the product
runner, band preparation, checkpoint store, and resume fingerprint directly
(``NEBRunner._get_calculator`` accepts a plain ASE calculator, so EMT drives
the product pipeline without a GPU model).

Scientific pins:
- EMT Cu vacancy-hop barrier must land in [0.45, 1.0] eV (literature ~0.66-0.77
  eV); the window absorbs the 5-image sampled-max underestimate.
- An inverted harmonic dimer E = -k(r-r0)^2/2 has exactly one robust negative
  eigenvalue -2k in the unweighted 6D FD Hessian plus 5 zero-like modes.
- The robust negative-mode gate (10x zero scale, persistent across 3 deltas,
  tangent overlap >= 0.8) drives ``saddle_validation``.
- barrier_metrics never invents a fitted saddle; reverse barrier is reported
  even when the endpoints are inequivalent.
- The resume fingerprint covers the full NEBOptions field set, so a budget
  change produces a different sha256 (resume correctly rejected).
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

_SCRIPTS = Path(__file__).resolve().parents[3] / "validation" / "science" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

neb_suite = importlib.import_module("neb_suite")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _SyntheticForceCalculator:
    """Minimal ASE calculator for a harmonic pair potential along the bond."""

    def __init__(self, atoms: Atoms, k: float, r0: float, sign: float):
        self._atoms = atoms
        self._k = k
        self._r0 = r0
        self._sign = sign

    def get_forces(self, atoms=None, **_unused):
        a = self._atoms if atoms is None else atoms
        vec = a.positions[1] - a.positions[0]
        dist = float(np.linalg.norm(vec))
        unit = vec / dist
        # E = sign * k * (dist - r0)^2 / 2  =>  dE/dr = sign * k * (dist - r0)
        de_dr = self._sign * self._k * (dist - self._r0)
        # dr/dx1 = -u, dr/dx2 = +u  =>  F0 = +dE/dr * u, F1 = -dE/dr * u
        return np.stack([de_dr * unit, -de_dr * unit])

    def get_potential_energy(self, atoms=None, **_unused):
        a = self._atoms if atoms is None else atoms
        dist = float(np.linalg.norm(a.positions[1] - a.positions[0]))
        return self._sign * 0.5 * self._k * (dist - self._r0) ** 2


@pytest.fixture()
def emt() -> EMT:
    return EMT()


# ---------------------------------------------------------------------------
# geometry builders
# ---------------------------------------------------------------------------


def test_cu_vacancy_pair_is_deterministic_and_documented():
    pair_a = neb_suite.cu_vacancy_pair()
    pair_b = neb_suite.cu_vacancy_pair()
    assert pair_a["natoms"] == 31
    assert pair_a["atom_map"] == list(range(31))
    assert pair_a["image_shifts"] == [[0, 0, 0]] * 31
    for key in ("natoms", "hop_distance_A", "atom_map", "image_shifts"):
        assert pair_a[key] == pair_b[key]
    assert np.array_equal(pair_a["initial"].positions, pair_b["initial"].positions)
    assert np.array_equal(pair_a["final"].positions, pair_b["final"].positions)
    # hop distance is exactly the fcc nearest-neighbour a/sqrt(2)
    assert abs(pair_a["hop_distance_A"] - 3.615 / math.sqrt(2)) < 1e-9
    # both endpoints have one vacancy and conserve composition
    syms_i = sorted(pair_a["initial"].get_chemical_symbols())
    syms_f = sorted(pair_a["final"].get_chemical_symbols())
    assert syms_i == syms_f == ["Cu"] * 31


def test_na3ps4_pair_keeps_composition_and_documents_hop():
    pair = neb_suite.na3ps4_vacancy_pair()
    assert pair["natoms"] == 15
    syms_i = sorted(pair["initial"].get_chemical_symbols())
    syms_f = sorted(pair["final"].get_chemical_symbols())
    assert syms_i == syms_f
    assert syms_i.count("Na") == 5
    assert abs(pair["hop_distance_A"] - 3.5082) < 1e-3
    # no shorter periodic image than the documented hop
    assert abs(pair["hop_mic_delta_A"]) < 1e-3


# ---------------------------------------------------------------------------
# band tangent + saddle verdict + barrier metrics (known answers)
# ---------------------------------------------------------------------------


def test_band_tangent_matches_ase_improved_tangent():
    """Monotone bands take the downhill spring; extrema weight the segments."""
    # image 1-atom x: 0.0, 0.5, 1.0, 0.75, 0.5.  At image 2 the two adjacent
    # segments point in opposite x directions, so the branch choice is visible
    # in the sign of the tangent component of atom 1 (flat index 3).
    xs = [0.0, 0.5, 1.0, 0.75, 0.5]
    images = [
        Atoms("H2", positions=[[0, 0, 0], [x, 0, 0]], cell=[20, 20, 20], pbc=True)
        for x in xs
    ]
    # monotone increasing through image 2 (5 > 4 > 1): spring2 = (2 -> 3) = -x
    t_up = neb_suite.band_tangent_unit(images, [0.0, 1.0, 4.0, 5.0, 2.0], 2)
    assert t_up.shape == (6,)
    assert t_up[3] == pytest.approx(-1.0)
    assert abs(t_up[0]) < 1e-12
    # monotone decreasing through image 2 (1 < 4 < 5): spring1 = (1 -> 2) = +x
    t_down = neb_suite.band_tangent_unit(images, [0.0, 5.0, 4.0, 1.0, 2.0], 2)
    assert t_down[3] == pytest.approx(1.0)
    # extremum with the ci+1 side higher (3 > 1): spring2 keeps the max weight
    t_ext = neb_suite.band_tangent_unit(images, [0.0, 1.0, 4.0, 3.0, 2.0], 2)
    assert t_ext[3] == pytest.approx(-1.0)
    # boundary images and mismatched energies are rejected
    with pytest.raises(ValueError):
        neb_suite.band_tangent_unit(images, [0.0, 1.0, 2.0, 3.0, 4.0], 0)
    with pytest.raises(ValueError):
        neb_suite.band_tangent_unit(images, [0.0, 1.0, 2.0], 2)


def _robust_case(value_eV_A2: float) -> dict:
    return {
        "n_robust_negative_modes": 1 if value_eV_A2 < neb_suite.NEG_ROBUST_EV_A2 else 0,
        "min_eigenvalue_eV_A2": value_eV_A2,
    }


def test_saddle_verdict_known_answers():
    persistent = [
        _robust_case(-0.5),
        _robust_case(-0.55),
        _robust_case(-0.6),
    ]
    good = neb_suite.saddle_verdict(persistent, 0.95)
    assert good["saddle_validation"] == "validated_first_order_candidate"
    assert good["n_persistent_negative_deltas"] == 3
    assert good["min_eigenvalue_eV_A2"] == pytest.approx(-0.6)
    assert good["spread_ratio"] == pytest.approx(0.1 / 0.55)
    # weak tangent overlap disqualifies the first-order claim
    weak = neb_suite.saddle_verdict(persistent, 0.3)
    assert weak["saddle_validation"] == "ci_neb_candidate_only"
    # two robust negative modes => bad saddle
    two_mode = [{"n_robust_negative_modes": 2, "min_eigenvalue_eV_A2": -0.5}] * 3
    bad = neb_suite.saddle_verdict(two_mode, 0.95)
    assert bad["saddle_validation"] == "bad_saddle_candidate"
    # nothing robust
    none = neb_suite.saddle_verdict([_robust_case(1e-5)] * 3, 0.95)
    assert none["saddle_validation"] == "no_robust_negative_mode"
    # non-persistence across deltas
    non_persistent = [_robust_case(-0.5), _robust_case(1e-5), _robust_case(-0.5)]
    assert (
        neb_suite.saddle_verdict(non_persistent, 0.95)["saddle_validation"]
        == "ci_neb_candidate_only"
    )


def test_barrier_metrics_semantics_and_no_invented_saddle():
    from mlipx.neb.results import barrier_metrics

    # symmetric profile: forward == reverse, reaction == 0
    m = barrier_metrics(np.array([0.0, 0.4, 0.7, 0.35, 0.0]))
    assert m["barrier_forward_sampled_eV"] == pytest.approx(0.7)
    assert m["barrier_reverse_sampled_eV"] == pytest.approx(0.7)
    assert m["reaction_energy_eV"] == pytest.approx(0.0)
    assert m["highest_energy_image_index"] == 2
    assert m["barrier_status"] == "internal_peak_sampled"
    assert m["barrier_forward_fitted_eV"] is None
    assert m["barrier_reverse_fitted_eV"] is None
    # inequivalent endpoints: reverse barrier still reported, not zeroed
    m2 = barrier_metrics(np.array([0.0, 0.5, 0.6, 0.55, 0.3]))
    assert m2["barrier_forward_sampled_eV"] == pytest.approx(0.6)
    assert m2["barrier_reverse_sampled_eV"] == pytest.approx(0.3)
    assert m2["reaction_energy_eV"] == pytest.approx(0.3)
    # monotone band: no internal peak => status says so, no barrier invented
    m3 = barrier_metrics(np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
    assert m3["barrier_status"] == "no_internal_saddle_identified"
    assert m3["highest_energy_image_index"] == 4
    assert m3["barrier_forward_sampled_eV"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        barrier_metrics(np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# FD Hessian known answer: inverted harmonic dimer
# ---------------------------------------------------------------------------


def test_fd_hessian_inverted_dimer_has_single_robust_negative_mode():
    k = 2.0
    atoms = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[20, 20, 20], pbc=True)
    atoms.calc = _SyntheticForceCalculator(atoms, k=k, r0=1.5, sign=-1.0)
    hessian = neb_suite.fd_hessian(atoms, 0.005)
    evals = np.linalg.eigvalsh(hessian)
    negative = [v for v in evals if v < neb_suite.NEG_ROBUST_EV_A2]
    zero_like = [v for v in evals if abs(v) <= neb_suite.ZERO_SCALE_EV_A2]
    # unweighted 6D hessian of E = -k(r-r0)^2/2: stretch mode eigenvalue -2k
    assert len(negative) == 1
    assert negative[0] == pytest.approx(-2.0 * k, abs=2e-2)
    # 3 translations + 2 bendings stay at zero scale
    assert len(zero_like) == 5
    analysis = neb_suite.negative_mode_analysis(evals)
    assert analysis["n_robust_negative_modes"] == 1
    # tangent overlap: the softest direction IS the bond direction
    unit = np.zeros(6)
    unit[3] = 1.0
    unit[0] = -1.0
    unit /= np.linalg.norm(unit)
    _, vecs = np.linalg.eigh(hessian)
    overlap = abs(float(np.dot(vecs[:, 0], unit)))
    assert overlap == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# end-to-end through the product NEB runner on EMT/CPU
# ---------------------------------------------------------------------------


def test_emt_ci_neb_end_to_end_barrier_and_saddle(emt):
    """prepare_band -> product NEBRunner (climb) -> barrier + saddle verdict."""
    from ase.optimize import FIRE

    from mlipx.neb.prepare import prepare_band
    from mlipx.neb.schema import NEBOptions
    from mlipx.runners.neb import NEBRunner

    pair = neb_suite.cu_vacancy_pair()
    # endpoint_policy="validate" requires raw forces below endpoint_fmax, so
    # both endpoints must be pre-relaxed exactly as the t5a workflow does
    initial = pair["initial"].copy()
    initial.calc = emt
    FIRE(initial, logfile=None).run(fmax=neb_suite.ENDPOINT_FMAX, steps=500)
    final = pair["final"].copy()
    final.calc = emt
    FIRE(final, logfile=None).run(fmax=neb_suite.ENDPOINT_FMAX, steps=500)
    options = NEBOptions(
        n_intermediate_images=3,
        climb=True,
        interpolation="linear",
        fmax_eV_A=neb_suite.CI_FMAX,
        pre_fmax_eV_A=neb_suite.NEB_PRE_FMAX,
        max_steps=neb_suite.NEB_MAX_STEPS,
        pre_max_steps=neb_suite.NEB_MAX_STEPS,
        endpoint_policy="validate",
        endpoint_fmax_eV_A=neb_suite.ENDPOINT_FMAX,
    )
    band = prepare_band(initial, final, options)
    assert band.source == "endpoints"
    assert len(band.images) == 5
    assert band.atom_map.tolist() == list(range(31))
    assert band.image_shifts.tolist() == [[0, 0, 0]] * 31

    runner = NEBRunner(emt, options, verbose=False)
    result = runner.run(band)
    assert result.status == "completed"
    assert result.converged is True
    energies = np.asarray(result.energies_eV, dtype=float)
    assert energies.shape == (5,)

    metrics = result.to_dict()
    assert metrics["barrier_status"] == "converged_path_estimate"
    barrier = metrics["barrier_forward_sampled_eV"]
    # EMT Cu vacancy hop literature window (sampled max underestimates the
    # true saddle by up to ~0.1 eV at 5 images)
    assert 0.45 <= barrier <= 1.0, barrier
    assert abs(metrics["reaction_energy_eV"]) < 0.1  # symmetric vacancy hop
    ci = result.climbing_image_index
    assert ci is not None
    assert energies[ci] == pytest.approx(energies.max())
    assert result.max_neb_force_eV_A <= neb_suite.CI_FMAX * 1.05
    assert metrics["barrier_forward_fitted_eV"] is None

    # saddle analysis on the final band's climbing image
    final_band = runner.last_band
    assert final_band is not None and len(final_band) == 5
    ci_image = final_band[ci].copy()
    ci_image.calc = emt
    tangent = neb_suite.band_tangent_unit(final_band, energies.tolist(), ci)
    per_delta = []
    overlap = None
    for delta in neb_suite.SADDLE_DELTAS:
        hessian = neb_suite.fd_hessian(ci_image, delta)
        evals, vecs = np.linalg.eigh(hessian)
        per_delta.append(neb_suite.negative_mode_analysis(evals))
        overlap = abs(float(np.dot(vecs[:, 0], tangent)))
    verdict = neb_suite.saddle_verdict(per_delta, overlap)
    assert verdict["saddle_validation"] in {
        "validated_first_order_candidate",
        "ci_neb_candidate_only",
    }, verdict


def test_emt_neb_budget_not_converged_is_recorded(emt):
    """A bounded budget that cannot converge is recorded, never run forever."""
    from ase.optimize import FIRE

    from mlipx.neb.prepare import prepare_band
    from mlipx.neb.schema import NEBOptions
    from mlipx.runners.neb import NEBRunner

    pair = neb_suite.cu_vacancy_pair()
    initial = pair["initial"].copy()
    initial.calc = emt
    FIRE(initial, logfile=None).run(fmax=neb_suite.ENDPOINT_FMAX, steps=500)
    final = pair["final"].copy()
    final.calc = emt
    FIRE(final, logfile=None).run(fmax=neb_suite.ENDPOINT_FMAX, steps=500)
    options = NEBOptions(
        n_intermediate_images=3,
        climb=True,
        interpolation="linear",
        fmax_eV_A=neb_suite.CI_FMAX,
        pre_fmax_eV_A=neb_suite.NEB_PRE_FMAX,
        max_steps=1,
        pre_max_steps=1,
        endpoint_policy="validate",
        endpoint_fmax_eV_A=neb_suite.ENDPOINT_FMAX,
    )
    band = prepare_band(initial, final, options)
    result = NEBRunner(emt, options, verbose=False).run(band)
    assert result.status == "not_converged"
    assert result.converged is False
    metrics = result.to_dict()
    assert metrics["barrier_status"] == "unconverged_path_sample"
    assert metrics["failure_reason"] is not None


# ---------------------------------------------------------------------------
# resume fingerprint + checkpoint store identity (product-level, CPU)
# ---------------------------------------------------------------------------


def _model_record(**overrides):
    base = {
        "model_type": "mace",
        "model_sha256": "0" * 64,
        "backend_distribution": "mace-torch",
        "backend_version": "0.3.10",
        "framework_versions": {"torch": "2.4.0"},
        "task": "bulk",
        "head_effective": None,
        "dtype_effective": "float64",
        "inference_mode": "eager",
        "compile_enabled": False,
    }
    base.update(overrides)
    return base


def _fresh_band():
    from mlipx.neb.prepare import prepare_band
    from mlipx.neb.schema import NEBOptions

    pair = neb_suite.cu_vacancy_pair()
    options = NEBOptions(n_intermediate_images=3, interpolation="linear")
    return prepare_band(pair["initial"], pair["final"], options), options


def test_resume_fingerprint_covers_full_option_set():
    """Budget is immutable: any NEBOptions field change alters the sha256."""
    from mlipx.neb.workflow import _fingerprint

    band, options = _fresh_band()
    model = _model_record()
    fp_a, sha_a = _fingerprint(band, options, model)
    fp_b, sha_b = _fingerprint(band, options, model)
    assert sha_a == sha_b
    assert fp_a["neb_options"] == fp_b["neb_options"]
    assert fp_a["schema"] == "mlipx.neb-resume-fingerprint/1"
    assert fp_a["band"]["image_count"] == 5
    assert fp_a["band"]["atom_map"] == list(range(31))

    # budget change must change the fingerprint (resume rejected downstream)
    from dataclasses import replace

    _, sha_c = _fingerprint(
        band, replace(options, max_steps=options.max_steps + 1), model
    )
    assert sha_c != sha_a
    # model identity change must change the fingerprint
    _, sha_d = _fingerprint(band, options, _model_record(dtype_effective="float32"))
    assert sha_d != sha_a
    # band identity change must change the fingerprint
    band2, options2 = _fresh_band()
    band2.images[1].positions[0, 0] += 1e-6
    _, sha_e = _fingerprint(
        type(band2)(
            images=band2.images,
            atom_map=band2.atom_map,
            image_shifts=band2.image_shifts,
            path_convention=band2.path_convention,
            source=band2.source,
        ),
        options2,
        model,
    )
    # geometry is continuity-checked from the checkpoint band, not the
    # fingerprint (composition identity only); document that explicitly
    assert sha_e == sha_a


def test_checkpoint_store_roundtrip_bitwise(tmp_path):
    """NEBCheckpointStore preserves band geometry and fingerprint metadata."""
    from mlipx.neb.io import NEBCheckpointStore, load_checkpoint
    from mlipx.neb.prepare import prepare_band
    from mlipx.neb.schema import NEBOptions

    pair = neb_suite.cu_vacancy_pair()
    options = NEBOptions(n_intermediate_images=3, interpolation="linear")
    band = prepare_band(pair["initial"], pair["final"], options)
    run_dir = tmp_path / "run"
    store = NEBCheckpointStore(run_dir)
    fingerprint = {"schema": "mlipx.neb-resume-fingerprint/1", "probe": True}
    path = store.write(
        band.images,
        run_id="run-a",
        attempt_id="attempt-1",
        stage="final",
        stage_step=42,
        resume_fingerprint=fingerprint,
        resume_fingerprint_sha256="abc123",
        resolved_config={"calc_type": "neb"},
    )
    assert path.is_dir()
    meta, images = load_checkpoint(path)
    assert meta["stage"] == "final"
    assert meta["stage_step"] == 42
    assert meta["run_id"] == "run-a"
    assert meta["resume_fingerprint_sha256"] == "abc123"
    assert meta["resume_fingerprint"]["probe"] is True
    assert meta["complete"] is True
    assert len(images) == 5
    for saved, original in zip(images, band.images, strict=False):
        assert np.array_equal(saved.positions, original.positions)
        assert saved.get_chemical_symbols() == original.get_chemical_symbols()

    # stage scan helper used by the resume-identity workflow
    positions = neb_suite.checkpoint_stage_positions(run_dir, "final")
    assert positions is not None and len(positions) == 5
    for found, original in zip(positions, band.images, strict=False):
        assert np.array_equal(found, original.positions)
    assert neb_suite.checkpoint_stage_positions(run_dir, "prepared") is None

    # second write must create a new sequence and 'latest' must follow it
    path2 = store.write(
        band.images,
        run_id="run-b",
        attempt_id="attempt-2",
        stage="prepared",
        stage_step=0,
        resume_fingerprint=fingerprint,
        resume_fingerprint_sha256="def456",
        resolved_config={"calc_type": "neb"},
    )
    assert path2 != path
    prepared = neb_suite.checkpoint_stage_positions(run_dir, "prepared")
    assert prepared is not None and len(prepared) == 5
    latest_meta, _ = load_checkpoint(run_dir / "checkpoints" / "latest")
    assert latest_meta["run_id"] == "run-b"


# ---------------------------------------------------------------------------
# suite CLI plumbing
# ---------------------------------------------------------------------------


def test_run_suite_registers_t5_tier():
    run_suite = importlib.import_module("run_suite")
    assert run_suite.TIER_COMMANDS["t5"] == ["neb_suite.py"]


def test_neb_suite_module_constants_match_taskbook():
    assert neb_suite.CI_FMAX == 0.03
    assert neb_suite.NEB_PRE_FMAX == 0.10
    assert neb_suite.ENDPOINT_FMAX == 0.02
    assert neb_suite.NEB_MAX_STEPS == 800
    assert neb_suite.IMAGE_STABILIZATION_EV == 0.05
    assert neb_suite.SADDLE_DELTAS == (0.005, 0.010, 0.020)
    assert neb_suite.ZERO_SCALE_EV_A2 == 1e-3
    assert neb_suite.NEG_ROBUST_EV_A2 == -1e-2
    assert neb_suite.TANGENT_OVERLAP_MIN == 0.8
