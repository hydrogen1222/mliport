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


def _robust_case(
    value_eV_A2: float, *, overlap: float = 0.95, mode_overlap: float = 0.99
) -> dict:
    return {
        "n_robust_negative_modes": 1 if value_eV_A2 < neb_suite.NEG_ROBUST_EV_A2 else 0,
        "min_eigenvalue_eV_A2": value_eV_A2,
        "tangent_overlap": overlap,
        "mode_overlap_vs_first": mode_overlap,
    }


def _stationary(**overrides):
    """Keyword inputs for a converged, stationary, finite band."""
    kwargs = {
        "ci_converged": True,
        "physical_fmax_eV_A": 1e-4,
        "band_fmax_eV_A": 1e-4,
        "finite": True,
        "model_identity_ok": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_saddle_verdict_known_answers():
    persistent = [
        _robust_case(-0.5),
        _robust_case(-0.55),
        _robust_case(-0.6),
    ]
    good = neb_suite.saddle_verdict(persistent, **_stationary())
    assert good["saddle_validation"] == "validated_first_order_candidate"
    assert good["validated"] is True
    assert good["n_persistent_negative_deltas"] == 3
    assert good["min_eigenvalue_eV_A2"] == pytest.approx(-0.6)
    assert good["spread_ratio"] == pytest.approx(0.1 / 0.55)
    assert good["tangent_overlap"] == pytest.approx(0.95)
    assert good["min_mode_overlap_vs_first"] == pytest.approx(0.99)
    # weak tangent overlap disqualifies the first-order claim
    weak = neb_suite.saddle_verdict(
        [_robust_case(-0.5, overlap=0.3)] * 3, **_stationary()
    )
    assert weak["saddle_validation"] == "ci_neb_candidate_only"
    assert weak["validated"] is False
    assert "tangent overlap" in " ".join(weak["rejection_reasons"])
    # two robust negative modes => bad saddle
    two_mode = [
        {
            "n_robust_negative_modes": 2,
            "min_eigenvalue_eV_A2": -0.5,
            "tangent_overlap": 0.95,
            "mode_overlap_vs_first": 0.99,
        }
    ] * 3
    bad = neb_suite.saddle_verdict(two_mode, **_stationary())
    assert bad["saddle_validation"] == "bad_saddle_candidate"
    # nothing robust
    none = neb_suite.saddle_verdict([_robust_case(1e-5)] * 3, **_stationary())
    assert none["saddle_validation"] == "no_robust_negative_mode"
    assert none["validated"] is False
    # non-persistence across deltas
    non_persistent = [_robust_case(-0.5), _robust_case(1e-5), _robust_case(-0.5)]
    assert (
        neb_suite.saddle_verdict(non_persistent, **_stationary())["saddle_validation"]
        == "ci_neb_candidate_only"
    )
    # a failure at one scale is never overridden by later good scales
    late_bad = [_robust_case(-0.5), _robust_case(-0.55), _robust_case(-0.6)]
    late_bad[-1]["n_robust_negative_modes"] = 2
    assert (
        neb_suite.saddle_verdict(late_bad, **_stationary())["saddle_validation"]
        == "bad_saddle_candidate"
    )
    late_bad[-1]["n_robust_negative_modes"] = 0
    assert (
        neb_suite.saddle_verdict(late_bad, **_stationary())["saddle_validation"]
        == "ci_neb_candidate_only"
    )
    # eigenvector rotation between deltas is a scale inconsistency
    rotated = [
        _robust_case(-0.5, mode_overlap=0.99),
        _robust_case(-0.5, mode_overlap=0.1),
        _robust_case(-0.5, mode_overlap=0.99),
    ]
    verdict = neb_suite.saddle_verdict(rotated, **_stationary())
    assert verdict["saddle_validation"] == "ci_neb_candidate_only"
    assert "eigenvector" in " ".join(verdict["rejection_reasons"])


def test_saddle_verdict_rejects_non_stationary_negative_curvature():
    """R01 red case: negative curvature at x=0.3 of E=(x^2-1)^2+y^2+z^2.

    gradient 4x(x^2-1) = -1.092 eV/A at x=0.3 and Hxx = 12x^2-4 = -2.92,
    so exactly one robust negative mode exists -- and none of that makes the
    point a stationarity-validated saddle.
    """
    nonstationary = [_robust_case(-2.92)] * 3
    # CI-NEB never converged: may export the Hessian, but only as a
    # non-stationary characterization, never as a validated saddle.
    verdict = neb_suite.saddle_verdict(
        nonstationary,
        **_stationary(
            ci_converged=False,
            physical_fmax_eV_A=1.092,
            band_fmax_eV_A=0.5,
        ),
    )
    assert verdict["saddle_validation"] == "nonstationary_characterization"
    assert verdict["validated"] is False
    assert "nonstationary_characterization" not in neb_suite.SADDLE_FAIL_VERDICTS
    # Even if the converged flag were claimed, the physical gradient rejects.
    lying = neb_suite.saddle_verdict(
        nonstationary,
        **_stationary(physical_fmax_eV_A=1.092, band_fmax_eV_A=0.5),
    )
    assert lying["saddle_validation"] == "not_stationary"
    assert lying["validated"] is False
    assert "not_stationary" in neb_suite.SADDLE_FAIL_VERDICTS
    # whole-band non-convergence also blocks certification
    band_bad = neb_suite.saddle_verdict(
        nonstationary, **_stationary(band_fmax_eV_A=0.2)
    )
    assert band_bad["saddle_validation"] == "band_not_converged"
    # missing/NaN observables are fail-closed
    assert (
        neb_suite.saddle_verdict(nonstationary, **_stationary(finite=False))[
            "saddle_validation"
        ]
        == "invalid_nonfinite_inputs"
    )
    assert (
        neb_suite.saddle_verdict(nonstationary, **_stationary(model_identity_ok=False))[
            "saddle_validation"
        ]
        == "model_identity_mismatch"
    )
    assert (
        neb_suite.saddle_verdict(nonstationary, **_stationary(physical_fmax_eV_A=None))[
            "saddle_validation"
        ]
        == "stationarity_not_measured"
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
    dof = neb_suite.active_dof_indices(ci_image)
    tangent = neb_suite.band_tangent_unit(final_band, energies.tolist(), ci)[dof]
    tangent = tangent / np.linalg.norm(tangent)
    per_delta = []
    first_mode = None
    for delta in neb_suite.SADDLE_DELTAS:
        hessian = neb_suite.fd_hessian(ci_image, delta, dof)
        evals, vecs = np.linalg.eigh(hessian)
        entry = neb_suite.negative_mode_analysis(evals)
        entry["tangent_overlap"] = abs(float(np.dot(vecs[:, 0], tangent)))
        if first_mode is None:
            first_mode = vecs[:, 0]
            entry["mode_overlap_vs_first"] = 1.0
        else:
            entry["mode_overlap_vs_first"] = abs(float(np.dot(vecs[:, 0], first_mode)))
        per_delta.append(entry)
    verdict = neb_suite.saddle_verdict(
        per_delta,
        ci_converged=result.converged,
        physical_fmax_eV_A=result.climbing_physical_fmax_eV_A,
        band_fmax_eV_A=result.max_neb_force_eV_A,
        finite=True,
        model_identity_ok=True,
    )
    assert verdict["saddle_validation"] in {
        "validated_first_order_candidate",
        "ci_neb_candidate_only",
    }, verdict
    # no failure at any scale may be overridden by a later one
    assert verdict["n_deltas"] == len(neb_suite.SADDLE_DELTAS)


# ---------------------------------------------------------------------------
# R01: the real workflow must reject a non-stationary negative-curvature point
# ---------------------------------------------------------------------------


class _DoubleWellCalculator:
    """Analytic E = sum_i [(x_i^2 - 1)^2 + y_i^2 + z_i^2] (eV, Angstrom).

    At x=0 the point is the true first-order saddle between x=+-1
    (Hxx = -4, force 0).  At x=0.3 the gradient is 4x(x^2-1) = -1.092 eV/A
    and Hxx = -2.92: one robust negative mode at a *non-stationary* point.
    """

    def get_potential_energy(self, atoms=None, **_kwargs):
        pos = np.asarray(
            (self._atoms if atoms is None else atoms).positions, dtype=float
        )
        return float(
            np.sum((pos[:, 0] ** 2 - 1.0) ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
        )

    def get_forces(self, atoms=None, **_kwargs):
        pos = np.asarray(
            (self._atoms if atoms is None else atoms).positions, dtype=float
        )
        forces = np.zeros_like(pos)
        forces[:, 0] = -4.0 * pos[:, 0] * (pos[:, 0] ** 2 - 1.0)
        forces[:, 1] = -2.0 * pos[:, 1]
        forces[:, 2] = -2.0 * pos[:, 2]
        return forces


def _double_well_band(x_values):
    images = []
    for x in x_values:
        atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
        atoms.calc = _DoubleWellCalculator()
        images.append(atoms)
    return images


def _write_band_checkpoint(tmp_path, images):
    from mlipx.neb.io import NEBCheckpointStore

    store = NEBCheckpointStore(tmp_path / "neb_run")
    return store.write(
        images,
        run_id="run-r01",
        attempt_id="attempt-r01",
        stage="ci_neb",
        stage_step=1,
        resume_fingerprint={"schema": "mlipx.neb-resume-fingerprint/1"},
        resume_fingerprint_sha256="0" * 64,
        resolved_config={"calc_type": "neb"},
    )


def _saddle_workflow_ctx(canonical_calc):
    from types import SimpleNamespace

    engines_mod = importlib.import_module("engines")
    return (
        engines_mod.EngineContext(
            engine="mace",
            profile_id="r01-toy",
            profile={
                "model_sha256": "a" * 64,
                "upstream_model_id": "r01-toy",
                "identity": "R01 analytic toy",
            },
            wrapper=None,
            calculator=canonical_calc,
            device="cpu",
            dtype="float64",
            task="bulk",
            head=None,
            backend_version="toy",
            framework_version="toy",
        ),
        SimpleNamespace(out=None, tag=None),
    )


def _run_saddle_workflow(tmp_path, *, x_values, ci_converged, two_atoms=False):
    import json

    if two_atoms:
        images = []
        for x in x_values:
            atoms = Atoms(
                "H2",
                positions=[[x, 0.0, 0.0], [x, 0.0, 0.0]],
                cell=[20.0, 20.0, 20.0],
                pbc=True,
            )
            atoms.calc = _DoubleWellCalculator()
            images.append(atoms)
    else:
        images = _double_well_band(x_values)
    checkpoint = _write_band_checkpoint(tmp_path, images)
    ci_calc = _DoubleWellCalculator()
    ci_image = images[2].copy()
    ci_image.calc = ci_calc
    payload = {
        "converged": bool(ci_converged),
        "climbing_image_index": 2,
        "energies_eV": [float(img.get_potential_energy()) for img in images],
        "max_neb_force_eV_A": 1e-4 if ci_converged else 0.5,
        "barrier_forward_sampled_eV": 1.0,
        "model": {
            "model_type": "mace",
            "model_sha256": "a" * 64,
            "task": "bulk",
            "head_effective": None,
            "dtype_requested": "float64",
            "inference_mode": "ase",
        },
        "latest_checkpoint": str(checkpoint),
    }
    ctx, args = _saddle_workflow_ctx(ci_calc)
    args.out = tmp_path / "records"
    neb_suite._STATE[(ctx.engine, ctx.profile_id)] = {"ci_neb5": {"payload": payload}}
    code = neb_suite.workflow_saddle_hessian(ctx, args, args.out)
    written = sorted((tmp_path / "records").glob("*.json"))
    assert written, "the workflow must always publish its verdict"
    rec = json.loads(written[-1].read_text(encoding="utf-8"))
    return code, rec


def test_workflow_saddle_hessian_validates_true_saddle(tmp_path):
    code, rec = _run_saddle_workflow(
        tmp_path, x_values=[-1.0, -0.5, 0.0, 0.5, 1.0], ci_converged=True
    )
    assert code == 0
    assert rec["status"] == "pass"
    metrics = rec["metrics"]
    assert metrics["saddle_validation"] == "validated_first_order_candidate"
    assert metrics["validated"] is True
    assert metrics["active_physical_fmax_eV_A"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["n_active_dof"] == 3
    assert len(metrics["per_delta"]) == 3
    # every displacement scale is preserved, not just the last overlap
    assert len(metrics["per_delta_tangent_overlap"]) == 3
    assert all(
        o == pytest.approx(1.0, abs=1e-6) for o in metrics["per_delta_tangent_overlap"]
    )


def test_workflow_saddle_hessian_rejects_nonstationary_point(tmp_path):
    """x=0.3: one robust negative mode at force 1.092 eV/A must not certify."""
    code, rec = _run_saddle_workflow(
        tmp_path, x_values=[-1.0, -0.5, 0.3, 0.5, 1.0], ci_converged=False
    )
    assert code == 0  # characterization is a completed workflow ...
    assert rec["status"] == "characterized"  # ... but never a pass
    metrics = rec["metrics"]
    assert metrics["saddle_validation"] == "nonstationary_characterization"
    assert metrics["validated"] is False
    assert metrics["ci_converged"] is False
    # the Hessian is still exported with its negative curvature
    assert metrics["min_eigenvalue_eV_A2"] < neb_suite.NEG_ROBUST_EV_A2
    assert metrics["active_physical_fmax_eV_A"] == pytest.approx(1.092, abs=1e-6)

    # same point, but with the convergence flag claimed: fail closed
    code2, rec2 = _run_saddle_workflow(
        tmp_path / "claimed", x_values=[-1.0, -0.5, 0.3, 0.5, 1.0], ci_converged=True
    )
    assert code2 == 1
    assert rec2["status"] == "fail"
    assert rec2["metrics"]["saddle_validation"] == "not_stationary"
    assert rec2["metrics"]["validated"] is False


def test_workflow_saddle_hessian_rejects_two_negative_modes(tmp_path):
    code, rec = _run_saddle_workflow(
        tmp_path,
        x_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        ci_converged=True,
        two_atoms=True,
    )
    assert code == 1
    assert rec["status"] == "fail"
    assert rec["metrics"]["saddle_validation"] == "bad_saddle_candidate"


def test_active_dof_helpers_exclude_fixed_atoms():
    from ase.constraints import FixAtoms

    atoms = Atoms(
        "H2",
        positions=[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[1]))
    dof = neb_suite.active_dof_indices(atoms)
    assert dof.tolist() == [0, 1, 2]

    class _FixedForceCalculator:
        def get_forces(self, atoms=None, **_kwargs):
            # the frozen atom carries a huge unrelaxed force; it must not
            # enter the active-DOF stationarity gate
            return np.array([[0.001, 0.0, 0.0], [99.0, 0.0, 0.0]])

    atoms.calc = _FixedForceCalculator()
    assert neb_suite.active_force_fmax(atoms) == pytest.approx(0.001)


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
