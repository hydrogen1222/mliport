"""Synthetic known-answer tests for the mliport analysis families (PR5, t7).

Taskbook requirement (analysis tier): every analysis method must have a
synthetic known-answer test independent of the four MLIPs, so analysis bugs
are distinguishable from model physics. The families below follow the
product task registry (``AnalysisRequest.VALID_TASKS``):

    validate, thermo, rdf, rmsd, msd, transport, density, arrhenius,
    electrolyte, vacf, spectrum

All trajectories are analytic constructions (random walks with known D,
lattices with known neighbor shells, exact sinusoidal velocity signals);
no MLIP calculator is involved anywhere in this module.

Semantics pinned here (and never relaxed):
- saved-frame interval != integration timestep (both recorded explicitly)
- native OLS MSD stays ``publication_grade = false`` (kinisi remains the
  transport authority; the native fit is diagnostic only)
- GEMDAT diffusion/Haven outputs are diagnostic crosschecks only
- an Arrhenius physical Ea requires >= 3 temperatures; two-temperature
  fits are mathematically determined only and must warn
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units

from mliport.analysis.arrhenius import fit_arrhenius
from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.electrolyte import gemdat_electrolyte
from mliport.analysis.msd import calculate_msd
from mliport.analysis.spectral import calculate_vacf, velocity_spectrum
from mliport.analysis.structure import (
    density_map,
    periodic_rmsd_rmsf,
    radial_distribution,
)
from mliport.analysis.thermo import thermodynamic_diagnostics
from mliport.analysis.transport import (
    kinisi_transport,
    nernst_einstein_tracer_conductivity,
    particle_number_density_m3,
)
from mliport.analysis.validation import (
    InsufficientTrajectoryInformationError,
    InvalidTrajectoryError,
    UnsupportedAnalysisError,
    require_analysis,
    validate_time_axis,
)


def _dataset(frames, times_fs, convention="wrapped", **kwargs):
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=times_fs,
        positions_convention=convention,
        **kwargs,
    )


def _rocksalt_supercell():
    """2x2x2 conventional rocksalt (a=5 A): 32 Na + 32 Cl, cell 10 A."""
    base = Atoms(
        symbols=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        scaled_positions=[
            (0, 0, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
        ],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    return base.repeat((2, 2, 2))


# ---------------------------------------------------------------------------
# 28.1 Time-axis and units
# ---------------------------------------------------------------------------


def test_time_axis_saved_interval_differs_from_integration_timestep():
    frames = [
        Atoms("Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
        for _ in range(12)
    ]
    dataset = _dataset(
        frames,
        np.arange(12) * 10.0,
        md_timestep_fs=2.0,
        frame_stride_steps=5,
    )
    report = validate_time_axis(dataset)
    assert report.available and report.uniform and report.finite
    assert report.strictly_increasing
    assert report.frame_interval_fs == pytest.approx(10.0)
    assert report.maximum_interval_deviation_fs == pytest.approx(0.0)
    # The saved interval (10 fs) must be distinguishable from the integration
    # timestep (2 fs); the dataset keeps both.
    assert dataset.frame_interval_fs == pytest.approx(10.0)
    assert dataset.md_timestep_fs == pytest.approx(2.0)
    assert dataset.frame_stride_steps == 5


def test_time_axis_nonuniform_is_reported_and_rejected_by_downstream_tasks():
    frames = [
        Atoms("Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
        for _ in range(20)
    ]
    times = np.arange(20) * 10.0
    times[10] += 3.0
    dataset = _dataset(frames, times)
    report = validate_time_axis(dataset)
    assert not report.uniform
    assert report.frame_interval_fs is None
    assert report.maximum_interval_deviation_fs == pytest.approx(3.0)
    with pytest.raises(InvalidTrajectoryError, match="uniform time axis"):
        calculate_msd(dataset)


def test_ps_fs_conversion_is_exact_fs_to_ps():
    frames = [
        Atoms("Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
        for _ in range(51)
    ]
    dataset = _dataset(frames, np.arange(51) * 200.0)
    result = thermodynamic_diagnostics(dataset)
    assert result["summary"]["time_start_ps"] == pytest.approx(0.0)
    assert result["summary"]["time_end_ps"] == pytest.approx(10.0)
    assert result["summary"]["duration_ps"] == pytest.approx(10.0)
    assert result["columns"]["time_ps"][-1] == pytest.approx(10.0)


def test_variable_cell_is_ineligible_for_transport():
    frames = [
        Atoms(
            "Na2",
            positions=[[1, 1, 1], [3, 1, 1]],
            cell=[10 + 0.1 * k, 10, 10],
            pbc=True,
        )
        for k in range(10)
    ]
    dataset = _dataset(frames, np.arange(10) * 10.0)
    with pytest.raises(InvalidTrajectoryError, match="Variable-cell transport"):
        require_analysis(dataset, "transport")


# ---------------------------------------------------------------------------
# 28.2 Thermodynamics
# ---------------------------------------------------------------------------


def _thermo_dataset(n_frames, energies_eV, dt_fs=20.0):
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = []
    for energy in energies_eV:
        atoms = Atoms(
            "Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, forces=np.zeros((2, 3))
        )
        frames.append(atoms)
    return _dataset(frames, np.arange(len(frames)) * dt_fs)


def test_thermo_exact_constant_energy():
    dataset = _thermo_dataset(40, np.full(40, -12.5))
    result = thermodynamic_diagnostics(dataset)
    summary = result["summary"]
    assert summary["potential_energy_mean_eV_atom"] == pytest.approx(-6.25)
    assert summary["potential_energy_std_eV_atom"] == pytest.approx(0.0, abs=1e-12)
    assert summary["potential_energy_minimum_eV_atom"] == pytest.approx(-6.25)
    assert summary["potential_energy_maximum_eV_atom"] == pytest.approx(-6.25)


def test_thermo_known_linear_drift_mean_and_std():
    # E(k) = -10 + 0.02 k eV on 2 atoms -> eV/atom ramp from -5.0 to -4.51.
    n = 50
    energies = -10.0 + 0.02 * np.arange(n)
    dataset = _thermo_dataset(n, energies)
    dataset.total_energy_eV = energies.copy()
    dataset.kinetic_energy_eV = np.full(n, 1.5)
    result = thermodynamic_diagnostics(dataset)
    summary = result["summary"]
    per_atom = energies / 2.0
    assert summary["total_energy_mean_eV_atom"] == pytest.approx(per_atom.mean())
    assert summary["total_energy_std_eV_atom"] == pytest.approx(per_atom.std(ddof=1))
    assert summary["kinetic_energy_mean_eV_atom"] == pytest.approx(0.75)
    assert summary["kinetic_energy_std_eV_atom"] == pytest.approx(0.0, abs=1e-12)
    # eV/atom and ps columns carry the documented units.
    assert summary["potential_energy_mean_eV_atom"] == pytest.approx(-4.755)


def test_thermo_known_temperature_series():
    dataset = _thermo_dataset(20, np.full(20, -10.0))
    dataset.temperature_K = 300.0 + np.arange(20) * 1.0
    result = thermodynamic_diagnostics(dataset)
    summary = result["summary"]
    assert summary["temperature_mean_K"] == pytest.approx(309.5)
    assert summary["temperature_std_K"] == pytest.approx(np.arange(20).std(ddof=1))
    assert summary["temperature_minimum_K"] == pytest.approx(300.0)
    assert summary["temperature_maximum_K"] == pytest.approx(319.0)


# ---------------------------------------------------------------------------
# 28.3 RDF / coordination
# ---------------------------------------------------------------------------


def _rocksalt_dataset(n_frames=3):
    base = _rocksalt_supercell()
    frames = [base.copy() for _ in range(n_frames)]
    return _dataset(frames, np.arange(n_frames) * 10.0)


def test_rdf_rocksalt_first_shell_coordination_exact():
    # Conventional rocksalt a=5 A: nearest Na-Cl shell at a/2 = 2.5 A with
    # coordination 6; in the 2x2x2 supercell the MIC limit is 5.0 A.
    dataset = _rocksalt_dataset()
    result = radial_distribution(
        dataset,
        center_species="Na",
        neighbor_species="Cl",
        r_max_A=3.0,
        cn_cutoff_A=3.0,
    )
    assert result["coordination_number_at_cutoff"] == pytest.approx(6.0, abs=1e-9)
    assert result["r_max_safe_A"] == pytest.approx(5.0, abs=1e-9)


def test_rdf_species_selection_excludes_same_species_shells():
    dataset = _rocksalt_dataset()
    na_na = radial_distribution(
        dataset,
        center_species="Na",
        neighbor_species="Na",
        r_max_A=3.0,
        cn_cutoff_A=3.0,
    )
    # First Na-Na shell sits at a*sqrt(2)/2 = 3.54 A: zero Na neighbors at 3.0 A.
    assert na_na["coordination_number_at_cutoff"] == pytest.approx(0.0, abs=1e-9)
    na_na_wide = radial_distribution(
        dataset,
        center_species="Na",
        neighbor_species="Na",
        r_max_A=3.7,
        cn_cutoff_A=3.7,
    )
    assert na_na_wide["coordination_number_at_cutoff"] == pytest.approx(12.0, abs=1e-9)


def test_rdf_coordination_curve_is_monotone_cumulative():
    dataset = _rocksalt_dataset()
    result = radial_distribution(
        dataset, center_species="Na", neighbor_species="Cl", r_max_A=3.0
    )
    coordination = result["coordination_number_center_neighbor"]
    assert np.all(np.diff(coordination) >= -1e-12)
    assert coordination[-1] == pytest.approx(6.0, abs=1e-9)


def test_rdf_triclinic_limit_is_enforced_and_legal_radius_runs():
    base = _rocksalt_supercell()
    # Shear the cubic cell into a triclinic one (volume and nearest-neighbor
    # distances preserved; the conservative height-based MIC limit shrinks).
    shear = np.eye(3)
    shear[0, 1] = 0.4
    triclinic = base.copy()
    triclinic.set_cell(base.cell[:] @ shear.T, scale_atoms=False)
    triclinic.wrap()
    frames = [triclinic.copy() for _ in range(2)]
    dataset = _dataset(frames, np.arange(2) * 10.0)
    from mliport.analysis.structure import safe_r_max_A

    safe = safe_r_max_A(dataset.cells)
    assert safe < 5.0  # sheared cell is strictly more restrictive than cubic
    with pytest.raises(UnsupportedAnalysisError, match="triclinic minimum-image"):
        radial_distribution(
            dataset, center_species="Na", neighbor_species="Cl", r_max_A=safe * 1.05
        )
    legal = radial_distribution(
        dataset, center_species="Na", neighbor_species="Cl", r_max_A=safe
    )
    assert legal["r_max_A"] == pytest.approx(safe)


# ---------------------------------------------------------------------------
# 28.4 RMSD / RMSF
# ---------------------------------------------------------------------------


def test_rmsd_rigid_translation_by_lattice_vector_is_zero():
    base = _rocksalt_supercell()
    shifted = base.copy()
    shifted.positions += base.cell[0]  # exactly one lattice vector
    frames = [base.copy(), base.copy(), shifted.copy(), shifted]
    dataset = _dataset(frames, np.arange(4) * 10.0)
    result = periodic_rmsd_rmsf(dataset)
    assert result["periodic_displacement_rmsd_A"][-1] == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(result["periodic_displacement_rmsf_A"])) == pytest.approx(
        0.0, abs=1e-10
    )


def test_rmsd_rigid_fractional_translation_is_exact():
    base = _rocksalt_supercell()
    shifted = base.copy()
    shifted.positions += np.array([0.3, -0.2, 0.1])
    frames = [base.copy(), base.copy(), shifted.copy(), shifted]
    dataset = _dataset(frames, np.arange(4) * 10.0)
    result = periodic_rmsd_rmsf(dataset)
    expected = float(np.sqrt(0.3**2 + 0.2**2 + 0.1**2))
    assert result["periodic_displacement_rmsd_A"][-1] == pytest.approx(
        expected, abs=1e-10
    )


def test_rmsd_wrapped_boundary_crossing_uses_minimum_image():
    cell = 10.0
    # A single frame-to-frame step crosses the x boundary (0.3 -> 9.7):
    # the per-step minimum-image displacement is -0.6 A, so the accumulated
    # unwrapped displacement is 0.8 A, not the naive 9.2 A difference.
    frames = [
        Atoms("Na", positions=[[0.5, 5.0, 5.0]], cell=[cell] * 3, pbc=True),
        Atoms("Na", positions=[[0.4, 5.0, 5.0]], cell=[cell] * 3, pbc=True),
        Atoms("Na", positions=[[0.4, 5.0, 5.0]], cell=[cell] * 3, pbc=True),
        Atoms("Na", positions=[[0.3, 5.0, 5.0]], cell=[cell] * 3, pbc=True),
        Atoms("Na", positions=[[9.7, 5.0, 5.0]], cell=[cell] * 3, pbc=True),
    ]
    dataset = _dataset(frames, np.arange(5) * 10.0)
    result = periodic_rmsd_rmsf(dataset)
    assert result["periodic_displacement_rmsd_A"][-1] == pytest.approx(0.8, abs=1e-10)


# ---------------------------------------------------------------------------
# 28.5 Native MSD (diagnostic; publication_grade must stay false)
# ---------------------------------------------------------------------------


def test_msd_brownian_random_walk_recovers_known_diffusion():
    rng = np.random.default_rng(42)
    n_particles, n_frames, dt_fs = 100, 200, 1000.0
    d_true_m2_s = 1.0e-8
    d_true_a2_fs = d_true_m2_s * 1.0e5  # 1 m^2/s = 1e5 A^2/fs
    step_std = np.sqrt(2.0 * d_true_a2_fs * dt_fs)
    positions = np.cumsum(rng.normal(0.0, step_std, (n_frames, n_particles, 3)), axis=0)
    frames = [
        Atoms(f"Na{n_particles}", positions=positions[k], cell=[80, 80, 80], pbc=True)
        for k in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * dt_fs, convention="unwrapped")
    result = calculate_msd(
        dataset, mobile_species="Na", fit_start_ps=20.0, fit_stop_ps=190.0
    )
    fit = result["diagnostic_linear_diffusion_fits"]["xyz"]
    assert fit["publication_grade"] is False
    assert fit["fit_start_ps"] == pytest.approx(20.0)
    assert fit["fit_stop_ps"] == pytest.approx(190.0)
    # Ensemble of 100 particles recovers D to a few permille; the pinned gate
    # is deliberately loose (25%) so the test stays a physics check, not a
    # noise-seed check.
    assert fit["D_diagnostic_m2_s"] == pytest.approx(d_true_m2_s, rel=0.25)
    assert fit["r_squared"] > 0.99
    assert not fit["diffusive_regime_warning"]
    alpha = result["log_log_alpha_by_axes"]["xyz"]
    mid = alpha[~np.isnan(alpha)]
    assert 0.8 < np.median(mid[10:-10]) < 1.2
    # Final-frame MSD matches the analytic 2 d D t (d = 3) within 10%.
    expected_last = 6.0 * d_true_a2_fs * (n_frames - 1) * dt_fs
    assert result["msd_by_axes_A2"]["xyz"][-1] == pytest.approx(expected_last, rel=0.10)


def test_msd_fft_and_direct_methods_agree():
    rng = np.random.default_rng(11)
    n_frames, dt_fs = 120, 1000.0
    positions = np.cumsum(rng.normal(0.0, 0.05, (n_frames, 20, 3)), axis=0)
    frames = [
        Atoms("Na20", positions=positions[k], cell=[60, 60, 60], pbc=True)
        for k in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * dt_fs, convention="unwrapped")
    fft = calculate_msd(dataset, method="fft")
    direct = calculate_msd(dataset, method="direct")
    assert set(fft["msd_by_axes_A2"]) == set(direct["msd_by_axes_A2"])
    assert "xyz" in fft["msd_by_axes_A2"]
    for axes, a in fft["msd_by_axes_A2"].items():
        assert np.allclose(
            a[1:], direct["msd_by_axes_A2"][axes][1:], rtol=1e-10, atol=1e-10
        )


# ---------------------------------------------------------------------------
# 28.6 VACF
# ---------------------------------------------------------------------------


def _cosine_velocity_dataset(
    n_frames=200, dt_fs=5.0, amplitude_a_fs=0.01, freq_thz=2.0
):
    """v(t) = A cos(2 pi f t) on every Cartesian axis (f in THz)."""
    t_fs = np.arange(n_frames) * dt_fs
    velocity = amplitude_a_fs * np.cos(2.0 * np.pi * freq_thz * t_fs * 1.0e-3)
    frames = []
    for k in range(n_frames):
        atoms = Atoms("Na", positions=[[3, 3, 3]], cell=[10, 10, 10], pbc=True)
        # Dataset contract: velocities are stored in A/fs; ASE velocities are
        # A/(ASE time unit) and the importer multiplies by units.fs.
        atoms.set_velocities(np.full((1, 3), velocity[k] / units.fs))
        frames.append(atoms)
    dataset = _dataset(frames, t_fs)
    assert dataset.velocities is not None
    assert dataset.velocities[0, 0, 0] == pytest.approx(amplitude_a_fs, rel=1e-6)
    return dataset, velocity, amplitude_a_fs, freq_thz, dt_fs


def test_vacf_cosine_sequence_analytic_and_normalized_convention():
    dataset, _, amplitude, freq_thz, dt_fs = _cosine_velocity_dataset()
    result = calculate_vacf(dataset)
    assert result["frame_interval_fs"] == pytest.approx(dt_fs)
    vacf_raw = result["vacf_raw_A2_fs2"]
    vacf_norm = result["vacf_normalized"]
    # C(0) = <v.v> time-averaged over the sinusoid = A^2/2 per axis (x3).
    # The 200-frame window covers the period count up to a sample boundary,
    # so allow a 2% windowing bias on the absolute value.
    assert vacf_raw[0] == pytest.approx(1.5 * amplitude**2, rel=0.02)
    # Normalized convention: C(0) is exactly 1.
    assert vacf_norm[0] == pytest.approx(1.0, rel=1e-12)
    # The documented convention is the time-origin average of v(t+tau).v(t);
    # compute that definition independently of the code under test.
    v = dataset.velocities[:, :, :]
    n = v.shape[0]
    expected_def = np.array(
        [float(np.mean(np.sum(v[: n - lag] * v[lag:], axis=2))) for lag in range(n)]
    )
    assert np.allclose(vacf_raw, expected_def, rtol=1e-10, atol=1e-14)
    # Sinusoidal structure: first zero crossing at T/4 = 125 fs (frame 25),
    # minimum at T/2 = 250 fs (frame 50), period 500 fs.
    lags_fs = np.arange(len(vacf_raw)) * dt_fs
    crossings = np.where(np.diff(np.sign(vacf_raw[:100])))[0]
    assert len(crossings) >= 1
    assert crossings[0] == pytest.approx(25, abs=2)
    assert int(np.argmin(vacf_raw[:100])) == pytest.approx(50, abs=2)


def test_vacf_direct_and_fft_agree():
    dataset, *_ = _cosine_velocity_dataset(n_frames=128)
    fft = calculate_vacf(dataset, method="fft")
    direct = calculate_vacf(dataset, method="direct")
    assert np.allclose(
        fft["vacf_raw_A2_fs2"][:40],
        direct["vacf_raw_A2_fs2"][:40],
        rtol=1e-8,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 28.7 Spectrum
# ---------------------------------------------------------------------------


def test_spectrum_peak_frequency_and_axis_conventions():
    n_frames, dt_fs, freq_thz = 200, 5.0, 2.0  # df = 1 THz exactly on-grid
    dataset, *_ = _cosine_velocity_dataset(
        n_frames=n_frames, dt_fs=dt_fs, freq_thz=freq_thz
    )
    vacf = calculate_vacf(dataset)
    spectrum = velocity_spectrum(vacf)
    frequencies = spectrum["frequency_THz"]
    peak = frequencies[int(np.argmax(spectrum["spectrum"]))]
    assert peak == pytest.approx(freq_thz, abs=0.5 * (1000.0 / (n_frames * dt_fs)))
    # Frequency axis: cm^-1 == THz * (1e10 cm/m) * 1e-12 s/ps ... conversion
    # consistency per saved axis.
    assert spectrum["frequency_cm^-1"] == pytest.approx(
        frequencies * 33.3564095, rel=1e-5
    )
    assert vacf["nyquist_THz"] == pytest.approx(0.5 * 1000.0 / dt_fs)
    assert spectrum["normalization"] == "normalized_area"
    # Exact sinusoid has no negative-time VACF content beyond numerical noise.
    assert spectrum["negative_fraction"] == pytest.approx(0.0, abs=0.2)


def test_spectrum_taper_changes_windowed_vacf_envelope():
    dataset, *_ = _cosine_velocity_dataset(n_frames=200)
    vacf = calculate_vacf(dataset)
    tapered = velocity_spectrum(vacf, taper="one-sided-cosine")
    raw = velocity_spectrum(vacf, taper="none")
    assert tapered["taper"] == "one-sided-cosine"
    # The one-sided cosine taper keeps w(0)=1 and forces w(last)=0 on the
    # normalized windowed VACF, damping the spectral tails.
    windowed = tapered["windowed_vacf"]
    assert windowed[0] == pytest.approx(1.0, rel=1e-12)
    assert windowed[-1] == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(tapered["spectrum"], raw["spectrum"])


# ---------------------------------------------------------------------------
# 28.8 Density
# ---------------------------------------------------------------------------


def test_density_map_known_periodic_sites_normalization():
    sites = np.array(
        [[10.0, 10.0, 10.0], [40.0, 40.0, 40.0], [10.0, 40.0, 10.0], [40.0, 10.0, 40.0]]
    )
    n_frames = 4
    frames = [
        Atoms("Na4", positions=sites, cell=[50, 50, 50], pbc=True)
        for _ in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * 10.0)
    result = density_map(
        dataset, mobile_species="Na", spacing_A=2.0, smoothing_sigma_A=None
    )
    counts = result["counts_per_voxel"]
    assert counts.shape == (25, 25, 25)
    # Total integrated occupancy == frames x selected atoms.
    assert int(counts.sum()) == n_frames * 4
    # Each site voxel is occupied at every frame.
    occupied = counts[counts > 0]
    assert occupied.size == 4
    assert np.all(occupied == n_frames)
    # number_density integrates to the mean mobile atom count.
    number_density = result["number_density_A^-3"]
    voxel_volume = result["voxel_volume_A3"]
    assert result["number_of_mobile_particles"] == 4
    integrated = float(np.sum(number_density)) * voxel_volume
    assert integrated == pytest.approx(4.0, rel=1e-9)


def test_density_map_triclinic_cell_is_handled():
    sites = np.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]])
    shear = np.eye(3)
    shear[0, 1] = 0.25
    cell = np.eye(3) * 20.0 @ shear.T
    frames = []
    for _ in range(3):
        atoms = Atoms("Na2", positions=sites, cell=cell, pbc=True)
        atoms.wrap()
        frames.append(atoms)
    dataset = _dataset(frames, np.arange(3) * 10.0)
    result = density_map(
        dataset, mobile_species="Na", spacing_A=2.0, smoothing_sigma_A=None
    )
    assert int(result["counts_per_voxel"].sum()) == 3 * 2


# ---------------------------------------------------------------------------
# 29 / 28.4 kinisi transport (quantitative transport authority)
# ---------------------------------------------------------------------------


def test_nernst_einstein_tracer_formula_exact():
    import scipy.constants as constants

    density_m3 = 1.0e28
    d_m2_s = 2.0e-9
    temperature = 900.0
    charge_e = 1.0
    result = nernst_einstein_tracer_conductivity(
        particle_density_m3=density_m3,
        tracer_diffusion_m2_s=d_m2_s,
        temperature_K=temperature,
        ionic_charge_e=charge_e,
    )
    expected = (
        density_m3
        * (charge_e * constants.e) ** 2
        * d_m2_s
        / (constants.k * temperature)
    )
    assert result["sigma_NE_tracer_S_m"] == pytest.approx(expected, rel=1e-12)
    assert result["sigma_NE_tracer_S_cm"] == pytest.approx(expected * 0.01, rel=1e-12)
    assert result["sigma_NE_tracer_mS_cm"] == pytest.approx(expected * 10.0, rel=1e-12)
    with pytest.raises(ValueError, match="ionic_charge_e is required"):
        nernst_einstein_tracer_conductivity(
            particle_density_m3=density_m3,
            tracer_diffusion_m2_s=d_m2_s,
            temperature_K=temperature,
            ionic_charge_e=None,
        )


def _brownian_dataset(
    n_particles=30, n_frames=300, dt_fs=1000.0, d_m2_s=1.0e-8, cell=60.0
):
    rng = np.random.default_rng(7)
    d_a2_fs = d_m2_s * 1.0e5
    step_std = np.sqrt(2.0 * d_a2_fs * dt_fs)
    positions = np.cumsum(rng.normal(0.0, step_std, (n_frames, n_particles, 3)), axis=0)
    frames = [
        Atoms(f"Na{n_particles}", positions=positions[k], cell=[cell] * 3, pbc=True)
        for k in range(n_frames)
    ]
    return _dataset(frames, np.arange(n_frames) * dt_fs, convention="unwrapped")


def test_kinisi_transport_brownian_known_diffusion_and_ne_consistency():
    pytest.importorskip("kinisi")
    dataset = _brownian_dataset()
    result = kinisi_transport(
        dataset,
        mobile_species="Na",
        ionic_charge_e=1.0,
        fit_start_ps=20.0,
        temperature_K=900.0,
        random_seed=0,
        n_samples=256,
        n_walkers=16,
        n_burn=200,
        n_thin=10,
    )
    assert result["quality"]["fixed_cell"] is True
    assert result["quality"]["uniform_sampling"] is True
    # Exact unwrapped reconstruction equivalence is verified, not assumed.
    semantics = result["kinisi_position_semantics"]
    assert semantics["exact_unwrapped_reconstruction_equivalent"] is True
    assert semantics["maximum_exact_vs_mic_difference_A"] < 1e-9
    posterior = result["tracer_diffusion"]["D_posterior_m2_s"]
    mean_d = (
        float(posterior["mean"])
        if isinstance(posterior, dict)
        else float(np.mean(result["D_tracer_samples_m2_s"]))
    )
    assert mean_d == pytest.approx(1.0e-8, rel=0.25)
    # sigma_NE must be the documented n (z e)^2 D / (k_B T) of the posterior
    # mean, with n from the dataset geometry (30 atoms in 60^3 A^3).
    density = particle_number_density_m3(dataset, mobile_species="Na")
    assert density == pytest.approx(30.0 / (60.0**3 * 1.0e-30), rel=1e-9)
    expected = nernst_einstein_tracer_conductivity(
        particle_density_m3=density,
        tracer_diffusion_m2_s=mean_d,
        temperature_K=900.0,
        ionic_charge_e=1.0,
    )["sigma_NE_tracer_S_m"]
    assert result["nernst_einstein"]["sigma_NE_tracer_S_m"] == pytest.approx(
        expected, rel=1e-9
    )


def test_kinisi_transport_rejects_large_unsafe_wrapped_steps():
    pytest.importorskip("kinisi")
    # A wrapped source with a per-frame step beyond 0.8 * half the minimum
    # cell height cannot be unwrapped safely: it must be refused, not guessed.
    rng = np.random.default_rng(5)
    sites = np.array([[10.0, 10.0, 10.0], [40.0, 10.0, 10.0]])  # 30 A hop
    n_frames = 60
    positions = np.tile(sites, (n_frames, 1, 1))
    positions[30:, 0] = sites[1]
    frames = [
        Atoms("Na2", positions=positions[k], cell=[50, 50, 50], pbc=True)
        for k in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * 100.0)
    with pytest.raises(
        InsufficientTrajectoryInformationError, match="unwrap safety ratio"
    ) as excinfo:
        kinisi_transport(
            dataset,
            mobile_species="Na",
            ionic_charge_e=1.0,
            fit_start_ps=1.0,
            temperature_K=900.0,
            random_seed=0,
            n_samples=32,
            n_walkers=8,
            n_burn=10,
            n_thin=10,
        )
    assert excinfo.value.status == "insufficient_trajectory_information"


# ---------------------------------------------------------------------------
# 31 GEMDAT electrolyte mechanisms (diagnostic crosscheck)
# ---------------------------------------------------------------------------


_SITES_8 = np.array(
    [[x, y, z] for x in (10.0, 22.0) for y in (10.0, 22.0) for z in (10.0, 22.0)]
)


def _site_hopping_dataset(n_frames=600, jitter=0.3, seed=3):
    """Four particles resident on eight sites; one 12 A hop each."""
    rng = np.random.default_rng(seed)
    homes = np.array([0, 1, 2, 3])
    positions = np.tile(_SITES_8[homes], (n_frames, 1, 1)).astype(float)
    positions += rng.normal(0.0, jitter, positions.shape)
    hops = ((150, 0, 0, 4), (400, 1, 1, 5), (280, 2, 2, 6), (500, 3, 3, 7))
    for frame, particle, site_from, site_to in hops:
        positions[frame:, particle] = _SITES_8[site_to] + rng.normal(
            0.0, jitter, (n_frames - frame, 3)
        )
    frames = [
        Atoms("Na4", positions=positions[k], cell=[50, 50, 50], pbc=True)
        for k in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * 100.0)
    return dataset, hops


@pytest.fixture()
def known_sites_poscar(tmp_path: Path) -> Path:
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(50.0), ["Na"] * len(_SITES_8), _SITES_8, coords_are_cartesian=True
    )
    path = tmp_path / "sites_known.vasp"
    structure.to(fmt="poscar", filename=str(path))
    return path


def test_gemdat_recovers_exact_injected_jump_events(known_sites_poscar: Path):
    pytest.importorskip("gemdat")
    dataset, hops = _site_hopping_dataset()
    result = gemdat_electrolyte(
        dataset,
        mobile_species="Na",
        sites_path=known_sites_poscar,
        temperature_K=900.0,
        site_radius_A=2.5,
    )
    summary = result.summary
    assert summary["backend"] == "GEMDAT"
    assert summary["number_of_sites"] == len(_SITES_8)
    assert summary["frame_interval_fs"] == pytest.approx(100.0)
    events = result.tables["transition_events"]
    assert len(events) == len(hops)
    for (_, row), (frame, particle, site_from, site_to) in zip(
        events.iterrows(), hops, strict=True
    ):
        assert int(row["atom index"]) == particle
        assert int(row["start site"]) == site_from
        assert int(row["destination site"]) == site_to
        assert int(row["time"]) == frame - 1
    jumps = result.tables["jumps"]
    assert np.allclose(jumps["jump_distance_A"].to_numpy(), 12.0, atol=1e-6)
    # Explicit reproducible site source is recorded in the summary.
    assert "sites_known.vasp" in str(summary["site_source"])


def test_gemdat_refuses_unsafe_wrapped_unwrap(known_sites_poscar: Path):
    pytest.importorskip("gemdat")
    # One 30 A instant hop inside a 50 A cell exceeds the 0.8 unwrap-safety
    # heuristic for wrapped sources: refuse instead of guessing images.
    rng = np.random.default_rng(5)
    n_frames = 60
    positions = np.tile(_SITES_8[:4], (n_frames, 1, 1)).astype(float)
    positions += rng.normal(0.0, 0.3, positions.shape)
    positions[30:, 0] = (
        _SITES_8[6]
        + np.array([0.0, 9.0, 9.0])
        + rng.normal(0.0, 0.3, (n_frames - 30, 3))
    )
    frames = [
        Atoms("Na4", positions=positions[k], cell=[50, 50, 50], pbc=True)
        for k in range(n_frames)
    ]
    dataset = _dataset(frames, np.arange(n_frames) * 100.0)
    with pytest.raises(UnsupportedAnalysisError, match="unwrap safety ratio"):
        gemdat_electrolyte(
            dataset,
            mobile_species="Na",
            sites_path=known_sites_poscar,
            temperature_K=900.0,
            site_radius_A=2.5,
        )


# ---------------------------------------------------------------------------
# 32 Arrhenius
# ---------------------------------------------------------------------------


def test_arrhenius_noiseless_recovery_exact():
    import scipy.constants as constants

    k_eV_per_K = constants.k / constants.e  # 8.617e-5 eV/K
    d0_true = 1.0e-8
    ea_true = 0.30
    temperatures = np.array([500.0, 600.0, 700.0, 800.0])
    diffusivities = d0_true * np.exp(-ea_true / (k_eV_per_K * temperatures))
    result = fit_arrhenius(
        temperatures_K=temperatures, diffusivities_m2_s=diffusivities
    )
    assert result["activation_energy_eV"] == pytest.approx(ea_true, rel=1e-6)
    assert result["preexponential_factor_m2_s"] == pytest.approx(d0_true, rel=1e-4)
    assert result["r_squared"] > 1.0 - 1e-9
    assert result["number_of_independent_temperature_runs"] == 4


def test_arrhenius_noisy_fit_with_supplied_uncertainty():
    import scipy.constants as constants

    k_eV_per_K = constants.k / constants.e
    rng = np.random.default_rng(13)
    d0_true = 5.0e-8
    ea_true = 0.25
    temperatures = np.array([500.0, 600.0, 700.0, 800.0])
    diffusivities = d0_true * np.exp(-ea_true / (k_eV_per_K * temperatures))
    noisy = diffusivities * (1.0 + rng.normal(0.0, 0.01, diffusivities.shape))
    result = fit_arrhenius(
        temperatures_K=temperatures,
        diffusivities_m2_s=noisy,
        diffusivity_std_m2_s=0.01 * noisy,
    )
    assert result["activation_energy_eV"] == pytest.approx(ea_true, abs=0.05)
    assert result["activation_energy_std_eV"] > 0.0
    assert result["r_squared"] > 0.99
    assert "uncertainty_method" in result


def test_arrhenius_two_temperature_fit_warns_no_goodness_of_fit():
    temperatures = [500.0, 600.0]
    diffusivities = [1.0e-9, 2.2e-9]
    result = fit_arrhenius(
        temperatures_K=temperatures, diffusivities_m2_s=diffusivities
    )
    warnings = result["warnings"]
    assert any("two temperatures" in warning.lower() for warning in warnings)
    # A physical Ea claim requires >= 3 temperatures: the two-point line is
    # recorded as mathematically determined only.
    assert result["number_of_independent_temperature_runs"] == 2


# ---------------------------------------------------------------------------
# Product task registry is the source of truth for the tested families
# ---------------------------------------------------------------------------


def test_analysis_request_task_registry_covers_tested_families():
    from mliport.analysis.schema import AnalysisRequest

    expected = {
        "validate",
        "thermo",
        "rdf",
        "rmsd",
        "msd",
        "transport",
        "density",
        "arrhenius",
        "electrolyte",
        "vacf",
        "spectrum",
    }
    assert expected == AnalysisRequest.VALID_TASKS
    assert dataclasses.is_dataclass(AnalysisRequest)
