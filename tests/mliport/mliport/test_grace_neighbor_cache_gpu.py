"""GPU integration gate for the GRACE neighbor cache.

Requires: a CUDA GPU, the GRACE SavedModel at models/grace/GRACE-2L-SMAX-OMAT-large
and a periodic LGPS structure (LGPS222.vasp) in the repository root. Skipped
automatically when any of these is missing.

Verifies, over a short MD-like walk, that the cached path is physically
equivalent to the cache-less path:
  * complete neighbour periodic-image multiset identical (0 tolerance);
  * bond vectors within float64 rounding (<=1e-12);
  * FP64 energy/force/stress differences remain at the model/runtime's small
    numerical-noise scale (separate model instances are not bitwise
    deterministic on TensorFlow/XLA GPU).

This expensive engine test is explicitly opt-in so the generic test suite
cannot accidentally initialise TensorFlow from the wrong environment. Run it
with ``MLIPORT_RUN_GRACE_GPU_TESTS=1`` under ``.venv-grace``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase.io import read

from mliport.calculators.factory import CalculatorFactory

MODEL = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "models",
    "grace",
    "GRACE-2L-SMAX-OMAT-large",
)
STRUCT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "LGPS222.vasp",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("MLIPORT_RUN_GRACE_GPU_TESTS") != "1"
    or not (os.path.isdir(MODEL) and os.path.isfile(STRUCT)),
    reason="opt-in GRACE GPU test/model/structure unavailable",
)


def test_grace_neighbor_cache_physical_equivalence():
    atoms0 = read(STRUCT)
    cw = CalculatorFactory.create(
        model_type="grace",
        model_path=MODEL,
        device="cuda",
        task="bulk",
        cpu_threads=8,
        neighbor_cache=True,
        neighbor_skin=1.5,
    )
    pw = CalculatorFactory.create(
        model_type="grace",
        model_path=MODEL,
        device="cuda",
        task="bulk",
        cpu_threads=8,
        neighbor_cache=False,
    )
    cached, plain = cw.get_calculator(), pw.get_calculator()

    ac = atoms0.copy()
    ac.calc = cached
    ap = atoms0.copy()
    ap.calc = plain
    rng = np.random.default_rng(42)
    vel = rng.normal(0, 0.12, (len(atoms0), 3))

    max_image_diff = 0
    max_bv_diff = 0.0
    max_df = 0.0
    max_de = 0.0
    max_ds = 0.0
    for step in range(200):
        if step > 0:
            vel = 0.99 * vel + rng.normal(0, 0.03, vel.shape)
            dr = vel + rng.normal(0, 0.01, vel.shape)
            ac.positions += dr
            ap.positions += dr
        ec = ac.get_potential_energy()
        fc = ac.get_forces()
        sc = ac.get_stress()
        ep = ap.get_potential_energy()
        fp = ap.get_forces()
        sp = ap.get_stress()
        dc, dp = cached.data, plain.data
        n_real = int(np.asarray(dc["batch_tot_nat_real"]))
        ic = np.asarray(dc["ind_i"]).ravel()
        jc = np.asarray(dc["ind_j"]).ravel()
        ip = np.asarray(dp["ind_i"]).ravel()
        jp = np.asarray(dp["ind_j"]).ravel()
        bc = np.asarray(dc["bond_vector"]).reshape(-1, 3)
        bp = np.asarray(dp["bond_vector"]).reshape(-1, 3)
        mc = ic < n_real
        mp = ip < n_real
        ci, cj, cb = _sort_images(ic[mc], jc[mc], bc[mc])
        pi, pj, pb = _sort_images(ip[mp], jp[mp], bp[mp])
        if not (np.array_equal(ci, pi) and np.array_equal(cj, pj)):
            max_image_diff = max(max_image_diff, 1)
        elif len(cb):
            max_bv_diff = max(max_bv_diff, float(np.abs(cb - pb).max()))
        max_de = max(max_de, float(abs(ec - ep)))
        max_df = max(max_df, float(np.abs(fc - fp).max()))
        max_ds = max(max_ds, float(np.abs(sc - sp).max()))

    print(
        "GRACE cache cross-check maxima:",
        {
            "d_bond_A": max_bv_diff,
            "dE_eV": max_de,
            "dF_eV_A": max_df,
            "dS_eV_A3": max_ds,
        },
    )
    assert max_image_diff == 0, "periodic-image multiset differs from fresh search"
    assert max_bv_diff <= 2e-12, f"bond vectors differ: {max_bv_diff}"
    assert max_de <= 1e-3, f"|dE| exceeds the validated GPU noise bound: {max_de}"
    assert max_df <= 5e-3, f"|dF| exceeds the validated GPU noise bound: {max_df}"
    assert max_ds <= 1e-6, f"|dS| exceeds the validated GPU noise bound: {max_ds}"


def _sort_images(ind_i, ind_j, vectors):
    """Canonicalise repeated periodic images without collapsing them."""
    vectors = np.asarray(vectors, dtype=float)
    order = np.lexsort(
        (
            np.round(vectors[:, 2], 10),
            np.round(vectors[:, 1], 10),
            np.round(vectors[:, 0], 10),
            np.asarray(ind_j),
            np.asarray(ind_i),
        )
    )
    return np.asarray(ind_i)[order], np.asarray(ind_j)[order], vectors[order]
