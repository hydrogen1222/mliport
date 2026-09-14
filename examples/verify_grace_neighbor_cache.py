"""Verify the GRACE neighbour-list cache is physically equivalent to a
cache-less run.

This script walks a short 200-frame MD-like trajectory through cached and
cache-less calculators and asserts:

  * the complete neighbour periodic-image multiset is identical (0 tolerance),
    including repeated images of the same atom pair;
  * bond vectors agree within float64 reconstruction rounding (<= 2e-12 Å);
  * energy/force/stress differences remain inside explicit numerical bounds.

The two TensorFlow/XLA model instances are not bitwise deterministic on GPU,
and ~1e-14 Å bond-vector rounding can be amplified by a high-order model. The
default output bounds match the checked V100/FP64 model and are deliberately
reported rather than described as exact equality.

Usage (use the GRACE environment, e.g. ``.venv-grace``):

    python examples/verify_grace_neighbor_cache.py \
        --model models/grace/GRACE-2L-SMAX-OMAT-large \
        --structure local-data/LGPS222.vasp --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read

# Allow running from a source checkout without installing mliport.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mliport.calculators.factory import CalculatorFactory  # noqa: E402


def _sorted_images(data):
    """Return real-atom model inputs in a deterministic multiset order."""
    ind_i = np.asarray(data["ind_i"]).ravel()
    ind_j = np.asarray(data["ind_j"]).ravel()
    vectors = np.asarray(data["bond_vector"]).reshape(-1, 3)
    n_real = int(np.asarray(data["batch_tot_nat_real"]))
    real = ind_i < n_real
    ind_i = ind_i[real]
    ind_j = ind_j[real]
    vectors = vectors[real]
    order = np.lexsort(
        (
            np.round(vectors[:, 2], 12),
            np.round(vectors[:, 1], 12),
            np.round(vectors[:, 0], 12),
            ind_j,
            ind_i,
        )
    )
    return ind_i[order], ind_j[order], vectors[order]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GRACE SavedModel directory")
    parser.add_argument("--structure", required=True, help="Periodic structure file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--skin", type=float, default=1.5)
    parser.add_argument("--energy-atol", type=float, default=1e-3, help="eV")
    parser.add_argument("--force-atol", type=float, default=5e-3, help="eV/Å")
    parser.add_argument("--stress-atol", type=float, default=1e-6, help="eV/Å³")
    args = parser.parse_args()

    atoms0 = read(args.structure)
    cached_wrap = CalculatorFactory.create(
        model_type="grace",
        model_path=args.model,
        device=args.device,
        task="bulk",
        cpu_threads=8,
        neighbor_cache=True,
        neighbor_skin=args.skin,
    )
    plain_wrap = CalculatorFactory.create(
        model_type="grace",
        model_path=args.model,
        device=args.device,
        task="bulk",
        cpu_threads=8,
        neighbor_cache=False,
    )
    cached = cached_wrap.get_calculator()
    plain = plain_wrap.get_calculator()

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
    for step in range(args.steps):
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
        ci, cj, cb = _sorted_images(cached.data)
        pi, pj, pb = _sorted_images(plain.data)
        if not (np.array_equal(ci, pi) and np.array_equal(cj, pj)):
            max_image_diff = max(max_image_diff, 1)
        elif len(cb):
            max_bv_diff = max(max_bv_diff, float(np.abs(cb - pb).max()))
        max_de = max(max_de, float(abs(ec - ep)))
        max_df = max(max_df, float(np.abs(fc - fp).max()))
        max_ds = max(max_ds, float(np.abs(sc - sp).max()))

    print(f"frames: {args.steps}")
    print(f"periodic-image multiset mismatch: {max_image_diff} (must be 0)")
    print(f"bond-vector max diff: {max_bv_diff:.3e} Å (must be <= 2e-12)")
    print(f"max |dE|: {max_de:.3e} eV (limit {args.energy_atol:.1e})")
    print(f"max |dF|: {max_df:.3e} eV/Å (limit {args.force_atol:.1e})")
    print(f"max |dS|: {max_ds:.3e} eV/Å³ (limit {args.stress_atol:.1e})")

    ok = (
        max_image_diff == 0
        and max_bv_diff <= 2e-12
        and max_de <= args.energy_atol
        and max_df <= args.force_atol
        and max_ds <= args.stress_atol
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
