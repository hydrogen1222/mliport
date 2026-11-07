"""Known-answer test for the real Calorine CPUNEP path.

This test only runs where ``calorine`` is installed (the dedicated
``.venv-nep`` environment) and where the GPUMD repository's NEP fixtures are
present locally. Both locations can be overridden with
``MLIPORT_NEP_TEST_MODEL`` / ``MLIPORT_NEP_TEST_STRUCTURE``; nothing is assumed
about the machine, so a checkout without the fixtures simply skips.

The reference energy is the value GPUMD's own NEP test suite reports for the
BaZrO3 fixture. CPUNEP is float64 and evaluates in-process, so it may differ
from that float32 GPUMD number in the last digits; the tolerance is therefore
relative, not absolute.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

# The dedicated .venv-nep environment deliberately contains only the numerical
# backend (calorine + ase) and no mliport installation, so make the checkout
# importable from the test itself; this keeps the documented
# `.venv-nep/bin/python -m pytest tests/mliport/mliport/test_nep_real_cpu.py`
# command working without a second install step.
_CHECKOUT = Path(__file__).resolve().parents[3] / "mliport"
if _CHECKOUT.is_dir() and str(_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT))

from mliport.calculators.nep_calc import NEPCalculatorWrapper  # noqa: E402

_CALORINE_AVAILABLE = find_spec("calorine") is not None

_DEFAULT_MODEL = Path("/tmp/gpumd-src/tests_pytest/fixtures/models/nep_BaZrO3.txt")
_DEFAULT_STRUCTURE = Path(
    "/tmp/gpumd-src/tests_pytest/fixtures/structures/BaZrO3-nat40-rattled.xyz"
)

MODEL_PATH = Path(os.environ.get("MLIPORT_NEP_TEST_MODEL", _DEFAULT_MODEL))
STRUCTURE_PATH = Path(os.environ.get("MLIPORT_NEP_TEST_STRUCTURE", _DEFAULT_STRUCTURE))

#: Energy GPUMD's NEP test suite reports for this structure/model pair (eV).
REFERENCE_ENERGY_EV = -258.17361808
RELATIVE_ENERGY_TOLERANCE = 1e-6

pytestmark = [
    pytest.mark.skipif(
        not _CALORINE_AVAILABLE, reason="calorine is not installed (use .venv-nep)"
    ),
    pytest.mark.skipif(
        not MODEL_PATH.exists(), reason=f"NEP model fixture not found: {MODEL_PATH}"
    ),
    pytest.mark.skipif(
        not STRUCTURE_PATH.exists(),
        reason=f"NEP structure fixture not found: {STRUCTURE_PATH}",
    ),
]


def test_cpunep_single_point_matches_gpumd_reference():
    """Float64 CPUNEP reproduces the reference energy, forces and stress."""
    import numpy as np
    from ase.io import read

    atoms = read(STRUCTURE_PATH)
    wrapper = NEPCalculatorWrapper(
        MODEL_PATH, device="cpu", task="bulk", nep_runtime="ase"
    )

    info = wrapper.info()
    assert info["model_type"] == "nep"
    assert info["default_dtype"] == "float64"
    assert info["per_call_subprocess"] is False
    assert info["nep_element_types"] == ["Ba", "Zr", "O"]
    assert info["nep_model_version"] == "nep4"

    # Pre-flight element guard must accept the fixture before any allocation.
    wrapper.require_elements(atoms.get_chemical_symbols())

    atoms.calc = wrapper.get_calculator()

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert energy == pytest.approx(
        REFERENCE_ENERGY_EV, rel=RELATIVE_ENERGY_TOLERANCE
    ), f"CPUNEP energy {energy!r} eV differs from the GPUMD reference"

    forces = atoms.get_forces()
    assert np.isfinite(forces).all()
    assert forces.shape == (len(atoms), 3)

    assert wrapper.has_stress is True
    stress = atoms.get_stress()
    assert np.isfinite(stress).all()
