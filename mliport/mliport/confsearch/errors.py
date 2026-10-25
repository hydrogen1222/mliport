"""Exceptions and scope gates for the confsearch workflow.

Section 2.2 of the specification lists explicit v1 non-goals.  A request that
falls outside v1 must raise a clear error that *names* the non-goal; it must
never fall back to a half-implementation or a silently different calculation.
"""

from __future__ import annotations


class ConfsearchError(ValueError):
    """Base class for invalid or unsupported confsearch inputs."""


class ConfsearchPreparationError(ConfsearchError):
    """Inputs cannot be prepared without changing their physical meaning."""


class ConfsearchNotImplementedError(ConfsearchError):
    """A requested feature is an explicit v1 non-goal."""


class ConfsearchGradientConsistencyError(ConfsearchError):
    """The model's forces are not (verifiably) the gradient of its energy."""


class ConfsearchInternalError(RuntimeError):
    """A violated invariant that is a bug, not a scientific outcome.

    Example: a fixed atom moved during MD/quench (spec section 8.2 rule 1).  The
    whole search aborts; this must never be reported as a result.
    """


#: v1 non-goals (spec section 2.2), mapped to the error text a user sees.
NON_GOALS: dict[str, str] = {
    "npt": (
        "NPT or any variable-cell MD is a v1 non-goal; the cell stays fixed "
        "(NVT, and quenches/refinement are fixed-cell)"
    ),
    "variable_cell": (
        "variable-cell refinement is a v1 non-goal; a slab cell must stay fixed"
    ),
    "bulk_mobile": (
        "bulk modes (mobile-ion arrangement, substitutional ordering) are a v1 "
        "non-goal; only cs_mode='adsorbate' is implemented"
    ),
    "melt_quench": (
        "amorphous melt-quench is a v1 non-goal; only cs_mode='adsorbate' is "
        "implemented"
    ),
    "interface": (
        "interface searches are a v1 non-goal; only cs_mode='adsorbate' is "
        "implemented"
    ),
    "multi_adsorbate": (
        "multiple adsorbates/coverage are a v1 non-goal; place exactly one "
        "molecule on one side of the slab"
    ),
    "two_sided": (
        "two-sided slabs are a v1 non-goal; the adsorbate must be above the "
        "slab's +z surface"
    ),
    "replica_exchange": (
        "replica exchange is a v1 non-goal; replicas are independent MD chains"
    ),
    "metadynamics": (
        "metadynamics, bias potentials, restraints and walls are v1 non-goals; "
        "no term may change the potential energy surface"
    ),
    "free_energy": (
        "free energies, Boltzmann populations and adsorption energies are v1 "
        "non-goals; use the existing opt workflow for separately computed "
        "references"
    ),
    "gas_phase": (
        "gas-phase conformer generation is a v1 non-goal; bring conformers "
        "from CREST/xtb as cs_molecule frames"
    ),
    "molecular": (
        "molecular (non-periodic) tasks are a v1 non-goal; confsearch requires "
        "a 3D-periodic slab"
    ),
}


def reject_non_goal(name: str) -> ConfsearchNotImplementedError:
    """Return the fail-closed error for a named v1 non-goal."""
    detail = NON_GOALS.get(name, f"{name!r} is outside the confsearch v1 scope")
    return ConfsearchNotImplementedError(f"confsearch: {detail}")


__all__ = [
    "NON_GOALS",
    "ConfsearchError",
    "ConfsearchGradientConsistencyError",
    "ConfsearchInternalError",
    "ConfsearchNotImplementedError",
    "ConfsearchPreparationError",
    "reject_non_goal",
]
