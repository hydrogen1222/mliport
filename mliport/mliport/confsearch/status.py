"""Minimum/cycle status vocabulary shared by the confsearch core and engine.

The values are stable strings: they appear in ``search_results.json``,
``registry.json`` and ``minima/all_minima.extxyz``, so they are part of the
output contract (schema ``mliport.confsearch-results/1``).
"""

from __future__ import annotations

#: Classification of a converged quench (spec section 8.2).
VALID = "valid"
REACTED = "reacted"
DESORBED = "desorbed"
IMAGE_CONTACT = "image_contact"

#: Quench outcomes (spec section 7.4).
QUENCH_FAILED = "quench_failed"
QUENCH_NOT_CONVERGED = "quench_not_converged"

#: Start-stage outcomes (spec section 7.1).
FAILED_START = "failed_start"

#: MD segment outcomes (spec section 7.3 step 6/7).
MD_FORCE_ABORT = "md_force_abort"
MD_INTEGRITY_ABORT = "md_integrity_abort"
MD_NONFINITE_ABORT = "md_nonfinite_abort"

#: Statuses produced by the classification rules, in rule order.
CLASSIFICATION_STATUSES = (VALID, REACTED, DESORBED, IMAGE_CONTACT)

#: Every status the engine can write for a stored structure.
ALL_STATUSES = (
    *CLASSIFICATION_STATUSES,
    QUENCH_FAILED,
    QUENCH_NOT_CONVERGED,
    FAILED_START,
    MD_FORCE_ABORT,
    MD_INTEGRITY_ABORT,
    MD_NONFINITE_ABORT,
)

#: Statuses that are neither ranked nor chain states, ever.
NEVER_RANKED = frozenset(
    {
        QUENCH_FAILED,
        QUENCH_NOT_CONVERGED,
        FAILED_START,
        MD_FORCE_ABORT,
        MD_INTEGRITY_ABORT,
        MD_NONFINITE_ABORT,
    }
)


def rankable_statuses(reaction_policy: str) -> frozenset[str]:
    """Statuses that enter the ranking for a given ``cs_reaction_policy``."""
    if reaction_policy == "include":
        return frozenset({VALID, REACTED})
    return frozenset({VALID})


__all__ = [
    "ALL_STATUSES",
    "CLASSIFICATION_STATUSES",
    "DESORBED",
    "FAILED_START",
    "IMAGE_CONTACT",
    "MD_FORCE_ABORT",
    "MD_INTEGRITY_ABORT",
    "MD_NONFINITE_ABORT",
    "NEVER_RANKED",
    "QUENCH_FAILED",
    "QUENCH_NOT_CONVERGED",
    "REACTED",
    "VALID",
    "rankable_statuses",
]
