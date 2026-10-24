"""Pinning tests for the generalized gradient pre-flight (P0 refactor).

``run_gradient_preflight`` now accepts one or more structures and takes a
caller-supplied naming/error class, while NEB keeps its historical behaviour:
at least two images, the ``mliport.neb-gradient-preflight/1`` schema and the
exact NEB refusal wording.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mliport.neb.preflight import (
    NEB_PREFLIGHT_NAMING,
    PREFLIGHT_SCHEMA,
    NEBGradientConsistencyError,
    PreflightNaming,
    run_gradient_preflight,
)

R0 = 1.2


class ConservativePair(Calculator):
    """Harmonic pair potential whose forces are the exact -grad(E)."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = atoms.positions
        energy = 0.0
        forces = np.zeros_like(positions)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                displacement = distance - R0
                energy += 2.0 * displacement**2
                pair_force = -(4.0 * displacement) * delta / distance
                forces[i] += pair_force
                forces[j] -= pair_force
        self.results = {"energy": energy, "forces": forces}


class SwirlPair(ConservativePair):
    """Conservative pair forces plus a non-conservative rotational field."""

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = atoms.positions
        extra = np.zeros_like(positions)
        for index, position in enumerate(positions):
            extra[index] = 0.7 * np.array([-position[1], position[0], 0.0])
        self.results["forces"] = self.results["forces"] + extra


class ConfsearchPreflightError(ValueError):
    """Stand-in for the confsearch error class raised by its pre-flight."""


CONFSEARCH_NAMING = PreflightNaming(
    context="confsearch",
    item_label="Structure",
    item_label_lower="structure",
    subject="structure",
    schema="mliport.confsearch-gradient-preflight/1",
    sampled_key="sampled_structures",
    error_class=ConfsearchPreflightError,
    scope=(
        "spot check on the first prepared start and sampled directions; not a "
        "proof of consistency over the whole potential energy surface"
    ),
    consequence=(
        "confsearch ranks force-converged minima by energy, so the ranking "
        "would be meaningless. Use a conservative (autograd-force) model or "
        "precision."
    ),
)


def _structure(offset: float = 0.0) -> Atoms:
    return Atoms(
        "HHe",
        positions=[[0.0, 0.0, 0.0], [R0 + 0.05 + offset, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


def test_generic_preflight_accepts_a_single_structure():
    summary = run_gradient_preflight(
        ConservativePair(),
        [_structure()],
        dtype="float64",
        minimum_structures=1,
        naming=CONFSEARCH_NAMING,
    )
    assert summary["status"] == "passed"
    assert summary["sampled_structures"] == [0]
    assert summary["schema"] == "mliport.confsearch-gradient-preflight/1"
    assert summary["n_failed"] == 0


def test_default_preflight_still_requires_two_structures_for_neb():
    with pytest.raises(NEBGradientConsistencyError, match="needs a band"):
        run_gradient_preflight(ConservativePair(), [_structure()], dtype="float64")


def test_minimum_structures_must_be_positive():
    with pytest.raises(ValueError, match="minimum_structures"):
        run_gradient_preflight(
            ConservativePair(), [_structure()], dtype="float64", minimum_structures=0
        )


def test_neb_naming_defaults_are_the_historical_wording():
    naming = PreflightNaming()
    assert naming == NEB_PREFLIGHT_NAMING
    assert naming.context == "NEB"
    assert naming.item_label == "Image"
    assert naming.item_label_lower == "image"
    assert naming.subject == "band"
    assert naming.schema == PREFLIGHT_SCHEMA == "mliport.neb-gradient-preflight/1"
    assert naming.sampled_key == "sampled_images"
    assert naming.error_class is NEBGradientConsistencyError
    assert naming.scope == (
        "spot check on the sampled band structures and directions; not a "
        "proof of consistency over the whole potential energy surface"
    )
    assert naming.consequence == (
        "NEB optimizes with these forces, so the band and any barrier would be "
        "meaningless. Use a conservative (autograd-force) model or precision."
    )


def test_neb_refusal_message_and_summary_are_unchanged():
    with pytest.raises(NEBGradientConsistencyError) as excinfo:
        run_gradient_preflight(
            SwirlPair(), [_structure(), _structure(0.1)], dtype="float64"
        )
    message = str(excinfo.value)
    assert message.startswith("NEB refused: the model's forces are not verifiably")
    assert "worst: image " in message
    assert (
        "NEB optimizes with these forces, so the band and any barrier would be"
        in message
    )
    assert excinfo.value.summary["status"] == "failed"
    assert excinfo.value.summary["schema"] == PREFLIGHT_SCHEMA
    assert "sampled_images" in excinfo.value.summary


def test_confsearch_naming_raises_its_own_error_and_wording():
    with pytest.raises(ConfsearchPreflightError) as excinfo:
        run_gradient_preflight(
            SwirlPair(),
            [_structure()],
            dtype="float64",
            minimum_structures=1,
            naming=CONFSEARCH_NAMING,
        )
    message = str(excinfo.value)
    assert message.startswith("confsearch refused:")
    assert "NEB" not in message
    assert "confsearch ranks force-converged minima by energy" in message
    assert excinfo.value.summary["schema"] == "mliport.confsearch-gradient-preflight/1"
    assert excinfo.value.summary["sampled_structures"] == [0]


def test_non_finite_energy_uses_the_caller_wording():
    class NotANumber(ConservativePair):
        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = float("nan")

    with pytest.raises(
        ConfsearchPreflightError, match="Structure 0: the model returned a"
    ):
        run_gradient_preflight(
            NotANumber(),
            [_structure()],
            dtype="float64",
            minimum_structures=1,
            naming=CONFSEARCH_NAMING,
        )
