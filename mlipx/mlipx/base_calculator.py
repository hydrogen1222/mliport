# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).

"""
Abstract base class for MLIP calculator wrappers.

All Runners depend on this interface only, never on a concrete model
implementation. This is the dependency-inversion seam that lets mlipx support
UMA (FAIRChem), MACE, DPA (DeepMD-kit) and GRACE behind one ASE-compatible API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class BaseMLIPCalculator(ABC):
    """
    Abstract base class for all MLIP calculator wrappers.

    The Runners layer only depends on this contract; it never touches any
    concrete model implementation. Every wrapper returns a standard ASE
    ``Calculator`` from :meth:`get_calculator`, so the rest of the pipeline
    (single point, optimization, MD, batch, writers) is engine-agnostic.
    """

    @abstractmethod
    def get_calculator(self) -> Calculator:
        """
        Return a standard ASE Calculator instance.

        Implementations should lazily build and cache the calculator so the
        heavy backend import only happens on first use.

        Returns:
            An ``ase.calculators.calculator.Calculator`` subclass instance.
        """

    @property
    @abstractmethod
    def task(self) -> str:
        """
        Current task name.

        For UMA this is one of ``omat/omol/oc20/oc25/odac/omc``. For the
        generic engines (MACE/DPA/GRACE) it collapses to ``bulk`` (periodic)
        or ``molecule`` (non-periodic), which controls PBC setup. Model-specific
        MACE heads or DPA branches are selected separately with ``head``.
        """

    @property
    @abstractmethod
    def has_stress(self) -> bool:
        """Whether this model supports stress (virial) calculation."""

    @property
    def inference_mode(self) -> str:
        """
        Inference mode.

        Only UMA distinguishes ``default``/``turbo``. Other engines return
        ``default`` and ignore the value.
        """
        return "default"

    @abstractmethod
    def info(self) -> dict:
        """
        Return model metadata for writing OUTCAR/JSON output.

        Returns:
            Dictionary with at least ``model_type``, ``model_path``,
            ``device`` and ``implemented_properties`` keys.
        """

    @property
    def implemented_properties(self) -> list[str]:
        """
        List of physical quantities this model can compute.

        Defaults to reading ``implemented_properties`` off the underlying ASE
        calculator. Subclasses may override for cheaper access.
        """
        calc = self.get_calculator()
        return list(calc.implemented_properties)

    def capabilities(self, model_record: dict):
        """Return evidence-bound capabilities for this exact loaded model."""
        from mlipx.capabilities import resolve_capabilities

        return resolve_capabilities(model_record, self.implemented_properties)
