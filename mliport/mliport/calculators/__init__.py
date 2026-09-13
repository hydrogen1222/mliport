# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: multi-engine calculator wrappers.

"""
MLIP engine wrappers and the calculator factory.

Each wrapper adapts a backend (UMA/MACE/DPA/GRACE) ASE Calculator to the
``BaseMLIPCalculator`` contract. The factory selects one from a model type.
"""

from __future__ import annotations

from mliport.base_calculator import BaseMLIPCalculator
from mliport.calculators.factory import CalculatorFactory, SUPPORTED_TYPES

# All four backend wrappers are exported symmetrically; they are loaded
# lazily so importing this package never imports a backend framework.
_BACKEND_EXPORTS = {
    "UMACalculator": ".uma",
    "MACECalculatorWrapper": ".mace_calc",
    "DPACalculatorWrapper": ".dpa_calc",
    "GRACECalculatorWrapper": ".grace_calc",
}


def __getattr__(name: str):
    module = _BACKEND_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib  # noqa: PLC0415

    return getattr(importlib.import_module(module, __package__), name)


__all__ = [
    "BaseMLIPCalculator",
    "CalculatorFactory",
    "DPACalculatorWrapper",
    "GRACECalculatorWrapper",
    "MACECalculatorWrapper",
    "SUPPORTED_TYPES",
    "UMACalculator",
]
