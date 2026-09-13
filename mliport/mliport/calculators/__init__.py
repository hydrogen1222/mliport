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

__all__ = [
    "BaseMLIPCalculator",
    "CalculatorFactory",
    "SUPPORTED_TYPES",
]
