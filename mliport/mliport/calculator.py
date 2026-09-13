# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project.

"""Deprecated compatibility module.

``UMACalculator`` now lives in :mod:`mliport.calculators.uma`, next to the
MACE/DPA/GRACE wrappers, so no backend is privileged.  New code should use
``CalculatorFactory`` (backend selected by ``model_type``) or import from
``mliport.calculators``; this module only exists so pre-rename/pre-move
imports keep working.
"""

from __future__ import annotations

from mliport.calculators.uma import UMACalculator

__all__ = ["UMACalculator"]
