"""Validated, calculator-independent trajectory analysis for mliport.

Optional backends such as kinisi and GEMDAT are intentionally not imported at
package import time.
"""

from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.validation import (
    AnalysisError,
    InvalidTrajectoryError,
    OptionalDependencyError,
    UnsupportedAnalysisError,
    ValidationReport,
    require_analysis,
    validate_trajectory,
)

__all__ = [
    "AnalysisError",
    "InvalidTrajectoryError",
    "OptionalDependencyError",
    "TrajectoryDataset",
    "UnsupportedAnalysisError",
    "ValidationReport",
    "require_analysis",
    "validate_trajectory",
]
