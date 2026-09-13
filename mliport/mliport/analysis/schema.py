"""Small, explicit schema for reproducible Analysis v2 requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass(slots=True)
class AnalysisRequest:
    task: str
    source: str
    parameters: dict[str, Any] = field(default_factory=dict)
    force: bool = False

    VALID_TASKS = frozenset(
        {
            "validate",
            "thermo",
            "rdf",
            "rmsd",
            "msd",
            "transport",
            "density",
            "arrhenius",
            "electrolyte",
            "vacf",
            "spectrum",
        }
    )

    def __post_init__(self) -> None:
        self.task = self.task.lower()
        if self.task not in self.VALID_TASKS:
            raise ValueError(
                f"Unknown analysis task {self.task!r}; choose from "
                + ", ".join(sorted(self.VALID_TASKS))
            )
        if not self.source:
            raise ValueError("Analysis source path is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def source_path(self) -> Path:
        return Path(self.source).expanduser().resolve()
