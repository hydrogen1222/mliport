"""Source locations for raw configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceLocation:
    """Where a raw configuration token was declared."""

    base_dir: Path
    path: Path | None = None
    line: int | None = None
    label: str = ""

    @classmethod
    def from_file(cls, path: str | Path, line: int | None = None) -> SourceLocation:
        """Create a location rooted at the declaring file's directory."""
        resolved = Path(path).expanduser().resolve()
        return cls(
            base_dir=resolved.parent,
            path=resolved,
            line=line,
            label=str(resolved),
        )

    @classmethod
    def synthetic(
        cls, label: str, base_dir: str | Path | None = None
    ) -> SourceLocation:
        """Create a location for an in-memory/CLI layer."""
        base = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
        return cls(base_dir=base.resolve(), label=label)

    @property
    def display(self) -> str:
        """Human-readable file/line or synthetic source label."""
        target = str(self.path) if self.path is not None else self.label
        if self.line is not None:
            return f"{target}:{self.line}"
        return target
