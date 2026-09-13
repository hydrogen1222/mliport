"""Final environment inventory; never mistake an installed RECORD for a wheel hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def snapshot() -> dict:
    packages = {}
    records = {}
    for dist in distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if name in packages and packages[name] != dist.version:
            raise RuntimeError(f"Conflicting installed distributions for {name}")
        packages[name] = dist.version
        record = dist.read_text("RECORD")
        if record is not None:
            records[name] = hashlib.sha256(record.encode()).hexdigest()
    return {
        "schema": "mliport.install-inventory/1",
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "installed_record_sha256": dict(sorted(records.items())),
        "wheel_hashes": {},
        "wheel_hash_status": "not_available_from_installed_metadata",
    }


def validate_constraints(inventory: dict, text: str) -> None:
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        actual = inventory["packages"].get(canonicalize_name(requirement.name))
        if actual is None or actual not in requirement.specifier:
            raise RuntimeError(f"Final inventory violates {line}: installed {actual}")


def verify_inventory(path: Path | None = None) -> bool:
    target = path or Path(sys.prefix) / "mliport-install-inventory.json"
    if not target.exists():
        return False
    recorded = json.loads(target.read_text(encoding="utf-8"))
    actual = snapshot()
    for key in ("schema", "python", "packages", "installed_record_sha256"):
        if recorded.get(key) != actual[key]:
            raise RuntimeError(
                f"Installed environment differs from saved inventory: {key}"
            )
    validate_constraints(actual, recorded["constraints"])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    target = Path(sys.prefix) / "mliport-install-inventory.json"
    if args.write:
        if args.constraints is None:
            parser.error("--write requires --constraints")
        inventory = snapshot()
        constraints = args.constraints.read_text(encoding="utf-8")
        validate_constraints(inventory, constraints)
        inventory["constraints"] = constraints
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    if not verify_inventory(target):
        raise RuntimeError("No installer inventory exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
