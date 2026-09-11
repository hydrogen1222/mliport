"""Generate the committed case manifests (deterministic fixtures).

The fixture identities (structure_id) are pinned in
``validation/science/cases/manifests/fixtures.json``; a CPU regression test
compares the live fixtures against it, so any accidental change to lattice
constants or displacement seeds breaks the test instead of silently
invalidating prior evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import fixtures  # noqa: E402

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "cases" / "manifests"


def build_fixtures_manifest() -> dict:
    bulk = {name: fixtures.build_fixture(name)[1] for name in fixtures.FIXTURES}
    extra = {
        name: fixtures.build_extra(name)[1] for name in fixtures.INVARIANCE_SYSTEMS
        if name not in fixtures.FIXTURES
    }
    return {
        "schema": "mlipx.beta-case-fixtures/1",
        "suite_revision": common.BETA_VALIDATION_SUITE_REVISION,
        "bulk_fixtures": bulk,
        "extra_fixtures": extra,
        "invariance_systems": list(fixtures.INVARIANCE_SYSTEMS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest matches the live fixtures",
    )
    args = parser.parse_args()

    manifest = build_fixtures_manifest()
    target = MANIFEST_DIR / "fixtures.json"
    if args.check:
        committed = json.loads(target.read_text(encoding="utf-8"))
        if committed != manifest:
            print(
                "fixture manifest drift: regenerate via build_cases.py",
                file=sys.stderr,
            )
            return 1
        print(f"[cases] fixtures manifest up to date: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[cases] wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
