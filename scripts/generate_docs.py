"""Regenerate machine-generated documentation from repository code.

Currently generates ``docs/installation.md`` (backend/environment/Python
matrix from the compatibility registry).  The docs-contract tests compare the
committed files against these renderers, so never edit a generated block by
hand.

Usage::

    python scripts/generate_docs.py [--check]

``--check`` exits non-zero when a generated file is out of date instead of
rewriting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mliport"))

from mliport.install.docs import render_installation_markdown  # noqa: E402

GENERATED_FILES = {
    REPO / "docs" / "installation.md": render_installation_markdown,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when a generated file is stale.",
    )
    args = parser.parse_args()
    failures = 0
    for path, render in GENERATED_FILES.items():
        content = render()
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == content:
            print(f"[docs] up to date: {path.relative_to(REPO)}")
            continue
        if args.check:
            print(
                f"[docs] STALE: {path.relative_to(REPO)} "
                "(run python scripts/generate_docs.py)",
                file=sys.stderr,
            )
            failures += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[docs] wrote {path.relative_to(REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
