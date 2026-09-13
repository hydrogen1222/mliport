"""PR-J acceptance tests: interface capability matrix (section 12).

The matrix is the single machine-readable statement of which entry point can
express which feature.  It must not be an equivalence claim, and features the
INCAR key set cannot express must say so explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MATRIX = REPO / "validation" / "science" / "capability_matrix.json"
ALLOWED = {"yes", "no", "partial", "not_applicable"}


def _matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def _feature(matrix, name):
    return next(entry for entry in matrix["features"] if entry["feature"] == name)


def test_capability_matrix_shape_and_notes():
    matrix = _matrix()
    assert matrix["schema"] == "mliport.capability-matrix/1"
    features = matrix["features"]
    assert features
    for entry in features:
        assert set(entry) >= {"feature", "api", "direct_cli", "incar", "tui"}
        for interface in ("api", "direct_cli", "incar", "tui"):
            assert entry[interface] in ALLOWED, (entry["feature"], interface)
        if any(entry[key] != "yes" for key in ("api", "direct_cli", "incar", "tui")):
            assert entry.get(
                "note"
            ), f"{entry['feature']} needs a note for its non-yes entries"
    names = [entry["feature"] for entry in features]
    assert len(names) == len(set(names))


def test_advanced_neb_control_limitation_is_explicit():
    matrix = _matrix()
    for name in ("neb_atom_map", "neb_image_shifts"):
        entry = _feature(matrix, name)
        assert entry["api"] == "yes"
        assert entry["direct_cli"] == "yes"
        assert entry["incar"] == "no"
        assert "INCAR" in entry["note"]
    joined = " ".join(matrix["limitations"])
    assert "Advanced explicit atom mapping/image-shift control" in joined
    assert "not an equivalence claim" in joined


def test_matrix_matches_the_incar_schema_and_cli():
    from mliport.config.schema import get_schema

    incar_keys = set(get_schema().canonical_names())
    assert "neb_initial" in incar_keys and "neb_final" in incar_keys
    assert "neb_atom_map" not in incar_keys
    assert "neb_image_shifts" not in incar_keys

    cli = (REPO / "mliport" / "mliport" / "cli.py").read_text(encoding="utf-8")
    assert '"--atom-map"' in cli
    assert '"--image-shifts"' in cli
    assert '"--resume"' in cli

    prepare = (REPO / "mliport" / "mliport" / "neb" / "prepare.py").read_text(
        encoding="utf-8"
    )
    assert "atom_map" in prepare and "image_shifts" in prepare


def test_readmes_state_the_incar_limitation():
    for name in ("README.md", "README_CN.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "capability_matrix.json" in text, name
    normalized = " ".join((REPO / "README.md").read_text(encoding="utf-8").split())
    assert (
        "Advanced explicit atom mapping/image-shift control is available "
        "through API/direct CLI/TUI only"
    ) in normalized
