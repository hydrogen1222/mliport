from __future__ import annotations

import json

import pytest

from mliport.install import inventory
from mliport.install.plan import generate_plan


def test_every_resolver_stage_shares_backend_constraints():
    plan = generate_plan(None, ["uma", "mace", "dpa", "grace"], verify=False)
    for engine in ("uma", "mace", "dpa", "grace"):
        venv = ".venv" if engine == "uma" else f".venv-{engine}"
        pip = [
            s for s in plan.steps if s.stage == "pip" and f"{venv}/bin/python" in s.argv
        ]
        assert pip
        assert all(
            s.argv[s.argv.index("--constraint") + 1]
            == f"{venv}/mliport-constraints.txt"
            for s in pip
        )
        assert any(
            s.stage == "check" and f"{venv}/bin/python" in s.argv for s in plan.steps
        )
        assert any(
            s.stage == "inventory" and f"{venv}/bin/python" in s.argv
            for s in plan.steps
        )


def test_local_cuda_build_must_match():
    with pytest.raises(RuntimeError, match="violates"):
        inventory.validate_constraints(
            {"packages": {"torch": "2.8.0+cu128"}}, "torch==2.8.0+cu126"
        )


def test_inventory_rejects_changed_environment(tmp_path, monkeypatch):
    current = {
        "schema": "mliport.install-inventory/1",
        "python": "3.12.0",
        "packages": {"torch": "2.8.0+cpu"},
        "installed_record_sha256": {},
    }
    target = tmp_path / "inventory.json"
    target.write_text(json.dumps({**current, "constraints": "torch==2.8.0+cpu"}))
    monkeypatch.setattr(inventory, "snapshot", lambda: current)
    assert inventory.verify_inventory(target)
    current["packages"]["torch"] = "2.8.0+cu126"
    with pytest.raises(RuntimeError, match="differs"):
        inventory.verify_inventory(target)
