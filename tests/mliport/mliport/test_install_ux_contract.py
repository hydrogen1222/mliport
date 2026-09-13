"""PR3 acceptance tests: installer UX contract (DOC-01/02/03, section 7.2).

The compatibility registry is the only source for environment names, Python
constraints and runtime pins; the docs are generated from it, the README does
not hand-copy them, the installer fails closed on backend-incompatible Python
versions, and the launchers only select the Python runtime.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _backends():
    from mliport.install.compatibility import BACKENDS

    return BACKENDS


# --------------------------------------------------------------- DOC-01
def test_installation_doc_is_generated_from_the_registry():
    from mliport.install.docs import render_installation_markdown

    path = REPO / "docs" / "installation.md"
    assert path.is_file(), "docs/installation.md must exist"
    assert path.read_text(encoding="utf-8") == render_installation_markdown()
    committed = path.read_text(encoding="utf-8")
    for engine, backend in _backends().items():
        assert backend.venv_name in committed, engine
        assert backend.requirement in committed, engine


def test_environment_matrix_is_backend_specific():
    from mliport.install.docs import backend_environment_rows

    rows = {row["engine"]: row for row in backend_environment_rows()}
    assert rows["uma"]["environment"] == ".venv"
    assert rows["uma"]["python"] == "3.11, 3.12"
    assert rows["mace"]["environment"] == ".venv-mace"
    assert rows["dpa"]["environment"] == ".venv-dpa"
    assert rows["grace"]["environment"] == ".venv-grace"
    for engine in ("mace", "dpa", "grace"):
        assert rows[engine]["python"] == "3.10, 3.11, 3.12", engine


# --------------------------------------------------------------- DOC-02
@pytest.mark.parametrize("readme", ["README.md", "README_CN.md"])
def test_readme_states_the_real_installation_model(readme):
    text = (REPO / readme).read_text(encoding="utf-8")
    # the old, wrong claims must be gone
    assert ".venv-uma" not in text
    assert "pip install ./mliport" not in text
    # per-backend runtime invocation is documented
    assert "docs/installation.md" in text
    assert ".venv-mace/bin/mliport" in text
    assert ".venv/bin/mliport" in text
    assert "Scripts" in text and "mliport.exe" in text
    # launchers are advertised
    assert "bin/mliport-mace" in text


# --------------------------------------------------------------- DOC-03
def test_build_plan_fails_closed_for_backend_incompatible_python():
    from mliport.install.plan import InstallPlanError, generate_plan

    with pytest.raises(InstallPlanError, match="requires-python"):
        generate_plan(
            gpus=None,
            engines=["uma"],
            source="auto",
            python_version="3.10",
            device="cpu",
            clean=False,
            verify=False,
        )
    with pytest.raises(InstallPlanError, match="requires-python"):
        generate_plan(
            gpus=None,
            engines=["mace", "uma"],
            source="auto",
            python_version="3.10",
            device="cpu",
            clean=False,
            verify=False,
        )
    # backends that do support 3.10 still plan normally
    plan = generate_plan(
        gpus=None,
        engines=["mace", "dpa", "grace"],
        source="auto",
        python_version="3.10",
        device="cpu",
        clean=False,
        verify=False,
    )
    assert plan.python_version == "3.10"


def test_installer_help_mentions_backend_constraints():
    from mliport.install.cli import _parser

    help_text = _parser().format_help()
    assert "requires-python" in help_text
    assert "UMA" in help_text or "uma" in help_text


# ------------------------------------------------------------- launchers
def test_posix_launcher_only_selects_the_runtime(tmp_path):
    from mliport.install.compatibility import BACKENDS
    from mliport.install.launcher import create_launcher

    backend = BACKENDS["mace"]
    path = create_launcher(backend, tmp_path)
    assert path == tmp_path / "bin" / "mliport-mace"
    assert path.stat().st_mode & stat.S_IXUSR
    script = path.read_text(encoding="utf-8")
    assert f"{backend.venv_name}/bin/mliport" in script
    # the wrapper may not set/export the visibility variable or inject options
    assert "CUDA_VISIBLE_DEVICES=" not in script
    assert "export CUDA_VISIBLE_DEVICES" not in script
    assert "--model-type" not in script
    assert "--dtype" not in script

    stub = tmp_path / backend.venv_name / "bin" / "mliport"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        "#!/bin/sh\n" 'echo "CUDA=$CUDA_VISIBLE_DEVICES"\n' 'echo "ARGS=$*"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [str(path), "sp", "s.cif", "--model", "m.model", "--model-type", "mace"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-keep-me"},
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA=GPU-keep-me" in result.stdout
    assert "ARGS=sp s.cif --model m.model --model-type mace" in result.stdout


def test_windows_launcher_uses_the_scripts_entry_point():
    from mliport.install.compatibility import BACKENDS
    from mliport.install.launcher import windows_launcher_script

    script = windows_launcher_script(BACKENDS["uma"])
    assert "Scripts\\mliport.exe" in script
    assert "CUDA_VISIBLE_DEVICES=" not in script
    assert "--dtype" not in script
