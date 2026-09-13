"""Build/install integration regression for packaged capability evidence.

RC-01's most important test: the curated capability registry must reach an
*installed wheel*, not just the source tree. This test builds the mliport wheel,
installs it into a clean venv outside the repository, and resolves capabilities
there without any ``evidence_dir`` — then proves the resolver stays fail-closed
when a single identity field is changed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mliport.capabilities import resolve_capabilities

pytestmark = pytest.mark.wheel_integration


def _assert_hermetic_wheel(wheel: Path) -> None:
    """The wheel ships the package + curated evidence and nothing else."""
    import zipfile

    names = zipfile.ZipFile(wheel).namelist()
    leaked = [
        name
        for name in names
        if name.startswith(("build/", "tests/", "validation/", "templates/"))
        or ".egg-info" in name
    ]
    assert not leaked, f"wheel contains repository artifacts: {leaked[:5]}"
    # The data directory must be a real package so that
    # importlib.resources.files("mliport.data") has Package semantics on
    # Python 3.10/3.11 (plain-module anchors are a 3.12 addition).
    assert "mliport/data/__init__.py" in names
    packaged_evidence = [
        name for name in names if name.startswith("mliport/data/validation/")
    ]
    assert sorted(packaged_evidence) == [
        "mliport/data/validation/dpa-v100.json",
        "mliport/data/validation/grace-v100.json",
        "mliport/data/validation/mace-float32-v100.json",
        "mliport/data/validation/mace-v100.json",
    ]


def _build_wheel(out_dir: Path) -> Path:
    source_dir = Path(__file__).resolve().parents[3] / "mliport"
    # Build from a sanitized copy so the build never touches (or reads a stale
    # build/ cache of) the repository checkout.
    build_dir = out_dir / "src"
    shutil.copytree(
        source_dir,
        build_dir,
        ignore=shutil.ignore_patterns(
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            ".venv",
        ),
    )
    if shutil.which("uv") is not None:
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(out_dir),
                str(build_dir),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    else:
        try:
            import pip  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("neither uv nor pip is available to build a wheel")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "-w",
                str(out_dir),
                str(build_dir),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    wheels = sorted(out_dir.glob("mliport-*.whl"))
    assert wheels, "wheel build produced no mliport wheel"
    _assert_hermetic_wheel(wheels[-1])
    return wheels[-1]


def _create_clean_venv(env_dir: Path, wheel: Path, python_version: str) -> Path:
    """Install the wheel into a venv pinned to an explicit Python version.

    ``uv venv`` without ``--python`` would silently reuse whatever interpreter
    runs the test suite, so the packaged-evidence contract would only ever be
    proven for the CI runner's Python. Pinning 3.10/3.11/3.12 keeps the
    ``importlib.resources`` package-anchor contract under test on every
    supported version.
    """
    python = env_dir / "bin" / "python"
    if shutil.which("uv") is not None:
        subprocess.run(
            ["uv", "venv", "--quiet", "--python", python_version, str(env_dir)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--quiet",
                "--python",
                str(python),
                "--no-deps",
                str(wheel),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    else:  # pragma: no cover - CI always has uv; local fallback
        if python_version != f"{sys.version_info.major}.{sys.version_info.minor}":
            pytest.skip("no uv available: only the running interpreter can be venv'd")
        import venv

        venv.create(str(env_dir), with_pip=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", "-q", str(wheel)],
            check=True,
            capture_output=True,
            timeout=600,
        )
    assert python.exists(), "clean venv was not created"
    return python


def _packaged_registry_identities() -> list[dict]:
    from mliport.capabilities import _package_evidence_dir

    identities = []
    for entry in _package_evidence_dir().iterdir():
        if entry.name.endswith(".json"):
            record = json.loads(entry.read_text(encoding="utf-8"))
            identities.append(record["model_identity"])
    assert identities, "packaged curated registry is empty"
    return identities


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build the wheel once; every pinned venv reuses the same artifact."""
    tmp = tmp_path_factory.mktemp("wheel-integration")
    return _build_wheel(tmp / "dist")


@pytest.fixture(scope="module", params=["3.10", "3.11", "3.12"])
def installed_venv(request, tmp_path_factory, built_wheel):
    """Install the wheel into a clean venv pinned to each supported Python."""
    tmp = tmp_path_factory.mktemp(f"wheel-integration-{request.param}")
    return _create_clean_venv(tmp / "clean-venv", built_wheel, request.param)


def _run_in_venv(python: Path, snippet: str) -> object:
    env = dict(os.environ)
    # Nothing from the repository may leak into the clean venv resolution.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [str(python), "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return json.loads(proc.stdout)


def test_installed_wheel_resolves_curated_evidence_without_evidence_dir(
    installed_venv,
):
    snippet = (
        "import json\n"
        "from mliport.capabilities import _package_evidence_dir, resolve_capabilities\n"
        "out = []\n"
        "for entry in sorted(_package_evidence_dir().iterdir()):\n"
        "    if not entry.name.endswith('.json'):\n"
        "        continue\n"
        "    record = json.loads(entry.read_text(encoding='utf-8'))\n"
        "    cap = resolve_capabilities(record['model_identity'], ['energy', 'forces'])\n"
        "    out.append({'evidence_id': record['evidence_id'],\n"
        "                'state': cap.energy_gradient_consistency,\n"
        "                'resolved_id': cap.validation_evidence_id})\n"
        "print(json.dumps(out))\n"
    )
    results = _run_in_venv(installed_venv, snippet)
    assert results, "installed wheel saw no packaged evidence"
    for result in results:
        assert result["state"] == "validated", result
        assert result["resolved_id"] == result["evidence_id"], result


def test_installed_wheel_stays_fail_closed_on_identity_mismatch(installed_venv):
    snippet = (
        "import json\n"
        "from mliport.capabilities import _package_evidence_dir, resolve_capabilities\n"
        "record = next(\n"
        "    json.loads(e.read_text(encoding='utf-8'))\n"
        "    for e in sorted(_package_evidence_dir().iterdir())\n"
        "    if e.name.endswith('.json')\n"
        ")\n"
        "identity = dict(record['model_identity'])\n"
        "identity['model_sha256'] = '0' * 64\n"
        "cap = resolve_capabilities(identity, ['energy', 'forces'])\n"
        "print(json.dumps({'state': cap.energy_gradient_consistency,\n"
        "                  'id': cap.validation_evidence_id}))\n"
    )
    result = _run_in_venv(installed_venv, snippet)
    assert result == {"state": "unknown", "id": None}


def test_source_and_installed_wheel_agree():
    """The editable/source resolution must equal what the wheel resolves."""
    for identity in _packaged_registry_identities():
        capability = resolve_capabilities(identity, ["energy", "forces"])
        assert capability.energy_gradient_consistency == "validated"
