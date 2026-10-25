"""Resume identity, run.log and run_context contracts (spec 11.1/11.2 revision 5).

Review items on PR #65:

* the resume fingerprint must use the model-identity subset, so resuming on a
  different GPU or with a moved checkpoint of identical content is allowed,
  while a different model/dtype/dispersion is refused;
* ``run.log`` is the live append-only ``LiveRunLogger`` (resume appends);
* ``run_context.json`` is created once and updated in place (``resumes[]``);
* ``resolved_config.json`` is written for new runs only.
"""

from __future__ import annotations

import json

import pytest

from mliport.confsearch.checkpoint import (
    DISPERSION_IDENTITY_KEYS,
    MODEL_IDENTITY_KEYS,
    build_resume_fingerprint,
    compare_fingerprints,
    model_identity,
)

BASE_MODEL = {
    "model_type": "mace",
    "model_sha256": "a" * 64,
    "backend_distribution": "mace-torch",
    "backend_version": "0.3.10",
    "framework_versions": {"torch": "2.4.0"},
    "task": "bulk",
    "head_effective": None,
    "head_requested": None,
    "dtype_effective": "float64",
    "dtype_requested": "float64",
    "inference_mode": "default",
    "compile_enabled": False,
    "direct_forces": None,
    "model_path": "/models/a.model",
    "model_alias": None,
    "model_source": "cli --model",
    "actual_device_type": "cpu",
    "actual_device_logical_index": 0,
    "actual_device_uuid": None,
    "backend_reported_device": "cpu",
    "dispersion": None,
}


def _fingerprint(model: dict) -> str:
    _payload, sha = build_resume_fingerprint(
        options=_options(),
        model=model,
        input_hashes={"cs_starts": {"start.extxyz": "abc"}},
        regions={"adsorbate": [16, 17, 18], "fixed": [0, 1]},
        seed=7,
        mliport_version="2.0.0b4",
    )
    return sha


def _options():
    from tests.mliport.mliport.confsearch.structures import mode_a_options

    return mode_a_options()


def test_model_identity_is_the_spec_subset():
    identity = model_identity(BASE_MODEL)
    for key in MODEL_IDENTITY_KEYS:
        assert key in identity
    for excluded in (
        "model_path",
        "model_alias",
        "model_source",
        "actual_device_type",
        "actual_device_logical_index",
        "actual_device_uuid",
        "backend_reported_device",
        "backend_distribution",
        "head_requested",
        "dtype_requested",
    ):
        assert excluded not in identity
    assert identity["dispersion"] is None


def test_dispersion_identity_keeps_only_the_physics():
    model = dict(
        BASE_MODEL,
        dispersion={
            "method": "d3",
            "xc": "pbe",
            "damping": "bj",
            "implementation": "torch-dftd",
            "version": "1.2.3",
            "device": "cuda:0",
        },
    )
    dispersion = model_identity(model)["dispersion"]
    assert set(dispersion) == set(DISPERSION_IDENTITY_KEYS)
    assert dispersion["xc"] == "pbe"
    assert "device" not in dispersion


def test_resume_fingerprint_ignores_where_the_model_ran_or_was_stored():
    moved = dict(
        BASE_MODEL,
        model_path="/elsewhere/copy-of-a.model",
        model_alias="mace_alias",
        model_source="settings.ini",
        actual_device_type="cuda",
        actual_device_logical_index=3,
        actual_device_uuid="GPU-1234",
        backend_reported_device="cuda:3",
        dispersion={
            "method": "d3",
            "xc": "pbe",
            "damping": "bj",
            "implementation": "torch-dftd",
            "version": "1.2.3",
            "device": "cuda:3",
        },
    )
    reference = dict(
        BASE_MODEL,
        dispersion={
            "method": "d3",
            "xc": "pbe",
            "damping": "bj",
            "implementation": "torch-dftd",
            "version": "1.2.3",
            "device": "cpu",
        },
    )
    assert _fingerprint(reference) == _fingerprint(moved)


@pytest.mark.parametrize(
    "change",
    [
        {"model_sha256": "b" * 64},
        {"dtype_effective": "float32"},
        {"head_effective": "other"},
        {"inference_mode": "turbo"},
        {"task": "omol"},
        {"framework_versions": {"torch": "2.5.0"}},
        {"backend_version": "0.3.11"},
        {"direct_forces": True},
    ],
)
def test_resume_fingerprint_refuses_a_different_model(change):
    changed = dict(BASE_MODEL, **change)
    assert _fingerprint(changed) != _fingerprint(BASE_MODEL)


def test_resume_fingerprint_refuses_a_different_dispersion():
    reference = dict(
        BASE_MODEL,
        dispersion={
            "method": "d3",
            "xc": "pbe",
            "damping": "bj",
            "implementation": "torch-dftd",
            "version": "1.2.3",
            "device": "cpu",
        },
    )
    other_xc = dict(reference, dispersion=dict(reference["dispersion"], xc="rpbe"))
    assert _fingerprint(other_xc) != _fingerprint(reference)


def test_compare_fingerprints_reports_the_differing_keys():
    first, _sha = build_resume_fingerprint(
        options=_options(),
        model=BASE_MODEL,
        input_hashes={},
        regions={},
        seed=1,
        mliport_version="2.0.0b4",
    )
    second, _sha = build_resume_fingerprint(
        options=_options(),
        model=dict(BASE_MODEL, model_sha256="c" * 64),
        input_hashes={},
        regions={},
        seed=1,
        mliport_version="2.0.0b4",
    )
    differences = compare_fingerprints(first, second)
    assert any("model.model_sha256" in item for item in differences)


class _SyntheticWrapper:
    """Backend-shaped wrapper around a synthetic calculator (no MLIP needed)."""

    task = "bulk"

    def __init__(self, calculator, model_path) -> None:
        self._calculator = calculator
        self._model_path = model_path

    def info(self) -> dict:
        return {
            "model_type": "mace",
            "task": "bulk",
            "backend_version": "test",
            "framework_versions": {"torch": "test"},
            "default_dtype": "float64",
            "model_path": str(self._model_path),
        }

    def get_calculator(self):
        return self._calculator


def _workflow_setup(tmp_path, *, cycles: int = 1):
    from ase.io import write

    from mliport.config import resolve_config
    from tests.mliport.mliport.confsearch.calculators import EggCrateCalculator
    from tests.mliport.mliport.confsearch.test_engine import _start

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"identity model")
    start_file = tmp_path / "start.extxyz"
    write(start_file, _start((0.5, 0.5)), format="extxyz")
    resolved = resolve_config(
        calc_type="confsearch",
        cli={
            "model_type": "mace",
            "model_path": str(model_file),
            "task": "bulk",
            "cs_starts": str(start_file),
            "cs_adsorbate_indices": "16-18",
            "cs_cycles": cycles,
            "cs_hold_ps": 0.05,
            "cs_cool_ps": 0.05,
            "cs_snapshot_interval_ps": 0.02,
            "timestep": 0.5,
            "cs_temperature_high": 800.0,
            "cs_temperature_low": 300.0,
            "cs_quench_fmax": 0.01,
            "cs_quench_steps": 2000,
            "cs_refine": False,
            "cs_refine_fmax": 0.005,
            "seed": 7,
        },
    )
    return resolved, _SyntheticWrapper(EggCrateCalculator(), model_file)


def test_run_log_is_append_only_and_run_context_is_updated(tmp_path):
    from mliport.confsearch.workflow import run_confsearch_workflow

    resolved, wrapper = _workflow_setup(tmp_path, cycles=1)
    output = tmp_path / "run"
    run_confsearch_workflow(wrapper, resolved, output_dir=output)

    log_first = (output / "run.log").read_text()
    resolved_config_first = (output / "resolved_config.json").read_text()
    context_first = json.loads((output / "run_context.json").read_text())
    assert log_first.strip()
    assert "gradient pre-flight" in log_first or "start" in log_first
    assert context_first["resumes"] == []
    assert context_first["finished_at"]

    run_confsearch_workflow(wrapper, resolved, output_dir=output, resume=True)

    log_second = (output / "run.log").read_text()
    assert log_second.startswith(log_first)
    assert len(log_second) > len(log_first)
    # A resume never rewrites resolved_config.json.
    assert (output / "resolved_config.json").read_text() == resolved_config_first
    context_second = json.loads((output / "run_context.json").read_text())
    assert len(context_second["resumes"]) == 1
    assert context_second["resumes"][0]["seed"] == context_first["seed"]
    # The recorded identity, starts and options are kept, not rebuilt.
    for key in ("model", "starts", "regions", "options", "seed"):
        assert context_second[key] == context_first[key]
