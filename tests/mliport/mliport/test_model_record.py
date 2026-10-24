"""Pinning tests for the shared model-identity record (P0 refactor).

``build_model_record`` is the extracted NEB ``_model_record`` plus the
``dispersion`` field required by the confsearch comparability key.  NEB must
keep raising :class:`NEBPreparationError` with the same messages, and its
resume fingerprint must not change because a new provenance field appeared.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mliport.config import resolve_config
from mliport.model_record import (
    build_model_record,
    distribution_version,
    framework_distributions,
    hash_model_path,
)
from mliport.neb.io import canonical_fingerprint
from mliport.neb.prepare import BandInput
from mliport.neb.schema import NEBOptions, NEBPreparationError
from mliport.neb.workflow import _fingerprint, _model_record


class ModelWrapperStub:
    """Minimal BaseMLIPCalculator-like wrapper with a configurable info()."""

    task = "bulk"

    def __init__(self, info: dict, inference_mode: str = "default") -> None:
        self._info = dict(info)
        self.inference_mode = inference_mode

    def info(self) -> dict:
        return dict(self._info)


def _resolved(tmp_path, *, model_type: str = "mace", task: str = "bulk", **cli):
    model = tmp_path / "model.pt"
    model.write_bytes(b"identity model")
    overrides = {
        "model_type": model_type,
        "model_path": str(model),
        "task": task,
        **cli,
    }
    return resolve_config(calc_type="opt", cli=overrides)


def _wrapper(tmp_path, **overrides) -> ModelWrapperStub:
    info = {
        "model_type": "mace",
        "task": "bulk",
        "backend_version": "99.1-test",
        "framework_versions": {"torch": "99.2-test"},
        "default_dtype": "float64",
        "active_head": "default",
        "model_path": str((tmp_path / "model.pt").resolve()),
    }
    info.update(overrides)
    return ModelWrapperStub(info)


def test_build_model_record_keeps_the_historical_keys_and_adds_dispersion(tmp_path):
    record = build_model_record(
        _wrapper(tmp_path), _resolved(tmp_path), error_class=ValueError
    )
    expected = {
        "model_type",
        "actual_device_type",
        "actual_device_logical_index",
        "actual_device_uuid",
        "inference_mode",
        "compile_enabled",
        "direct_forces",
        "model_path",
        "model_sha256",
        "model_alias",
        "model_source",
        "backend_distribution",
        "backend_version",
        "framework_versions",
        "task",
        "head_requested",
        "head_effective",
        "dtype_requested",
        "dtype_effective",
        "backend_reported_device",
    }
    assert expected <= set(record)
    assert "dispersion" in record
    assert record["dispersion"] is None
    assert record["model_type"] == "mace"
    assert record["backend_distribution"] == "mace-torch"
    assert record["model_sha256"] == hash_model_path(
        tmp_path / "model.pt", error_class=ValueError
    )


def test_build_model_record_records_the_dispersion_correction(tmp_path):
    correction = {"xc": "pbe", "damping": "bj", "version": "6.2.0"}
    record = build_model_record(
        _wrapper(tmp_path, dispersion=correction),
        _resolved(tmp_path),
        error_class=ValueError,
    )
    assert record["dispersion"] == correction


def test_build_model_record_uses_the_caller_error_class_with_neb_messages(tmp_path):
    wrapper = _wrapper(tmp_path, model_type="uma")
    with pytest.raises(ValueError, match="requires 'mace'"):
        build_model_record(wrapper, _resolved(tmp_path), error_class=ValueError)


def test_neb_model_record_still_raises_neb_preparation_error(tmp_path):
    wrapper = _wrapper(tmp_path, task="omol")
    with pytest.raises(NEBPreparationError, match="requires 'bulk'"):
        _model_record(wrapper, _resolved(tmp_path))


def test_neb_model_record_includes_dispersion(tmp_path):
    record = _model_record(_wrapper(tmp_path), _resolved(tmp_path))
    assert record["dispersion"] is None


def test_neb_resume_fingerprint_is_unaffected_by_the_new_field(tmp_path):
    """NEB's fingerprint selects identity keys explicitly, so dispersion is inert."""
    band = BandInput(
        images=(
            Atoms(
                "HHe",
                positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                cell=[6.0, 6.0, 6.0],
                pbc=True,
            ),
            Atoms(
                "HHe",
                positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                cell=[6.0, 6.0, 6.0],
                pbc=True,
            ),
            Atoms(
                "HHe",
                positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                cell=[6.0, 6.0, 6.0],
                pbc=True,
            ),
        ),
        atom_map=np.array([0, 1]),
        image_shifts=np.zeros((2, 3), dtype=int),
        path_convention="mic",
        source="unit-test",
    )
    options = NEBOptions(n_intermediate_images=1, interpolation="linear")
    base = build_model_record(
        _wrapper(tmp_path), _resolved(tmp_path), error_class=ValueError
    )
    with_dispersion = dict(base, dispersion={"xc": "pbe", "damping": "bj"})

    fingerprint_a, sha_a = _fingerprint(band, options, base)
    fingerprint_b, sha_b = _fingerprint(band, options, with_dispersion)

    assert sha_a == sha_b
    assert fingerprint_a["model"] == fingerprint_b["model"]
    assert "dispersion" not in fingerprint_a["model"]


def test_hash_model_path_is_content_addressed_and_fails_closed(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "a.bin").write_bytes(b"alpha")
    first = hash_model_path(model_dir, error_class=ValueError)
    (model_dir / "a.bin").write_bytes(b"beta")
    assert hash_model_path(model_dir, error_class=ValueError) != first

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        hash_model_path(empty, error_class=ValueError)

    with pytest.raises(FileNotFoundError, match="Model path not found"):
        hash_model_path(tmp_path / "missing.pt", error_class=ValueError)


def test_framework_and_distribution_helpers():
    assert framework_distributions("mace", "model.pt") == ("torch",)
    assert framework_distributions("dpa", "model.pb") == ("tensorflow",)
    assert framework_distributions("dpa", "model.pt") == ("torch",)
    assert framework_distributions("grace", "model") == ("tensorflow",)
    assert distribution_version("definitely-not-a-distribution") is None
    assert isinstance(distribution_version("numpy"), str)


def test_canonical_fingerprint_round_trips_through_json():
    normalized, digest = canonical_fingerprint({"b": [1, 2], "a": {"c": None}})
    assert normalized == {"a": {"c": None}, "b": [1, 2]}
    assert len(digest) == 64
