"""DPA has two environments; the runtime switch is explicit and fail-closed.

#61 DPA-SSE spec §5: ``DPA_RUNTIME=legacy|current`` (default ``current``), no
automatic fallback in either direction, the legacy environment only ever loads
the trust-gated conversion, and only DPA-2 checkpoints are accepted there.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import mliport.calculators.dpa_calc as dpa_calc
from mliport.calculators.dpa_calc import (
    DPA_RUNTIMES,
    DPACalculatorWrapper,
    dpa_runtime_problems,
)


@pytest.fixture(autouse=True)
def _isolated_process(monkeypatch):
    """The CLI restarts itself with CUDA_VISIBLE_DEVICES; tests run in-process."""
    monkeypatch.setattr(DPACalculatorWrapper, "_apply_device_env", lambda self: None)


def _wrapper(tmp_path, **kwargs):
    model = tmp_path / "model.pt"
    model.write_bytes(b"stub")
    return DPACalculatorWrapper(model, device="cpu", task="bulk", **kwargs)


# --------------------------------------------------------------- vocabulary
def test_runtime_vocabulary_is_the_two_environments():
    assert DPA_RUNTIMES == ("current", "legacy")


def test_default_runtime_is_current(tmp_path):
    assert _wrapper(tmp_path)._dpa_runtime == "current"


def test_invalid_runtime_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="Invalid DPA runtime"):
        _wrapper(tmp_path, dpa_runtime="auto")


# ------------------------------------------------------- environment matching
def test_current_runtime_matches_the_pinned_environment():
    assert dpa_runtime_problems("current", "3.2.1.dev63+g0edeaef56") == []


def test_legacy_runtime_matches_only_the_pinned_release():
    assert dpa_runtime_problems("legacy", "3.0.0b3") == []
    assert dpa_runtime_problems("legacy", "3.0.0b3+local") == []
    # b0 does not know ``use_tebd_bias``; b4 already changed the repformer layout.
    for other in ("3.0.0b0", "3.0.0b1", "3.0.0b2", "3.0.0b4", "3.0.0"):
        assert dpa_runtime_problems("legacy", other), other


def test_legacy_runtime_in_the_current_environment_is_refused():
    problems = dpa_runtime_problems("legacy", "3.2.1.dev63+g0edeaef56")
    assert problems and "install_mliport.sh --engines dpa-legacy" in problems[0]
    assert "does not fall back" in problems[0] or "do not fall back" in problems[0]


def test_current_runtime_in_the_legacy_environment_is_refused():
    problems = dpa_runtime_problems("current", "3.0.0b3")
    assert problems and "DPA_RUNTIME=legacy" in problems[0]


def test_missing_deepmd_is_only_an_error_for_the_legacy_runtime():
    # ``current`` lets the adapter's own import error explain the missing
    # package; ``legacy`` cannot be verified without it.
    assert dpa_runtime_problems("current", None) == []
    problems = dpa_runtime_problems("legacy", None)
    assert problems and "dpa-legacy" in problems[0]


def test_get_calculator_enforces_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.2.1")
    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    with pytest.raises(RuntimeError, match="dpa-legacy"):
        wrapper.get_calculator()
    assert wrapper._calculator is None


# ---------------------------------------------------------- safe checkpoints
def test_legacy_runtime_refuses_a_checkpoint_needing_unsafe_unpickling(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Weights only load failed")),
    )
    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    with pytest.raises(RuntimeError, match="convert_dpa_checkpoint.py"):
        wrapper.get_calculator()
    assert wrapper._calculator is None


def test_legacy_runtime_refuses_a_non_pt_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    model = tmp_path / "model.pth"
    model.write_bytes(b"stub")
    wrapper = DPACalculatorWrapper(
        model, device="cpu", task="bulk", dpa_runtime="legacy"
    )
    with pytest.raises(ValueError, match="only loads '.pt' checkpoints"):
        wrapper.get_calculator()
    assert wrapper._calculator is None


def test_legacy_runtime_refuses_to_run_without_tensor_verification(
    tmp_path, monkeypatch
):
    """A skipped strict-tensor check is a failure in the legacy runtime."""
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    torch = pytest.importorskip("torch")
    payload = {
        "model": {"_extra_state": {"model_params": {"descriptor": {"type": "dpa2"}}}}
    }
    monkeypatch.setattr(torch, "load", lambda *a, **k: payload)
    module = types.ModuleType("deepmd.calculator")

    class DP:
        def __init__(self, **kwargs):
            self.dp = types.SimpleNamespace(
                deep_eval=types.SimpleNamespace(_dpmodel=None)
            )
            self.implemented_properties = ["energy", "forces", "stress"]

    module.DP = DP
    monkeypatch.setitem(sys.modules, "deepmd", types.ModuleType("deepmd"))
    monkeypatch.setitem(sys.modules, "deepmd.calculator", module)

    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    with pytest.raises(RuntimeError, match="without tensor verification"):
        wrapper.get_calculator()
    assert wrapper._checkpoint_tensors_verified is False
    assert wrapper._calculator is None


@pytest.mark.parametrize("descriptor", ["dpa4", "dpa4c", "sezm", None])
def test_legacy_runtime_refuses_non_dpa2_checkpoints(tmp_path, monkeypatch, descriptor):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    torch = pytest.importorskip("torch")
    payload = {
        "model": {
            "_extra_state": {"model_params": {"descriptor": {"type": descriptor}}}
        }
    }
    monkeypatch.setattr(torch, "load", lambda *a, **k: payload)
    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    with pytest.raises(ValueError, match="DPA_RUNTIME=legacy can only reconstruct"):
        wrapper.get_calculator()
    assert wrapper._calculator is None


def test_legacy_runtime_accepts_a_dpa2_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    torch = pytest.importorskip("torch")
    payload = {
        "model": {"_extra_state": {"model_params": {"descriptor": {"type": "dpa2"}}}}
    }
    monkeypatch.setattr(torch, "load", lambda *a, **k: payload)
    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    # Both guards pass; the (stubbed) deepmd import is what stops it next.
    wrapper._require_safe_checkpoint()
    wrapper._require_legacy_compatible_definition()


# -------------------------------------------------------------- provenance
def test_info_records_runtime_and_deepmd_version(tmp_path, monkeypatch):
    monkeypatch.setattr(dpa_calc, "installed_deepmd_version", lambda: "3.0.0b3")
    wrapper = _wrapper(tmp_path, dpa_runtime="legacy")
    fake = types.SimpleNamespace(
        deep_eval=None,
        implemented_properties=["energy", "forces", "stress"],
    )
    monkeypatch.setattr(wrapper, "get_calculator", lambda: fake)
    monkeypatch.setattr(
        "mliport.devices.torch_model_identity",
        lambda model, device: {
            "actual_device_type": "cpu",
            "actual_device_logical_index": None,
            "actual_device_uuid": None,
            "model_precision": ["float64"],
        },
    )
    info = wrapper.info()
    assert info["dpa_runtime"] == "legacy"
    assert info["deepmd_version"] == "3.0.0b3"


def test_blackwell_has_no_build_for_the_legacy_runtime():
    """torch 2.3.1+cu121 stops at sm_90; Blackwell must not claim support."""
    from mliport.install.compatibility import lookup_backend

    backend = lookup_backend("dpa-legacy")
    assert backend is not None
    assert backend.arch_profiles["blackwell"].upstream_supported is False
    assert backend.arch_profiles["hopper"].upstream_supported is True
    assert backend.arch_profiles["maxwell"].upstream_supported is False


def test_model_record_carries_the_dpa_runtime():
    """The identity record must say which environment produced a number."""
    import mliport.model_record as model_record

    source_path = Path(model_record.__file__)
    assert '"dpa_runtime": info.get("dpa_runtime")' in source_path.read_text(
        encoding="utf-8"
    )


def test_every_entry_point_can_express_the_runtime():
    """CLI, queue and the Python API must all be able to say ``legacy``."""
    from mliport import api, queue
    from mliport.backend_selection import CALCULATOR_OPTION_KEYS
    from mliport.config.schema import get_schema

    assert "dpa_runtime" in CALCULATOR_OPTION_KEYS["dpa"]
    spec = get_schema().resolve("dpa_runtime")
    assert spec is not None and spec.choices == ("current", "legacy")
    assert queue._OPT_FLAGS["dpa_runtime"] == ("--dpa-runtime",)
    for function in (
        api.run_single_point,
        api.run_optimization,
        api.run_md,
        api.run_neb,
    ):
        assert "dpa_runtime" in function.__code__.co_varnames
