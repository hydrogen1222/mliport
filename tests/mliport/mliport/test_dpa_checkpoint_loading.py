"""Reject scientific changes hidden by permissive upstream state loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mliport.calculators.dpa_calc import DPACalculatorWrapper

torch = pytest.importorskip("torch")


def _fixture(tmp_path, *, definition=None):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    model.weight.data[:] = torch.tensor([[1.25, -2.5]], dtype=torch.float64)
    state = {
        "model.Default.weight": model.weight.detach().clone(),
        "_extra_state": {"model_params": definition or {"type": "standard"}},
    }
    path = tmp_path / "model.pt"
    torch.save({"model": state}, path)
    calculator = SimpleNamespace(
        dp=SimpleNamespace(deep_eval=SimpleNamespace(_dpmodel=model))
    )
    return DPACalculatorWrapper(path), calculator, model


def test_exportable_eager_weights_have_actual_device_and_dtype(tmp_path):
    wrapper, calculator, model = _fixture(tmp_path)
    wrapper._validate_pt_checkpoint(calculator)
    assert wrapper._checkpoint_tensors_verified
    # An exportable evaluator exposes an eager callable, not its weight module,
    # through get_model(). Identity must use the actual _dpmodel tensors.
    calculator.dp.deep_eval.get_model = lambda: (lambda x: x)
    assert wrapper._torch_model(calculator) is model
    from mliport.devices import torch_model_identity

    info = torch_model_identity(wrapper._torch_model(calculator), "cpu")
    assert info["actual_device_type"] == "cpu"
    assert info["model_precision"] == ["float64"]


@pytest.mark.parametrize("change", ["missing", "unused", "dtype", "value"])
def test_partial_or_changed_network_is_rejected(tmp_path, change):
    wrapper, calculator, model = _fixture(tmp_path)
    if change == "missing":
        model.register_buffer("new_reference", torch.tensor([0.0]))
    elif change == "unused":
        payload = torch.load(wrapper.model_path, weights_only=True)
        payload["model"]["model.Default.old_weight"] = torch.tensor([1.0])
        torch.save(payload, wrapper.model_path)
    elif change == "dtype":
        model.float()
    else:
        model.weight.data[0, 0] += 0.25
    with pytest.raises(RuntimeError, match="partial load|tensor changed"):
        wrapper._validate_pt_checkpoint(calculator)
    assert not wrapper._checkpoint_tensors_verified


def test_exportable_selected_definition_cannot_hide_multitask_checkpoint(tmp_path):
    wrapper, calculator, _ = _fixture(
        tmp_path, definition={"model_dict": {"Default": {}, "other": {}}}
    )
    with pytest.raises(ValueError, match="multi-task.*required"):
        wrapper._validate_pt_checkpoint(calculator)
    wrapper._head = "Default"
    wrapper._validate_pt_checkpoint(calculator)
    assert wrapper._active_head == "Default"
    assert wrapper._available_heads == ["Default", "other"]


@pytest.mark.parametrize("spin", [False, True])
def test_only_documented_nonspin_descriptor_version_migration_is_allowed(
    tmp_path, spin
):
    definition = {
        "type": "dpa4",
        "descriptor": {"type": "dpa4", "add_chg_spin_ebd": spin},
    }
    wrapper, calculator, model = _fixture(tmp_path, definition=definition)
    model.atomic_model = torch.nn.Module()
    model.atomic_model.descriptor = torch.nn.Module()
    model.atomic_model.descriptor.register_buffer(
        "version_tensor", torch.tensor(1.3, dtype=torch.float64)
    )
    payload = torch.load(wrapper.model_path, weights_only=True)
    payload["model"]["model.Default.atomic_model.descriptor.version_tensor"] = (
        torch.tensor(1.1, dtype=torch.float64)
    )
    torch.save(payload, wrapper.model_path)
    if spin:
        with pytest.raises(RuntimeError, match="tensor changed"):
            wrapper._validate_pt_checkpoint(calculator)
    else:
        wrapper._validate_pt_checkpoint(calculator)
        assert wrapper._checkpoint_migrations == [
            "upstream unbridged descriptor version 1.1 -> 1.3"
        ]


def test_lossless_stat_widening_is_recorded_not_refused(tmp_path):
    """deepmd 3.2 builds the fp32 DPA-3.1 statistics tensors as float64.

    The stored values are bit-identical after widening (float32 -> float64 is
    exact), so refusing the model would reject a checkpoint the runtime loaded
    correctly. The widening is recorded as a migration because it changes the
    precision the run computes in.
    """
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    model.weight.data[:] = torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    state = {
        "model.Default.weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        "_extra_state": {"model_params": {"type": "standard"}},
    }
    path = tmp_path / "model.pt"
    torch.save({"model": state}, path)
    calculator = SimpleNamespace(
        dp=SimpleNamespace(deep_eval=SimpleNamespace(_dpmodel=model))
    )
    wrapper = DPACalculatorWrapper(path)
    wrapper._validate_pt_checkpoint(calculator)
    assert wrapper._checkpoint_tensors_verified
    assert len(wrapper._checkpoint_migrations) == 1
    assert "widened 1 tensor(s) float32->float64" in wrapper._checkpoint_migrations[0]
    assert "model.Default.weight" not in wrapper._checkpoint_migrations[0]


@pytest.mark.parametrize(
    "source,loaded,reason",
    [
        (torch.ones(3, dtype=torch.float32), torch.ones(3, dtype=torch.float64), None),
        (
            torch.ones(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float32),
            "dtype",
        ),
        (
            torch.ones(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
            "changed",
        ),
        (
            torch.ones(3, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
            "shape",
        ),
        (
            torch.ones(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.int32),
            "dtype",
        ),
    ],
)
def test_tensor_load_verdict(source, loaded, reason):
    from mliport.calculators.dpa_calc import tensor_load_verdict

    if reason is None:
        note = tensor_load_verdict("t", source, loaded)
        assert note and "float32->float64" in note
        return
    with pytest.raises(RuntimeError, match=reason):
        tensor_load_verdict("t", source, loaded)
