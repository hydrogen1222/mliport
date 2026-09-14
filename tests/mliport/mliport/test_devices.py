from __future__ import annotations

import pytest

from mliport.devices import VisibleGpu, verify_runtime_uuid, visible_gpus


@pytest.fixture()
def inventory():
    return [
        VisibleGpu(0, "GPU-a", 0, "V100", (7, 0)),
        VisibleGpu(1, "GPU-b", 1, "4090", (8, 9)),
        VisibleGpu(2, "GPU-c", 2, "V100", (7, 0)),
        VisibleGpu(3, "GPU-d", 3, "4090", (8, 9)),
    ]


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, [0, 1, 2, 3]),
        ({"CUDA_VISIBLE_DEVICES": "2,3"}, [2, 3]),
        ({"CUDA_VISIBLE_DEVICES": "GPU-d,GPU-a"}, [3, 0]),
        ({"CUDA_VISIBLE_DEVICES": ""}, []),
        ({"CUDA_VISIBLE_DEVICES": "-1"}, []),
        ({"CUDA_VISIBLE_DEVICES": "2"}, [2]),
    ],
)
def test_visible_logical_order(inventory, env, expected):
    resolved = visible_gpus(inventory, env)
    assert [g.physical_index for g in resolved] == expected
    assert [g.logical_index for g in resolved] == list(range(len(expected)))
    assert [g.uuid for g in resolved] == [inventory[i].uuid for i in expected]


@pytest.mark.parametrize("selector", ["4", "GPU-", "0,0", "0,", "-2"])
def test_invalid_visibility_fails(inventory, selector):
    with pytest.raises(RuntimeError):
        visible_gpus(inventory, {"CUDA_VISIBLE_DEVICES": selector})


def test_framework_uuid_must_match_request_and_lease(monkeypatch):
    monkeypatch.setenv("MLIPORT_LEASED_GPU_UUID", "GPU-a")
    assert verify_runtime_uuid("GPU-a", "GPU-a") == "GPU-a"
    for actual, requested in [(None, "GPU-a"), ("GPU-b", "GPU-a"), ("GPU-b", "GPU-b")]:
        with pytest.raises(RuntimeError):
            verify_runtime_uuid(actual, requested)


def test_visibility_is_isolated_truth_table(monkeypatch):
    """DPA/GRACE isolation preflight predicate (fresh-install acceptance)."""
    from mliport.devices import visibility_is_isolated

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("DEVICE", raising=False)
    assert visibility_is_isolated("cuda") is False
    assert visibility_is_isolated("cpu") is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert visibility_is_isolated("cuda") is True
    assert visibility_is_isolated("cpu") is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visibility_is_isolated("cpu") is True
    assert visibility_is_isolated("cuda") is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    assert visibility_is_isolated("cuda") is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert visibility_is_isolated("cuda") is False
    assert visibility_is_isolated("cpu") is True


# ---------------------------------------------------------------------------
# GPU-01: one CUDA_VISIBLE_DEVICES resolver for CLI and queue
# ---------------------------------------------------------------------------


def test_mg01_without_cvd_uses_requested_physical_ordinal(inventory):
    from mliport.devices import resolve_visible_device

    env = {}
    assert (
        resolve_visible_device("cuda:0", inventory=inventory, environment=env).uuid
        == "GPU-a"
    )
    assert (
        resolve_visible_device("cuda", inventory=inventory, environment=env).uuid
        == "GPU-a"
    )
    assert (
        resolve_visible_device("cuda:3", inventory=inventory, environment=env).uuid
        == "GPU-d"
    )
    assert resolve_visible_device("cpu", inventory=inventory, environment=env) is None


def test_mg02_numeric_cvd_uses_local_ordinals(inventory):
    from mliport.devices import resolve_visible_device

    env = {"CUDA_VISIBLE_DEVICES": "2,3"}
    assert (
        resolve_visible_device("cuda:0", inventory=inventory, environment=env).uuid
        == "GPU-c"
    )
    assert (
        resolve_visible_device("cuda:1", inventory=inventory, environment=env).uuid
        == "GPU-d"
    )


def test_mg03_uuid_cvd_uses_local_ordinals(inventory):
    from mliport.devices import resolve_visible_device

    env = {"CUDA_VISIBLE_DEVICES": "GPU-c,GPU-d"}
    assert (
        resolve_visible_device("cuda:0", inventory=inventory, environment=env).uuid
        == "GPU-c"
    )
    assert (
        resolve_visible_device("cuda:1", inventory=inventory, environment=env).uuid
        == "GPU-d"
    )


def test_mg04_unique_uuid_prefix_resolves(inventory):
    from mliport.devices import resolve_visible_device

    env = {"CUDA_VISIBLE_DEVICES": "GPU-c"}
    assert (
        resolve_visible_device("cuda:0", inventory=inventory, environment=env).uuid
        == "GPU-c"
    )
    with pytest.raises(RuntimeError):
        # Too short to be unique.
        resolve_visible_device(
            "cuda:0", inventory=inventory, environment={"CUDA_VISIBLE_DEVICES": "GPU-"}
        )


def test_mg05_ordinal_outside_cvd_fails(inventory):
    from mliport.devices import resolve_visible_device

    with pytest.raises(RuntimeError, match="not visible"):
        resolve_visible_device(
            "cuda:1", inventory=inventory, environment={"CUDA_VISIBLE_DEVICES": "2"}
        )


@pytest.mark.parametrize("selector", ["2,,3", "0,0", "GPU-z", "-2", "4"])
def test_mg06_malformed_or_unknown_cvd_fails(inventory, selector):
    from mliport.devices import resolve_visible_device

    with pytest.raises(RuntimeError):
        resolve_visible_device(
            "cuda:0",
            inventory=inventory,
            environment={"CUDA_VISIBLE_DEVICES": selector},
        )


def test_mg07_queue_and_cli_resolvers_agree(inventory, monkeypatch):
    import mliport.devices as devices
    from mliport.cli import _device_visibility_value

    monkeypatch.setattr(devices, "query_physical_gpus", lambda: inventory)
    for env in (
        {},
        {"CUDA_VISIBLE_DEVICES": "2,3"},
        {"CUDA_VISIBLE_DEVICES": "GPU-d,GPU-a"},
    ):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", ""))
        if not env:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        for device in ("cuda:0", "cuda:1"):
            try:
                expected = devices.resolve_device_uuid(device, environment=env)
            except RuntimeError:
                continue
            assert devices.resolve_device_uuid(device, environment=env) == expected
            assert _device_visibility_value(device) == expected


def test_mig_selectors_fail_closed(inventory):
    from mliport.devices import resolve_visible_device

    with pytest.raises(RuntimeError, match="MIG"):
        resolve_visible_device(
            "cuda:0",
            inventory=inventory,
            environment={"CUDA_VISIBLE_DEVICES": "MIG-abc"},
        )
    with pytest.raises(ValueError, match="MIG"):
        resolve_visible_device("MIG-abc", inventory=inventory, environment={})


def test_invalid_device_string_fails(inventory):
    from mliport.devices import resolve_visible_device

    with pytest.raises(ValueError):
        resolve_visible_device("tpu", inventory=inventory, environment={})
