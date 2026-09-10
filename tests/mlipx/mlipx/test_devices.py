from __future__ import annotations

import pytest

from mlipx.devices import VisibleGpu, verify_runtime_uuid, visible_gpus


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
    monkeypatch.setenv("MLIPX_LEASED_GPU_UUID", "GPU-a")
    assert verify_runtime_uuid("GPU-a", "GPU-a") == "GPU-a"
    for actual, requested in [(None, "GPU-a"), ("GPU-b", "GPU-a"), ("GPU-b", "GPU-b")]:
        with pytest.raises(RuntimeError):
            verify_runtime_uuid(actual, requested)
