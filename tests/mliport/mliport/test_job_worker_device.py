"""The worker must execute the frozen lease, not a mutable CUDA ordinal."""

from __future__ import annotations

import pytest

from mliport.job_worker import leased_process_environment


@pytest.mark.parametrize("visible", [None, "2,3", "GPU-other", ""])
@pytest.mark.parametrize("flag", [["--device", "cuda:1"], ["--device=cuda:1"]])
def test_frozen_uuid_remaps_only_child(visible, flag):
    inherited = {"KEEP": "value"}
    if visible is not None:
        inherited["CUDA_VISIBLE_DEVICES"] = visible
    original = inherited.copy()
    command = ["python", "-m", "mliport", "sp", *flag]
    cmd, env = leased_process_environment(
        {"device": "cuda:1", "device_uuid": "GPU-leased"}, command, inherited
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-leased"
    assert env["MLIPORT_LEASED_GPU_UUID"] == "GPU-leased"
    assert env["KEEP"] == "value"
    assert "cuda:1" not in " ".join(cmd)
    assert "cuda:0" in " ".join(cmd)
    assert inherited == original
    assert command[-1] == flag[-1]


def test_cpu_hides_gpus_and_clears_inherited_lease():
    _, env = leased_process_environment(
        {"device": "cpu"},
        ["python"],
        {"CUDA_VISIBLE_DEVICES": "0", "MLIPORT_LEASED_GPU_UUID": "GPU-old"},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "MLIPORT_LEASED_GPU_UUID" not in env


@pytest.mark.parametrize("uuid", [None, "0", "GPU-a,GPU-b", "GPU-a b"])
def test_missing_or_invalid_gpu_lease_fails_closed(uuid):
    with pytest.raises(RuntimeError, match="lease"):
        leased_process_environment(
            {"device": "cuda", "device_uuid": uuid}, ["python"], {}
        )
