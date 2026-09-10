"""GPU identities without importing a numerical runtime or changing visibility."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class VisibleGpu:
    logical_index: int
    uuid: str
    physical_index: int | None
    name: str
    compute_capability: tuple[int, int] | None


def visible_gpus(
    physical: Sequence[VisibleGpu], environment: Mapping[str, str] | None = None
) -> list[VisibleGpu]:
    """Resolve an explicit CUDA visibility list, preserving its logical order.

    Unknown or ambiguous selectors fail closed; an empty list hides all GPUs.
    UUID prefixes are accepted only when uniquely resolvable in the inventory.
    """
    env = os.environ if environment is None else environment
    visibility = env.get("CUDA_VISIBLE_DEVICES")
    if visibility is None:
        selected = list(physical)
    elif visibility.strip() in {"", "-1"}:
        selected = []
    else:
        selected = []
        for token in visibility.split(","):
            selector = token.strip()
            matches = [
                gpu
                for gpu in physical
                if (selector.isdigit() and gpu.physical_index == int(selector))
                or (
                    selector.startswith(("GPU-", "MIG-"))
                    and gpu.uuid.startswith(selector)
                )
            ]
            if len(matches) != 1 or matches[0] in selected:
                raise RuntimeError(
                    f"CUDA_VISIBLE_DEVICES selector {selector!r} is unknown, "
                    "ambiguous, or duplicated"
                )
            selected.append(matches[0])
    return [
        VisibleGpu(
            index, gpu.uuid, gpu.physical_index, gpu.name, gpu.compute_capability
        )
        for index, gpu in enumerate(selected)
    ]


def verify_runtime_uuid(actual_uuid: str | None, requested_uuid: str) -> str:
    """Never substitute the requested identity for missing runtime evidence."""
    if not actual_uuid:
        raise RuntimeError("Framework did not report an actual GPU UUID")
    lease = os.environ.get("MLIPX_LEASED_GPU_UUID")
    if actual_uuid != requested_uuid or (lease and actual_uuid != lease):
        raise RuntimeError("Actual framework GPU UUID differs from request or lease")
    return actual_uuid


def require_isolated_visibility(device: str, backend: str) -> None:
    """Preflight backends whose ASE adapter has no per-calculator device.

    Such adapters cannot safely select another GPU inside an existing Python
    process. The caller must isolate it before importing the framework, as the
    queue worker does. No environment variable is modified here.
    """
    dev = str(device).lower()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev == "cpu":
        valid = visibility is not None and visibility.strip() in {"", "-1"}
    else:
        valid = (
            dev in {"cuda", "gpu", "cuda:0"}
            and visibility is not None
            and bool(visibility.strip())
            and visibility.strip() != "-1"
            and "," not in visibility
            and os.environ.get("LOCAL_RANK", "0") == "0"
            and os.environ.get("DEVICE") != "cpu"
        )
    if not valid:
        raise RuntimeError(
            f"{backend} requires process-level device isolation before framework "
            "imports. Use the queue worker, or start a fresh process with "
            "CUDA_VISIBLE_DEVICES='' for cpu / a single GPU UUID for cuda:0. "
            "Refusing to change the parent process GPU visibility."
        )


def torch_model_identity(model, requested: str) -> dict:
    """Read device identity from model tensors, never from requested settings."""
    unknown = {
        "actual_device_type": "unknown",
        "actual_device_logical_index": None,
        "actual_device_uuid": None,
    }
    if model is None or not callable(getattr(model, "parameters", None)):
        return unknown
    import torch

    if not isinstance(model, torch.nn.Module):
        return unknown
    tensors = list(model.parameters())
    if not tensors and callable(getattr(model, "buffers", None)):
        tensors = list(model.buffers())
    if not tensors or not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return unknown
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError("Model tensors span multiple devices; identity is ambiguous")
    actual = next(iter(devices))
    expected_type = "cpu" if requested == "cpu" else "cuda"
    if actual.type != expected_type:
        raise RuntimeError("Model tensor device type differs from requested device")
    identity = {
        "actual_device_type": actual.type,
        "actual_device_logical_index": actual.index,
        "actual_device_uuid": None,
        "model_precision": sorted(
            {
                str(t.dtype).removeprefix("torch.")
                for t in tensors
                if t.is_floating_point()
            }
        ),
    }
    if actual.type == "cuda":
        from mlipx.queue import resolve_device_uuid

        properties = torch.cuda.get_device_properties(actual.index)
        uuid = getattr(properties, "uuid", None)
        if uuid is not None:
            uuid = str(uuid)
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
        identity["actual_device_uuid"] = verify_runtime_uuid(
            uuid, resolve_device_uuid(requested)
        )
    return identity


def tensorflow_output_identity(outputs, requested: str) -> dict:
    """Confirm TensorFlow output placement via its runtime PCI bus identity."""
    unknown = {
        "actual_device_type": "unknown",
        "actual_device_logical_index": None,
        "actual_device_uuid": None,
    }
    if not isinstance(outputs, list) or not outputs:
        return unknown
    import re
    import subprocess
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    tensors = [
        value
        for output in outputs
        for value in output.values()
        if isinstance(value, tf.Tensor) and value.dtype.is_floating
    ]
    if not tensors:
        return unknown
    gpu_tensors = [tensor for tensor in tensors if "GPU:" in tensor.device.upper()]
    device_names = {tensor.device for tensor in gpu_tensors or tensors}
    if len(device_names) != 1:
        raise RuntimeError("TensorFlow output devices are ambiguous")
    name = next(iter(device_names))
    spec = tf.DeviceSpec.from_string(name)
    actual_type = "cuda" if spec.device_type == "GPU" else "cpu"
    if actual_type != ("cpu" if requested == "cpu" else "cuda"):
        raise RuntimeError("TensorFlow output device differs from request")
    identity = {
        "actual_device_type": actual_type,
        "actual_device_logical_index": spec.device_index
        if actual_type == "cuda"
        else None,
        "actual_device_uuid": None,
        "model_precision": sorted({tensor.dtype.name for tensor in tensors}),
    }
    if actual_type == "cpu":
        return identity
    local = [
        dev
        for dev in device_lib.list_local_devices()
        if dev.device_type == "GPU"
        and tf.DeviceSpec.from_string(dev.name).device_index == spec.device_index
    ]
    if len(local) != 1:
        raise RuntimeError("Cannot identify actual TensorFlow GPU")
    bus = re.search(r"pci bus id:\s*([0-9a-fA-F:.]+)", local[0].physical_device_desc)
    if bus is None:
        raise RuntimeError("TensorFlow runtime did not report a GPU PCI bus ID")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    matches = [
        uuid.strip()
        for line in result.stdout.splitlines()
        for address, uuid in [line.split(",", 1)]
        if address.strip().lower().lstrip("0") == bus.group(1).lower().lstrip("0")
    ]
    if len(matches) != 1:
        raise RuntimeError("TensorFlow PCI bus cannot be uniquely mapped to a GPU UUID")
    from mlipx.queue import resolve_device_uuid

    identity["actual_device_uuid"] = verify_runtime_uuid(
        matches[0], resolve_device_uuid(requested)
    )
    return identity
