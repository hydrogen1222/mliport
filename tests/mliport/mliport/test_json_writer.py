"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This code is licensed under the MIT license found in the LICENSE file in the
root directory of this source tree.
"""

from __future__ import annotations

import json

import torch

from mliport.writers.json_writer import NumpyEncoder


def test_numpy_encoder_serializes_torch_tensors():
    payload = {
        "scalar": torch.tensor(5.0, device="cpu"),
        "vector": torch.tensor([1, 2], device="cpu"),
    }

    decoded = json.loads(json.dumps(payload, cls=NumpyEncoder))

    assert decoded == {"scalar": 5.0, "vector": [1, 2]}
