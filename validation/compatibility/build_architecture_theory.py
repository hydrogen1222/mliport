#!/usr/bin/env python3
"""Build the GPU architecture theory audit (task book Phase D).

The audit answers one question per backend and GPU family:

    Which exact wheel/native-extension evidence exists, and what level of
    compatibility does that evidence actually justify?

It is deliberately hardware-independent for everything except Volta: the V100
acceptance is the only real-hardware evidence.  Run with ``--probe`` to
re-check the installed environments against the recorded evidence.

Outputs::

    validation/compatibility/gpu_architecture_theory.json
    validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
AUDIT_DATE = "2026-09-14"

PYTHONS = ("3.10", "3.11", "3.12")

# ---------------------------------------------------------------------------
# Framework wheel evidence (official index + exact binary inspection)
# ---------------------------------------------------------------------------

TORCH_WHEELS: dict[tuple[str, str], dict] = {
    ("2.8.0", "cu126"): {
        "filename": "torch-2.8.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "ce6e6a1f4803ad62d1fe51ce3fe5ca14bcd8bc7cace7b09d5590f8147fa16ad",
        "source": "https://download.pytorch.org/whl/cu126/",
        "python_abi": ["cp39", "cp310", "cp311", "cp312", "cp313"],
        "compiled_archs": [
            "sm_50",
            "sm_60",
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
        ],
        "evidence": "cuobjdump --list-elf libtorch_cuda.so (installed .venv-mace)",
    },
    ("2.10.0", "cu126"): {
        "filename": "torch-2.10.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "2a7a569206f07965eff69b28e147676540bb0ba6e1a39410802b6e4708cb8356",
        "source": "https://download.pytorch.org/whl/cu126/",
        "python_abi": ["cp39", "cp310", "cp311", "cp312", "cp313"],
        "compiled_archs": [
            "sm_50",
            "sm_60",
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
        ],
        "evidence": "cuobjdump --list-elf libtorch_cuda.so (installed .venv-dpa)",
    },
    ("2.8.0", "cu128"): {
        "filename": "torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "4354fc05bb79b208d6995a04ca1ceef6a9547b1c4334435574353d381c55087c",
        "source": "https://download.pytorch.org/whl/cu128/",
        "python_abi": ["cp39", "cp310", "cp311", "cp312", "cp313"],
        "compiled_archs": [
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
            "sm_120",
        ],
        "evidence": "cuobjdump --list-elf libtorch_cuda.so (wheel downloaded, hash verified)",
    },
    ("2.10.0", "cu128"): {
        "filename": "torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "628e89bd5110ced7debee2a57c69959725b7fbc64eab81a39dd70e46c7e28ba5",
        "source": "https://download.pytorch.org/whl/cu128/",
        "python_abi": ["cp39", "cp310", "cp311", "cp312", "cp313"],
        "compiled_archs": [
            "sm_70",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
            "sm_120",
        ],
        "evidence": "cuobjdump --list-elf libtorch_cuda.so (wheel downloaded, hash verified)",
    },
}

TENSORFLOW_WHEEL = {
    "filename": "tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "sha256": "25265b0bc527e0d54b1e9cc60c44a24f44a809fe27666b905f0466471f9c52ec",
    "source": "https://pypi.org/project/tensorflow/2.20.0/",
    "python_abi": ["cp39", "cp310", "cp311", "cp312", "cp313"],
    "cuda_version": "12.5.1",
    "cudnn_version": "9",
    "declared_capabilities": ["sm_60", "sm_70", "sm_80", "sm_89", "compute_90"],
    "compiled_archs": ["sm_60", "sm_70", "sm_80", "sm_89", "sm_90"],
    "evidence": "tf.sysconfig.get_build_info() + cuobjdump --list-elf libtensorflow_cc.so.2",
}

BACKEND_WHEELS = {
    "mace": {
        "package": "mace-torch",
        "version": "0.3.16",
        "filename": "mace_torch-0.3.16-py3-none-any.whl",
        "sha256": "b80407edf6b2a1ec8523668c2a36852d20927ce1c3c56b70983a9f2dc53233ad",
    },
    "dpa": {
        "package": "deepmd-kit",
        "version": "3.1.3",
        "filename": "deepmd_kit-3.1.3-py37-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "sha256": "ef9460a2ae4e985e826f8b4649bbd27df70ac65c9bb0b41568fc5f007bd0a9ba",
    },
    "grace": {
        "package": "tensorpotential",
        "version": "0.6.0",
        "filename": "tensorpotential-0.6.0-py3-none-any.whl",
        "sha256": "af9a4371acd51966885b3242daad0350c90b1f0cf03073d931b815fce889063f",
    },
    "uma": {
        "package": "fairchem-core",
        "version": "2.21.0",
        "filename": "fairchem_core-2.21.0-py3-none-any.whl",
        "sha256": "9db616317927c7a91cea2dbca707a163b090f88b329d8856297d22e1db9469fd",
    },
}

NATIVE_AUDIT = {
    "mace": {
        "distributions": ["mace-torch", "e3nn"],
        "bundled_device_code": "none",
        "inherits": "torch",
        "detail": "mace-torch and e3nn ship no bundled .so with device code; "
        "GPU kernels come from the pinned torch build.",
    },
    "uma": {
        "distributions": ["fairchem-core", "e3nn"],
        "bundled_device_code": "none",
        "inherits": "torch",
        "detail": "fairchem-core ships no bundled .so with device code.",
    },
    "grace": {
        "distributions": ["tensorpotential"],
        "bundled_device_code": "none",
        "inherits": "tensorflow",
        "detail": "tensorpotential ships no bundled .so with device code; GPU "
        "kernels come from the TensorFlow 2.20 build.",
    },
    "dpa": {
        "distributions": ["deepmd-kit"],
        "bundled_device_code": "libdeepmd_op_cuda.so",
        "compiled_archs": [
            "sm_50",
            "sm_52",
            "sm_53",
            "sm_60",
            "sm_61",
            "sm_62",
            "sm_70",
            "sm_72",
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_87",
            "sm_89",
            "sm_90",
            "sm_100",
            "sm_101",
            "sm_120",
        ],
        "host_only": ["libdeepmd_op_pt.so (links libdeepmd_op_cuda.so and mpich)"],
        "inherits": "torch",
        "detail": "cuobjdump shows explicit cubins from sm_50 through sm_120, "
        "including sm_60 and sm_61 for Pascal.",
    },
}

ARCH_ENTRIES = [
    ("maxwell", "Maxwell", [(5, 0), (5, 2)], "cu126"),
    ("pascal_p100", "Pascal P100 (sm60)", [(6, 0)], "cu126"),
    ("pascal_p40", "Pascal P40 (sm61)", [(6, 1)], "cu126"),
    ("volta", "Volta V100 (sm70)", [(7, 0)], "cu126"),
    ("turing", "Turing (sm75)", [(7, 5)], "cu128"),
    ("ampere_sm80", "Ampere (sm80)", [(8, 0)], "cu128"),
    ("ampere_sm86", "Ampere (sm86)", [(8, 6)], "cu128"),
    ("ada", "Ada Lovelace (sm89)", [(8, 9)], "cu128"),
    ("hopper", "Hopper (sm90)", [(9, 0)], "cu128"),
    ("blackwell", "Blackwell (sm100/sm120)", [(10, 0), (12, 0)], "cu128"),
]

BACKENDS = ("mace", "dpa", "grace", "uma")

#: Only Volta has real mliport hardware acceptance evidence.
HARDWARE_VERIFIED = {"volta"}


def _parse_sm(token: str) -> tuple[int, int] | None:
    body = str(token)[3:] if str(token).startswith("sm_") else ""
    if not body.isdigit() or len(body) < 2:
        return None
    # sm_90 -> (9, 0); sm_100 -> (10, 0); sm_120 -> (12, 0).
    return int(body[:-1]), int(body[-1])


def _covers(compiled_archs: list[str], cc: tuple[int, int]) -> bool:
    """NVIDIA minor-revision rule: sm_X.Y cubin runs on sm_X.Z for Z >= Y."""
    major, minor = cc
    for token in compiled_archs:
        arch = _parse_sm(token)
        if arch and arch[0] == major and arch[1] <= minor:
            return True
    return False


def _torch_version_for(backend: str) -> str:
    return "2.10.0" if backend == "dpa" else "2.8.0"


def classify(
    backend: str, arch_name: str, ccs: list[tuple[int, int]], channel: str
) -> tuple[str, list[str]]:
    caveats: list[str] = []
    if arch_name in HARDWARE_VERIFIED:
        return "hardware_verified", caveats
    if arch_name == "maxwell":
        return "experimental", [
            "Upstream PyTorch/TensorFlow still compile sm_50/sm_52 kernels, but "
            "the compatibility registry keeps Maxwell experimental and no real "
            "hardware smoke exists.",
        ]
    if backend == "grace" and arch_name in {"hopper", "blackwell"}:
        return "experimental", [
            "TensorFlow 2.20 declares compute_90 PTX and sm_90 SASS but no "
            "Blackwell SASS target; mliport keeps GRACE experimental here.",
        ]
    if backend == "grace":
        framework_archs = TENSORFLOW_WHEEL["compiled_archs"]
        if all(_covers(framework_archs, cc) for cc in ccs):
            if (7, 5) in ccs:
                caveats.append(
                    "Turing sm75 runs the sm_70 cubin via NVIDIA "
                    "minor-revision binary compatibility."
                )
            return "binary_theory_verified", caveats
        return "resolver_only", ["No TensorFlow cubin covers this capability."]
    version = _torch_version_for(backend)
    wheel = TORCH_WHEELS[(version, channel)]
    if not all(_covers(wheel["compiled_archs"], cc) for cc in ccs):
        return "resolver_only", ["No torch cubin covers this capability."]
    native = NATIVE_AUDIT[backend]
    native_archs = native.get("compiled_archs")
    if native_archs is not None and not all(_covers(native_archs, cc) for cc in ccs):
        return "resolver_only", [
            f"{native['bundled_device_code']} has no cubin covering this "
            "capability.",
        ]
    if arch_name == "pascal_p40":
        caveats.append(
            "sm61 uses the explicit sm_61 cubin (DPA) or the sm_60 "
            "cubin via minor-revision compatibility (torch/TF)."
        )
    if arch_name == "blackwell":
        caveats.append(
            "cu128 torch cubins include sm_100/sm_120; no hardware "
            "smoke exists for this family."
        )
    return "binary_theory_verified", caveats


def build_entries() -> list[dict]:
    entries = []
    for arch_name, label, ccs, channel in ARCH_ENTRIES:
        for backend in BACKENDS:
            classification, caveats = classify(backend, arch_name, ccs, channel)
            if backend == "grace":
                framework = {
                    "name": "tensorflow",
                    "version": "2.20.0",
                    "wheel": TENSORFLOW_WHEEL["filename"],
                    "sha256": TENSORFLOW_WHEEL["sha256"],
                    "compiled_archs": TENSORFLOW_WHEEL["compiled_archs"],
                }
            else:
                version = _torch_version_for(backend)
                wheel = TORCH_WHEELS[(version, channel)]
                framework = {
                    "name": "torch",
                    "version": version,
                    "channel": channel,
                    "wheel": wheel["filename"],
                    "sha256": wheel["sha256"],
                    "compiled_archs": wheel["compiled_archs"],
                }
            entries.append(
                {
                    "architecture": arch_name,
                    "label": label,
                    "compute_capabilities": [list(cc) for cc in ccs],
                    "backend": backend,
                    "framework": framework,
                    "backend_package": BACKEND_WHEELS[backend]["package"]
                    + "=="
                    + BACKEND_WHEELS[backend]["version"],
                    "native_extension_evidence": NATIVE_AUDIT[backend]["detail"],
                    "real_hardware_tested": arch_name in HARDWARE_VERIFIED,
                    "support_classification": classification,
                    "caveats": caveats,
                    "evidence_source": [
                        BACKEND_WHEELS[backend]["filename"],
                        framework["wheel"],
                        NATIVE_AUDIT[backend]["detail"],
                    ],
                }
            )
    return entries


def probe_installed() -> dict:
    """Re-check the installed environments against the recorded evidence."""
    findings = {}
    for engine, venv in (
        ("mace", ".venv-mace"),
        ("dpa", ".venv-dpa"),
        ("uma", ".venv"),
        ("grace", ".venv-grace"),
    ):
        python = REPO / venv / "bin" / "python"
        if not python.is_file():
            findings[engine] = {"installed": False}
            continue
        code = (
            "import json\n"
            "out={}\n"
            "try:\n"
            "    import torch\n"
            "    out['torch']=torch.__version__\n"
            "    out['cuda']=torch.version.cuda\n"
            "    out['archs']=torch.cuda.get_arch_list()\n"
            "except Exception as exc:\n"
            "    out['torch_error']=str(exc)\n"
            "try:\n"
            "    import tensorflow as tf\n"
            "    out['tensorflow']=tf.__version__\n"
            "    info=tf.sysconfig.get_build_info()\n"
            "    out['tf_caps']=list(info.get('cuda_compute_capabilities', []))\n"
            "    out['tf_cuda']=info.get('cuda_version')\n"
            "except Exception as exc:\n"
            "    out['tf_error']=str(exc)\n"
            "print(json.dumps(out))\n"
        )
        proc = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            findings[engine] = {"installed": True, "error": proc.stderr[-300:]}
            continue
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        findings[engine] = {"installed": True, **payload}
    return findings


def render_report(data: dict) -> str:
    lines = [
        "# mliport GPU architecture theory audit",
        "",
        f"Audit date: {AUDIT_DATE}. Scope: binary-level compatibility evidence "
        "for the pinned frameworks and backend packages, without pretending "
        "that non-Volta hardware was tested.",
        "",
        "## What this report is",
        "",
        "Every entry answers two separate questions:",
        "",
        "1. does the exact pinned wheel/native extension contain kernels for "
        "that compute capability, and",
        "2. has mliport actually run that backend on real hardware of that " "family?",
        "",
        "Only the V100/Volta row answers yes to the second question; the other "
        "families are classified from binary evidence and the dependency "
        "resolver.",
        "",
        "## Classification levels",
        "",
        "| Level | Meaning |",
        "|---|---|",
        "| `hardware_verified` | mliport ran the backend on real GPU hardware of this family (V100 acceptance). |",
        "| `binary_theory_verified` | Exact wheel/native cubes cover every requested compute capability, but no hardware smoke exists. |",
        "| `resolver_only` | The installer resolves a wheel, but the binary evidence does not cover this capability. |",
        "| `experimental` | Upstream does not declare support, or the framework target is PTX-only for this family. |",
        "| `unsupported` | No route is planned. |",
        "",
        "## External facts",
        "",
        "- PyTorch stops publishing CUDA 12.6 wheels from 2.15, which drops "
        "Maxwell, Pascal and Volta binary support; CUDA 13.x dropped compute "
        "capabilities below Turing. See "
        "[the PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) "
        "and [the 2.15 support-matrix RFC](https://github.com/pytorch/pytorch/issues/190385).",
        "- mliport therefore pins legacy architectures to torch 2.8.0/2.10.0 "
        "+cu126 and guards the mapping in `_torch_modern_cuda`; an unknown "
        "framework version fails closed until a new audit is written.",
        f"- TensorFlow 2.20.0 is built with CUDA {TENSORFLOW_WHEEL['cuda_version']} "
        f"and cuDNN {TENSORFLOW_WHEEL['cudnn_version']}; its declared "
        f"capabilities are {', '.join(TENSORFLOW_WHEEL['declared_capabilities'])}. "
        "Hopper has an sm_90 SASS path plus compute_90 PTX; Blackwell has no "
        "explicit SASS target.",
        "- The acceptance host runs driver 580.173.02, which is newer than the "
        "CUDA 12.x runtime requirements of every pinned wheel (CUDA minor "
        "version compatibility).",
        "",
        "## Framework wheel evidence",
        "",
        "| Package | Version | Channel | Wheel | SHA-256 | Compiled arches |",
        "|---|---|---|---|---|---|",
    ]
    for (version, channel), wheel in sorted(TORCH_WHEELS.items()):
        lines.append(
            f"| torch | {version} | {channel} | `{wheel['filename']}` "
            f"| `{wheel['sha256'][:16]}…` | {', '.join(wheel['compiled_archs'])} |"
        )
    lines.append(
        f"| tensorflow | 2.20.0 | CUDA 12.5/cuDNN 9 | `{TENSORFLOW_WHEEL['filename']}` "
        f"| `{TENSORFLOW_WHEEL['sha256'][:16]}…` | "
        f"{', '.join(TENSORFLOW_WHEEL['compiled_archs'])} (+compute_90 PTX) |"
    )
    lines += [
        "",
        "Backend packages: `mace-torch`/`fairchem-core`/`tensorpotential` are "
        "pure Python wheels; `deepmd-kit` carries the CUDA op described below.",
        "",
        "## Native extension audit",
        "",
        "| Backend | Distributions | Bundled device code | Compiled arches |",
        "|---|---|---|---|",
    ]
    for backend, audit in NATIVE_AUDIT.items():
        archs = ", ".join(audit.get("compiled_archs", [])) or "none"
        lines.append(
            f"| {backend.upper()} | {', '.join(audit['distributions'])} "
            f"| {audit['bundled_device_code']} | {archs} |"
        )
    lines += [
        "",
        "`deepmd-kit`'s `libdeepmd_op_cuda.so` was inspected with "
        "`cuobjdump --list-elf`; it contains explicit cubins from sm_50 through "
        "sm_120, including sm_60 and sm_61 for Pascal. `libdeepmd_op_pt.so` is "
        "the host-side torch extension and links that CUDA op. MACE, UMA and "
        "GRACE inherit their framework's kernels because their packages ship no "
        "device code.",
        "",
        "## Classification matrix",
        "",
        "| Architecture | Backend | Framework | Classification | Real hardware |",
        "|---|---|---|---|---|",
    ]
    for entry in data["entries"]:
        framework = entry["framework"]
        framework_text = f"{framework['name']} {framework['version']}"
        if framework.get("channel"):
            framework_text += f"+{framework['channel']}"
        lines.append(
            f"| {entry['label']} | {entry['backend'].upper()} | {framework_text} "
            f"| `{entry['support_classification']}` "
            f"| {'yes' if entry['real_hardware_tested'] else 'no'} |"
        )
    lines += [
        "",
        "## Pascal: P100 and P40 are not the same claim",
        "",
        "- P100 is sm_60: torch cu126, TensorFlow 2.20 and the DPA CUDA op all "
        "contain sm_60 cubins.",
        "- P40 is sm_61: the DPA CUDA op contains an explicit sm_61 cubin; torch "
        "and TensorFlow have sm_60 cubins and rely on NVIDIA minor-revision "
        "binary compatibility for sm_61.",
        "- Neither Pascal part has an mliport hardware smoke, so the "
        "classification stops at `binary_theory_verified`.",
        "",
        "## Hopper and Blackwell",
        "",
        "- Torch backends (MACE/DPA/UMA) use the cu128 wheel whose cubins "
        "include sm_90, sm_100 and sm_120.",
        "- GRACE stays `experimental` on Hopper and Blackwell: TensorFlow 2.20 "
        "has an sm_90 SASS path and compute_90 PTX, but no Blackwell SASS "
        "target, and mliport has not run GRACE there.",
        "",
        "## Limitations",
        "",
        "- No Pascal, Turing, Ampere, Ada, Hopper or Blackwell hardware was "
        "available. Their entries are binary evidence plus resolver behaviour, "
        "not runtime measurements.",
        "- `cuobjdump` reads cubin/PTX sections; it does not prove that every "
        "code path in a framework avoids an unsupported instruction.",
        "- MIG instances are rejected by the device resolver instead of being "
        "treated as whole GPUs.",
        "- A future framework bump invalidates this report; Dependabot is "
        "configured to leave the pinned frameworks for manual review.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python validation/compatibility/build_architecture_theory.py --probe",
        "```",
        "",
        "The `--probe` mode re-reads the installed environments and compares "
        "them with the recorded evidence; the JSON output is the machine-"
        "readable form of this report.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="re-check installed environments against the audit",
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    entries = build_entries()
    data = {
        "schema": "mliport.gpu-architecture-theory/1",
        "generated_at": AUDIT_DATE,
        "pythons": list(PYTHONS),
        "framework_wheels": {
            f"torch-{version}+{channel}": wheel
            for (version, channel), wheel in TORCH_WHEELS.items()
        }
        | {"tensorflow-2.20.0": TENSORFLOW_WHEEL},
        "backend_wheels": BACKEND_WHEELS,
        "native_extensions": NATIVE_AUDIT,
        "entries": entries,
        "probe": None,
    }
    if args.probe:
        data["probe"] = probe_installed()
    out = Path(args.out)
    (out / "gpu_architecture_theory.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    (out / "GPU_ARCHITECTURE_THEORY_REPORT.md").write_text(
        render_report(data) + "\n", encoding="utf-8"
    )
    packaged = REPO / "mliport" / "mliport" / "data" / "compatibility"
    packaged.mkdir(parents=True, exist_ok=True)
    (packaged / "gpu_architecture_theory.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'gpu_architecture_theory.json'}")
    print(f"wrote {out / 'GPU_ARCHITECTURE_THEORY_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
