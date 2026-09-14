# mliport GPU architecture theory audit

Audit date: 2026-09-14. Scope: binary-level compatibility evidence for the pinned frameworks and backend packages, without pretending that non-Volta hardware was tested.

## What this report is

Every entry answers two separate questions:

1. does the exact pinned wheel/native extension contain kernels for that compute capability, and
2. has mliport actually run that backend on real hardware of that family?

Only the V100/Volta row answers yes to the second question; the other families are classified from binary evidence and the dependency resolver.

## Classification levels

| Level | Meaning |
|---|---|
| `hardware_verified` | mliport ran the backend on real GPU hardware of this family (V100 acceptance). |
| `binary_theory_verified` | Exact wheel/native cubes cover every requested compute capability, but no hardware smoke exists. |
| `resolver_only` | The installer resolves a wheel, but the binary evidence does not cover this capability. |
| `experimental` | Upstream does not declare support, or the framework target is PTX-only for this family. |
| `unsupported` | No route is planned. |

## External facts

- PyTorch stops publishing CUDA 12.6 wheels from 2.15, which drops Maxwell, Pascal and Volta binary support; CUDA 13.x dropped compute capabilities below Turing. See [the PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) and [the 2.15 support-matrix RFC](https://github.com/pytorch/pytorch/issues/190385).
- mliport therefore pins legacy architectures to torch 2.8.0/2.10.0 +cu126 and guards the mapping in `_torch_modern_cuda`; an unknown framework version fails closed until a new audit is written.
- TensorFlow 2.20.0 is built with CUDA 12.5.1 and cuDNN 9; its declared capabilities are sm_60, sm_70, sm_80, sm_89, compute_90. Hopper has an sm_90 SASS path plus compute_90 PTX; Blackwell has no explicit SASS target.
- The acceptance host runs driver 580.173.02, which is newer than the CUDA 12.x runtime requirements of every pinned wheel (CUDA minor version compatibility).

## Framework wheel evidence

| Package | Version | Channel | Wheel | SHA-256 | Compiled arches |
|---|---|---|---|---|---|
| torch | 2.10.0 | cu126 | `torch-2.10.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl` | `2a7a569206f07965…` | sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 |
| torch | 2.10.0 | cu128 | `torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl` | `628e89bd5110ced7…` | sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 |
| torch | 2.8.0 | cu126 | `torch-2.8.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl` | `ce6e6a1f4803ad62…` | sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 |
| torch | 2.8.0 | cu128 | `torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl` | `4354fc05bb79b208…` | sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 |
| tensorflow | 2.20.0 | CUDA 12.5/cuDNN 9 | `tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `25265b0bc527e0d5…` | sm_60, sm_70, sm_80, sm_89, sm_90 (+compute_90 PTX) |

Backend packages: `mace-torch`/`fairchem-core`/`tensorpotential` are pure Python wheels; `deepmd-kit` carries the CUDA op described below.

## Native extension audit

| Backend | Distributions | Bundled device code | Compiled arches |
|---|---|---|---|
| MACE | mace-torch, e3nn | none | none |
| UMA | fairchem-core, e3nn | none | none |
| GRACE | tensorpotential | none | none |
| DPA | deepmd-kit | libdeepmd_op_cuda.so | sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100, sm_101, sm_120 |

`deepmd-kit`'s `libdeepmd_op_cuda.so` was inspected with `cuobjdump --list-elf`; it contains explicit cubins from sm_50 through sm_120, including sm_60 and sm_61 for Pascal. `libdeepmd_op_pt.so` is the host-side torch extension and links that CUDA op. MACE, UMA and GRACE inherit their framework's kernels because their packages ship no device code.

## Classification matrix

| Architecture | Backend | Framework | Classification | Real hardware |
|---|---|---|---|---|
| Maxwell | MACE | torch 2.8.0+cu126 | `experimental` | no |
| Maxwell | DPA | torch 2.10.0+cu126 | `experimental` | no |
| Maxwell | GRACE | tensorflow 2.20.0 | `experimental` | no |
| Maxwell | UMA | torch 2.8.0+cu126 | `experimental` | no |
| Pascal P100 (sm60) | MACE | torch 2.8.0+cu126 | `binary_theory_verified` | no |
| Pascal P100 (sm60) | DPA | torch 2.10.0+cu126 | `binary_theory_verified` | no |
| Pascal P100 (sm60) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Pascal P100 (sm60) | UMA | torch 2.8.0+cu126 | `binary_theory_verified` | no |
| Pascal P40 (sm61) | MACE | torch 2.8.0+cu126 | `binary_theory_verified` | no |
| Pascal P40 (sm61) | DPA | torch 2.10.0+cu126 | `binary_theory_verified` | no |
| Pascal P40 (sm61) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Pascal P40 (sm61) | UMA | torch 2.8.0+cu126 | `binary_theory_verified` | no |
| Volta V100 (sm70) | MACE | torch 2.8.0+cu126 | `hardware_verified` | yes |
| Volta V100 (sm70) | DPA | torch 2.10.0+cu126 | `hardware_verified` | yes |
| Volta V100 (sm70) | GRACE | tensorflow 2.20.0 | `hardware_verified` | yes |
| Volta V100 (sm70) | UMA | torch 2.8.0+cu126 | `hardware_verified` | yes |
| Turing (sm75) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Turing (sm75) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Turing (sm75) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Turing (sm75) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm80) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm80) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm80) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Ampere (sm80) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm86) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm86) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Ampere (sm86) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Ampere (sm86) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ada Lovelace (sm89) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Ada Lovelace (sm89) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Ada Lovelace (sm89) | GRACE | tensorflow 2.20.0 | `binary_theory_verified` | no |
| Ada Lovelace (sm89) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Hopper (sm90) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Hopper (sm90) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Hopper (sm90) | GRACE | tensorflow 2.20.0 | `experimental` | no |
| Hopper (sm90) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Blackwell (sm100/sm120) | MACE | torch 2.8.0+cu128 | `binary_theory_verified` | no |
| Blackwell (sm100/sm120) | DPA | torch 2.10.0+cu128 | `binary_theory_verified` | no |
| Blackwell (sm100/sm120) | GRACE | tensorflow 2.20.0 | `experimental` | no |
| Blackwell (sm100/sm120) | UMA | torch 2.8.0+cu128 | `binary_theory_verified` | no |

## Pascal: P100 and P40 are not the same claim

- P100 is sm_60: torch cu126, TensorFlow 2.20 and the DPA CUDA op all contain sm_60 cubins.
- P40 is sm_61: the DPA CUDA op contains an explicit sm_61 cubin; torch and TensorFlow have sm_60 cubins and rely on NVIDIA minor-revision binary compatibility for sm_61.
- Neither Pascal part has an mliport hardware smoke, so the classification stops at `binary_theory_verified`.

## Hopper and Blackwell

- Torch backends (MACE/DPA/UMA) use the cu128 wheel whose cubins include sm_90, sm_100 and sm_120.
- GRACE stays `experimental` on Hopper and Blackwell: TensorFlow 2.20 has an sm_90 SASS path and compute_90 PTX, but no Blackwell SASS target, and mliport has not run GRACE there.

## Limitations

- No Pascal, Turing, Ampere, Ada, Hopper or Blackwell hardware was available. Their entries are binary evidence plus resolver behaviour, not runtime measurements.
- `cuobjdump` reads cubin/PTX sections; it does not prove that every code path in a framework avoids an unsupported instruction.
- MIG instances are rejected by the device resolver instead of being treated as whole GPUs.
- A future framework bump invalidates this report; Dependabot is configured to leave the pinned frameworks for manual review.

## Reproduction

```bash
python validation/compatibility/build_architecture_theory.py --probe
```

The `--probe` mode re-reads the installed environments and compares them with the recorded evidence; the JSON output is the machine-readable form of this report.

