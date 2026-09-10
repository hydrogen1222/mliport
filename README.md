# mlipx — MLIP eXtended

**A VASP-style CLI / TUI / Python API for machine-learning interatomic potentials (MLIPs).**

mlipx wraps four MLIP engines behind one unified interface — **UMA (FAIRChem)** (default), **MACE**, **DPA (DeepMD-kit)**, and **GRACE** — and provides single-point (SP), geometry optimization (OPT), molecular dynamics (MD), batch processing, and validated trajectory analysis with VASP-compatible outputs (OUTCAR, CONTCAR, XDATCAR, OSZICAR).

```
structure.cif ──▶  MLIP engine (UMA/MACE/DPA/GRACE)  ──▶  energy, forces, stress
```

---

## Supported Engines

| `--model-type` | Engine | Backend package | Tasks |
|---|---|---|---|
| `uma` (default, alias `fairchem`) | UMA — FAIRChem | `fairchem-core` | `omat` / `omol` / `oc20` / `oc25` / `odac` / `omc` |
| `mace` | MACE | `mace-torch` | `bulk` / `molecule` |
| `dpa` | DPA — DeepMD-kit | `deepmd-kit` | `bulk` / `molecule` |
| `grace` | GRACE | `tensorpotential` | `bulk` / `molecule` |

Each engine runs in its **own isolated Python environment** because their dependencies conflict (UMA needs `e3nn>=0.5`; MACE pins `e3nn==0.4.4`; DPA pins `torch==2.10`; GRACE uses TensorFlow). Do **not** install all four into one environment.

---

## Installation

### One-command installer (recommended)

```bash
# Clone and install uv (skip if uv already works)
git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx
curl -LsSf https://astral.sh/uv/install.sh | sh

# Auto-detect GPU and install all four engines
./scripts/install_mlipx.sh
```

Common variants:

```bash
./scripts/install_mlipx.sh --device cpu        # CPU-only machine
./scripts/install_mlipx.sh --engines uma,mace  # only UMA + MACE
./scripts/install_mlipx.sh --source china      # numbered China-mirror menu
./scripts/install_mlipx.sh --source china-ustc --non-interactive  # fixed source
./scripts/install_mlipx.sh --clean             # rebuild every venv
./scripts/install_mlipx.sh --dry-run           # preview without installing
```

If an earlier installation stopped partway, rerun it with `--clean` so the
partially populated engine environments are rebuilt before verification.
GPU installations are large: reserve tens of GiB for all four isolated
environments. GRACE alone typically needs about 6–7 GiB after installation
and additional temporary space while CUDA wheels are downloaded and extracted.

Run `./scripts/install_mlipx.sh --help` for all options.

### Manual installation (four environments)

If you prefer to install by hand, create one venv per engine:

```bash
# UMA (default) — installed explicitly like every other engine
uv venv --python 3.12 .venv
uv pip install --no-config --python .venv/bin/python \
  "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv/bin/python \
  -e ./mlipx "fairchem-core==2.21.0"
.venv/bin/mlipx doctor --engine uma --device auto

# MACE
uv venv --python 3.12 .venv-mace
uv pip install --no-config --python .venv-mace/bin/python \
  "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-mace/bin/python \
  -e ./mlipx "e3nn==0.4.4" "mace-torch==0.3.16"
.venv-mace/bin/mlipx doctor --engine mace --device auto

# DPA / DeepMD
uv venv --python 3.12 .venv-dpa
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-dpa/bin/python \
  -e ./mlipx "deepmd-kit==3.1.3"
.venv-dpa/bin/mlipx doctor --engine dpa --device auto

# GRACE
uv venv --python 3.12 .venv-grace
uv pip install --no-config --python .venv-grace/bin/python \
  -e ./mlipx "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"
.venv-grace/bin/mlipx doctor --engine grace --device auto
```

Use the matching command prefix:

| Engine | Environment | Prefix |
|---|---|---|
| UMA | `.venv` | `.venv/bin/mlipx ...` |
| MACE | `.venv-mace` | `.venv-mace/bin/mlipx ...` |
| DPA | `.venv-dpa` | `.venv-dpa/bin/mlipx ...` |
| GRACE | `.venv-grace` | `.venv-grace/bin/mlipx ...` |

### GPU compatibility

The installer and `mlipx setup` choose the correct PyTorch/CUDA wheel automatically.

| GPU family | Examples | Compute capability | CUDA route |
|---|---|---|---|
| Maxwell | GTX 960, TITAN X | sm_50/52 | cu126 Legacy (⚠️ experimental) |
| Pascal | **Tesla P40**, GTX 1080 Ti, P100 | sm_60/61 | cu126 Legacy |
| Volta | **V100** | sm_70 | cu126 Legacy |
| Turing | RTX 20xx | sm_75 | cu128+ Modern |
| Ampere | **RTX 3080 Ti**, 30xx | sm_80/86 | cu128+ Modern |
| Ada | **RTX 4090**, 40xx | sm_89 | cu128+ Modern |
| Hopper | H100 | sm_90 | cu128+ Modern |
| Blackwell | RTX 50xx | sm_100/120 | cu128+ Modern |
| none | CPU only | — | CPU wheels |

> **Why two CUDA routes?** Maxwell/Pascal/Volta must use the **cu126 Legacy** channel: PyTorch 2.8+ removed Maxwell/Pascal from cu128 builds, and PyTorch 2.11+ removed Volta from cu128+. Turing+ use the **modern** channel (cu128 for torch 2.8–2.10, cu130 for torch 2.12+). Maxwell is Experimental because TensorFlow 2.20 official wheels start at sm_60.

**Per-engine install verification status** (from `mlipx/install/compatibility.py`; V100 and RTX 4090 have been tested on real hardware. P40 uses the corrected exact `+cu126` wheel pin but still needs a post-fix model smoke retest). "Verified" here means the engine installed and produced a real model prediction on that GPU family — not that every workload was exercised. Workload-level statuses for V100 are listed below.

| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ |
|---|---|---|---|---|---|
| UMA | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| MACE | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| DPA | experimental | needs smoke test | **verified** | **verified** | needs smoke test |
| GRACE | experimental | needs smoke test | **verified** | **verified** | needs smoke test |

**Workload-level verification (single V100-SXM2-16GB, sm_70, driver 580.173.02)** — short real-model runs with hard timeouts on one GPU. Sanitized records live in [`validation/runtime/v100/`](validation/runtime/v100/) (the GPU is identified only by a SHA-256 of its UUID); each status maps to exactly one record:

| Backend / model | SP | Short MD | E–F gradient | NEB smoke | GRACE cache | Records |
|---|---|---|---|---|---|---|
| UMA | not run | not run | not run | not run | n/a | [uma.json](validation/runtime/v100/uma.json) (no offline wheelhouse on this machine) |
| MACE float64 | passed | passed | passed | passed | n/a | [mace.json](validation/runtime/v100/mace.json) |
| MACE float32 | passed | passed | passed | passed | n/a | [mace-float32.json](validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | passed | passed | passed | passed | n/a | [dpa.json](validation/runtime/v100/dpa.json) |
| GRACE float32 | passed | passed | passed | passed | passed (ON and OFF) | [grace.json](validation/runtime/v100/grace.json), [grace-nocache.json](validation/runtime/v100/grace-nocache.json) |

The NEB smoke is a short 5-image fixed-cell run on a 4-atom Cu cell with `saddle_validation: not_performed` — see [NEB](#nudged-elastic-band-neb) for what that implies.

### Download sources

PyPI packages and PyTorch wheels are handled separately. The installer never
modifies your global `~/.config/uv/uv.toml`; it uses `UV_NO_CONFIG=1` and
per-process variables. In a real terminal, `--source china` presents a numbered
choice of TUNA, Aliyun, USTC, Tencent Cloud, or the official source. China
profiles all use the Aliyun PyTorch-wheel mirror, and Enter defaults to TUNA.
CI and piped input never prompt and retain TUNA as the deterministic default;
`--non-interactive` makes that behavior explicit.
Offline mode also requires the requested Python to be already discoverable by
`uv`; the bootstrap will not silently download an interpreter.

| `--source` | PyPI | PyTorch wheels | Use when |
|---|---|---|---|
| `auto` → `official` | pypi.org | download.pytorch.org | Default |
| `official` | pypi.org | download.pytorch.org | Explicit official source |
| `china` | Interactive; TUNA by default | mirrors.aliyun.com (`--find-links`) | Interactive use in China |
| `china-aliyun` | mirrors.aliyun.com | mirrors.aliyun.com | Fixed Aliyun source |
| `china-ustc` | mirrors.ustc.edu.cn | mirrors.aliyun.com | Fixed USTC source |
| `china-tencent` | mirrors.cloud.tencent.com | mirrors.aliyun.com | Fixed Tencent source |
| `offline` | cached only | cached only | Air-gapped machine with Python present |
| `custom` | your env vars | your env vars | Advanced |

---

## Quick Start

### Single-point energy (UMA)

```bash
.venv/bin/mlipx sp structure.cif --model uma-s-1.pt --task omat --device cpu
```

### Geometry optimization

```bash
.venv/bin/mlipx opt structure.cif --model uma-s-1.pt --task omat \
  --cell-opt --fmax 0.02
```

### Molecular dynamics

```bash
.venv/bin/mlipx md structure.cif --model uma-s-1.pt --task omat \
  --device cuda --steps 10000
```

### Nudged elastic band (NEB)

```bash
.venv/bin/mlipx neb --initial initial.vasp --final final.vasp \
  --model uma-s-1.pt --task omat --device cuda \
  --output results/hop --images 7 --climb --fmax 0.03
```

Engines without a matching validation record refuse to start a NEB run; pass
`--allow-unvalidated-neb` to override explicitly. Resume from the latest
complete-band checkpoint with `--resume results/hop/checkpoints/latest` — the
model and all scientific options are restored from the checkpoint (see
[Resume semantics](#resume-semantics)).

### Other engines

```bash
# MACE
.venv-mace/bin/mlipx sp bulk.cif --model mace.model \
  --model-type mace --task bulk --head default --device cuda:0

# DPA / DeepMD
.venv-dpa/bin/mlipx opt bulk.cif --model dpa.pt \
  --model-type dpa --task bulk --head Domains_SSE_PBE \
  --device cuda:0 --fmax 0.05

# GRACE (--model points to a SavedModel directory)
.venv-grace/bin/mlipx sp bulk.cif --model grace_model/ \
  --model-type grace --task bulk --device cuda:0 \
  --gpu-memory-limit-mb 6144
```

### INCAR files (VASP-style)

```bash
.venv/bin/mlipx template sp          # generate INCAR.sp
.venv/bin/mlipx run -i INCAR.sp -s structure.cif
```

Example `INCAR.sp`:

```ini
CALC_TYPE   = SP
MODEL_TYPE  = UMA        # or MACE / DPA / GRACE
MODEL_PATH  = uma-s-1.pt
TASK        = omat       # UMA: omat/omol/...; others: bulk/molecule
DEVICE      = cpu
```

NEB uses the same mechanism — `mlipx template neb` writes `INCAR.neb`
(`CALCULATION = NEB`) with endpoints, path, endpoint-policy, and convergence
keywords:

```ini
CALCULATION          = NEB
MODEL_TYPE           = UMA
MODEL_PATH           = uma-s-1.pt
NEB_INITIAL          = initial.vasp
NEB_FINAL            = final.vasp
NEB_IMAGES           = 7
NEB_INTERPOLATION    = linear   # or idpp
NEB_CLIMB            = .FALSE.  # .TRUE. enables CI-NEB
FMAX                 = 0.03
```

### Batch

```bash
.venv/bin/mlipx batch structures/ --model uma-s-1.pt \
  --model-type uma --task omat --device cuda \
  --calc-type sp --pattern "*.cif" --output batch_results
```

Each input gets its own output subdirectory; the root gets `batch_summary.json`.

---

## Trajectory Analysis

`mlipx analyze` is calculator-independent: you can analyze a trajectory produced by any engine from the UMA `.venv` without loading the model backend.

```bash
# Validate first (checks time axis, PBC, conventions, eligibility)
.venv/bin/mlipx analyze results/LGPS-800K validate

# Thermodynamics
.venv/bin/mlipx analyze results/LGPS-800K thermo

# RDF / coordination
.venv/bin/mlipx analyze results/LGPS-800K rdf \
  --center Li --neighbor S --rmax 6 --cn-cutoff 3

# MSD
.venv/bin/mlipx analyze results/LGPS-800K msd \
  --mobile Li --axes x,y,z,xyz --drift-reference nonmobile

# Density / VACF / spectrum / Arrhenius
.venv/bin/mlipx analyze results/LGPS-800K density --mobile Li --spacing 0.25
.venv/bin/mlipx analyze results/LGPS-800K vacf --species Li
.venv/bin/mlipx analyze results/LGPS-800K spectrum --species Li --taper one-sided-cosine

# Fit multi-temperature Arrhenius from independent transport results
# (each value is passed as a separate repeated flag)
.venv/bin/mlipx analyze RUN arrhenius \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 1e-10 --diffusivity 2e-10 --diffusivity 5e-10 \
  --diffusivity-std 0.1e-10 --diffusivity-std 0.2e-10 --diffusivity-std 0.5e-10
```

### Analysis hierarchy: MSD, transport, and electrolyte

The three commands have deliberately different scientific roles:

- `msd` is a diagnostic view: multiple-time-origin MSD, local `alpha`, and
  native OLS fit-window diagnostics. Its OLS result is explicitly
  `publication_grade=false` and is not the transport authority.
- `transport` is the quantitative authority. Kinisi provides tracer
  `D_tracer`; mlipx derives
  `sigma_NE_tracer = n (z e)^2 D_tracer/(k_B T)`. With
  `--collective-conductivity`, kinisi MSCD includes distinct charge
  correlations and provides `sigma_collective`, from which
  `D_sigma = sigma_collective k_B T/(n (z e)^2)` is derived. The reported
  Haven ratio is explicitly `H_R = D_tracer/D_sigma = sigma_NE_tracer/
  sigma_collective`; the correlation factor is
  `sigma_collective/sigma_NE_tracer`. `--jump-diffusion` adds MSTD and `D_J`
  as a total/jump-displacement diagnostic, not tracer diffusion.
- `electrolyte` is GEMDAT mechanism analysis: density, sites, occupancy,
  transitions, residence, jumps, collective events, and percolating paths.
  GEMDAT endpoint/COM diffusivities and Haven ratios are retained only under
  `diagnostic_crosscheck` (`publication_transport_authority=false`).

Covariance-aware transport uses [kinisi 2.x](https://joss.theoj.org/papers/10.21105/joss.05984). It requires an explicit fit start:

```bash
.venv/bin/mlipx analyze RUN transport --mobile Li --charge 1 \
  --drift-reference nonmobile --fit-start-ps 40 \
  --lag-step-ps 2 --lag-stop-ps 200 --random-seed 0

# LGPS: tracer + collective Einstein conductivity
mlipx analyze RUN transport \
  --mobile Li --charge 1 --drift-reference nonmobile \
  --fit-start-ps 40 --lag-step-ps 2 --lag-stop-ps 200 \
  --collective-conductivity --random-seed 0
```

Key scientific rules:

- **Never guess.** `--charge` is required; temperature comes from the run (or `--temperature-K` for external trajectories); `wrapped`/`unwrapped` must describe the actual file.
- **Fixed-cell only.** Variable-cell transport is unsupported.
- **Drift correction is explicit:** `none`, `nonmobile`, or `indices`.
- **`--lag-step-ps` / `--lag-stop-ps`** sparsify kinisi's lag-time grid, not the trajectory frames. They must be used together.
- **Native MSD OLS diagnostics** require both `--fit-start-ps` and `--fit-stop-ps`; mlipx does not auto-detect a publication fit window. Transport uses its explicit kinisi `--fit-start-ps` and selected lag grid.
- **Nernst–Einstein tracer conductivity** (`sigma_NE_tracer`) is reported with posterior mean / SD / 95% CI. It is not a total physical uncertainty and not automatically equal to experimental/collective conductivity.
- `sigma_NE_tracer` ignores distinct ion correlations. `sigma_collective` is a
  collective Einstein ionic conductivity within the analyzed classical MD
  trajectory and selected charge model; it is not automatically an
  experimental bulk/polycrystalline conductivity.
- Kinisi credible intervals are conditional posterior intervals, not
  model/finite-size/replica uncertainty. `--collective-system-particles`
  selects index-ordered kinisi statistical groups, not independent replicas.
  Haven ratio intervals, when available, are labelled independent-marginal
  approximations because tracer/collective covariance is not modeled.
- A safely reconstructed 0.1 ps saved interval can be valid for long-time
  Einstein/MSCD analysis. It is not dense sampling for short-time VACF or
  Green–Kubo analysis.

### Electrolyte mechanisms (optional GEMDAT)

GEMDAT is an optional backend for site mapping, jumps, and percolation:

```bash
python -m pip install -e './mlipx[analysis,electrolyte]'
.venv/bin/mlipx analyze RUN electrolyte --mobile Li --sites Li_sites.cif \
  --jump-dimensions 3 --percolation-axes xyz
```

A site source is mandatory (`--sites` or `--discover-sites-from-density`). GEMDAT endpoint diffusivity is never promoted over the kinisi estimate.
Explicit CIF sites are the reproducible pathway; density segmentation is
exploratory and records its resolution/background/peak metadata. GEMDAT
free-energy path barriers are finite-temperature occupancy-derived quantities,
not NEB potential-energy migration barriers.

### Outputs & reproducibility

Every analysis task writes under `RUN/analysis/TASK/REQUEST_HASH/`:

```
request.json
provenance.json
results.json
task-specific CSV/NPZ
PNG and SVG when applicable
diagnostics.json when applicable
```

Transport additionally writes `transport_summary.csv`, compressed
`kinisi_arrays.npz`, and `transport_msd`; collective and jump analyses add
`transport_mscd` and `transport_mstd`. Electrolyte writes compressed
`electrolyte_arrays.npz`, summary/transition/jump/percolation CSVs, CIF site
artifacts, and headless density/free-energy/mechanism plots. The transport
and electrolyte scientific revisions are part of the cache key, so old
results are not silently reused after these semantic changes.

The request hash includes the source fingerprint, selection, range, axes, drift definition, scientific parameters, and backend versions. Identical requests are reused unless `--force` is supplied.

---

## Interfaces

| Interface | Command | Best for |
|---|---|---|
| CLI | `mlipx sp/opt/md/batch/...` | Scripts, HPC, automation |
| TUI | `mlipx tui` | Interactive exploration |
| Python API | `from mlipx.api import run_single_point, ...` | Custom workflows |
| INCAR | `mlipx run -i INCAR` | VASP-style config |

### Python API

```python
from mlipx.api import run_single_point, run_md, run_neb, calculate_energy

result = run_single_point("structure.cif", "uma-s-1.pt", task="omat")
energy = calculate_energy("structure.cif", "uma-s-1.pt", task="omat")

neb = run_neb(
    "initial.vasp", "final.vasp", "uma-s-1.pt",
    task="omat", device="cuda:0", output_dir="results/hop",
    n_intermediate_images=7, climb=True,
)
print(neb["converged"], neb["barrier_forward_sampled_eV"])
```

---

## INCAR Configuration

| Category | Key | Default |
|---|---|---|
| Calculation | `CALC_TYPE` | — (`SP` / `OPT` / `MD`) |
| Model | `MODEL_TYPE` | `UMA` |
| Model | `MODEL_PATH` | — |
| Model | `TASK` | `omat` (UMA) / `bulk` (others) |
| Model | `DEVICE` | `cpu` |
| Model | `HEAD` | — (MACE/DPA multi-task) |
| Model | `DTYPE` | `float64` (MACE) |
| Output | `WRITE_OUTCAR` | `.TRUE.` |
| Output | `WRITE_XDATCAR` | `.TRUE.` |
| Output | `WRITE_TRAJECTORY` | `.TRUE.` |
| Output | `WRITE_JSON` | `.TRUE.` |
| OPT | `FMAX` | `0.05` |
| OPT | `MAX_STEPS` | `500` |
| OPT | `OPT_ALGO` | `FIRE` |
| OPT | `CELL_OPT` | `.FALSE.` |
| MD | `MD_ENSEMBLE` | `NVT` |
| MD | `TEMPERATURE` | `300` |
| MD | `TIMESTEP` | `1.0` |
| MD | `STEPS` | `1000` |
| MD | `THERMOSTAT` | `LANGEVIN` |
| MD | `SAVE_INTERVAL` | `10` |
| NEB | `NEB_IMAGES` | `7` |
| NEB | `NEB_CLIMB` | `.FALSE.` |
| NEB | `NEB_METHOD` | `improvedtangent` |
| NEB | `NEB_INTERPOLATION` | `linear` |
| NEB | `NEB_PATH_CONVENTION` | `mic` |
| NEB | `NEB_SPRING` | `0.1` |
| NEB | `FMAX` | `0.03` (NEB scope) |
| NEB | `MAX_STEPS` | `1000` (NEB scope) |
| NEB | `NEB_PRE_FMAX` / `NEB_PRE_MAX_STEPS` | `0.1` / `300` |
| NEB | `NEB_MAXSTEP` | `0.1` |
| NEB | `NEB_ENDPOINT_POLICY` | `validate` |
| NEB | `NEB_ENDPOINT_FMAX` / `NEB_ENDPOINT_STEPS` | `0.02` / `500` |
| NEB | `NEB_CHECKPOINT_INTERVAL` | `10` |
| NEB | `NEB_MIN_DISTANCE` | `0.5` |

See the generated templates (`mlipx template sp/opt/md/neb`) for the full keyword list with comments.

---

## Output Files

Each calculation writes a self-contained output directory:

```
OUTPUT/
├── OUTCAR                 VASP-like text output
├── CONTCAR                final structure
├── XDATCAR                trajectory in VASP layout (if enabled)
├── mlipx_results.json     machine-readable results
├── raw/
│   ├── trajectory.traj    canonical ASE trajectory
│   └── md.csv             MD time series (if MD)
└── artifacts.json         provenance / versions / semantics
```

For high-throughput runs, use `--no-write-outcar --no-write-xdatcar` to skip the VASP interoperability text I/O; the canonical trajectory remains enabled.

## Nudged Elastic Band (NEB)

`mlipx neb` runs a fixed-cell NEB/CI-NEB (`improvedtangent` spring method)
between two endpoints. Endpoints are validated or relaxed first
(`NEB_ENDPOINT_POLICY`), the initial band comes from linear or IDPP
interpolation, and complete bands are checkpointed every `NEB_CHECKPOINT_INTERVAL`
steps for crash-safe resume.

### Output layout

```
results/hop/
├── run_context.json     run/attempt IDs, model identity, resolved options
├── artifacts.json       status + pointers (checkpoint, result, VASP export)
├── checkpoints/
│   ├── step_000003/     complete band: band.traj + checkpoint.json
│   └── latest           symlink to the newest complete checkpoint
├── neb_results.json     mlipx.neb-results/2: converged, sampled barriers,
│                        reaction energy, highest-energy image, max NEB force
└── vasp_path/           00/POSCAR ... NN/POSCAR export, no MLIP energies
                         (vasp_path_<attempt>/ after a resume)
```

### Resume semantics

```text
geometry resume ≠ strict FIRE optimizer state restart
```

`--resume results/hop/checkpoints/latest` restores the band geometry, the model
identity, and every scientific option from the checkpoint, reuses the original
run ID and output directory, and starts a new attempt. It is a geometry-level
restart — not a bit-exact optimizer-state reload — so results depend only on
the restored geometry and options. On resume, endpoints, atom map, and image
shifts are ignored (the checkpoint's original path identity is authoritative).

### Scientific semantics

- The climbing image is a **candidate** saddle: `converged: true` means the NEB
  force criterion was met — a Hessian-validated first-order saddle point was
  *not* verified (`saddle_validation: not_performed`).
- MLIP barriers are model barriers, not VASP/DFT barriers; compare like with
  like.
- Absolute energies from different model tasks, heads, or reference levels must
  never be mixed — including between the endpoints and the images of one path.
- Engines without a matching validation record are refused for NEB by default;
  `--allow-unvalidated-neb` overrides explicitly (fail-closed elsewhere).

---

## Background Jobs & Queue

Background jobs are submitted through the queue JSON interface:

```bash
# 1. Describe tasks in JSON
cat > tasks.json <<'JSON'
{
  "max_concurrent": 1,
  "tasks": [
    {
      "name": "opt-uma-1",
      "calc_type": "opt",
      "structure": "/path/a.cif",
      "model": "/path/uma-s-1.pt",
      "model_type": "uma",
      "device": "cuda:0",
      "options": {"fmax": 0.05}
    }
  ]
}
JSON

# 2. Submit and start
.venv/bin/mlipx queue submit tasks.json
.venv/bin/mlipx queue start            # background scheduler
.venv/bin/mlipx queue status

# 3. Manage
.venv/bin/mlipx jobs                   # list running/done/failed
.venv/bin/mlipx kill <job-id>          # terminate a running job
.venv/bin/mlipx clean                  # remove completed/failed records
```

The TUI also has built-in queue controls. Each task may use its own Python environment/engine/model.

---

## Resource Control

| Option | Effect |
|---|---|
| `--cpu-threads N` | CPU intra-op threads (PyTorch for UMA/MACE/DPA; TF for GRACE) |
| `--gpu-memory-growth` | GRACE: grow TF GPU memory on demand (default enabled) |
| `--gpu-memory-limit-mb MIB` | GRACE: hard limit on TF GPU memory |
| `--inference-mode turbo` | UMA only: fast inference preset |
| `--activation-checkpointing` | UMA only: save GPU memory |
| `--dtype float32` | MACE: opt in to float32 for speed (default float64) |

---

## Troubleshooting

### "no kernel image is available for execution on the device"

Your PyTorch build has no kernel for your GPU. Use the cu126 Legacy channel for Maxwell/Pascal/Volta, or the modern channel for Turing+. Run:

```bash
.venv/bin/mlipx setup     # machine-specific report
./scripts/install_mlipx.sh --dry-run
```

### "No edges found in structure"

Atoms are too far apart (> cutoff), the cell is invalid, or PBC is wrong. Check the input structure and use the correct `--task` (periodic `omat`/`bulk` vs molecular `omol`/`molecule`).

### CUDA out of memory

Use `--device cpu`, a smaller model, or UMA `--activation-checkpointing`. For GRACE set `--gpu-memory-limit-mb`.

### Installer reports `No space left on device`

Inspect the repository filesystem and uv cache before retrying:

```bash
df -h . "$(uv cache dir)"
du -sh "$(uv cache dir)" .venv* 2>/dev/null
uv cache clean
```

For an interrupted GRACE install, keep the already verified UMA/MACE/DPA
environments and rebuild only GRACE. Aim for at least 10–12 GiB free during
the installation; exact usage depends on wheel versions and uv's link mode.

```bash
./scripts/install_mlipx.sh --source china --engines grace --clean
```

If the uv cache is on a small root filesystem but another filesystem has more
space, set `UV_CACHE_DIR=/larger/path/uv-cache` for the retry. The repository
filesystem still needs room for the final `.venv-grace` environment.

### MACE environment incompatible

MACE must not share the UMA environment. Use `.venv-mace/bin/mlipx ...` (the installer creates it automatically).

### Atom explosion in MD

Pre-relaxation is on by default for NVT (up to 50 FIRE steps). For NVE it is off; enable it if needed.

---

## Development

Use a dedicated dev venv so it never collides with the UMA runtime `.venv`:

```bash
# 1. Create a dev environment
uv venv --python 3.12 .venv-dev

# 2. Install mlipx with dev + analysis extras
uv pip install --python .venv-dev/bin/python -e './mlipx[dev,analysis]'

# 3. Run tests (no heavy ML backend required — backend tests are mocked/skipped)
.venv-dev/bin/python -m pytest tests -q
```

UMA is consumed through the external `fairchem-core` dependency. Core code
lives in `mlipx/mlipx/`; installation/compatibility logic is in
`mlipx/mlipx/install/`.

Analysis extras (optional): `./mlipx[analysis]` (scipy/matplotlib),
`./mlipx[transport]` (kinisi), `./mlipx[electrolyte]` (gemdat), or
`./mlipx[analysis-all]` for all three.


## Versioning & revisions

Provenance does not hang off the package version alone. Four independent
revision axes are recorded in outputs, results, and checkpoints:

- Package version (SemVer, e.g. `2.0.0`), recorded as `mlipx_version`.
- Result schema revisions (`mlipx.neb-results/2`, `mlipx.runtime-validation/1`,
  ...).
- NEB checkpoint schema revision (`mlipx.neb-checkpoint/1`).
- NEB scientific revision — bumped for changes that alter path preparation,
  force, or convergence semantics (see `mlipx/mlipx/neb/revisions.py`).

The NEB resume fingerprint pins all of these together (plus model identity and
path identity). A mismatch in any axis fails resume closed rather than
silently resuming with changed semantics.
---

## License

MIT License. mlipx builds on [FAIRChem](https://github.com/FAIR-Chem/fairchem) (Copyright © Meta Platforms, Inc. and affiliates), licensed under MIT. See [`LICENSE.md`](LICENSE.md) and [`mlipx/LICENSE`](mlipx/LICENSE).
