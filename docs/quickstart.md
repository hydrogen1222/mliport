# Quickstart (5-10 minutes)

This path installs **one** backend and runs a first single point plus a short
relaxation. MACE is used as the first example because its checkpoints are open
and light; equivalent commands for the other backends are listed at the end.
You do not need all four backends to start.

## 1. Clone and install one backend

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

The installer creates the backend's own environment, installs mliport inside
it, writes a launcher and runs `mliport doctor` unless `--skip-doctor` is
given. See [installation.md](installation.md) for CPU-only, all-backend,
offline and manual options.

## 2. Check the environment

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

`doctor` fails closed when the runtime it finds cannot actually run the
backend. Fix any failing check before continuing; see
[troubleshooting.md](troubleshooting.md).

## 3. Provide a model and a structure

Put a small periodic structure (CIF/POSCAR/XYZ) and a MACE checkpoint
somewhere convenient:

```bash
ls structure.cif
ls mace-omat-0-medium.model   # SHA-256 pinned in validation/science/model_manifest.json
```

mliport never downloads or bundles model weights implicitly; supply the path
you obtained from the upstream project.

## 4. Single point

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --output runs/mace-sp
```

A successful run prints the energy/forces summary and writes
`runs/mace-sp/` with `OUTCAR`, `OSZICAR`, `resolved_config.json`, the result
JSON and `arrays.npz` (see [outputs.md](outputs.md)).

## 5. Short relaxation

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --fmax 0.05 --max-steps 50 \
  --output runs/mace-opt
```

Check `runs/mace-opt/result.json` for `converged` and the final
`CONTCAR`/XDATCAR. `converged: false` is reported honestly and is not a
successful relaxation.

## 6. Inspect the result

```bash
python - <<'PY'
import json
result = json.load(open("runs/mace-opt/result.json"))
print(result["status"], result["converged"], result.get("energy"))
PY
```

The full provenance — model identity, resolved configuration sources,
requested vs actual device and software commit — is in
`resolved_config.json` and the result JSON.

## 7. Equivalent first runs for the other backends

Install the backend you want (`--engines dpa`, `grace` or `uma`) and use its
environment/launcher:

```bash
# DPA (multi-task checkpoints need the intended branch)
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --output runs/dpa-sp

# GRACE (MODEL_PATH is the SavedModel directory)
.venv-grace/bin/mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA (task family is explicit)
.venv/bin/mliport sp structure.cif --model uma.pt \
  --model-type uma --task omat --output runs/uma-sp
```

There is no default backend: omitting `--model-type` (and any model alias)
is a fatal configuration error.

## Next steps

- Choose a backend in detail: [models.md](models.md)
- Configure runs and strictness: [configuration.md](configuration.md)
- MD, NEB and batch: [workflows/](workflows/single-point.md)
- Analyse a trajectory: [analysis/overview.md](analysis/overview.md)
- Know the evidence behind the claims: [validation.md](validation.md)

## The LGPS acceptance structure

The repository ships a real periodic LGPS (Li10GeP2S12) primitive cell with
50 atoms:

```text
examples/structures/li10gep2s12_primitive.vasp
```

It is derived from a 2x2x2 LGPS supercell; `examples/structures/README.md`
records the exact provenance and hashes. The cell is small enough for CPU
smoke tests and is the structure used by the repository acceptance suite.

A complete first run on it looks like this:

```bash
# tested example
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda --output runs/lgps-mace-sp
```

Expected outcome: `runs/lgps-mace-sp/mliport_results.json` with a finite
`energy` (about -216 eV for the MACE-OMAT-0 checkpoint), finite forces, the
50-atom formula `Ge2Li20P4S24`, and `actual_device_type: cuda` when a GPU was
requested.
