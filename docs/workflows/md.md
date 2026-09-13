# Molecular dynamics (MD)

**Purpose.** Run NVE or NVT molecular dynamics with the selected backend and
record a reproducible trajectory for analysis (MSD/transport/VACF).

**Minimal command.**

```bash
.venv-mace/bin/mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

**Important options.**

| Option | Meaning |
|---|---|
| `--ensemble` | `NVT` (default) or `NVE` |
| `--temp` | target temperature in K (NVT) |
| `--timestep` | integration step in fs (default 1.0) |
| `--steps` | production steps (default 1000) |
| `--equilibration-steps` | same-ensemble equilibration before production (default 0) |
| `--thermostat` | `LANGEVIN` (default), `BUSSI`, `NHC` |
| `--friction`, `--bussi-tau`, `--nhc-tdamp`, `--nhc-tchain`, `--nhc-tloop` | thermostat parameters |
| `--save-interval` | trajectory save interval in steps (default 10) |
| `--seed` | random seed; auto-generated and recorded when omitted |
| `--velocity-policy`, `--com-policy` | velocity/centre-of-mass policies |
| `--pre-relax`, `--pre-relax-steps`, `--pre-relax-fmax` | pre-relaxation before production (default on for NVT, off for NVE) |
| `--fmax-abort` | abort threshold for unphysical forces (default 20 eV/Å) |
| `--write-outcar`, `--write-xdatcar`, `--write-trajectory` | output selection |

**Inputs.** A periodic structure, a model and an explicit backend. MD on a
molecule requires a molecular task (`omol`/`molecule`) and sensible
charge/spin.

**Outputs.** Canonical ASE trajectory (`trajectory.traj`), `XDATCAR`,
`OUTCAR`/`OSZICAR`, `resolved_config.json` and result JSON with
energy/temperature traces. The seed, thermostat parameters, timestep and
save interval are part of the record.

**Failure semantics.** NaN/exploding forces above `--fmax-abort` abort the
run instead of writing corrupt frames. Output-write failures abort rather
than silently dropping frames. A missing model/structure or an unsupported
option combination exits before dynamics start.

**Scientific caveats — timestep is not the save interval.**

- `--timestep` is the integration step; `--save-interval` is how often a
  frame is written. A 1 fs timestep with `--save-interval 10` produces frames
  every 10 fs; transport analysis needs the **saved** spacing, not the
  integration step.
- Equilibration frames must not be mixed into production statistics; use
  `--equilibration-steps` and analyse only the production segment.
- NVT thermostats change the dynamics (LANGEVIN is stochastic, BUSSI/NHC are
  deterministic); transport coefficients depend on that choice.
- `--com-policy` matters for drift; NHC requires an explicit policy.
- Constraints/fixed atoms are applied through the calculator/runners; state
  them when reporting diffusion.
- NVE and NVT are different ensembles: never average them together.
- Restart from a checkpoint with `--resume`/checkpoint commands; a restarted
  run keeps the original seed/provenance.

**Example (production-quality segmenting).**

```bash
.venv/bin/mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC --com-policy none \
  --output runs/md-800K
```
