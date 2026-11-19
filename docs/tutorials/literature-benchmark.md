# Literature benchmark

This tutorial shows how to benchmark your MLIP-computed ionic conductivity
against published experimental or DFT data. The workflow is:

1. Run NVT-MD at several temperatures
2. Extract diffusion coefficients with `transport`
3. Fit an Arrhenius relation to get activation energy *E*_a
4. Compare *D*(T) and *E*_a with published values

## Why benchmark

MLIPs can reproduce the PES of their training set very accurately, but ionic
conductivities are sensitive to small errors in barrier heights. Comparing
against established experimental values for a standard material (LGPS, LLZO,
or LATP) before using a model for research gives you confidence in the
force field's reliability for transport.

## Step 1 — Plan your temperature grid

Choose temperatures spaced evenly in 1/T (not in T), which gives roughly
equal spacing on an Arrhenius plot. A four-point grid from 400 K to 1000 K is
usually sufficient; use five or more for a publication-quality fit.

| Run | Temperature |
|---|---|
| `runs/md-400K` | 400 K |
| `runs/md-600K` | 600 K |
| `runs/md-800K` | 800 K |
| `runs/md-1000K` | 1000 K |

Each run should be long enough for the MSD to reach the diffusive (linear)
regime. For LGPS at 400 K, 1–2 ns is typically needed; at 1000 K, 100–200 ps
may suffice.

## Step 2 — Run MD at each temperature

```bash
for T in 400 600 800 1000; do
  ./bin/mliport-mace md \
    --structure runs/lgps-opt/CONTCAR \
    --model-type mace \
    --model-file path/to/mace-checkpoint.pt \
    --ensemble NVT \
    --temp $T \
    --timestep 1.0 \
    --steps 1000000 \
    --equilibration-steps 50000 \
    --thermostat BUSSI \
    --output runs/md-${T}K
done
```

## Step 3 — Extract diffusion coefficients

Run `transport` for each temperature. Record *D* from `summary.json`.

```bash
for T in 400 600 800 1000; do
  ./bin/mliport-mace analyze runs/md-${T}K transport \
    --mobile Li \
    --frame-interval-fs 1.0 \
    --drift-reference nonmobile \
    --drift-mode mass_weighted_com \
    --temperature $T
done
```

Collect the diffusion coefficients from each run's `summary.json`:

```bash
for T in 400 600 800 1000; do
  python -c "
import json, glob
path = sorted(glob.glob('runs/md-${T}K/analysis/transport/*/summary.json'))[-1]
d = json.load(open(path))
print(f'T={T}K  D={d[\"diffusivity_m2_s\"]:.3e}  sigma={d.get(\"conductivity_S_cm\",\"N/A\")}')
"
done
```

## Step 4 — Arrhenius fit

The `arrhenius` task takes diffusivity values explicitly as flags. Pass each
(temperature, diffusivity) pair in matching order:

```bash
mliport analyze runs/md-1000K arrhenius \
  --temperature 400 --temperature 600 \
  --temperature 800 --temperature 1000 \
  --diffusivity 1.5e-14 --diffusivity 3.2e-13 \
  --diffusivity 2.8e-12 --diffusivity 1.1e-11 \
  --extrapolate-temperature 300
```

!!! note "Flag syntax"
    Each `--temperature` and `--diffusivity` flag is repeated once per data
    point, in the same order. The *i*-th temperature pairs with the *i*-th
    diffusivity. The run you name is only the analysis output location; the
    numbers come from the flags.

The source run (`runs/md-1000K` above) is just where the output goes — it
does not read trajectory data. Any run directory works; using the highest
temperature run is conventional.

Output: `summary.json` with *E*_a (eV), *D*₀ (m² s⁻¹), and *D*(300 K) if
`--extrapolate-temperature 300` was set.

## Step 5 — Compare with literature

LGPS reference values from selected publications:

| Quantity | Experimental | MLIP benchmark target |
|---|---|---|
| σ (300 K) | ~12 mS cm⁻¹ (Kamaya et al., 2011) | within 50 % |
| *E*_a | 0.21–0.24 eV | ± 0.05 eV |
| *D*_Li (600 K) | ~10⁻¹² m² s⁻¹ (simulation) | — |

If your *E*_a is 0.20–0.26 eV and your room-temperature conductivity after
Nernst–Einstein extrapolation is in the 5–20 mS cm⁻¹ range, the MACE model
reproduces the experimental transport behaviour of LGPS well.

Significant deviations (> factor of 10 in *D*, > 0.1 eV in *E*_a) suggest
the MLIP is not well-calibrated for LGPS transport and results should be
treated with caution.

## Checklist before reporting

- [ ] At least 3 temperatures spanning ≥ 2× in temperature
- [ ] Each MSD clearly linear for at least one decade in time
- [ ] Thermodynamic check (`thermo`) shows no energy drift and stable *T*
- [ ] RMSD check shows the crystal did not melt at any temperature
- [ ] *E*_a reported with its statistical uncertainty from the kinisi fit
- [ ] Thermostat specified (results depend on thermostat type)
- [ ] Comparison with at least one independent benchmark

## See also

- [LGPS workflow tutorial](lgps-workflow.md) — single-temperature workflow
- [Transport analysis](../analysis/transport.md)
- [Arrhenius analysis](../analysis/arrhenius.md)
- [Validation](../validation.md)
