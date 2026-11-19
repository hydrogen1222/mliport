# Trajectory validation

The `validate` task inspects a trajectory directory and reports whether the
data meets the requirements for each downstream analysis. Run it first to
catch problems before they produce confusing errors.

```bash
mliport analyze runs/md validate
```

The output is a JSON file (`validate.json`) containing every check result, so
you can parse it programmatically if needed.

## What it checks

### Structural integrity

| Check | Detail |
|---|---|
| Finite positions | All atomic coordinates are finite (no NaN / Inf) |
| Finite cells | All lattice vectors are finite |
| Cell rank | Every frame has a non-singular (rank-3) cell |
| Fixed cell | All frames share the same cell — required for MSD and transport |
| PBC | Periodic boundary conditions in all three dimensions |

### Time axis

| Check | Detail |
|---|---|
| Available | An explicit or declared time axis exists |
| Finite | Time intervals contain no NaN or Inf |
| Strictly increasing | Timestamps never go backwards |
| Uniform | Constant frame spacing — required for FFT MSD, VACF, and kinisi transport |

If the time axis is uniform, the report includes the frame interval in
femtoseconds and the Nyquist limits (THz and cm⁻¹) for spectral analysis.

### Production and run status

- Frames are split into **equilibration** and **production** phases.
  Only production frames are used by MSD, transport, and VACF.
- The source run status (`completed`, `aborted`, etc.) is reported.
  Incomplete runs produce a warning and block MSD/transport.

### Velocities

Whether velocities are stored in the trajectory.  VACF and vibrational
spectrum require them; other tasks do not.

### Position convention

Whether positions are recorded as `wrapped` or `unwrapped`.  Tasks that
compute displacements (MSD, transport) need a known convention so they can
reconstruct continuous atomic paths.

## Eligibility flags

The report contains four Boolean flags summarising which tasks the trajectory
supports:

| Flag | Required conditions |
|---|---|
| `eligible_for_rdf` | Finite positions/cells, rank-3 cell, 3-D PBC, ≥ 1 frame |
| `eligible_for_msd` | All of the above, plus: fixed cell, uniform time, known position convention, ≥ 4 production frames, complete run |
| `eligible_for_transport` | Same as MSD (transport inherits the MSD gate) |
| `eligible_for_vacf` | Finite positions/cells, rank-3 cell, stored velocities, uniform time, ≥ 4 frames |

If a task is **not** eligible, the `errors` list explains exactly why. The
`warnings` list captures non-fatal issues such as nonuniform sampling or
unverified PBC.

## Example output (abbreviated)

```json
{
  "n_frames": 5001,
  "n_atoms": 64,
  "duration_ps": 50.0,
  "three_dimensional_pbc": true,
  "fixed_cell": true,
  "velocities_stored": false,
  "positions_convention": "wrapped",
  "run_status": "completed",
  "production_frames": 5001,
  "eligible_for_rdf": true,
  "eligible_for_msd": true,
  "eligible_for_transport": true,
  "eligible_for_vacf": false,
  "errors": [],
  "warnings": []
}
```

Here `eligible_for_vacf` is `false` because no velocities were saved. All
other tasks are eligible.

## Tips

- Run `validate` after every MD run before attempting analysis.
- If `eligible_for_msd` is `false`, check the `errors` array — common causes
  are variable cells (NPT trajectories analysed with structure tasks that
  require fixed cells), nonuniform time sampling, and incomplete runs.
- For external trajectories (not produced by mliport), the position
  convention may be `unknown`, which blocks MSD/transport. Use the lossless
  ASE trajectory format for reliable convention tracking.
