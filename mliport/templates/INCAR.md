# Molecular Dynamics Settings
# mliport - VASP-style input
# Select one engine explicitly: UMA, MACE, DPA or GRACE (fairchem is an alias for uma)

# Calculation Type
CALC_TYPE = MD
TASK = omat

# Model Settings
MODEL_PATH = uma-s-1.pt
MODEL_TYPE = REQUIRED     # choose: uma | mace | dpa | grace
DEVICE = cuda              # MD benefits from GPU
INFERENCE_MODE = turbo     # Turbo mode for better performance (UMA only)

# MD Settings
MD_ENSEMBLE = NVT          # NVT or NVE (Velocity Verlet)
TEMPERATURE = 300.0        # Temperature in Kelvin
TIMESTEP = 1.0             # Time step in femtoseconds
STEPS = 10000              # Number of production MD steps
EQUILIBRATION_STEPS = 0    # Same-ensemble steps before production
THERMOSTAT = LANGEVIN      # LANGEVIN, BUSSI, or NHC (NVT only)
FRICTION = 0.001           # Langevin friction (1/fs)
BUSSI_TAU = 1000.0         # Bussi/CSVR coupling time (fs)
NHC_TDAMP = 100.0          # Nose-Hoover-chain damping time (fs)
NHC_TCHAIN = 3             # Nose-Hoover chain length
NHC_TLOOP = 1              # Thermostat integration substeps
SAVE_INTERVAL = 10         # Save trajectory every N steps

# Pre-relaxation before MD
# PRE_RELAX = .TRUE.        # default: .TRUE. for NVT, .FALSE. for NVE
# PRE_RELAX_STEPS = 50
# PRE_RELAX_FMAX = 0.1

# Notes:
# - TASK: omat (materials, UMA), omol (molecules, UMA), oc20 (catalysis, UMA),
#   bulk (MACE/DPA/GRACE), molecule (MACE/DPA/GRACE)
# - MODEL_TYPE=mace: add HEAD=default
# - MODEL_TYPE=dpa:  add HEAD=<branch>
# - MODEL_TYPE=grace: MODEL_PATH is a SavedModel directory
# - DEVICE: cpu or cuda[:N]
