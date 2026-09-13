# Single Point Calculation Settings
# mliport - VASP-style input
# Supports engines: UMA (default), MACE, DPA, GRACE

# Calculation Type
CALC_TYPE = SP
TASK = omat

# Model Settings
MODEL_PATH = uma-s-1.pt
MODEL_TYPE = uma          # uma (default), mace, dpa, grace
DEVICE = cpu
INFERENCE_MODE = default  # UMA only; ignored by other engines

# Output Control
WRITE_FORCES = .TRUE.
WRITE_STRESS = .TRUE.
OUTPUT_FORMAT = VASP

# Notes:
# - TASK: omat (materials, UMA), omol (molecules, UMA), oc20 (catalysis, UMA),
#   bulk (MACE/DPA/GRACE), molecule (MACE/DPA/GRACE)
# - MODEL_TYPE=mace: MODEL_PATH should point to a .model file, add HEAD=default
# - MODEL_TYPE=dpa:  MODEL_PATH should point to a .pt/.pth file, add HEAD=<branch>
# - MODEL_TYPE=grace: MODEL_PATH should point to a SavedModel directory
# - DEVICE: cpu or cuda[:N]
# - INFERENCE_MODE: default (general) or turbo (fast, UMA only)
