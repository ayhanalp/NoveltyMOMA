#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Resolve repo root robustly
# -----------------------------
# This gets the directory where *this script* lives,
# then assumes the repo root is the parent (adjust if needed)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# -----------------------------
# (Optional) activate conda env
# -----------------------------
# Uncomment and edit if needed
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate env

# -----------------------------
# Experiment parameters
# -----------------------------
ALG_NAME="nsga2"
DOMAIN="rover"
DATA_DIR="$REPO_ROOT/data"

ALG_CONFIG="config/generic/generic_DMOConfig.yaml"
ENV_CONFIG="config/generic/generic_MORoverEnvConfig.yaml"

#SEED=2024
SEED=0
#LABEL="4ag_2poi_0p1b"
TRAJ_WRITE_FREQ=50
BETA=0.15

# -----------------------------
# Sanity checks
# -----------------------------
[[ -d "$DATA_DIR" ]] || { echo "Data dir not found: $DATA_DIR"; exit 1; }
[[ -f "$REPO_ROOT/$ALG_CONFIG" ]] || { echo "Alg config not found: $ALG_CONFIG"; exit 1; }
[[ -f "$REPO_ROOT/$ENV_CONFIG" ]] || { echo "Env config not found: $ENV_CONFIG"; exit 1; }

# -----------------------------
# Run experiment
# -----------------------------
PYTHONUNBUFFERED=1 python "$REPO_ROOT/main.py" \
    "$ALG_NAME" \
    "$DOMAIN" \
    "$DATA_DIR" \
    "$REPO_ROOT/$ALG_CONFIG" \
    "$REPO_ROOT/$ENV_CONFIG" \
    "$SEED" \
    "$TRAJ_WRITE_FREQ" \
    "$BETA"
